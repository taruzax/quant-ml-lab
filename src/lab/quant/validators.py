import polars as pl
from typing import Any

# pyrefly: ignore [missing-import]
from lab.core.config import PipelineConfig

# pyrefly: ignore [missing-import]
from lab.core.schemas import CANONICAL_BAR_COLUMNS, PRICE_COLUMNS, REQUIRED_DTYPES, REQUIRED_OHLCV_COLUMNS
from lab.quant.timing import as_utc, normalize_explicit_bar_times


class DataValidationError(Exception):
    """Raised when data fails pipeline quality checks."""


def validate_schema(df, expected_columns=None, expected_dtypes=None):
    if expected_columns is None:
        expected_columns = REQUIRED_OHLCV_COLUMNS

    if expected_dtypes is None:
        expected_dtypes = REQUIRED_DTYPES

    if df.is_empty():
        raise DataValidationError("DataFrame is empty. Cannot validate.")

    actual_cols = set(df.columns)
    missing = [c for c in expected_columns if c not in actual_cols]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}. Have: {sorted(actual_cols)}")

    dtype_errors = []
    for col, ex_dtype in expected_dtypes.items():
        if col in df.columns:
            actual_dtype = df[col].dtype
            if actual_dtype != ex_dtype:
                dtype_errors.append(f"  {col}: expected {ex_dtype}, got {actual_dtype}")
    if dtype_errors:
        raise DataValidationError("Dtype mismatches:\n" + "\n".join(dtype_errors))
    return df


def validate_nulls(df, tolerance, columns=None):
    if df.is_empty():
        raise DataValidationError("DataFrame is empty. Cannot validate.")
    check_cols = columns if columns is not None else df.columns
    n_rows = df.height

    for col in check_cols:
        if col not in df.columns:
            continue
        null_count = df[col].null_count()
        if df[col].dtype in (pl.Float32, pl.Float64):
            null_count += df.filter(pl.col(col).is_nan()).height
        null_ratio = null_count / n_rows
        if null_ratio > tolerance:
            raise DataValidationError(
                f"Column '{col}' has {null_count}/{n_rows} nulls/NaNs ({null_ratio:.4%}), exceeds tolerance {tolerance:.4%}"
            )
    return df


def validate_prices(df, price_columns=None, min_price=0.0):
    if price_columns is None:
        price_columns = PRICE_COLUMNS

    for col in price_columns:
        if col not in df.columns:
            continue
        violations = df.filter(pl.col(col).is_not_null() & (pl.col(col) < min_price))
        if violations.height > 0:
            sample = violations.head(5).select("timestamp", "ticker", col)
            raise DataValidationError(f"Column '{col}' has {violations.height} values below {min_price}.\nSample:\n{sample}")
    return df


def validate_monotonic_timestamps(df, date_col="timestamp", group_col="ticker"):
    if date_col not in df.columns:
        raise DataValidationError(f"Date column '{date_col}' not found.")

    violations = df.with_columns(prev_date=pl.col(date_col).shift(1).over(group_col)).filter(
        pl.col("prev_date").is_not_null() & (pl.col(date_col) < pl.col("prev_date"))
    )

    if violations.height > 0:
        sample = violations.head(3).select(group_col, "prev_date", date_col)
        raise DataValidationError(f"Non-monotonic dates detected in {violations.height} rows.\nSample:\n{sample}")

    return df


def run_all_validations(df: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Chain all validators using config values. Returns DataFrame if all pass."""
    df = validate_schema(df)
    df = validate_nulls(df, tolerance=config.null_tolerance, columns=PRICE_COLUMNS)
    df = validate_prices(df, min_price=config.min_price)
    df = validate_monotonic_timestamps(df)
    return df


def _timezone_aware_dtype(dtype: pl.DataType) -> bool:
    return isinstance(dtype, pl.Datetime) and dtype.time_zone is not None


def validate_canonical_bars(
    df: pl.DataFrame,
    *,
    calendar: str,
    timeframe,
    require_timezone: bool = True,
) -> pl.DataFrame:
    """Validate normalized OHLCV bars with explicit open and close times."""
    if df.is_empty():
        raise DataValidationError("Canonical bar data is empty")
    missing = sorted(set(CANONICAL_BAR_COLUMNS) - set(df.columns))
    if missing:
        raise DataValidationError(f"Canonical bars are missing columns: {missing}")
    for column in ("bar_open_time", "bar_close_time", "timestamp"):
        if not isinstance(df[column].dtype, pl.Datetime):
            raise DataValidationError(f"{column} must use a timezone-aware datetime dtype")
        if require_timezone and not _timezone_aware_dtype(df[column].dtype):
            raise DataValidationError(f"{column} must be timezone-aware UTC")
    if df["timestamp"].dtype.time_zone != "UTC":
        raise DataValidationError("Canonical timestamps must use UTC")
    if df.filter(pl.col("timestamp") != pl.col("bar_close_time")).height:
        raise DataValidationError("timestamp must equal bar_close_time")
    if df.filter(pl.any_horizontal(*[pl.col(column).is_null() for column in CANONICAL_BAR_COLUMNS])).height:
        raise DataValidationError("Canonical bars cannot contain null contract fields")
    for column in PRICE_COLUMNS:
        invalid = df.filter(~pl.col(column).is_finite() | (pl.col(column) <= 0))
        if invalid.height:
            raise DataValidationError(f"{column} must contain finite positive prices")
    if df.filter(
        (pl.col("high") < pl.max_horizontal("open", "close"))
        | (pl.col("low") > pl.min_horizontal("open", "close"))
        | (pl.col("high") < pl.col("low"))
    ).height:
        raise DataValidationError("OHLC values violate high/low ordering")
    if df.filter(~pl.col("volume").is_finite() | (pl.col("volume") < 0)).height:
        raise DataValidationError("volume must be finite and nonnegative")
    ordered = df.sort(["ticker", "bar_open_time"])
    if ordered.filter(
        pl.col("bar_open_time").shift(-1).over("ticker").is_not_null()
        & (pl.col("bar_open_time").shift(-1).over("ticker") < pl.col("bar_close_time"))
    ).height:
        raise DataValidationError("Bars overlap within a ticker")
    if ordered.filter(
        pl.col("raw_bar_index").shift(-1).over("ticker").is_not_null()
        & (pl.col("raw_bar_index").shift(-1).over("ticker") <= pl.col("raw_bar_index"))
    ).height:
        raise DataValidationError("raw_bar_index must increase within each ticker")
    return df


def validate_explicit_bar_times(df: pl.DataFrame, *, calendar: str, timeframe) -> pl.DataFrame:
    """Validate each explicit interval against the declared exchange calendar."""
    for row in df.select(["bar_open_time", "bar_close_time"]).iter_rows():
        try:
            normalize_explicit_bar_times(row[0], row[1], calendar, timeframe)
        except ValueError as exc:
            raise DataValidationError(str(exc)) from exc
    return df


def compare_bar_coverage(
    expected: pl.DataFrame,
    observed: pl.DataFrame,
    *,
    tickers: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Classify missing, unexpected, duplicate, and valid calendar bars."""
    expected_columns = {"session_id", "bar_open_time", "bar_close_time"}
    if not expected_columns.issubset(expected.columns):
        raise DataValidationError(f"Expected coverage keys are missing: {sorted(expected_columns - set(expected.columns))}")
    observed_columns = expected_columns | {"ticker"}
    if not observed_columns.issubset(observed.columns):
        raise DataValidationError(f"Observed coverage keys are missing: {sorted(observed_columns - set(observed.columns))}")
    selected_tickers = tuple(tickers or sorted(observed["ticker"].unique().to_list()))
    expected_keys = expected.select(sorted(expected_columns))
    expected_with_ticker = pl.concat(
        [expected_keys.with_columns(pl.lit(ticker).alias("ticker")) for ticker in selected_tickers],
        how="vertical",
    ) if selected_tickers else expected_keys.with_columns(pl.lit(None, dtype=pl.Utf8).alias("ticker"))
    observed_keys = observed.select(sorted(observed_columns))
    duplicates = observed_keys.group_by(sorted(observed_columns)).len().filter(pl.col("len") > 1)
    distinct_observed = observed_keys.unique(subset=sorted(observed_columns))
    missing = expected_with_ticker.join(distinct_observed, on=sorted(observed_columns), how="anti")
    unexpected = distinct_observed.join(expected_with_ticker, on=sorted(observed_columns), how="anti")
    valid = expected_with_ticker.join(distinct_observed, on=sorted(observed_columns), how="inner")
    return {
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "valid": valid,
        "missing_count": missing.height,
        "unexpected_count": unexpected.height,
        "duplicate_count": duplicates.height,
        "valid_count": valid.height,
    }
