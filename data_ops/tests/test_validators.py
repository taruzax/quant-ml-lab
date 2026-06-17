import polars as pl
import pytest
import numpy as np

# pyrefly: ignore [missing-import]
from src.data.validators import (
    DataValidationError,
    validate_schema,
    validate_nulls,
    validate_prices,
    validate_monotonic_dates,
    run_all_validations,
)
# pyrefly: ignore [missing-import]
from src.core.config import PipelineConfig


def _make_clean_ohlcv(n: int = 200, ticker: str = "TEST") -> pl.DataFrame:
    """Create a clean OHLCV DataFrame that passes all validations."""
    np.random.seed(42)
    dates = pl.date_range(
        start=pl.date(2024, 1, 1),
        end=pl.date(2024, 1, 1) + pl.duration(days=n - 1),
        interval="1d",
        eager=True,
    )
    close = np.abs(np.cumsum(np.random.normal(100, 1, n)))
    close = np.maximum(close, 1.0)  # ensure positive
    return pl.DataFrame({
        "date": dates,
        "ticker": [ticker] * n,
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.uniform(1e6, 1e7, n),
    })


def test_valid_data_passes():
    df = _make_clean_ohlcv()
    result = validate_schema(df)
    assert result.shape == df.shape


def test_missing_column_raises():
    df = _make_clean_ohlcv().drop("close")
    with pytest.raises(DataValidationError, match="Missing columns.*close"):
        validate_schema(df)


def test_null_above_tolerance_raises():
    df = _make_clean_ohlcv(n=1000)
    # Inject 1% nulls (10 out of 1000 rows)
    mask = [None if i < 10 else df["close"][i] for i in range(1000)]
    df = df.with_columns(pl.Series("close", mask))
    with pytest.raises(DataValidationError, match="close.*nulls"):
        validate_nulls(df, tolerance=0.001)


def test_null_below_tolerance_passes():
    df = _make_clean_ohlcv(n=10000)
    # Inject 0.005% nulls (well below 0.1% threshold)
    mask = [None if i == 0 else df["close"][i] for i in range(10000)]
    df = df.with_columns(pl.Series("close", mask))
    result = validate_nulls(df, tolerance=0.001)
    assert result.shape == df.shape


def test_negative_price_raises():
    df = _make_clean_ohlcv()
    df = df.with_columns(
        pl.when(pl.col("close") == pl.col("close").first())
        .then(pl.lit(-1.0))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    with pytest.raises(DataValidationError, match="close.*below"):
        validate_prices(df, min_price=0.0)


def test_non_monotonic_dates_raises():
    df = _make_clean_ohlcv(n=10)
    # Swap two dates within the same ticker to break monotonicity
    dates = df["date"].to_list() 
    dates[2], dates[5] = dates[5], dates[2]
    df = df.with_columns(pl.Series("date", dates))
    with pytest.raises(DataValidationError, match="Non-monotonic"):
        validate_monotonic_dates(df)


def test_empty_dataframe_raises():
    df = _make_clean_ohlcv().head(0)
    with pytest.raises(DataValidationError, match="empty"):
        validate_schema(df)


def test_run_all_validations_chains():
    df = _make_clean_ohlcv().drop("close")
    config = PipelineConfig()
    with pytest.raises(DataValidationError):
        run_all_validations(df, config)