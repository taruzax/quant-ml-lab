import polars as pl 
# pyrefly: ignore [missing-import]
from src.core.config import PipelineConfig
# pyrefly: ignore [missing-import]
from src.core.schemas import PRICE_COLUMNS, REQUIRED_OHLCV_COLUMNS, REQUIRED_DTYPES

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
        raise DataValidationError(
            "Dtype mismatches:\n" + "\n".join(dtype_errors)
        )        
    return df

def validate_nulls(df, tolerance, columns=None):
    if df.is_empty():
        raise DataValidationError("DataFrame is empty. Cannot validate.")

    check_cols = columns if columns is not None else df.columns
    n_rows = df.height

    for col in check_cols:
        if col not in check_cols:
            continue
        null_count = df[col].null_count()
        null_ratio = null_count/n_rows
        if null_ratio>tolerance:
            raise DataValidationError(
                f"Column '{col}' has {null_count}/{n_rows} nulls "
                f"({null_ratio:.4%}), exceeds tolerance {tolerance:.4%}"
            )
    return df

def validate_prices(df, price_columns=None, min_price=0.0):
    if price_columns is None:
        price_columns = PRICE_COLUMNS
    
    for col in price_columns:
        if col not in df.columns:
            continue
        violations = df.filter(pl.col(col).is_not_null() &(pl.col(col)< min_price))
        if violations.height>0:
            sample = violations.head(5).select('date', 'ticker', col)
            raise DataValidationError(
                f"Column '{col}' has {violations.height} values below {min_price}.\n"
                f"Sample:\n{sample}"
            )
    return df


def validate_monotonic_dates(df, date_col="date", group_col="ticker"):
    if date_col not in df.columns:
        raise DataValidationError(f"Date column '{date_col}' not found.")
    
    violations = (
        df.with_columns(prev_date = pl.col(date_col).shift(1).over(group_col)
        ).filter(pl.col('prev_date').is_not_null() & (pl.col(date_col)< pl.col('prev_date')))
    )

    if violations.height > 0:
        sample = violations.head(3).select(group_col, "prev_date", date_col)
        raise DataValidationError(
            f"Non-monotonic dates detected in {violations.height} rows.\n"
            f"Sample:\n{sample}"
        )

    return df

def run_all_validations(df: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Chain all validators using config values. Returns DataFrame if all pass."""
    df = validate_schema(df)
    df = validate_nulls(df, tolerance=config.null_tolerance, columns=PRICE_COLUMNS)
    df = validate_prices(df, min_price=config.min_price)
    df = validate_monotonic_dates(df)
    return df