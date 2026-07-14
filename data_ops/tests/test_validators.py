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


def test_valid_data_passes(single_ticker_df):
    df = single_ticker_df
    result = validate_schema(df)
    assert result.shape == df.shape


def test_missing_column_raises(single_ticker_df):
    df = single_ticker_df.drop("close")
    with pytest.raises(DataValidationError, match="Missing columns.*close"):
        validate_schema(df)


def test_null_above_tolerance_raises(single_ticker_df):
    df = single_ticker_df
    # Inject 1 null into 300 rows (1/300 = 0.33% > 0.1% tolerance)
    mask = [None if i == 0 else df["close"][i] for i in range(300)]
    df = df.with_columns(pl.Series("close", mask))
    with pytest.raises(DataValidationError, match="close.*nulls"):
        validate_nulls(df, tolerance=0.001)


def test_null_below_tolerance_passes(single_ticker_df):
    df = single_ticker_df
    # Inject 1 null into 300 rows (1/300 = 0.33%). With tolerance 1%, this should pass.
    mask = [None if i == 0 else df["close"][i] for i in range(300)]
    df = df.with_columns(pl.Series("close", mask))
    result = validate_nulls(df, tolerance=0.01)
    assert result.shape == df.shape


def test_negative_price_raises(single_ticker_df):
    df = single_ticker_df
    df = df.with_columns(
        pl.when(pl.col("close") == pl.col("close").first())
        .then(pl.lit(-1.0))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    with pytest.raises(DataValidationError, match="close.*below"):
        validate_prices(df, min_price=0.0)


def test_non_monotonic_dates_raises(single_ticker_df):
    df = single_ticker_df.head(10)
    # Swap two dates within the same ticker to break monotonicity
    dates = df["date"].to_list() 
    dates[2], dates[5] = dates[5], dates[2]
    df = df.with_columns(pl.Series("date", dates))
    with pytest.raises(DataValidationError, match="Non-monotonic"):
        validate_monotonic_dates(df)


def test_empty_dataframe_raises(single_ticker_df):
    df = single_ticker_df.head(0)
    with pytest.raises(DataValidationError, match="empty"):
        validate_schema(df)


def test_run_all_validations_chains(single_ticker_df):
    df = single_ticker_df.drop("close")
    config = PipelineConfig()
    with pytest.raises(DataValidationError):
        run_all_validations(df, config)