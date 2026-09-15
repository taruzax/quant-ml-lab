from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

# pyrefly: ignore [missing-import]
from lab.core.config import PipelineConfig

# pyrefly: ignore [missing-import]
from lab.data.validators import (
    DataValidationError,
    run_all_validations,
    validate_monotonic_timestamps,
    validate_nulls,
    validate_prices,
    validate_schema,
)
from lab.quant.validators import validate_canonical_bars


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
        pl.when(pl.col("close") == pl.col("close").first()).then(pl.lit(-1.0)).otherwise(pl.col("close")).alias("close")
    )
    with pytest.raises(DataValidationError, match="close.*below"):
        validate_prices(df, min_price=0.0)


def test_non_monotonic_dates_raises(single_ticker_df):
    df = single_ticker_df.head(10)
    # Swap two dates within the same ticker to break monotonicity
    dates = df["timestamp"].to_list()
    dates[2], dates[5] = dates[5], dates[2]
    df = df.with_columns(pl.Series("timestamp", dates))
    with pytest.raises(DataValidationError, match="Non-monotonic"):
        validate_monotonic_timestamps(df)


def test_empty_dataframe_raises(single_ticker_df):
    df = single_ticker_df.head(0)
    with pytest.raises(DataValidationError, match="empty"):
        validate_schema(df)


def test_run_all_validations_chains(single_ticker_df):
    df = single_ticker_df.drop("close")
    config = PipelineConfig()
    with pytest.raises(DataValidationError):
        run_all_validations(df, config)


def test_canonical_bars_accept_valid_contract():
    opened = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    closed = opened + timedelta(hours=1)
    result = validate_canonical_bars(
        pl.DataFrame(
            {
                "ticker": ["AAA"],
                "raw_bar_index": [0],
                "bar_open_time": [opened],
                "bar_close_time": [closed],
                "timestamp": [closed],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
                "session_id": ["2024-01-02"],
            }
        ),
        calendar="XNYS",
        timeframe="1h",
    )
    assert result.height == 1


def test_canonical_bars_reject_bad_ohlc():
    opened = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    closed = opened + timedelta(hours=1)
    invalid = pl.DataFrame(
        {
            "ticker": ["AAA"],
            "raw_bar_index": [0],
            "bar_open_time": [opened],
            "bar_close_time": [closed],
            "timestamp": [closed],
            "open": [100.0],
            "high": [98.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
            "session_id": ["2024-01-02"],
        }
    )
    with pytest.raises(DataValidationError, match="OHLC"):
        validate_canonical_bars(invalid, calendar="XNYS", timeframe="1h")
