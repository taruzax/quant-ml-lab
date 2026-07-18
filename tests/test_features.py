from datetime import datetime

import numpy as np
import polars as pl
import pytest

# pyrefly: ignore [missing-import]
from lab.core.schemas import lagged_col, return_col, target_col

# pyrefly: ignore [missing-import]
from lab.data.features import (
    calculate_dollar_volume,
    calculate_forward_targets,
    calculate_lagged_features,
    calculate_returns,
    calculate_technical_indicators,
    create_sector_dummies,
    create_time_cycles,
)


def test_dollar_volume_columns_exist(single_ticker_df):
    from lab.core.config import PipelineConfig

    config = PipelineConfig()
    df = single_ticker_df
    result = calculate_dollar_volume(df, config)
    for col in ["dollar_vol", "dollar_vol_1m", "dollar_vol_rank"]:
        assert col in result.columns, f"Missing column: {col}"


def test_returns_correct_values():
    """Manually compute 1-bar return for known prices and verify match."""
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2)],
            "ticker": ["T"] * 3,
            "close": [100.0, 110.0, 105.0],
        }
    )
    result = calculate_returns(df, lags=[1], clip_quantile=0.0)
    # 1-day return: 110/100 - 1 = 0.10, 105/110 - 1 = -0.0454...
    returns = result[return_col(1)].to_list()
    assert returns[0] is None  # first row has no prior
    assert abs(returns[1] - 0.10) < 0.01


def test_lagged_features_reference_correct_column(single_ticker_df):
    """CRITICAL: Verify return_1b_lag1 is actually a shift of return_1b, NOT return_63b."""
    df = single_ticker_df
    df = calculate_returns(df, lags=[1, 5, 10, 21, 42, 63])
    result = calculate_lagged_features(df, return_lags=[1], lookback_periods=[1])

    # return_1b_lag1 should be return_1b shifted by 1
    expected = df[return_col(1)].shift(1)
    actual = result[lagged_col(1, 1)]

    # Compare non-null values
    mask = expected.is_not_null() & actual.is_not_null()
    np.testing.assert_allclose(
        actual.filter(mask).to_numpy(),
        expected.filter(mask).to_numpy(),
        rtol=1e-6,
        err_msg="return_1b_lag1 does not match shifted return_1b! Bug not fixed.",
    )


def test_forward_targets_shift_correctly(single_ticker_df):
    """target_1b should be return_1b shifted by -1 (one step into the future)."""
    df = single_ticker_df
    df = calculate_returns(df, lags=[1])
    result = calculate_forward_targets(df, horizons=[1])

    expected = df[return_col(1)].shift(-1)
    actual = result[target_col(1)]

    mask = expected.is_not_null() & actual.is_not_null()
    np.testing.assert_allclose(
        actual.filter(mask).to_numpy(),
        expected.filter(mask).to_numpy(),
        rtol=1e-6,
    )


def test_time_features_extraction():
    from lab.core.config import PipelineConfig, Timeframe

    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 3, 15, 10), datetime(2024, 7, 4, 14)],
            "ticker": ["A", "A"],
        }
    )

    config = PipelineConfig(timeframe=Timeframe.D1)
    result = create_time_cycles(df, config)
    assert result["year_scaled"].to_list() == [4, 4]
    assert np.allclose(result["month_sin"].to_list(), [1.0, -0.5], atol=1e-5)
    assert "hour_sin" not in result.columns

    config_h1 = PipelineConfig(timeframe=Timeframe.H1)
    result_h1 = create_time_cycles(df, config_h1)
    assert "hour_sin" in result_h1.columns


def test_sector_dummies_missing_sector_raises():
    df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "ticker": ["A"], "year": [2024], "month": [1]})
    with pytest.raises(ValueError, match="sector"):
        create_sector_dummies(df)


def test_technical_indicators_columns_exist(single_ticker_df):
    """Verify that TA-Lib correctly attaches the new indicator columns."""
    df = single_ticker_df
    result = calculate_technical_indicators(df)
    expected_columns = ["ema5", "macd", "cdl2crows", "wclprice"]

    for col in expected_columns:
        assert col in result.columns, f"Missing technical indicator column: {col}"

        # Verify that TA-Lib actually calculated data and didn't just return all nulls
        # (It is completely normal for the first N rows to be null while indicators warm up)
        valid_count = result[col].is_not_null().sum()
        assert valid_count > 0, f"Column {col} was created but contains entirely null values!"

    print("\n\n--- Sample Data (Last 5 rows) ---")
    print(result.select(["timestamp"] + expected_columns).tail(5))
