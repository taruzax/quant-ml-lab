import numpy as np
import polars as pl
import pytest
import datetime
# pyrefly: ignore [missing-import]
from src.data.features import (
    calculate_dollar_volume,
    calculate_returns,
    calculate_lagged_features,
    calculate_forward_targets,
    create_time_cycles,
    create_sector_dummies,
)
# pyrefly: ignore [missing-import]
from src.core.schemas import return_col, lagged_col, target_col

def test_dollar_volume_columns_exist(single_ticker_df):
    df = single_ticker_df
    result = calculate_dollar_volume(df)
    for col in ["dollar_vol", "dollar_vol_1m", "dollar_vol_rank"]:
        assert col in result.columns, f"Missing column: {col}"


def test_returns_correct_values():
    """Manually compute 1-day return for known prices and verify match."""
    df = pl.DataFrame({
        "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 3), interval="1d", eager=True),
        "ticker": ["T"] * 3,
        "close": [100.0, 110.0, 105.0],
    })
    result = calculate_returns(df, lags=[1], clip_quantile=0.0)
    # 1-day return: 110/100 - 1 = 0.10, 105/110 - 1 = -0.0454...
    returns = result[return_col(1)].to_list()
    assert returns[0] is None  # first row has no prior
    assert abs(returns[1] - 0.10) < 0.01


def test_lagged_features_reference_correct_column(single_ticker_df):
    """CRITICAL: Verify return_1d_lag1 is actually a shift of return_1d, NOT return_63d."""
    df = single_ticker_df
    df = calculate_returns(df, lags=[1, 5, 10, 21, 42, 63])
    result = calculate_lagged_features(df, return_lags=[1], lookback_periods=[1])

    # return_1d_lag1 should be return_1d shifted by 1
    expected = df[return_col(1)].shift(1)
    actual = result[lagged_col(1, 1)]

    # Compare non-null values
    mask = expected.is_not_null() & actual.is_not_null()
    np.testing.assert_allclose(
        actual.filter(mask).to_numpy(),
        expected.filter(mask).to_numpy(),
        rtol=1e-6,
        err_msg="return_1d_lag1 does not match shifted return_1d! Bug not fixed.",
    )


def test_forward_targets_shift_correctly(single_ticker_df):
    """target_1d should be return_1d shifted by -1 (one step into the future)."""
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
    df = pl.DataFrame({
        "date": [datetime.date(2024, 3, 15), datetime.date(2024, 7, 4)],
        "ticker": ["A", "A"],
    })
    result = create_time_cycles(df)
    assert result["year_scaled"].to_list() == [4, 4]
    assert np.allclose(result["month_sin"].to_list(), [1.0, -0.5], atol=1e-5)




def test_sector_dummies_missing_sector_raises():
    df = pl.DataFrame({"date": [pl.date(2024, 1, 1)], "ticker": ["A"], "year": [2024], "month": [1]})
    with pytest.raises(ValueError, match="sector"):
        create_sector_dummies(df)