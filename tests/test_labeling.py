from datetime import datetime, timedelta

import polars as pl

from lab.core.config import PipelineConfig
from lab.data.labeling import calculate_volatility, triple_barrier_label


def _create_synthetic_series(prices: list[float], ticker: str = "AAPL") -> pl.DataFrame:
    n = len(prices)
    start_ts = datetime(2025, 1, 1, 10, 0)
    timestamps = [start_ts + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "ticker": [ticker] * n,
            "close": prices,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "volume": [1000.0] * n,
        }
    )


def test_barrier_labels_known_values():
    """Hand-computed test: verify exact +1, -1, and 0 labels on deterministic price path."""
    # 20 warm-up bars at 100, then clear moves
    prices = [100.0] * 20 + [100.0, 115.0, 100.0, 85.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    df = _create_synthetic_series(prices)

    config = PipelineConfig(
        profit_taking=1.0,
        stop_loss=1.0,
        vol_lookback_bars=20,
        expiry_bars=5,
        min_volatility=0.05,  # 5% barrier: upper = 105, lower = 95
    )

    labeled = triple_barrier_label(df, config)
    assert "label" in labeled.columns
    assert "t1" in labeled.columns
    assert "t0" in labeled.columns

    # Row index 20 (price=100.0, next price is 115.0 at step 1) -> should hit upper (+1) at step 1
    row_20 = labeled.filter(pl.col("timestamp") == df["timestamp"][20]).to_dicts()[0]
    assert row_20["label"] == 1
    assert row_20["t1"] == df["timestamp"][21]

    # Row index 22 (price=100.0, next price 85.0 at step 1) -> should hit lower (-1) at step 1
    row_22 = labeled.filter(pl.col("timestamp") == df["timestamp"][22]).to_dicts()[0]
    assert row_22["label"] == -1
    assert row_22["t1"] == df["timestamp"][23]


def test_barrier_expiry_in_bars_not_days():
    """Verify that vertical expiry works strictly in bar count across timeframes."""
    prices = [100.0] * 40
    df = _create_synthetic_series(prices)

    config_10 = PipelineConfig(vol_lookback_bars=20, expiry_bars=10, min_volatility=0.05)
    labeled_10 = triple_barrier_label(df, config_10)

    # For row 0 in labeled (which corresponds to row 0 in df), t1 must be exactly 10 bars ahead
    row_0 = labeled_10.to_dicts()[0]
    assert row_0["label"] == 0
    assert row_0["t1"] == df["timestamp"][10]


def test_barrier_volatility_scaling():
    """Verify barriers automatically scale with volatility levels."""
    # Low volatility sequence vs high volatility sequence
    low_vol_prices = [100.0 + (i % 2) * 0.1 for i in range(30)]
    df_low = _create_synthetic_series(low_vol_prices)

    vol_df = calculate_volatility(df_low, vol_lookback_bars=10, min_volatility=1e-5)
    assert vol_df["volatility"].mean() < 0.01


def test_barrier_short_group_skipped():
    """Verify ticker groups shorter than vol_lookback + expiry are skipped without raising errors."""
    prices = [100.0] * 10
    df = _create_synthetic_series(prices)
    config = PipelineConfig(vol_lookback_bars=20, expiry_bars=10)

    labeled = triple_barrier_label(df, config)
    assert labeled.is_empty()


def test_barrier_returns_t1():
    """Ensure t1 (first touch timestamp) is properly formatted and strictly greater than t0."""
    prices = [100.0 + i * 0.5 for i in range(50)]
    df = _create_synthetic_series(prices)
    config = PipelineConfig(vol_lookback_bars=20, expiry_bars=10, min_volatility=0.02)

    labeled = triple_barrier_label(df, config)
    assert labeled.height > 0

    t0_list = labeled["t0"].to_list()
    t1_list = labeled["t1"].to_list()
    for t0, t1 in zip(t0_list, t1_list):
        assert t1 > t0
