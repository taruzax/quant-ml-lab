from __future__ import annotations

import numpy as np
import polars as pl
import pytest

# pyrefly: ignore [missing-import]
from src.lab.core.config import PipelineConfig


@pytest.fixture
def single_ticker_df() -> pl.DataFrame:
    """300-row synthetic OHLCV DataFrame for ticker 'TEST'."""
    np.random.seed(42)
    n = 300
    dates = pl.date_range(
        start=pl.date(2024, 1, 1),
        end=pl.date(2024, 1, 1) + pl.duration(days=n - 1),
        interval="1d",
        eager=True,
    )
    close = np.exp(np.cumsum(np.random.normal(0, 0.01, n))) * 100
    return pl.DataFrame(
        {
            "date": dates,
            "ticker": ["TEST"] * n,
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.uniform(1e6, 1e7, n),
            "sector": ["Technology"] * n,
            "industry": ["Software"] * n,
        }
    )


@pytest.fixture
def multi_ticker_df(single_ticker_df) -> pl.DataFrame:
    """600-row DataFrame with tickers 'AAA' and 'BBB', 300 rows each."""
    df_a = single_ticker_df.with_columns(pl.lit("AAA").alias("ticker"))
    np.random.seed(99)
    n = 300
    dates = pl.date_range(
        start=pl.date(2024, 1, 1),
        end=pl.date(2024, 1, 1) + pl.duration(days=n - 1),
        interval="1d",
        eager=True,
    )
    close = np.exp(np.cumsum(np.random.normal(0, 0.015, n))) * 50
    df_b = pl.DataFrame(
        {
            "date": dates,
            "ticker": ["BBB"] * n,
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.uniform(1e6, 1e7, n),
            "sector": ["Healthcare"] * n,
            "industry": ["Biotech"] * n,
        }
    )
    return pl.concat([df_a, df_b])


@pytest.fixture
def synthetic_returns() -> np.ndarray:
    """(500, 5) array of synthetic daily returns for 5 assets."""
    np.random.seed(42)
    # Different volatilities to make HRP tests meaningful
    vols = [0.01, 0.02, 0.01, 0.015, 0.05]
    returns = np.column_stack([np.random.normal(0, vol, 500) for vol in vols])
    return returns


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Default PipelineConfig with test-appropriate values."""
    return PipelineConfig(
        sequence_len=10,
        batch_size=4,
        train_cutoff_date=None,
    )
