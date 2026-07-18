from datetime import timedelta
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from pydantic_settings import BaseSettings, SettingsConfigDict


class Timeframe(str, Enum):
    D1 = "1d"
    H1 = "1h"


class TimeframeConstantValues(TypedDict):
    bars_per_year: int
    annualization_factor: float
    typical_gap_tolerance: timedelta
    rolling_month_bars: int
    ingestion_interval: str


TIMEFRAME_CONSTANTS: dict[Timeframe, TimeframeConstantValues] = {
    Timeframe.D1: {
        "bars_per_year": 252,
        "annualization_factor": 252.0,
        "typical_gap_tolerance": timedelta(days=3),
        "rolling_month_bars": 21,
        "ingestion_interval": "1d",
    },
    Timeframe.H1: {
        "bars_per_year": 1638,
        "annualization_factor": 1638.0,
        "typical_gap_tolerance": timedelta(hours=2),
        "rolling_month_bars": 147,
        "ingestion_interval": "60m",
    },
}


class PipelineConfig(BaseSettings):
    # Read from .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Data paths
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    ticker_config_path: Path = Path("config/tickers.yaml")
    ingestion_start: str = "2025-01-01"
    timeframe: Timeframe = Timeframe.H1

    # FFD parameters
    ffd_threshold: float = 0.001
    ffd_max_d: float = 1.0
    ffd_min_d: float = 0.1
    adf_significance: float = 0.05
    ffd_coverage_threshold: float = 0.8

    # Validation thresholds
    null_tolerance: float = 0.001
    min_price: float = 0.0

    # Tensor factory
    sequence_len: int = 60
    batch_size: int = 32
    train_cutoff_date: str | None = None

    # Risk engine
    max_position_size: float = 0.20
    min_position_size: float = 0.05

    # Returns
    return_lags: list[int] = [1, 5, 10, 21, 42, 63]
    clip_quantile: float = 0.001
    lookback_periods: list[int] = [1, 2, 3, 4, 5]
    target_horizons: list[int] = [1, 5, 10, 21]

    @property
    def ingestion_interval(self) -> str:
        return TIMEFRAME_CONSTANTS[self.timeframe]["ingestion_interval"]

    @property
    def bars_per_year(self) -> int:
        return TIMEFRAME_CONSTANTS[self.timeframe]["bars_per_year"]

    @property
    def annualization_factor(self) -> float:
        return TIMEFRAME_CONSTANTS[self.timeframe]["annualization_factor"]

    @property
    def gap_tolerance(self) -> timedelta:
        return TIMEFRAME_CONSTANTS[self.timeframe]["typical_gap_tolerance"]

    @property
    def rolling_month_bars(self) -> int:
        return TIMEFRAME_CONSTANTS[self.timeframe]["rolling_month_bars"]


@lru_cache
def get_config():
    return PipelineConfig()
