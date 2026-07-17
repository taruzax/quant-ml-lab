from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseSettings):
    # Read from .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Data paths
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    ticker_config_path: Path = Path("config/tickers.yaml")

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


@lru_cache
def get_config():
    return PipelineConfig()
