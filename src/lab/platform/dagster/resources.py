from __future__ import annotations

from pathlib import Path

from dagster import ConfigurableResource
from pydantic import Field

from lab.core.config import PipelineConfig, Timeframe, _load_yaml_defaults

_YAML_DEFAULTS = _load_yaml_defaults()


def _default(name: str, fallback):
    return _YAML_DEFAULTS.get(name, fallback)


class PipelineConfigResource(ConfigurableResource):
    """
    Dagster resource that mirrors PipelineConfig with flat fields.

    Dagster cannot generate config schema for PipelineConfig directly because it
    inherits from BaseSettings. Keep this resource primitive/serializable, then
    convert it back to PipelineConfig at asset runtime.
    """

    raw_data_dir: str = _default("raw_data_dir", "data/raw")
    processed_data_dir: str = _default("processed_data_dir", "data/processed")
    ticker_config_path: str = _default("ticker_config_path", "config/tickers.yaml")
    ingestion_start: str = _default("ingestion_start", "2025-01-01")
    timeframe: str = _default("timeframe", "1h")

    ffd_threshold: float = _default("ffd_threshold", 0.001)
    ffd_max_d: float = _default("ffd_max_d", 1.0)
    ffd_min_d: float = _default("ffd_min_d", 0.1)
    adf_significance: float = _default("adf_significance", 0.05)
    ffd_coverage_threshold: float = _default("ffd_coverage_threshold", 0.8)

    null_tolerance: float = _default("null_tolerance", 0.001)
    min_price: float = _default("min_price", 0.0)

    sequence_len: int = _default("sequence_len", 60)
    batch_size: int = _default("batch_size", 32)
    train_cutoff_date: str | None = _default("train_cutoff_date", None)

    max_position_size: float = _default("max_position_size", 0.20)
    min_position_size: float = _default("min_position_size", 0.05)

    return_lags: list[int] = Field(default_factory=lambda: list(_default("return_lags", [1, 5, 10, 21, 42, 63])))
    clip_quantile: float = _default("clip_quantile", 0.001)
    lookback_periods: list[int] = Field(default_factory=lambda: list(_default("lookback_periods", [1, 2, 3, 4, 5])))
    target_horizons: list[int] = Field(default_factory=lambda: list(_default("target_horizons", [1, 5, 10, 21])))

    def to_pipeline_config(self) -> PipelineConfig:
        return PipelineConfig(
            raw_data_dir=Path(self.raw_data_dir),
            processed_data_dir=Path(self.processed_data_dir),
            ticker_config_path=Path(self.ticker_config_path),
            ingestion_start=self.ingestion_start,
            timeframe=Timeframe(self.timeframe),
            ffd_threshold=self.ffd_threshold,
            ffd_max_d=self.ffd_max_d,
            ffd_min_d=self.ffd_min_d,
            adf_significance=self.adf_significance,
            ffd_coverage_threshold=self.ffd_coverage_threshold,
            null_tolerance=self.null_tolerance,
            min_price=self.min_price,
            sequence_len=self.sequence_len,
            batch_size=self.batch_size,
            train_cutoff_date=self.train_cutoff_date,
            max_position_size=self.max_position_size,
            min_position_size=self.min_position_size,
            return_lags=self.return_lags,
            clip_quantile=self.clip_quantile,
            lookback_periods=self.lookback_periods,
            target_horizons=self.target_horizons,
        )
