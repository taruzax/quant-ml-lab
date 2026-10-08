from __future__ import annotations

from pathlib import Path
from typing import Any

from dagster import ConfigurableResource
from pydantic import Field

from lab.core.config import PipelineConfig


class PipelineConfigResource(ConfigurableResource):
    """Dagster resource that resolves the same canonical Pydantic config as Python callers."""

    config_path: str = "config/pipeline.yaml"
    overrides: dict[str, Any] = Field(default_factory=dict)
    timeframe: str | None = None
    sequence_len: int | None = None
    ingestion_config_path: str = "config/ingestion.yaml"
    ingestion_timeframe: str | None = None
    ingestion_stream_name: str | None = None

    def to_pipeline_config(self) -> PipelineConfig:
        overrides = dict(self.overrides)
        if self.timeframe is not None or self.sequence_len is not None:
            overrides["data"] = {
                **overrides.get("data", {}),
                **({"timeframe": self.timeframe} if self.timeframe is not None else {}),
            }
            overrides["tensor"] = {
                **overrides.get("tensor", {}),
                **({"sequence_len": self.sequence_len} if self.sequence_len is not None else {}),
            }
        return PipelineConfig.from_yaml(Path(self.config_path), overrides=overrides)
