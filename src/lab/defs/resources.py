from __future__ import annotations
from dagster import ConfigurableResource
from pydantic import Field
from lab.core.config import PipelineConfig


class PipelineConfigResource(ConfigurableResource):
    """
    Dagster wrapper for the core PipelineConfig.
    """
    core: PipelineConfig = Field(default_factory=PipelineConfig)
    def to_pipeline_config(self) -> PipelineConfig:
        return self.core
