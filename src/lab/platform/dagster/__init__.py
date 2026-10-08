from dagster import Definitions, define_asset_job

from lab.platform.dagster import assets
from lab.platform.dagster.resources import PipelineConfigResource
from lab.platform.dagster.schedules import build_ingestion_schedules

ingestion_job = define_asset_job("market_data_ingestion", selection=[assets.ingestion_publication])
daily_ingestion_schedule, hourly_ingestion_schedule = build_ingestion_schedules(ingestion_job)

defs = Definitions(
    assets=[assets.ingestion_publication],
    asset_checks=[assets.ingestion_calendar_coverage],
    schedules=[daily_ingestion_schedule, hourly_ingestion_schedule],
    resources={
        "config_py": PipelineConfigResource(),
    },
)

__all__ = ["defs", "ingestion_job"]
