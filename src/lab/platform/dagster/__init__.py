from dagster import Definitions, fs_io_manager, load_asset_checks_from_modules
from dagster_polars import PolarsParquetIOManager

from lab.platform.dagster import assets, checks
from lab.platform.dagster.resources import PipelineConfigResource

defs = Definitions(
    assets=[
        assets.raw_ohlcv,
        assets.validated_data,
        assets.consolidated_snapshot,
        assets.prepared_dataset,
        assets.development_run,
        assets.portfolio_evaluation,
    ],
    asset_checks=load_asset_checks_from_modules([checks]),
    resources={
        "config_py": PipelineConfigResource(),
        "io_manager": PolarsParquetIOManager(base_dir="data/dagster"),
        "fs_io_manager": fs_io_manager,
    },
)
