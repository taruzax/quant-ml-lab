from dagster import Definitions, load_asset_checks_from_modules, load_assets_from_modules
from dagster_polars import PolarsParquetIOManager

from lab.defs import assets, checks
from lab.defs.resources import PipelineConfigResource

defs = Definitions(
    assets=load_assets_from_modules([assets]),
    asset_checks=load_asset_checks_from_modules([checks]),
    resources={
        "config_py": PipelineConfigResource(),
        "io_manager": PolarsParquetIOManager(base_dir="data/dagster"),
    },
)
