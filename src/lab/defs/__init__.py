from dagster import Definitions, load_assets_from_modules
from dagster_polars import PolarsParquetIOManager

from lab.defs import assets
from lab.defs.resources import PipelineConfigResource

defs = Definitions(
    assets=load_assets_from_modules([assets]),
    resources={
        "config_py": PipelineConfigResource(),
        "io_manager": PolarsParquetIOManager(base_dir="data/dagster"),
    },
)
