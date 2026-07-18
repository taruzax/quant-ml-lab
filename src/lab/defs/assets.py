from __future__ import annotations
import dagster as dg
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import polars as pl
from lab.core.config import PipelineConfig
from lab.core.schemas import target_col
from lab.data.features import apply_all_features
from lab.data.ffd import find_global_d, frac_diff_polars
from lab.data.loader import load_market_data
from lab.data.tensor_loader import TimeSeriesDataset
from lab.data.validators import run_all_validations
from lab.defs.resources import PipelineConfigResource

DATA_PIPELINE_GROUP = "data_pipeline"

@dataclass(frozen = True)
class AssetWindow:
    ticker: str
    start: datetime
    end: datetime

def _pipeline_config(resource: PipelineConfigResource) -> PipelineConfig:
    return resource.to_pipeline_config()

def _default_window(config: PipelineConfig) -> AssetWindow:
    start = datetime.fromisoformat(config.ingestion_start)
    return AssetWindow(ticker="AAPL", start=start, end=start + timedelta(days=1))

def _clean_model_frame(df: pl.DataFrame, config: PipelineConfig) -> tuple[pl.DataFrame, list[str], list[str]]:
    target_cols = [target_col(horizon) for horizon in config.target_horizons if target_col(horizon) in df.columns]
    blocked = {"timestamp", "ticker", "sector", "industry", *target_cols}
    numeric_types = {pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
    feature_cols = [col for col, dtype in df.schema.items() if col not in blocked and dtype in numeric_types]

    clean_df = df.select(["timestamp", "ticker", *feature_cols, *target_cols]).drop_nulls()
    return clean_df, feature_cols, target_cols


@dg.asset(
    group_name=DATA_PIPELINE_GROUP,
    automation_condition=dg.AutomationCondition.eager(),
)
def raw_ohlcv(context: dg.AssetExecutionContext, config: PipelineConfigResource) -> pl.DataFrame:
    pipeline_config = _pipeline_config(config)
    window = _default_window(pipeline_config)
    return load_market_data(
        tickers=[window.ticker],
        interval=pipeline_config.ingestion_interval,
        start=window.start.isoformat(),
        end=window.end.isoformat(),
    )
