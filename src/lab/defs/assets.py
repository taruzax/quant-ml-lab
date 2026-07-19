
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
from dagster import AssetExecutionContext, MaterializeResult, AutomationCondition, MetadataValue, asset

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
    return AssetWindow(ticker="AAPL", start=start, end=start + timedelta(days=30))

def _clean_model_frame(df: pl.DataFrame, config: PipelineConfig) -> tuple[pl.DataFrame, list[str], list[str]]:
    target_cols = [target_col(horizon) for horizon in config.target_horizons if target_col(horizon) in df.columns]
    blocked = {"timestamp", "ticker", "sector", "industry", *target_cols}
    numeric_types = {pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
    feature_cols = [col for col, dtype in df.schema.items() if col not in blocked and dtype in numeric_types]

    clean_df = df.select(["timestamp", "ticker", *feature_cols, *target_cols]).drop_nulls()
    return clean_df, feature_cols, target_cols


@asset(
    group_name=DATA_PIPELINE_GROUP,
    automation_condition=AutomationCondition.eager(),
)
def raw_ohlcv(context: AssetExecutionContext, config_py: PipelineConfigResource) -> pl.DataFrame:
    pipeline_config = _pipeline_config(config_py)
    window = _default_window(pipeline_config)
    return load_market_data(
        tickers=[window.ticker],
        interval=pipeline_config.ingestion_interval,
        start=window.start.strftime("%Y-%m-%d"),
        end=window.end.strftime("%Y-%m-%d"),
    )
@asset(
    group_name=DATA_PIPELINE_GROUP,
    automation_condition=AutomationCondition.eager(),
)
def validated_data(raw_ohlcv: pl.DataFrame, config_py: PipelineConfigResource) -> pl.DataFrame:
    return run_all_validations(raw_ohlcv, _pipeline_config(config_py))


@asset(group_name=DATA_PIPELINE_GROUP)
def features(validated_data: pl.DataFrame, config_py: PipelineConfigResource) -> pl.DataFrame:
    return apply_all_features(validated_data, _pipeline_config(config_py))


@asset(group_name=DATA_PIPELINE_GROUP)
def ffd_features(features: pl.DataFrame, config_py: PipelineConfigResource) -> pl.DataFrame:
    pipeline_config = _pipeline_config(config_py)
    frame = features.with_columns(pl.col("close").log().alias("log_close"))
    d_value = find_global_d(
        frame,
        col_name="log_close",
        coverage_threshold=pipeline_config.ffd_coverage_threshold,
        max_d=pipeline_config.ffd_max_d,
        threshold=pipeline_config.ffd_threshold,
        min_d=pipeline_config.ffd_min_d,
        significance=pipeline_config.adf_significance,
    )
    return frac_diff_polars(frame, col_name="log_close", d=d_value, threshold=pipeline_config.ffd_threshold)


@asset(group_name=DATA_PIPELINE_GROUP)
def tensors(context: AssetExecutionContext, ffd_features: pl.DataFrame, config_py: PipelineConfigResource) -> MaterializeResult[pl.DataFrame]:
    pipeline_config = _pipeline_config(config_py)
    clean_df, feature_cols, target_cols = _clean_model_frame(ffd_features, pipeline_config)

    dataset = TimeSeriesDataset(
        clean_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        sequence_len=pipeline_config.sequence_len,
    )

    manifest = pl.DataFrame(
        {
            "timeframe": [pipeline_config.timeframe.value],
            "n_rows": [clean_df.height],
            "n_windows": [len(dataset)],
            "n_features": [len(feature_cols)],
            "n_targets": [len(target_cols)],
            "sequence_len": [pipeline_config.sequence_len],
        }
    )
    return MaterializeResult(
        value=manifest,
        metadata={
            "windows": MetadataValue.int(len(dataset)),
            "features": MetadataValue.int(len(feature_cols)),
            "targets": MetadataValue.int(len(target_cols)),
        },
    )