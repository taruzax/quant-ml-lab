"""Active ingestion asset plus deprecated, unregistered research assets for compatibility."""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any

import polars as pl
from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    AssetIn,
    AutomationCondition,
    DailyPartitionsDefinition,
    DimensionPartitionMapping,
    DynamicPartitionsDefinition,
    MaterializeResult,
    MetadataValue,
    MultiPartitionMapping,
    MultiPartitionsDefinition,
    TimeWindowPartitionMapping,
    asset,
    asset_check,
)

from lab.core.config import IngestionConfig, PipelineConfig, Timeframe, get_platform_config
from lab.core.schemas import target_col
from lab.platform.data_access import load_market_data
from lab.platform.ingestion import IngestionService
from lab.platform.market_store import ProviderBatchStore, SnapshotCatalog
from lab.platform.dagster.resources import PipelineConfigResource
from lab.quant.features import apply_all_features
from lab.quant.ffd import find_global_d, frac_diff_polars
from lab.quant.validators import run_all_validations
from lab.research.samples import TimeSeriesDataset

DATA_PIPELINE_GROUP = "data_pipeline"


def create_ingestion_service(config_py: PipelineConfigResource) -> IngestionService:
    platform = get_platform_config()
    return IngestionService(
        batch_store=ProviderBatchStore(platform.paths.snapshot_dir.parent / "market_batches"),
        catalog=SnapshotCatalog(platform.paths.snapshot_dir),
    )


@asset(group_name="market_data")
def ingestion_publication(context: AssetExecutionContext, config_py: PipelineConfigResource) -> dict[str, list[dict[str, Any]]]:
    """Ingest one configured stream and publish its immutable snapshot when coverage permits."""
    ingestion_config = IngestionConfig.from_yaml(config_py.ingestion_config_path)
    matching = [stream for stream in ingestion_config.streams if stream.enabled]
    if config_py.ingestion_timeframe is not None:
        selected_timeframe = Timeframe(config_py.ingestion_timeframe)
        matching = [stream for stream in matching if stream.timeframe == selected_timeframe]
    if config_py.ingestion_stream_name is not None:
        matching = [stream for stream in matching if stream.name == config_py.ingestion_stream_name]
    service = create_ingestion_service(config_py)
    results = [(stream.gap_policy, service.execute(stream)) for stream in matching]
    metadata = {
        "stream_count": MetadataValue.int(len(results)),
        "attempts": MetadataValue.json([item.model_dump(mode="json") for _, item in results]),
        "failed_count": MetadataValue.int(sum(item.status == "failed" for _, item in results)),
        "published_snapshot_ids": MetadataValue.json([item.snapshot_id for _, item in results if item.snapshot_id]),
    }
    if not results:
        metadata["status"] = MetadataValue.text("noop: no enabled streams matched selection")
    context.add_output_metadata(metadata)
    return {"attempts": [{"gap_policy": policy, **item.model_dump(mode="json")} for policy, item in results]}


@asset_check(asset=ingestion_publication, name="ingestion_calendar_coverage", blocking=False)
def ingestion_calendar_coverage(ingestion_publication: dict[str, list[dict[str, Any]]]) -> AssetCheckResult:
    """Flag failed ingestion and incomplete recorded coverage without blocking snapshot history."""
    failures = [
        {"stream_id": item["stream_id"], "status": item["status"], "missing_rows": item["missing_rows"], "error": item.get("error")}
        for item in ingestion_publication.get("attempts", [])
        if item["status"] == "failed" or (item["gap_policy"] == "record" and item["missing_rows"] > 0)
    ]
    return AssetCheckResult(
        passed=not failures,
        metadata={"coverage_failures": MetadataValue.json(failures), "checked_stream_count": len(ingestion_publication.get("attempts", []))},
    )

ticker_partitions = DynamicPartitionsDefinition(name="tickers")
time_window_partitions = DailyPartitionsDefinition(start_date="2020-01-01")
pipeline_partitions = MultiPartitionsDefinition(
    {
        "ticker": ticker_partitions,
        "time": time_window_partitions,
    }
)
LOOKBACK_MAPPING = MultiPartitionMapping(
    {
        "time": DimensionPartitionMapping(
            dimension_name="time", partition_mapping=TimeWindowPartitionMapping(start_offset=-50, end_offset=0)
        )
    }
)


@dataclass(frozen=True)
class AssetWindow:
    ticker: str
    start: datetime
    end: datetime


def partition_window(context: AssetExecutionContext) -> AssetWindow:
    key = context.partition_key
    ticker = key.keys_by_dimension["ticker"]
    day = datetime.combine(datetime.fromisoformat(key.keys_by_dimension["time"]).date(), time.min)
    return AssetWindow(ticker=ticker, start=day, end=day + timedelta(days=1))


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
    partitions_def=pipeline_partitions,
    automation_condition=AutomationCondition.eager(),
)
def raw_ohlcv(context: AssetExecutionContext, config_py: PipelineConfigResource) -> pl.DataFrame:
    pipeline_config = _pipeline_config(config_py)
    window = partition_window(context)
    return load_market_data(
        tickers=[window.ticker],
        interval=pipeline_config.ingestion_interval,
        start=window.start.strftime("%Y-%m-%d"),
        end=window.end.strftime("%Y-%m-%d"),
    )


@asset(
    group_name=DATA_PIPELINE_GROUP,
    partitions_def=pipeline_partitions,
    automation_condition=AutomationCondition.eager(),
)
def validated_data(raw_ohlcv: pl.DataFrame, config_py: PipelineConfigResource) -> pl.DataFrame:
    return run_all_validations(raw_ohlcv, _pipeline_config(config_py))


@asset(
    group_name=DATA_PIPELINE_GROUP,
    partitions_def=pipeline_partitions,
    ins={"validated_data": AssetIn(partition_mapping=LOOKBACK_MAPPING)},
)
def features(
    context: AssetExecutionContext, validated_data: dict[str, pl.DataFrame], config_py: PipelineConfigResource
) -> pl.DataFrame:
    combined_df = pl.concat(list(validated_data.values()), how="diagonal")
    df = apply_all_features(combined_df, _pipeline_config(config_py))
    time_str = context.partition_key.keys_by_dimension["time"]
    target_date = datetime.fromisoformat(time_str).date()
    return df.filter(pl.col("timestamp").dt.date() == target_date)


@asset(
    group_name=DATA_PIPELINE_GROUP,
    partitions_def=pipeline_partitions,
    ins={"features": AssetIn(partition_mapping=LOOKBACK_MAPPING)},
)
def ffd_features(
    context: AssetExecutionContext, features: dict[str, pl.DataFrame], config_py: PipelineConfigResource
) -> pl.DataFrame:
    combined_df = pl.concat(list(features.values()), how="diagonal")
    pipeline_config = _pipeline_config(config_py)
    frame = combined_df.with_columns(pl.col("close").log().alias("log_close"))
    d_value = find_global_d(
        frame,
        col_name="log_close",
        coverage_threshold=pipeline_config.ffd_coverage_threshold,
        max_d=pipeline_config.ffd_max_d,
        threshold=pipeline_config.ffd_threshold,
        min_d=pipeline_config.ffd_min_d,
        significance=pipeline_config.adf_significance,
    )

    df = frac_diff_polars(frame, col_name="log_close", d=d_value, threshold=pipeline_config.ffd_threshold)
    time_str = context.partition_key.keys_by_dimension["time"]
    target_date = datetime.fromisoformat(time_str).date()
    return df.filter(pl.col("timestamp").dt.date() == target_date)


@asset(
    group_name=DATA_PIPELINE_GROUP,
    partitions_def=pipeline_partitions,
)
def tensors(
    context: AssetExecutionContext, ffd_features: pl.DataFrame, config_py: PipelineConfigResource
) -> MaterializeResult[pl.DataFrame]:
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


@asset(group_name="research_pipeline", io_manager_key="fs_io_manager")
def consolidated_snapshot(config_py: PipelineConfigResource):
    """Load the canonical validated market snapshot used by research callers."""
    from lab.platform.data_access import load_market_snapshot

    return load_market_snapshot(_pipeline_config(config_py))


@asset(group_name="research_pipeline", io_manager_key="fs_io_manager")
def prepared_dataset(consolidated_snapshot, config_py: PipelineConfigResource):
    """Prepare causal features, labels and split boundaries through the Python API."""
    from lab.research.preprocessing import prepare_dataset

    return prepare_dataset(_pipeline_config(config_py), snapshot=consolidated_snapshot)


@asset(group_name="research_pipeline", io_manager_key="fs_io_manager")
def development_run(prepared_dataset, config_py: PipelineConfigResource):
    """Run development folds and return an immutable local run reference."""
    from lab.research.experiment import run_experiment

    result = run_experiment(_pipeline_config(config_py), dataset=prepared_dataset)
    return result.ref.model_dump(mode="json")


@asset(group_name="research_pipeline", io_manager_key="fs_io_manager")
def portfolio_evaluation(development_run):
    """Expose saved development allocation/evaluation artifact references."""
    return {"run_id": development_run["run_id"], "status": "completed", "source": "shared_research_api"}
