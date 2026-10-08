from datetime import datetime, timezone
import json
from math import isfinite
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lab.core.config import Timeframe
from lab.core.schemas import validate_feature_names


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, validate_assignment=True)


def _require_text(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class SampleKey(ContractModel):
    ticker: str
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Sample timestamps must be timezone-aware")
        return value


class MarketSnapshot(ContractModel):
    schema_version: str = "market-snapshot.v1"
    snapshot_id: str
    snapshot_hash: str
    bars: pl.DataFrame
    ticker_order: tuple[str, ...]
    calendar: str
    timeframe: Timeframe
    provenance: dict[str, Any] = Field(default_factory=dict)
    validation_diagnostics: dict[str, Any] = Field(default_factory=dict)
    source_manifest: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("ticker_order")
    @classmethod
    def unique_tickers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("ticker_order must contain unique tickers")
        return value


class PreparedDataset(ContractModel):
    schema_version: str = "prepared-dataset.v1"
    snapshot: MarketSnapshot
    features: pl.DataFrame
    feature_specification: tuple[str, ...]
    labels: pl.DataFrame
    split_plan: Any
    exclusions: pl.DataFrame | dict[str, Any] = Field(default_factory=dict)
    config: Any | None = None

    @field_validator("feature_specification")
    @classmethod
    def validate_features(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return validate_feature_names(value)


class LabelResult(ContractModel):
    schema_version: str = "label-result.v2"
    labels: pl.DataFrame
    exclusions: pl.DataFrame
    target_name: str
    conventions: dict[str, Any] = Field(default_factory=dict)


class FoldSpec(ContractModel):
    schema_version: str = "fold-spec.v1"
    fold_id: int
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime

    @model_validator(mode="after")
    def validate_order(self) -> "FoldSpec":
        if not self.train_start < self.train_end <= self.validation_start < self.validation_end:
            raise ValueError("Fold intervals must be ordered half-open intervals")
        return self


class ExecutionPeriod(ContractModel):
    schema_version: str = "execution-period.v1"
    name: str
    first_decision: datetime
    last_decision: datetime
    final_liquidation_open: datetime


class SplitPlan(ContractModel):
    schema_version: str = "split-plan.v1"
    decision_timestamps: tuple[datetime, ...]
    development_start: datetime
    holdout_start: datetime
    holdout_end: datetime
    folds: tuple[FoldSpec, ...]
    execution_periods: tuple[ExecutionPeriod, ...]
    embargo_bars: int = 0

    @model_validator(mode="after")
    def validate_plan(self) -> "SplitPlan":
        if tuple(sorted(set(self.decision_timestamps))) != self.decision_timestamps:
            raise ValueError("Split decision timestamps must be unique and chronological")
        if not self.development_start < self.holdout_start < self.holdout_end:
            raise ValueError("Development and holdout boundaries must be ordered")
        if not self.folds:
            raise ValueError("Split plan requires at least one validation fold")
        return self


class PreprocessingState(ContractModel):
    schema_version: str = "preprocessing-state.v1"
    feature_order: tuple[str, ...]
    clip_thresholds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    vocabularies: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    means: dict[str, float] = Field(default_factory=dict)
    scales: dict[str, float] = Field(default_factory=dict)
    ffd_orders: dict[str, float | None] = Field(default_factory=dict)
    ffd_truncation: dict[str, int] = Field(default_factory=dict)
    ffd_features: tuple[str, ...] = ()
    ffd_threshold: float = 0.001
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class SampleSet(ContractModel):
    schema_version: str = "sample-set.v1"
    X: np.ndarray
    keys: tuple[SampleKey, ...]
    window_start: tuple[datetime, ...]
    window_end: tuple[datetime, ...]
    raw_row_indices: tuple[tuple[int, ...], ...]
    feature_order: tuple[str, ...]
    y: np.ndarray | None = None
    label_metadata: tuple[dict[str, Any], ...] | None = None

    @model_validator(mode="after")
    def validate_sample_shape(self) -> "SampleSet":
        if self.X.ndim != 3:
            raise ValueError("SampleSet.X must have shape [N, T, F]")
        if not np.isfinite(self.X).all():
            raise ValueError("SampleSet.X must contain only finite values")
        count = self.X.shape[0]
        if len(self.keys) != count or len(self.window_start) != count or len(self.window_end) != count:
            raise ValueError("SampleSet metadata must have one row per sample")
        if len(self.raw_row_indices) != count:
            raise ValueError("SampleSet.raw_row_indices must have one row per sample")
        if self.y is not None and self.y.shape[0] != count:
            raise ValueError("SampleSet.y must align with SampleSet.keys")
        if len(self.feature_order) != self.X.shape[2]:
            raise ValueError("feature_order length must match the final SampleSet dimension")
        validate_feature_names(self.feature_order)
        if len({(key.ticker, key.timestamp) for key in self.keys}) != len(self.keys):
            raise ValueError("SampleSet keys must be unique")
        return self


class FoldBundle(ContractModel):
    schema_version: str = "fold-bundle.v1"
    train_samples: SampleSet
    evaluation_samples: SampleSet
    scoring_keys: tuple[SampleKey, ...]
    preprocessor: Any
    boundaries: dict[str, Any]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class PredictionFrame(ContractModel):
    schema_version: str = "prediction-frame.v1"
    keys: tuple[SampleKey, ...]
    model_ref: str
    fold_ref: str
    predictions: np.ndarray

    @model_validator(mode="after")
    def validate_predictions(self) -> "PredictionFrame":
        if len({(key.ticker, key.timestamp) for key in self.keys}) != len(self.keys):
            raise ValueError("Prediction keys must be unique")
        if self.predictions.ndim == 1:
            if self.predictions.shape[0] != len(self.keys) or not np.isfinite(self.predictions).all():
                raise ValueError("Regression predictions must be finite and key-aligned")
        elif self.predictions.ndim == 2:
            if self.predictions.shape != (len(self.keys), 3):
                raise ValueError("Classification predictions must have shape [N, 3] in [-1, 0, +1] order")
            if not np.isfinite(self.predictions).all() or (self.predictions < 0).any():
                raise ValueError("Classification probabilities must be finite and nonnegative")
            if not np.allclose(self.predictions.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("Classification probabilities must sum to one")
        else:
            raise ValueError("Predictions must be one- or two-dimensional")
        return self


class AllocationFrame(ContractModel):
    schema_version: str = "allocation-frame.v1"
    decision_time: datetime
    valid_from: datetime
    ticker_budgets: dict[str, float]
    unconstrained_budgets: dict[str, float] = Field(default_factory=dict)
    eligibility: dict[str, str] = Field(default_factory=dict)
    exclusions: dict[str, str] = Field(default_factory=dict)
    covariance_reference: str | None = None
    history_reference: str | None = None
    config_hash: str
    cash_capacity: float = 1.0
    turnover: float = 0.0
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class BacktestResult(ContractModel):
    schema_version: str = "backtest-result.v1"
    orders: pl.DataFrame
    trade_events: pl.DataFrame
    equity: pl.DataFrame
    gross_returns: pl.Series | None = None
    net_returns: pl.Series | None = None
    exposure: pl.DataFrame | None = None
    turnover: pl.Series | None = None
    period: dict[str, Any]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class MetricResult(ContractModel):
    schema_version: str = "metric-result.v1"
    name: str
    value: float | None
    status: Literal["available", "unavailable", "failed"]
    reason: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    conventions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_value(self) -> "MetricResult":
        if self.status == "available":
            if self.value is None or not isfinite(self.value):
                raise ValueError("Available metrics require a finite numeric value")
        elif self.value is not None:
            raise ValueError("Unavailable or failed metrics must use value=None")
        return self


class RunRef(ContractModel):
    schema_version: str = "run-ref.v1"
    run_id: str
    snapshot_hash: str
    config_hash: str
    split_hash: str
    model_ref: str
    parent_run_id: str | None = None


class RunResult(ContractModel):
    schema_version: str = "run-result.v1"
    ref: RunRef
    predictions: tuple[PredictionFrame, ...] = ()
    allocations: tuple[AllocationFrame, ...] = ()
    metrics: tuple[MetricResult, ...] = ()
    backtest: BacktestResult | None = None
    snapshot: MarketSnapshot | None = None
    artifact_manifest: dict[str, Any] = Field(default_factory=dict)


class SnapshotCatalogEntry(ContractModel):
    schema_version: str = "snapshot-catalog-entry.v1"
    snapshot_id: str
    snapshot_hash: str
    stream_id: str
    provider: str
    calendar: str
    timeframe: Timeframe
    ticker_order: tuple[str, ...]
    first_completed_bar: datetime
    last_completed_bar: datetime
    published_at: datetime
    coverage_status: Literal["accepted", "rejected", "incomplete"]
    batch_ids: tuple[str, ...] = ()
    bundle_path: str
    manifest_checksum: str

    @field_validator(
        "snapshot_id",
        "snapshot_hash",
        "stream_id",
        "provider",
        "calendar",
        "bundle_path",
        "manifest_checksum",
    )
    @classmethod
    def validate_text_fields(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)

    @field_validator("ticker_order")
    @classmethod
    def validate_tickers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not ticker.strip() for ticker in value) or len(set(value)) != len(value):
            raise ValueError("ticker_order must contain unique non-empty tickers")
        return value

    @field_validator("first_completed_bar", "last_completed_bar", "published_at")
    @classmethod
    def validate_timestamps(cls, value: datetime, info: Any) -> datetime:
        return _require_aware(value, info.field_name)

    @model_validator(mode="after")
    def validate_window(self) -> "SnapshotCatalogEntry":
        if self.first_completed_bar > self.last_completed_bar:
            raise ValueError("Snapshot completed-bar boundaries must be ordered")
        return self


class IngestionAttemptSummary(ContractModel):
    schema_version: str = "ingestion-attempt-summary.v1"
    attempt_id: str
    stream_id: str
    status: Literal["planned", "running", "succeeded", "noop", "failed"]
    planned_start: datetime
    planned_end: datetime
    completed_boundary: datetime | None = None
    batch_id: str | None = None
    snapshot_id: str | None = None
    observed_rows: int = Field(default=0, ge=0)
    missing_rows: int = Field(default=0, ge=0)
    unexpected_rows: int = Field(default=0, ge=0)
    duplicate_rows: int = Field(default=0, ge=0)
    revised_rows: int = Field(default=0, ge=0)
    batch_ids: tuple[str, ...] = ()
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None

    @field_validator("attempt_id", "stream_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)

    @field_validator("batch_id", "snapshot_id", "error")
    @classmethod
    def validate_optional_text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _require_text(value, info.field_name)

    @field_validator("planned_start", "planned_end", "completed_boundary", "started_at", "finished_at")
    @classmethod
    def validate_optional_timestamps(cls, value: datetime | None, info: Any) -> datetime | None:
        return None if value is None else _require_aware(value, info.field_name)

    @model_validator(mode="after")
    def validate_attempt(self) -> "IngestionAttemptSummary":
        if self.planned_start >= self.planned_end:
            raise ValueError("Ingestion planned window must be ordered")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("Ingestion finished_at must not precede started_at")
        if self.status == "failed" and not self.error:
            raise ValueError("Failed ingestion attempts require an error")
        if self.status in {"succeeded", "noop"} and self.error is not None:
            raise ValueError("Successful or no-op ingestion attempts cannot contain an error")
        return self


class PreparedDatasetRef(ContractModel):
    schema_version: str = "prepared-dataset-ref.v1"
    dataset_id: str
    snapshot_id: str
    snapshot_hash: str
    config_hash: str
    feature_spec_hash: str
    split_hash: str
    bundle_path: str
    checksum: str
    created_at: datetime

    @field_validator(
        "dataset_id",
        "snapshot_id",
        "snapshot_hash",
        "config_hash",
        "feature_spec_hash",
        "split_hash",
        "bundle_path",
        "checksum",
    )
    @classmethod
    def validate_reference_text(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _require_aware(value, "created_at")


class ExperimentVariation(ContractModel):
    schema_version: str = "experiment-variation.v1"
    variation_id: str
    campaign: str
    snapshot_id: str
    snapshot_hash: str
    effective_config_hash: str
    feature_specification: tuple[str, ...]
    seed: int
    parameter_overrides: dict[str, Any] = Field(default_factory=dict)
    allocation_overrides: dict[str, Any] = Field(default_factory=dict)
    backtest_overrides: dict[str, Any] = Field(default_factory=dict)
    resource_request: dict[str, Any] = Field(default_factory=dict)

    @field_validator("variation_id", "campaign", "snapshot_id", "snapshot_hash", "effective_config_hash")
    @classmethod
    def validate_variation_text(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)

    @field_validator("feature_specification")
    @classmethod
    def validate_feature_specification(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("feature_specification must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("feature_specification must not contain duplicates")
        return validate_feature_names(value)


class CampaignPlan(ContractModel):
    schema_version: str = "campaign-plan.v1"
    campaign_id: str
    campaign: str
    base_config_hash: str
    variations: tuple[ExperimentVariation, ...]
    max_planned_experiments: int = Field(gt=0)
    worker_count: int = Field(gt=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), exclude=True)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json", exclude={"created_at"}), sort_keys=True, separators=(",", ":"))

    @field_validator("campaign_id", "campaign", "base_config_hash")
    @classmethod
    def validate_campaign_text(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)

    @field_validator("created_at")
    @classmethod
    def validate_campaign_time(cls, value: datetime) -> datetime:
        return _require_aware(value, "created_at")

    @model_validator(mode="after")
    def validate_variations(self) -> "CampaignPlan":
        if not self.variations:
            raise ValueError("Campaign plan requires at least one variation")
        variation_ids = [variation.variation_id for variation in self.variations]
        if len(set(variation_ids)) != len(variation_ids):
            raise ValueError("Campaign variation IDs must be unique")
        if len(self.variations) > self.max_planned_experiments:
            raise ValueError("Campaign plan exceeds max_planned_experiments")
        if any(variation.campaign != self.campaign for variation in self.variations):
            raise ValueError("All campaign variations must use the plan campaign")
        return self


CampaignRunStatus = Literal["planned", "running", "completed", "failed", "cancelled"]


class CampaignRunEntry(ContractModel):
    schema_version: str = "campaign-run-entry.v1"
    variation_id: str
    planned_identity: str
    effective_config_hash: str
    snapshot_hash: str
    status: CampaignRunStatus
    created_at: datetime
    updated_at: datetime
    run_id: str | None = None
    error: str | None = None
    previous_status: CampaignRunStatus | None = None

    @field_validator("variation_id", "planned_identity", "effective_config_hash", "snapshot_hash")
    @classmethod
    def validate_run_text(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)

    @field_validator("run_id", "error")
    @classmethod
    def validate_optional_run_text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _require_text(value, info.field_name)

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_run_times(cls, value: datetime, info: Any) -> datetime:
        return _require_aware(value, info.field_name)

    @model_validator(mode="after")
    def validate_status_transition(self) -> "CampaignRunEntry":
        if self.updated_at < self.created_at:
            raise ValueError("Campaign run updated_at must not precede created_at")
        if self.status == "completed" and self.run_id is None:
            raise ValueError("Completed campaign entries require a run_id")
        if self.status == "failed" and not self.error:
            raise ValueError("Failed campaign entries require an error")
        if self.previous_status is not None:
            allowed = {
                "planned": {"running", "cancelled"},
                "running": {"running", "completed", "failed", "cancelled"},
                "completed": set(),
                "failed": {"running", "failed"},
                "cancelled": {"running", "cancelled"},
            }
            if self.status not in allowed[self.previous_status]:
                raise ValueError(
                    f"Illegal campaign status transition: {self.previous_status} -> {self.status}"
                )
        return self


class CampaignResultRef(ContractModel):
    schema_version: str = "campaign-result-ref.v1"
    campaign_id: str
    result_id: str
    plan_hash: str
    bundle_path: str
    checksum: str
    created_at: datetime

    @field_validator("campaign_id", "result_id", "plan_hash", "bundle_path", "checksum")
    @classmethod
    def validate_result_text(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)

    @field_validator("created_at")
    @classmethod
    def validate_result_time(cls, value: datetime) -> datetime:
        return _require_aware(value, "created_at")


class DeviceResolution(ContractModel):
    schema_version: str = "device-resolution.v1"
    model: str
    requested_device: Literal["auto", "cpu", "mps", "cuda"]
    resolved_device: Literal["cpu", "mps", "cuda"]
    reason: str
    resolved_at: datetime
    available_devices: tuple[str, ...] = ()

    @field_validator("model", "reason")
    @classmethod
    def validate_device_text(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)

    @field_validator("resolved_at")
    @classmethod
    def validate_device_time(cls, value: datetime) -> datetime:
        return _require_aware(value, "resolved_at")

    @field_validator("available_devices")
    @classmethod
    def validate_available_devices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        allowed = {"cpu", "mps", "cuda"}
        if any(device not in allowed for device in value) or len(set(value)) != len(value):
            raise ValueError("available_devices must contain unique supported devices")
        return value


class CampaignEvidenceReport(ContractModel):
    schema_version: str = "campaign-evidence-report.v1"
    source_run_id: str
    campaign: str
    comparison_dimensions: dict[str, Any]
    included_logical_trial_ids: tuple[str, ...]
    selected_attempt_ids: tuple[str, ...]
    exclusions: tuple[dict[str, Any], ...] = ()
    population_counts: dict[str, int]
    metric_values: dict[str, float | None]
    metric_details: dict[str, dict[str, Any]] = Field(default_factory=dict)
    calculated_at: datetime

    @field_validator("source_run_id", "campaign")
    @classmethod
    def validate_evidence_text(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)

    @field_validator("included_logical_trial_ids", "selected_attempt_ids")
    @classmethod
    def validate_evidence_ids(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        if any(not item.strip() for item in value) or len(set(value)) != len(value):
            raise ValueError(f"{info.field_name} must contain unique non-empty IDs")
        return value

    @field_validator("population_counts")
    @classmethod
    def validate_population_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("population_counts must not contain negative values")
        return value

    @field_validator("calculated_at")
    @classmethod
    def validate_evidence_time(cls, value: datetime) -> datetime:
        return _require_aware(value, "calculated_at")
