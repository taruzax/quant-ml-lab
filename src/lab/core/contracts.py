from datetime import datetime
from math import isfinite
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lab.core.config import Timeframe
from lab.core.schemas import validate_feature_names


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, validate_assignment=True)


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
