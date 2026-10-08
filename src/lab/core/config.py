import hashlib
import json
import warnings
from copy import deepcopy
from datetime import timedelta
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Timeframe(str, Enum):
    D1 = "1d"
    H1 = "1h"


class TimeframeConstantValues(TypedDict):
    bars_per_year: int
    annualization_factor: float
    typical_gap_tolerance: timedelta
    rolling_month_bars: int
    ingestion_interval: str


TIMEFRAME_CONSTANTS: dict[Timeframe, TimeframeConstantValues] = {
    Timeframe.D1: {
        "bars_per_year": 252,
        "annualization_factor": 252.0,
        "typical_gap_tolerance": timedelta(days=3),
        "rolling_month_bars": 21,
        "ingestion_interval": "1d",
    },
    Timeframe.H1: {
        "bars_per_year": 1638,
        "annualization_factor": 1638.0,
        "typical_gap_tolerance": timedelta(hours=2),
        "rolling_month_bars": 147,
        "ingestion_interval": "60m",
    },
}

PIPELINE_CONFIG_PATH = Path("config/pipeline.yaml")
PLATFORM_CONFIG_PATH = Path("config/platform.yaml")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataConfig(StrictModel):
    source: Literal["bundled_demo", "local", "provider", "snapshot"] = "bundled_demo"
    snapshot_id: str | None = None
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    ticker_config_path: Path = Path("config/tickers.yaml")
    input_path: Path | None = None
    metadata_path: Path | None = None
    ingestion_start: str = "2025-01-01"
    timeframe: Timeframe = Timeframe.H1
    calendar: str = "XNYS"
    price_adjustment: Literal["adjusted", "unadjusted"] = "adjusted"

    @model_validator(mode="after")
    def validate_snapshot_source(self) -> "DataConfig":
        if self.source == "snapshot" and not self.snapshot_id:
            raise ValueError("data.snapshot_id is required when data.source is 'snapshot'")
        if self.source != "snapshot" and self.snapshot_id is not None:
            raise ValueError("data.snapshot_id is only valid when data.source is 'snapshot'")
        return self


class ValidationConfig(StrictModel):
    null_tolerance: float = Field(default=0.001, ge=0.0, le=1.0)
    min_price: float = Field(default=0.0, ge=0.0)
    require_timezone: bool = True


class FeaturesConfig(StrictModel):
    return_lags: list[PositiveInt] = Field(default_factory=lambda: [1, 5, 10, 21, 42, 63])
    clip_quantile: float = Field(default=0.001, ge=0.0, lt=0.5)
    lookback_periods: list[PositiveInt] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    target_horizons: list[PositiveInt] = Field(default_factory=lambda: [1, 5, 10, 21])
    feature_columns: list[str] | None = None


class FFDConfig(StrictModel):
    threshold: float = Field(default=0.001, gt=0.0)
    max_d: float = Field(default=1.0, ge=0.0)
    min_d: float = Field(default=0.1, ge=0.0)
    adf_significance: float = Field(default=0.05, gt=0.0, lt=1.0)
    coverage_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_range(self) -> "FFDConfig":
        if self.min_d > self.max_d:
            raise ValueError("ffd.min_d must not exceed ffd.max_d")
        return self


class TensorConfig(StrictModel):
    sequence_len: PositiveInt = 60
    batch_size: PositiveInt = 32
    train_cutoff_date: str | None = None


class TripleBarrierConfig(StrictModel):
    profit_taking: float = Field(default=2.0, gt=0.0)
    stop_loss: float = Field(default=2.0, gt=0.0)
    expiry_bars: PositiveInt = 10
    volatility_span: PositiveInt = 20
    volatility_warmup: PositiveInt = 20
    volatility_floor: float = Field(default=0.0001, gt=0.0)
    ewm_adjust: bool = False
    ewm_bias: bool = False


class TaskConfig(StrictModel):
    kind: Literal["regression", "classification"] = "regression"
    triple_barrier: TripleBarrierConfig = Field(default_factory=TripleBarrierConfig)


class SplitsConfig(StrictModel):
    n_folds: PositiveInt = 3
    holdout_fraction: float = Field(default=0.20, gt=0.0, lt=1.0)
    development_train_fraction: float = Field(default=0.50, gt=0.0, lt=1.0)
    embargo_bars: int = Field(default=0, ge=0)
    diagnostic_mode: bool = False
    explicit_boundaries: dict[str, str] | None = None


class ModelSpec(StrictModel):
    enabled: bool
    params: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(StrictModel):
    baseline: ModelSpec = Field(default_factory=lambda: ModelSpec(enabled=True))
    xgboost: ModelSpec = Field(
        default_factory=lambda: ModelSpec(
            enabled=True,
            params={
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            },
        )
    )
    gru: ModelSpec = Field(default_factory=lambda: ModelSpec(enabled=False, params={"hidden_size": 32, "num_layers": 1}))
    lstm: ModelSpec = Field(default_factory=lambda: ModelSpec(enabled=False, params={"hidden_size": 32, "num_layers": 1}))
    device: Literal["auto", "cpu", "mps", "cuda"] = "cpu"
    seed: int = 42
    early_stopping: bool = False
    early_stopping_fraction: float = Field(default=0.20, gt=0.0, lt=1.0)
    early_stopping_patience: PositiveInt = 5
    deterministic_policy: Literal["strict", "warning"] = "warning"


class AllocationConfig(StrictModel):
    method: Literal["equal_weight", "hrp"] = "equal_weight"
    lookback_bars: PositiveInt = 252
    rebalance_every_bars: PositiveInt = 21
    max_position_size: float = Field(default=0.20, gt=0.0, le=1.0)
    min_position_size: float = Field(default=0.05, ge=0.0, le=1.0)
    gross_limit: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def validate_limits(self) -> "AllocationConfig":
        if self.min_position_size > self.max_position_size:
            raise ValueError("allocation.min_position_size must not exceed max_position_size")
        if self.max_position_size > self.gross_limit:
            raise ValueError("allocation.max_position_size must not exceed gross_limit")
        return self


class BacktestConfig(StrictModel):
    initial_capital: float = Field(default=100_000.0, gt=0.0)
    dead_zone: float = Field(default=0.0, ge=0.0)
    classification_confidence: float | None = Field(default=0.5, ge=0.0, le=1.0)
    fees_bps: float = Field(default=5.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)


class StatisticsConfig(StrictModel):
    frequency: Literal["native", "daily"] = "daily"
    min_observations: PositiveInt = 30


class CampaignConfig(StrictModel):
    name: str = "local-research"
    seed: int = 42


class IngestionStreamConfig(StrictModel):
    name: str
    provider: str = "yfinance"
    tickers: tuple[str, ...] = ()
    ticker_config_path: Path | None = None
    calendar: str = "XNYS"
    timeframe: Timeframe = Timeframe.D1
    timestamp_role: Literal["session_date", "bar_open", "bar_close"] | None = None
    provider_timezone: str = "America/New_York"
    start_boundary: str
    refresh_overlap_bars: PositiveInt = 1
    completion_delay_minutes: int = Field(default=15, ge=0)
    enabled: bool = True
    gap_policy: Literal["reject", "allow", "record"] = "reject"

    @model_validator(mode="after")
    def validate_stream(self) -> "IngestionStreamConfig":
        if not self.name.strip():
            raise ValueError("Ingestion stream name must not be empty")
        if bool(self.tickers) == (self.ticker_config_path is not None):
            raise ValueError("Each ingestion stream must define tickers or ticker_config_path, but not both")
        if any(not ticker.strip() for ticker in self.tickers) or len(set(self.tickers)) != len(self.tickers):
            raise ValueError("Ingestion stream tickers must be unique and non-empty")
        return self


class IngestionConfig(StrictModel):
    streams: tuple[IngestionStreamConfig, ...]

    @model_validator(mode="after")
    def validate_streams(self) -> "IngestionConfig":
        names = [stream.name for stream in self.streams]
        if not names:
            raise ValueError("At least one ingestion stream is required")
        if len(set(names)) != len(names):
            raise ValueError("Ingestion stream names must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path = Path("config/ingestion.yaml")) -> "IngestionConfig":
        return cls.model_validate(_read_yaml(path))


class CampaignMatrixConfig(StrictModel):
    name: str
    base_pipeline_config: Path = PIPELINE_CONFIG_PATH
    snapshot_ids: tuple[str, ...] = ()
    feature_column_sets: tuple[tuple[str, ...], ...] = ()
    model_parameter_overrides: tuple[dict[str, Any], ...] = ()
    seeds: tuple[int, ...] = ()
    allocation_overrides: tuple[dict[str, Any], ...] = ()
    backtest_overrides: tuple[dict[str, Any], ...] = ()
    max_planned_experiments: PositiveInt = 1
    worker_count: PositiveInt = 1

    @model_validator(mode="after")
    def validate_matrix(self) -> "CampaignMatrixConfig":
        if not self.name.strip():
            raise ValueError("Campaign matrix name must not be empty")
        for field_name in (
            "snapshot_ids",
            "feature_column_sets",
            "model_parameter_overrides",
            "seeds",
            "allocation_overrides",
            "backtest_overrides",
        ):
            values = getattr(self, field_name)
            keys = [json.dumps(value, sort_keys=True, default=str) for value in values]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{field_name} must not contain duplicate entries")
        dimensions = (
            len(self.snapshot_ids) or 1,
            len(self.feature_column_sets) or 1,
            len(self.model_parameter_overrides) or 1,
            len(self.seeds) or 1,
            len(self.allocation_overrides) or 1,
            len(self.backtest_overrides) or 1,
        )
        planned = 1
        for size in dimensions:
            planned *= size
        if planned > self.max_planned_experiments:
            raise ValueError(
                f"Campaign Cartesian product has {planned} experiments; "
                f"max_planned_experiments is {self.max_planned_experiments}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CampaignMatrixConfig":
        return cls.model_validate(_read_yaml(path))


class PlatformPathsConfig(StrictModel):
    artifacts_dir: Path = Path("artifacts")
    snapshot_dir: Path = Path("data/snapshots")
    mlflow_tracking_uri: str = "sqlite:///data/mlflow/mlflow.db"


class MLflowConfig(StrictModel):
    enabled: bool = True
    experiment_name: str = "quant-ml-lab"


class PlatformConfig(StrictModel):
    paths: PlatformPathsConfig = Field(default_factory=PlatformPathsConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    @classmethod
    def from_yaml(cls, path: Path = PLATFORM_CONFIG_PATH) -> "PlatformConfig":
        return cls.model_validate(_read_yaml(path))


CANONICAL_SECTIONS = {
    "data",
    "validation",
    "features",
    "ffd",
    "tensor",
    "task",
    "splits",
    "models",
    "allocation",
    "backtest",
    "statistics",
    "campaign",
}


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return raw


def _set_nested(mapping: dict[str, Any], path: tuple[str, ...], value: Any, source: str) -> None:
    target = mapping
    for key in path[:-1]:
        if key in target and not isinstance(target[key], dict):
            raise ValueError(f"Cannot translate {source}: {'.'.join(path)} conflicts with a scalar")
        target = target.setdefault(key, {})
    leaf = path[-1]
    if leaf in target and target[leaf] != value:
        raise ValueError(f"Conflicting settings for {'.'.join(path)} from {source}")
    target[leaf] = value


def _translate_yaml(raw: dict[str, Any]) -> dict[str, Any]:
    translated: dict[str, Any] = {key: deepcopy(value) for key, value in raw.items() if key in CANONICAL_SECTIONS}
    legacy_sections = {
        "labeling": {
            "profit_taking": ("task", "triple_barrier", "profit_taking"),
            "stop_loss": ("task", "triple_barrier", "stop_loss"),
            "vol_lookback_bars": ("task", "triple_barrier", "volatility_span"),
            "expiry_bars": ("task", "triple_barrier", "expiry_bars"),
            "min_volatility": ("task", "triple_barrier", "volatility_floor"),
        },
        "returns": {
            "return_lags": ("features", "return_lags"),
            "clip_quantile": ("features", "clip_quantile"),
            "lookback_periods": ("features", "lookback_periods"),
            "target_horizons": ("features", "target_horizons"),
        },
        "risk": {
            "max_position_size": ("allocation", "max_position_size"),
            "min_position_size": ("allocation", "min_position_size"),
        },
        "cv": {
            "n_splits": ("splits", "n_folds"),
            "embargo_bars": ("splits", "embargo_bars"),
            "cv_mode": ("splits", "diagnostic_mode"),
        },
    }
    for section, fields in legacy_sections.items():
        if section not in raw:
            continue
        warnings.warn(f"'{section}' is legacy configuration; use canonical nested sections", UserWarning, stacklevel=3)
        values = raw[section]
        if not isinstance(values, dict):
            raise ValueError(f"Legacy section '{section}' must be a mapping")
        for key, value in values.items():
            if key not in fields:
                raise ValueError(f"Unknown key '{section}.{key}' in legacy configuration")
            _set_nested(translated, fields[key], value, f"legacy {section}")

    flat_fields = {
        "raw_data_dir": ("data", "raw_data_dir"),
        "processed_data_dir": ("data", "processed_data_dir"),
        "ticker_config_path": ("data", "ticker_config_path"),
        "ingestion_start": ("data", "ingestion_start"),
        "timeframe": ("data", "timeframe"),
        "ffd_threshold": ("ffd", "threshold"),
        "ffd_max_d": ("ffd", "max_d"),
        "ffd_min_d": ("ffd", "min_d"),
        "adf_significance": ("ffd", "adf_significance"),
        "ffd_coverage_threshold": ("ffd", "coverage_threshold"),
        "null_tolerance": ("validation", "null_tolerance"),
        "min_price": ("validation", "min_price"),
        "sequence_len": ("tensor", "sequence_len"),
        "batch_size": ("tensor", "batch_size"),
        "train_cutoff_date": ("tensor", "train_cutoff_date"),
        "profit_taking": ("task", "triple_barrier", "profit_taking"),
        "stop_loss": ("task", "triple_barrier", "stop_loss"),
        "vol_lookback_bars": ("task", "triple_barrier", "volatility_span"),
        "expiry_bars": ("task", "triple_barrier", "expiry_bars"),
        "min_volatility": ("task", "triple_barrier", "volatility_floor"),
        "cv_mode": ("splits", "diagnostic_mode"),
        "n_splits": ("splits", "n_folds"),
        "embargo_bars": ("splits", "embargo_bars"),
    }
    for key, path in flat_fields.items():
        if key in raw:
            warnings.warn(f"'{key}' is a legacy flat setting; use '{'.'.join(path)}'", UserWarning, stacklevel=3)
            _set_nested(translated, path, raw[key], "legacy flat setting")

    unknown = set(raw) - CANONICAL_SECTIONS - set(legacy_sections) - set(flat_fields)
    if unknown:
        raise ValueError(f"Unknown configuration sections or keys: {sorted(unknown)}")
    return translated


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_yaml_defaults(path: str | Path = PIPELINE_CONFIG_PATH) -> dict[str, Any]:
    """Load nested YAML settings, translating legacy sections explicitly."""
    return _translate_yaml(_read_yaml(path))


def _yaml_settings_source() -> dict[str, Any]:
    return _load_yaml_defaults()


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="forbid",
        case_sensitive=False,
    )

    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    ffd: FFDConfig = Field(default_factory=FFDConfig)
    tensor: TensorConfig = Field(default_factory=TensorConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    allocation: AllocationConfig = Field(default_factory=AllocationConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)
    campaign: CampaignConfig = Field(default_factory=CampaignConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return init_settings, env_settings, dotenv_settings, _yaml_settings_source, file_secret_settings

    def __init__(self, **data: Any):
        super().__init__(**_translate_yaml(data))

    @classmethod
    def from_yaml(cls, path: str | Path = PIPELINE_CONFIG_PATH, overrides: dict[str, Any] | None = None) -> "PipelineConfig":
        values = _load_yaml_defaults(path)
        if overrides:
            values = _deep_merge(values, _translate_yaml(overrides))
        return cls(**values)

    def model_hash(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def resolved_config_hash(self) -> str:
        return self.model_hash()

    @property
    def raw_data_dir(self) -> Path:
        return self.data.raw_data_dir

    @property
    def processed_data_dir(self) -> Path:
        return self.data.processed_data_dir

    @property
    def ticker_config_path(self) -> Path:
        return self.data.ticker_config_path

    @property
    def ingestion_start(self) -> str:
        return self.data.ingestion_start

    @property
    def timeframe(self) -> Timeframe:
        return self.data.timeframe

    @property
    def calendar(self) -> str:
        return self.data.calendar

    @property
    def ingestion_interval(self) -> str:
        return TIMEFRAME_CONSTANTS[self.timeframe]["ingestion_interval"]

    @property
    def bars_per_year(self) -> int:
        return TIMEFRAME_CONSTANTS[self.timeframe]["bars_per_year"]

    @property
    def annualization_factor(self) -> float:
        return TIMEFRAME_CONSTANTS[self.timeframe]["annualization_factor"]

    @property
    def gap_tolerance(self) -> timedelta:
        return TIMEFRAME_CONSTANTS[self.timeframe]["typical_gap_tolerance"]

    @property
    def rolling_month_bars(self) -> int:
        return TIMEFRAME_CONSTANTS[self.timeframe]["rolling_month_bars"]

    @property
    def ffd_threshold(self) -> float:
        return self.ffd.threshold

    @property
    def ffd_max_d(self) -> float:
        return self.ffd.max_d

    @property
    def ffd_min_d(self) -> float:
        return self.ffd.min_d

    @property
    def adf_significance(self) -> float:
        return self.ffd.adf_significance

    @property
    def ffd_coverage_threshold(self) -> float:
        return self.ffd.coverage_threshold

    @property
    def null_tolerance(self) -> float:
        return self.validation.null_tolerance

    @property
    def min_price(self) -> float:
        return self.validation.min_price

    @property
    def sequence_len(self) -> int:
        return self.tensor.sequence_len

    @property
    def batch_size(self) -> int:
        return self.tensor.batch_size

    @property
    def train_cutoff_date(self) -> str | None:
        return self.tensor.train_cutoff_date

    @property
    def max_position_size(self) -> float:
        return self.allocation.max_position_size

    @property
    def min_position_size(self) -> float:
        return self.allocation.min_position_size

    @property
    def return_lags(self) -> list[int]:
        return self.features.return_lags

    @property
    def clip_quantile(self) -> float:
        return self.features.clip_quantile

    @property
    def lookback_periods(self) -> list[int]:
        return self.features.lookback_periods

    @property
    def target_horizons(self) -> list[int]:
        return self.features.target_horizons

    @property
    def profit_taking(self) -> float:
        return self.task.triple_barrier.profit_taking

    @property
    def stop_loss(self) -> float:
        return self.task.triple_barrier.stop_loss

    @property
    def vol_lookback_bars(self) -> int:
        return self.task.triple_barrier.volatility_span

    @property
    def expiry_bars(self) -> int:
        return self.task.triple_barrier.expiry_bars

    @property
    def min_volatility(self) -> float:
        return self.task.triple_barrier.volatility_floor

    @property
    def cv_mode(self) -> bool:
        return self.splits.diagnostic_mode

    @property
    def n_splits(self) -> int:
        return self.splits.n_folds

    @property
    def embargo_bars(self) -> int:
        return self.splits.embargo_bars


@lru_cache
def get_config() -> PipelineConfig:
    return PipelineConfig()


@lru_cache
def get_platform_config() -> PlatformConfig:
    return PlatformConfig.from_yaml()
