from __future__ import annotations

import math
import platform
from importlib.metadata import PackageNotFoundError, version
from dataclasses import dataclass
from typing import Any

import polars as pl

from lab.core.config import PipelineConfig
from lab.core.contracts import FoldBundle, FoldSpec, PreparedDataset, SampleSet
from lab.models.base import BaseModel
from lab.models.devices import resolve_device
from lab.models.registry import create_model
from lab.quant.cv import purge_training_labels, scoreable_labels
from lab.research.dataset import build_sample_set
from lab.research.preprocessing import FoldPreprocessor, fit_preprocessor


@dataclass(frozen=True)
class InnerStoppingData:
    train_samples: SampleSet
    stopping_samples: SampleSet
    preprocessor: FoldPreprocessor
    diagnostics: dict[str, Any]


def _target_name(dataset: PreparedDataset) -> str:
    return "target_1b_v2" if "target_1b_v2" in dataset.labels.columns else "label"


def build_inner_stopping_data(
    dataset: PreparedDataset,
    fold: FoldSpec,
    config: PipelineConfig,
) -> InnerStoppingData:
    """Build an inner chronological train/stopping split before outer preprocessing."""
    candidates = dataset.labels.filter(
        (pl.col("decision_time") >= fold.train_start) & (pl.col("decision_time") < fold.train_end)
    )
    timestamps = sorted(set(candidates.get_column("decision_time").to_list()))
    if len(timestamps) < 2:
        raise ValueError("C15 early stopping requires at least two unique training decision timestamps")
    stopping_count = max(1, math.ceil(len(timestamps) * config.models.early_stopping_fraction))
    if stopping_count >= len(timestamps):
        raise ValueError("C15 early stopping leaves no chronological inner-training timestamps")
    stopping_start = timestamps[-stopping_count]
    inner_candidates = candidates.filter(pl.col("decision_time") < stopping_start)
    inner_train = purge_training_labels(inner_candidates, stopping_start)
    stopping, stopping_diagnostics = scoreable_labels(dataset.labels, stopping_start, fold.train_end)
    stopping = stopping.filter(pl.col("decision_time") >= fold.train_start)
    if inner_train.is_empty() or stopping.is_empty():
        raise ValueError("C15 early stopping requires non-empty purged inner train and stopping sets")
    preprocessor = fit_preprocessor(
        dataset.features,
        dataset.feature_specification,
        fold.train_start,
        stopping_start,
        config,
    )
    transformed = preprocessor.transform(dataset.features)
    train_allowed = {(row["ticker"], row["decision_time"]) for row in inner_train.to_dicts()}
    stopping_allowed = {(row["ticker"], row["decision_time"]) for row in stopping.to_dicts()}
    train_samples = build_sample_set(
        transformed,
        dataset.feature_specification,
        config.tensor.sequence_len,
        labels=inner_train,
        allowed_decisions=train_allowed,
        target_name=_target_name(dataset),
    )
    stopping_samples = build_sample_set(
        transformed,
        dataset.feature_specification,
        config.tensor.sequence_len,
        labels=stopping,
        allowed_decisions=stopping_allowed,
        target_name=_target_name(dataset),
    )
    if len(train_samples.X) == 0 or len(stopping_samples.X) == 0:
        raise ValueError("C15 early stopping produced an empty supervised inner split")
    purged_count = int(inner_candidates.height - inner_train.height)
    diagnostics = {
        "train_start": fold.train_start,
        "stopping_start": stopping_start,
        "stopping_end": fold.train_end,
        "purged_inner_rows": purged_count,
        "train_rows": len(train_samples.X),
        "stopping_rows": len(stopping_samples.X),
        "metric": "mse" if config.task.kind == "regression" else "cross_entropy",
        "stopping_score_diagnostics": stopping_diagnostics,
    }
    return InnerStoppingData(train_samples, stopping_samples, preprocessor, diagnostics)


def _model_params(config: PipelineConfig, model_name: str) -> dict[str, Any]:
    params = dict(getattr(config.models, model_name).params)
    if model_name != "baseline":
        params["seed"] = config.models.seed
        params["device"] = config.models.device
        if model_name in {"gru", "lstm"}:
            params["deterministic_policy"] = config.models.deterministic_policy
    return params


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("numpy", "torch", "xgboost", "scikit-learn"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _new_model(config: PipelineConfig, model_name: str) -> BaseModel:
    model = create_model(model_name, task=config.task.kind, params=_model_params(config, model_name))
    requested_device = "cpu" if model_name == "baseline" and config.models.device in {"mps", "cuda"} else config.models.device
    resolution = resolve_device(model_name, requested_device)
    model.device_resolution = resolution.model_dump(mode="json")
    model.runtime_metadata = {
        "requested_device": config.models.device,
        "resolved_device": resolution.resolved_device,
        "device_reason": resolution.reason,
        "seed": config.models.seed,
        "deterministic_policy": config.models.deterministic_policy,
        "package_versions": _runtime_versions(),
    }
    return model


def train_candidate(
    bundle: FoldBundle,
    config: PipelineConfig,
    *,
    model_name: str,
    dataset: PreparedDataset | None = None,
    fold: FoldSpec | None = None,
    duration: int | None = None,
) -> BaseModel:
    """Train one candidate, optionally selecting and refitting its duration."""
    target = bundle.train_samples.y
    if target is None:
        raise ValueError("Training samples must contain targets")
    if config.models.early_stopping and model_name != "baseline" and duration is None:
        if dataset is None or fold is None:
            raise ValueError("Early stopping requires the raw prepared dataset and fold specification")
        inner = build_inner_stopping_data(dataset, fold, config)
        selector = _new_model(config, model_name)
        selector.fit(
            inner.train_samples.X,
            inner.train_samples.y.reshape(-1),
            stopping_data=(inner.stopping_samples.X, inner.stopping_samples.y.reshape(-1)),
            stopping_patience=config.models.early_stopping_patience,
        )
        selected_duration = selector.selected_duration
        if selected_duration is None or selected_duration <= 0:
            raise ValueError("Early stopping did not produce a positive selected duration")
        model = _new_model(config, model_name)
        model.fit(bundle.train_samples.X, target.reshape(-1), duration=selected_duration)
        model.selected_duration = selected_duration
        model.stopping_history = selector.stopping_history
        model.stopping_config = {
            "enabled": True,
            "mode": "selected_duration_refit",
            "fraction": config.models.early_stopping_fraction,
            "patience": config.models.early_stopping_patience,
            "metric": inner.diagnostics["metric"],
        }
        model.stopping_diagnostics = {**inner.diagnostics, "selected_duration": selected_duration}
        return model
    model = _new_model(config, model_name)
    model.fit(
        bundle.train_samples.X,
        target.reshape(-1),
        duration=duration,
        stopping_patience=config.models.early_stopping_patience,
    )
    stopping_enabled = config.models.early_stopping and model_name != "baseline"
    model.stopping_config = {
        "enabled": stopping_enabled,
        "mode": "fixed_duration_refit" if stopping_enabled and duration is not None else "disabled",
        "fraction": config.models.early_stopping_fraction,
        "patience": config.models.early_stopping_patience,
    }
    return model
