from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import polars as pl

from lab.core.config import PipelineConfig
from lab.core.contracts import FoldBundle, MarketSnapshot, PreparedDataset, PreparedDatasetRef, PreprocessingState
from lab.core.schemas import validate_feature_names
from lab.platform.data_access import load_market_snapshot
from lab.platform.dataset_store import DatasetStore
from lab.quant.cv import fold_training_labels, scoreable_labels
from lab.quant.features import apply_all_features, feature_specification
from lab.quant.ffd import find_min_d, frac_diff_ffd, get_weights_ffd
from lab.quant.labeling import compute_labels
from lab.research.dataset import build_sample_set
from lab.research.splits import build_split_plan


@dataclass
class FoldPreprocessor:
    state: PreprocessingState

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        result = _apply_fitted_ffd(frame, self.state)
        expressions: list[pl.Expr] = []
        for column in self.state.feature_order:
            if column not in result.columns:
                raise ValueError(f"Feature column disappeared before transform: {column}")
            expression = pl.col(column)
            if column in self.state.vocabularies:
                vocabulary = self.state.vocabularies[column]
                expression = pl.when(pl.col(column).is_null()).then(None)
                for code, value in enumerate(vocabulary):
                    expression = expression.when(pl.col(column) == value).then(float(code))
                expression = expression.otherwise(-1.0)
            if column in self.state.clip_thresholds:
                lower, upper = self.state.clip_thresholds[column]
                expression = expression.clip(lower, upper)
            if column in self.state.means:
                mean = self.state.means[column]
                scale = self.state.scales[column]
                expression = (expression - mean) / scale
            expressions.append(expression.alias(column))
        return result.with_columns(expressions)


def _apply_fitted_ffd(frame: pl.DataFrame, state: PreprocessingState) -> pl.DataFrame:
    """Apply the fitted FFD state before any fold-fitted postprocessing."""
    if not state.ffd_features:
        return frame
    if state.ffd_features != ("close",):
        raise ValueError(f"Unsupported FFD feature set: {state.ffd_features}")
    if any(order is None for order in state.ffd_orders.values()):
        raise ValueError("FFD state contains a ticker without an acceptable fitted order")
    tickers = set(frame.get_column("ticker").cast(pl.Utf8).to_list())
    missing = sorted(tickers - set(state.ffd_orders))
    if missing:
        raise ValueError(f"FFD state is missing fitted orders for tickers: {missing}")
    return frac_diff_ffd(
        frame,
        col_name="close",
        orders_by_ticker={ticker: float(order) for ticker, order in state.ffd_orders.items() if order is not None},
        threshold=state.ffd_threshold,
        output_col="close",
        log_input=True,
    )


def _numeric_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    values = frame.get_column(column).drop_nulls().cast(pl.Float64).to_numpy()
    return values[np.isfinite(values)]


def _fit_ffd_diagnostics(
    frame: pl.DataFrame, feature_columns: tuple[str, ...], config: PipelineConfig
) -> tuple[dict[str, float | None], dict[str, int], dict[str, Any]]:
    orders: dict[str, float | None] = {}
    truncation: dict[str, int] = {}
    diagnostics: dict[str, Any] = {}
    if "close" not in feature_columns:
        return orders, truncation, diagnostics
    for ticker, group in frame.group_by("ticker", maintain_order=True):
        ticker_name = str(ticker[0] if isinstance(ticker, tuple) else ticker)
        values = _numeric_values(group, "close")
        if values.size < 50:
            diagnostics[ticker_name] = {"status": "insufficient_history", "rows": int(values.size)}
            continue
        try:
            orders[ticker_name] = float(
                find_min_d(
                    group.with_columns(pl.col("close").log().alias("log_close")),
                    col_name="log_close",
                    max_d=config.ffd.max_d,
                    threshold=config.ffd.threshold,
                    min_d=config.ffd.min_d,
                    significance=config.ffd.adf_significance,
                )
            )
            truncation[ticker_name] = len(get_weights_ffd(orders[ticker_name], config.ffd.threshold))
            diagnostics[ticker_name] = {"status": "available", "rows": int(values.size)}
        except Exception as exc:
            orders[ticker_name] = None
            diagnostics[ticker_name] = {"status": "failed", "reason": str(exc)}
    return orders, truncation, diagnostics


def fit_preprocessor(
    frame: pl.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    train_start,
    train_end,
    config: PipelineConfig,
) -> FoldPreprocessor:
    feature_columns = validate_feature_names(tuple(feature_columns), set(frame.columns))
    train = frame.filter((pl.col("timestamp") >= train_start) & (pl.col("timestamp") < train_end))
    if train.is_empty():
        raise ValueError("Fold has no train history available for preprocessing")
    ffd_orders, ffd_truncation, ffd_diagnostics = _fit_ffd_diagnostics(train, feature_columns, config)
    ffd_state = PreprocessingState(
        feature_order=feature_columns,
        ffd_orders=ffd_orders,
        ffd_truncation=ffd_truncation,
        ffd_features=("close",) if "close" in feature_columns else (),
        ffd_threshold=config.ffd.threshold,
        diagnostics={"ffd": ffd_diagnostics, "fit_start": train_start, "fit_end": train_end},
    )
    train = _apply_fitted_ffd(train, ffd_state)
    clip_thresholds: dict[str, tuple[float, float]] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    vocabularies: dict[str, tuple[str, ...]] = {}
    for column in feature_columns:
        if train.schema[column] in {pl.Utf8, pl.Categorical}:
            values = tuple(sorted(set(train.get_column(column).drop_nulls().cast(pl.Utf8).to_list())))
            vocabularies[column] = values
            continue
        values = _numeric_values(train, column)
        if values.size == 0:
            raise ValueError(f"Feature {column} has no finite train history")
        lower = float(np.quantile(values, config.features.clip_quantile))
        upper = float(np.quantile(values, 1.0 - config.features.clip_quantile))
        clipped = np.clip(values, lower, upper)
        mean = float(np.mean(clipped))
        scale = float(np.std(clipped))
        clip_thresholds[column] = (lower, upper)
        means[column] = mean
        scales[column] = scale if scale > 0.0 else 1.0
    state = PreprocessingState(
        feature_order=feature_columns,
        clip_thresholds=clip_thresholds,
        vocabularies=vocabularies,
        means=means,
        scales=scales,
        ffd_orders=ffd_orders,
        ffd_truncation=ffd_truncation,
        ffd_features=ffd_state.ffd_features,
        ffd_threshold=config.ffd.threshold,
        diagnostics={"ffd": ffd_diagnostics, "fit_start": train_start, "fit_end": train_end},
    )
    return FoldPreprocessor(state=state)


def prepare_dataset(
    config: PipelineConfig,
    snapshot: MarketSnapshot | None = None,
    *,
    dataset_ref: PreparedDatasetRef | None = None,
) -> PreparedDataset:
    """Load one snapshot and produce causal features, labels, and split plan."""
    if dataset_ref is not None:
        dataset = DatasetStore().load(dataset_ref)
        if dataset_ref.config_hash != config.resolved_config_hash:
            raise ValueError("Prepared dataset was created from a different effective pipeline configuration")
        if dataset_ref.snapshot_hash != dataset.snapshot.snapshot_hash:
            raise ValueError("Prepared dataset snapshot identity does not match its reference")
        return dataset
    snapshot = snapshot if snapshot is not None else load_market_snapshot(config)
    feature_frame = apply_all_features(snapshot.bars, config)
    feature_columns = feature_specification(feature_frame, config)
    label_result = compute_labels(snapshot, config.task)
    split_plan = build_split_plan(snapshot, config.splits)
    return PreparedDataset(
        snapshot=snapshot,
        features=feature_frame,
        feature_specification=feature_columns,
        labels=label_result.labels,
        split_plan=split_plan,
        exclusions=label_result.exclusions,
        config=config,
    )


def prepare_and_store_dataset(
    config: PipelineConfig,
    snapshot: MarketSnapshot | None = None,
    *,
    store_root=None,
) -> tuple[PreparedDatasetRef, PreparedDataset]:
    """Prepare a dataset once and persist its immutable reusable bundle."""
    dataset = prepare_dataset(config, snapshot=snapshot)
    split_hash = hashlib.sha256(dataset.split_plan.model_dump_json().encode()).hexdigest()
    reference = DatasetStore(store_root or "artifacts").write(
        dataset,
        config_hash=config.resolved_config_hash,
        split_hash=split_hash,
    )
    return reference, dataset


def prepare_fold(dataset: PreparedDataset, fold_spec) -> FoldBundle:
    """Fit fold state on train history and construct aligned train/evaluation windows."""
    config = dataset.config if isinstance(dataset.config, PipelineConfig) else PipelineConfig()
    preprocessor = fit_preprocessor(
        dataset.features,
        dataset.feature_specification,
        fold_spec.train_start,
        fold_spec.train_end,
        config,
    )
    transformed = preprocessor.transform(dataset.features)
    train_labels = fold_training_labels(dataset.labels, fold_spec)
    evaluation_labels, score_diagnostics = scoreable_labels(
        dataset.labels,
        fold_spec.validation_start,
        fold_spec.validation_end,
    )
    train_allowed = {(row["ticker"], row["decision_time"]) for row in train_labels.to_dicts()}
    evaluation_allowed = {(row["ticker"], row["decision_time"]) for row in evaluation_labels.to_dicts()}
    target_name = "target_1b_v2" if "target_1b_v2" in dataset.labels.columns else "label"
    train_samples = build_sample_set(
        transformed,
        dataset.feature_specification,
        config.tensor.sequence_len,
        labels=train_labels,
        allowed_decisions=train_allowed,
        target_name=target_name,
    )
    evaluation_samples = build_sample_set(
        transformed,
        dataset.feature_specification,
        config.tensor.sequence_len,
        labels=evaluation_labels,
        allowed_decisions=evaluation_allowed,
        target_name=target_name,
    )
    return FoldBundle(
        train_samples=train_samples,
        evaluation_samples=evaluation_samples,
        scoring_keys=evaluation_samples.keys,
        preprocessor=preprocessor,
        boundaries={
            "train_start": fold_spec.train_start,
            "train_end": fold_spec.train_end,
            "validation_start": fold_spec.validation_start,
            "validation_end": fold_spec.validation_end,
        },
        diagnostics={"scoring": score_diagnostics},
    )
