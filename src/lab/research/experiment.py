"""Reusable saved baseline experiment and explicit holdout APIs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lab.core.config import PipelineConfig, get_platform_config
from lab.core.contracts import AllocationFrame, BacktestResult, CampaignEvidenceReport, FoldBundle, FoldSpec, MarketSnapshot, MetricResult, PredictionFrame, PreparedDataset, PreparedDatasetRef, RunRef, RunResult, SampleKey
from lab.models.registry import create_model
from lab.platform.artifacts import ArtifactStore, sha256_file
from lab.platform.evidence_store import EvidenceStore
from lab.platform.mlflow_adapter import log_local_model
from lab.quant.backtest import simulate_strategy
from lab.quant.metrics import classification_metrics, regression_metrics, sharpe_evidence_metrics, sharpe_statistics
from lab.research.preprocessing import prepare_dataset, prepare_fold
from lab.research.portfolio import build_allocations
from lab.research.trial_ledger import TrialLedger
from lab.research.training import train_candidate


@dataclass(frozen=True)
class FoldRun:
    fold_id: int
    bundle: FoldBundle
    model_ref: str
    predictions: PredictionFrame
    metrics: tuple[MetricResult, ...]


def _split_hash(dataset: PreparedDataset) -> str:
    payload = json.dumps(dataset.split_plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _code_identity() -> tuple[str, str]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True, text=True, check=False,
    )
    if revision.returncode != 0 or status.returncode != 0:
        return "unknown", "unknown"
    return revision.stdout.strip(), hashlib.sha256(status.stdout.encode()).hexdigest()


def train_fold(
    bundle: FoldBundle,
    config: PipelineConfig,
    *,
    model_name: str = "baseline",
    dataset: PreparedDataset | None = None,
    fold: FoldSpec | None = None,
    duration: int | None = None,
):
    return train_candidate(bundle, config, model_name=model_name, dataset=dataset, fold=fold, duration=duration)


def predict_fold(model: Any, bundle: FoldBundle, *, model_ref: str, fold_ref: str, task: str) -> PredictionFrame:
    if task == "classification":
        predictions = model.predict_proba(bundle.evaluation_samples.X)
    else:
        predictions = model.predict(bundle.evaluation_samples.X)
    return PredictionFrame(keys=bundle.evaluation_samples.keys, model_ref=model_ref, fold_ref=fold_ref, predictions=predictions)


def evaluate_predictions(prediction_frames: tuple[PredictionFrame, ...], bundles: tuple[FoldBundle, ...], *, task: str) -> tuple[MetricResult, ...]:
    actual: list[float] = []
    predicted: list[Any] = []
    for prediction, bundle in zip(prediction_frames, bundles):
        if bundle.evaluation_samples.y is None:
            continue
        actual.extend(bundle.evaluation_samples.y.reshape(-1).tolist())
        predicted.extend(prediction.predictions.tolist())
    if not actual:
        return (MetricResult(name="predictive_loss", value=None, status="unavailable", reason="no scoreable samples"),)
    if task == "classification":
        return tuple(classification_metrics(actual, np.asarray(predicted)).values())
    return tuple(regression_metrics(actual, predicted).values())


def _write_prediction(
    store: ArtifactStore,
    run_id: str,
    predictions: tuple[PredictionFrame, ...],
    relative_path: str = "predictions.json",
) -> dict[str, Any]:
    records = []
    for frame in predictions:
        for key, value in zip(frame.keys, frame.predictions):
            records.append(
                {
                    "ticker": key.ticker,
                    "timestamp": key.timestamp.isoformat(),
                    "model_ref": frame.model_ref,
                    "fold_ref": frame.fold_ref,
                    "prediction": np.asarray(value).tolist(),
                }
            )
    return store.write_json(run_id, relative_path, records)


def _combined_predictions(frames: tuple[PredictionFrame, ...]) -> list[tuple[SampleKey, np.ndarray]]:
    return [(key, np.asarray(value)) for frame in frames for key, value in zip(frame.keys, frame.predictions)]


def _strategy_result(
    dataset: PreparedDataset,
    predictions: tuple[PredictionFrame, ...],
    config: PipelineConfig,
    *,
    period_name: str,
) -> tuple[tuple[AllocationFrame, ...], BacktestResult]:
    rows = _combined_predictions(predictions)
    allocations = build_allocations(
        dataset.snapshot, rows, config.allocation,
        task=config.task.kind, backtest_config=config.backtest,
        config_hash=config.resolved_config_hash,
    )
    period = next(item for item in dataset.split_plan.execution_periods if item.name == period_name)
    backtest = simulate_strategy(dataset.snapshot, rows, allocations, config.task, config.backtest, period)
    return allocations, backtest


def persist_run(
    *,
    store: ArtifactStore,
    run_id: str,
    config: PipelineConfig,
    dataset: PreparedDataset,
    predictions: tuple[PredictionFrame, ...],
    metrics: tuple[MetricResult, ...],
    model_dirs: list[str],
    allocations: tuple[AllocationFrame, ...] = (),
    backtest: BacktestResult | None = None,
    bundles: tuple[FoldBundle, ...] = (),
    candidate_status: list[dict[str, Any]] | None = None,
    candidate_predictions: dict[str, tuple[PredictionFrame, ...]] | None = None,
    parent_run_id: str | None = None,
    parent_model_ref: str | None = None,
) -> RunResult:
    prediction_record = _write_prediction(store, run_id, predictions)
    candidate_prediction_records = []
    for candidate_name, candidate_frames in (candidate_predictions or {}).items():
        candidate_prediction_records.append(
            _write_prediction(store, run_id, candidate_frames, f"candidates/{candidate_name}/predictions.json")
        )
    metrics_record = store.write_json(run_id, "metrics.json", [metric.model_dump(mode="json") for metric in metrics])
    snapshot_record = store.write_parquet(run_id, "snapshot/bars.parquet", dataset.snapshot.bars)
    snapshot_metadata = store.write_json(run_id, "snapshot/metadata.json", dataset.snapshot.model_dump(mode="json", exclude={"bars"}))
    split_record = store.write_json(run_id, "split.json", dataset.split_plan.model_dump(mode="json"))
    allocation_record = store.write_json(run_id, "allocations.json", [item.model_dump(mode="json") for item in allocations])
    preprocessor_records = [
        store.write_json(run_id, f"preprocessors/fold-{index}.json", bundle.preprocessor.state.model_dump(mode="json"))
        for index, bundle in enumerate(bundles)
    ]
    records = [prediction_record, metrics_record, snapshot_record, snapshot_metadata, split_record, allocation_record, *preprocessor_records, *candidate_prediction_records]
    for model_dir in model_dirs:
        for name in ("manifest.json", "model.pkl"):
            path = store.run_dir(run_id) / model_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Incomplete local model bundle: {path}")
            records.append({"path": str(path.relative_to(store.run_dir(run_id))), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    backtest_paths: list[str] = []
    if backtest is not None:
        for name in ("orders", "trade_events", "equity", "exposure"):
            frame = getattr(backtest, name)
            if frame is not None:
                record = store.write_parquet(run_id, f"backtest/{name}.parquet", frame)
                records.append(record)
                backtest_paths.append(record["path"])
        series_record = store.write_json(
            run_id, "backtest/series.json",
            {
                "gross_returns": backtest.gross_returns.to_list() if backtest.gross_returns is not None else None,
                "net_returns": backtest.net_returns.to_list() if backtest.net_returns is not None else None,
                "turnover": backtest.turnover.to_list() if backtest.turnover is not None else None,
                "period": backtest.period,
                "diagnostics": backtest.diagnostics,
            },
        )
        records.append(series_record)
        backtest_paths.append(series_record["path"])
    revision, dirty_identity = _code_identity()
    manifest = {
        "config": config.model_dump(mode="json"),
        "resolved_config": config.model_dump(mode="json"),
        "config_hash": config.resolved_config_hash,
        "snapshot_hash": dataset.snapshot.snapshot_hash,
        "split_hash": _split_hash(dataset),
        "code_revision": revision,
        "dirty_state_identity": dirty_identity,
        "dependency_lock_hash": sha256_file("uv.lock") if Path("uv.lock").exists() else None,
        "seed": config.models.seed,
        "task": config.task.kind,
        "model_bundles": model_dirs,
        "parent_model_ref": parent_model_ref,
        "preprocessor_bundles": [item["path"] for item in preprocessor_records],
        "prediction_keys": [
            {"ticker": key.ticker, "timestamp": key.timestamp.isoformat()}
            for frame in predictions
            for key in frame.keys
        ],
        "candidates": candidate_status or [],
        "candidate_prediction_paths": {
            name: record["path"]
            for name, record in zip((candidate_predictions or {}).keys(), candidate_prediction_records)
        },
        "allocation_paths": [allocation_record["path"]],
        "backtest_paths": backtest_paths,
        "report_paths": [],
        "artifacts": records,
        "parent_run_id": parent_run_id,
    }
    store.finalize(run_id, manifest, required_artifacts=[item["path"] for item in records])
    ref = RunRef(
        run_id=run_id,
        snapshot_hash=dataset.snapshot.snapshot_hash,
        config_hash=config.resolved_config_hash,
        split_hash=_split_hash(dataset),
        model_ref=model_dirs[0] if model_dirs else parent_model_ref or "baseline",
        parent_run_id=parent_run_id,
    )
    return RunResult(ref=ref, predictions=predictions, allocations=allocations, metrics=metrics, backtest=backtest, snapshot=dataset.snapshot, artifact_manifest=manifest)


def run_experiment(
    config: PipelineConfig,
    dataset: PreparedDataset | None = None,
    *,
    tracking_mode: str = "immediate",
    dataset_ref: PreparedDatasetRef | None = None,
) -> RunResult:
    """Evaluate every enabled candidate, persist failures, and select one result."""
    if tracking_mode not in {"immediate", "deferred", "disabled"}:
        raise ValueError("tracking_mode must be immediate, deferred, or disabled")
    if dataset is not None and dataset_ref is not None:
        raise ValueError("Pass either dataset or dataset_ref, not both")
    prepared = dataset or prepare_dataset(config, dataset_ref=dataset_ref)
    store = ArtifactStore(get_platform_config().paths.artifacts_dir)
    run_id = store.create_run()
    ledger = TrialLedger(get_platform_config().paths.artifacts_dir / "trials.sqlite")
    logical_trial, attempt_id, _ = ledger.register_attempt(
        campaign=config.campaign.name,
        config_hash=config.resolved_config_hash,
        snapshot_hash=prepared.snapshot.snapshot_hash,
        evaluation_hash=_split_hash(prepared),
        settings={
            "task": config.task.kind,
            "models": config.models.model_dump(mode="json"),
            "frequency": config.statistics.frequency,
            "cost_treatment": "fees+slippage",
        },
    )
    bundles: list[FoldBundle] = []
    candidate_status: list[dict[str, Any]] = []
    try:
        for fold in prepared.split_plan.folds:
            bundle = prepare_fold(prepared, fold)
            bundles.append(bundle)
        candidates: list[tuple[str, tuple[PredictionFrame, ...], tuple[MetricResult, ...], list[str]]] = []
        candidate_prediction_sets: dict[str, tuple[PredictionFrame, ...]] = {}
        for model_name in ("baseline", "xgboost", "gru", "lstm"):
            spec = getattr(config.models, model_name)
            if not spec.enabled:
                candidate_status.append({"model": model_name, "status": "disabled"})
                continue
            candidate_predictions: list[PredictionFrame] = []
            candidate_dirs: list[str] = []
            tracking_refs: list[dict[str, str]] = []
            selected_durations: list[int] = []
            stopping_diagnostics: list[dict[str, Any]] = []
            try:
                for fold_index, bundle in enumerate(bundles):
                    model = train_fold(
                        bundle,
                        config,
                        model_name=model_name,
                        dataset=prepared,
                        fold=prepared.split_plan.folds[fold_index],
                    )
                    if model.selected_duration is not None:
                        selected_durations.append(int(model.selected_duration))
                    if model.stopping_diagnostics:
                        stopping_diagnostics.append(model.stopping_diagnostics)
                    model_dir = store.run_dir(run_id) / "models" / f"{model_name}-fold-{len(candidate_dirs)}"
                    model.save(model_dir)
                    model_ref = str(model_dir.relative_to(store.run_dir(run_id)))
                    candidate_dirs.append(model_ref)
                    prediction = predict_fold(model, bundle, model_ref=model_ref, fold_ref=str(len(candidate_dirs) - 1), task=config.task.kind)
                    candidate_predictions.append(prediction)
                    platform = get_platform_config()
                    if platform.mlflow.enabled and tracking_mode == "immediate":
                        fold_metrics = evaluate_predictions((prediction,), (bundle,), task=config.task.kind)
                        tracking_refs.append(
                            log_local_model(
                                model, artifact_path=f"{model_name}-fold-{len(candidate_dirs) - 1}",
                                model_dir=model_dir,
                                tracking_uri=platform.paths.mlflow_tracking_uri,
                                experiment_name=platform.mlflow.experiment_name,
                                params={
                                    "config_hash": config.resolved_config_hash,
                                    "snapshot_hash": prepared.snapshot.snapshot_hash,
                                    "split_hash": _split_hash(prepared),
                                    "task": config.task.kind,
                                    "model": model_name,
                                    "seed": config.models.seed,
                                    "fold": len(candidate_dirs) - 1,
                                    "model_bundle": model_ref,
                                },
                                metrics={item.name: item.value for item in fold_metrics if item.status == "available" and item.value is not None},
                            )
                        )
                candidate_metrics = evaluate_predictions(tuple(candidate_predictions), tuple(bundles), task=config.task.kind)
                selection_metric = next(
                    (metric for metric in candidate_metrics if metric.name in {"mse", "log_loss"} and metric.status == "available"),
                    None,
                )
                candidate_status.append(
                    {
                        "model": model_name,
                        "status": "completed",
                        "metrics": [metric.model_dump(mode="json") for metric in candidate_metrics],
                        "selection_metric": selection_metric.name if selection_metric else None,
                        "selection_value": selection_metric.value if selection_metric else None,
                        "selected_durations": selected_durations,
                        "stopping_diagnostics": stopping_diagnostics,
                        "stopping_policy": {
                            "enabled": config.models.early_stopping,
                            "fraction": config.models.early_stopping_fraction,
                            "patience": config.models.early_stopping_patience,
                        },
                        "model_bundles": candidate_dirs,
                        "tracking_mode": tracking_mode if get_platform_config().mlflow.enabled else "disabled",
                        "tracking_refs": tracking_refs,
                    }
                )
                candidates.append((model_name, tuple(candidate_predictions), candidate_metrics, candidate_dirs))
                candidate_prediction_sets[model_name] = tuple(candidate_predictions)
            except Exception as exc:
                candidate_status.append({"model": model_name, "status": "failed", "error": repr(exc)})
        if not candidates:
            raise RuntimeError("No enabled model candidate completed successfully")
        def candidate_score(candidate: tuple[str, tuple[PredictionFrame, ...], tuple[MetricResult, ...], list[str]]) -> float:
            metric = next((item for item in candidate[2] if item.name in {"mse", "log_loss"} and item.status == "available"), None)
            return float(metric.value) if metric is not None and metric.value is not None else float("inf")

        selected_name, predictions, metrics, model_dirs = min(candidates, key=lambda candidate: (candidate_score(candidate), candidate[0]))
        candidate_status.append({"selected_model": selected_name})
        allocations, backtest = _strategy_result(prepared, predictions, config, period_name="development")
        evidence_metrics = ()
        if backtest.net_returns is not None:
            evidence_metrics = tuple(sharpe_evidence_metrics(
                backtest.net_returns.to_numpy(),
                frequency=config.statistics.frequency,
                min_observations=config.statistics.min_observations,
                trial_sharpes=[],
            ).values())
            metrics = tuple(metrics) + evidence_metrics
        result = persist_run(
            store=store,
            run_id=run_id,
            config=config,
            dataset=prepared,
            predictions=tuple(predictions),
            metrics=metrics,
            model_dirs=model_dirs,
            allocations=allocations,
            backtest=backtest,
            bundles=tuple(bundles),
            candidate_status=candidate_status,
            candidate_predictions=candidate_prediction_sets,
        )
        ledger.update_attempt(
            attempt_id,
            status="completed",
            result={
                "run_id": run_id,
                "logical_trial_id": logical_trial,
                "development_net_returns": backtest.net_returns.to_list() if backtest.net_returns is not None else None,
                "development_gross_returns": backtest.gross_returns.to_list() if backtest.gross_returns is not None else None,
            },
        )
        return result
    except Exception as exc:
        store.mark_failed(run_id, repr(exc), {"logical_trial_id": logical_trial, "config_hash": config.resolved_config_hash, "candidates": candidate_status})
        ledger.update_attempt(attempt_id, status="failed", error=repr(exc))
        raise
    finally:
        ledger.close()


def load_run(run_id: str, *, artifacts_dir: str | Path | None = None) -> RunResult:
    """Load a completed run and its verified local snapshot and strategy results."""
    store = ArtifactStore(artifacts_dir or get_platform_config().paths.artifacts_dir)
    manifest = store.load_manifest(run_id)
    for record in manifest.get("artifacts", []):
        store.verify_artifact(run_id, record)
    records = json.loads((store.run_dir(run_id) / "predictions.json").read_text())
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["model_ref"], record["fold_ref"]), []).append(record)
    predictions = []
    for (model_ref, fold_ref), items in grouped.items():
        keys = tuple(SampleKey(ticker=item["ticker"], timestamp=item["timestamp"]) for item in items)
        values = np.asarray([item["prediction"] for item in items])
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.reshape(-1)
        predictions.append(PredictionFrame(keys=keys, model_ref=model_ref, fold_ref=fold_ref, predictions=values))
    metrics = tuple(MetricResult.model_validate(item) for item in json.loads((store.run_dir(run_id) / "metrics.json").read_text()))
    snapshot_metadata = json.loads((store.run_dir(run_id) / "snapshot/metadata.json").read_text())
    snapshot_metadata["bars"] = pl.read_parquet(store.run_dir(run_id) / "snapshot/bars.parquet")
    snapshot = MarketSnapshot.model_validate(snapshot_metadata)
    if snapshot.snapshot_hash != manifest["snapshot_hash"]:
        raise ValueError("Saved snapshot identity does not match the run manifest")
    allocations = tuple(
        AllocationFrame.model_validate(item)
        for item in json.loads((store.run_dir(run_id) / "allocations.json").read_text())
    )
    backtest = None
    if manifest.get("backtest_paths"):
        series = json.loads((store.run_dir(run_id) / "backtest/series.json").read_text())
        backtest = BacktestResult(
            orders=pl.read_parquet(store.run_dir(run_id) / "backtest/orders.parquet"),
            trade_events=pl.read_parquet(store.run_dir(run_id) / "backtest/trade_events.parquet"),
            equity=pl.read_parquet(store.run_dir(run_id) / "backtest/equity.parquet"),
            exposure=pl.read_parquet(store.run_dir(run_id) / "backtest/exposure.parquet"),
            gross_returns=pl.Series("gross_returns", series["gross_returns"]) if series["gross_returns"] is not None else None,
            net_returns=pl.Series("net_returns", series["net_returns"]) if series["net_returns"] is not None else None,
            turnover=pl.Series("turnover", series["turnover"]) if series["turnover"] is not None else None,
            period=series["period"],
            diagnostics=series["diagnostics"],
        )
    ref = RunRef(
        run_id=run_id,
        snapshot_hash=manifest["snapshot_hash"],
        config_hash=manifest["config_hash"],
        split_hash=manifest["split_hash"],
        model_ref=manifest.get("model_bundles", ["baseline"])[0] if manifest.get("model_bundles") else manifest.get("parent_model_ref") or "baseline",
        parent_run_id=manifest.get("parent_run_id"),
    )
    return RunResult(ref=ref, predictions=tuple(predictions), allocations=allocations, metrics=metrics, backtest=backtest, snapshot=snapshot, artifact_manifest=manifest)


def _load_prediction_frames(store: ArtifactStore, run_id: str, relative_path: str) -> tuple[PredictionFrame, ...]:
    records = json.loads((store.run_dir(run_id) / relative_path).read_text())
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["model_ref"], record["fold_ref"]), []).append(record)
    frames = []
    for (model_ref, fold_ref), items in grouped.items():
        keys = tuple(SampleKey(ticker=item["ticker"], timestamp=item["timestamp"]) for item in items)
        values = np.asarray([item["prediction"] for item in items])
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.reshape(-1)
        frames.append(PredictionFrame(keys=keys, model_ref=model_ref, fold_ref=fold_ref, predictions=values))
    return tuple(frames)


def load_candidate_predictions(run_id: str, model_name: str, *, artifacts_dir: str | Path | None = None) -> tuple[PredictionFrame, ...]:
    """Load a completed candidate's saved development predictions without retraining."""
    store = ArtifactStore(artifacts_dir or get_platform_config().paths.artifacts_dir)
    manifest = store.load_manifest(run_id)
    candidate = next(
        (item for item in manifest.get("candidates", []) if item.get("model") == model_name and item.get("status") == "completed"),
        None,
    )
    if candidate is None:
        raise ValueError(f"Candidate '{model_name}' is not a completed development candidate")
    relative_path = manifest.get("candidate_prediction_paths", {}).get(model_name)
    if relative_path is None:
        raise FileNotFoundError(f"Saved predictions are missing for candidate '{model_name}'")
    return _load_prediction_frames(store, run_id, relative_path)


def campaign_evidence(run_id: str, *, artifacts_dir: str | Path | None = None) -> dict[str, Any]:
    """Compute comparable campaign evidence from the frozen local ledger snapshot."""
    result = load_run(run_id, artifacts_dir=artifacts_dir)
    ledger = TrialLedger(Path(artifacts_dir or get_platform_config().paths.artifacts_dir) / "trials.sqlite")
    try:
        series = ledger.comparable_return_series(
            campaign=result.artifact_manifest["config"]["campaign"]["name"],
            snapshot_hash=result.ref.snapshot_hash,
            evaluation_hash=result.ref.split_hash,
            frequency=result.artifact_manifest["config"]["statistics"]["frequency"],
            cost_treatment="fees+slippage",
            task=result.artifact_manifest.get("task"),
        )
        trial_sharpes = []
        for item in series["complete"]:
            try:
                trial_sharpes.append(sharpe_statistics(item["returns"], min_observations=2)["sharpe"])
            except ValueError:
                continue
        current_returns = result.backtest.net_returns.to_numpy() if result.backtest is not None and result.backtest.net_returns is not None else []
        metrics = sharpe_evidence_metrics(
            current_returns,
            trial_sharpes=trial_sharpes,
            frequency=result.artifact_manifest["config"]["statistics"]["frequency"],
            min_observations=result.artifact_manifest["config"]["statistics"]["min_observations"],
        )
        report = CampaignEvidenceReport(
            source_run_id=run_id,
            campaign=result.artifact_manifest["config"]["campaign"]["name"],
            comparison_dimensions={
                "snapshot_hash": result.ref.snapshot_hash,
                "evaluation_hash": result.ref.split_hash,
                "task": result.artifact_manifest.get("task"),
                "frequency": result.artifact_manifest["config"]["statistics"]["frequency"],
                "cost_treatment": "fees+slippage",
            },
            included_logical_trial_ids=tuple(item["logical_trial_id"] for item in series["complete"]),
            selected_attempt_ids=tuple(item["attempt_id"] for item in series["complete"]),
            exclusions=tuple(series["exclusions"]),
            population_counts={
                "logical_trials": series["population"],
                "selected_attempts": len(series["complete"]),
                "excluded_attempts": len(series["exclusions"]),
            },
            metric_values={name: metric.value for name, metric in metrics.items()},
            metric_details={
                name: {"status": metric.status, "reason": metric.reason, "conventions": metric.conventions}
                for name, metric in metrics.items()
            },
            calculated_at=datetime.now(timezone.utc),
        )
        sidecar = EvidenceStore(artifacts_dir or get_platform_config().paths.artifacts_dir).write(report.model_dump(mode="json"))
        return {**report.model_dump(mode="json"), "evidence_sidecar": sidecar, "metrics": metrics}
    finally:
        ledger.close()


def index_deferred_run(run_id: str, *, artifacts_dir: str | Path | None = None) -> dict[str, Any]:
    """Index deferred campaign fold bundles serially without changing the run manifest."""
    store = ArtifactStore(artifacts_dir or get_platform_config().paths.artifacts_dir)
    run_dir = store.run_dir(run_id)
    sidecar_path = run_dir / "tracking" / "index.json"
    if sidecar_path.exists():
        return json.loads(sidecar_path.read_text())
    result = load_run(run_id, artifacts_dir=artifacts_dir)
    platform = get_platform_config()
    if not platform.mlflow.enabled:
        return {"status": "disabled", "run_id": run_id, "references": []}
    references: list[dict[str, str]] = []
    errors: list[str] = []
    config = PipelineConfig.model_validate(result.artifact_manifest["config"])
    for candidate in result.artifact_manifest.get("candidates", []):
        model_name = candidate.get("model")
        if candidate.get("status") != "completed" or model_name not in {"baseline", "xgboost", "gru", "lstm"}:
            continue
        model_bundles = candidate.get("model_bundles", [])
        metrics = {
            metric["name"]: float(metric["value"])
            for metric in candidate.get("metrics", [])
            if metric.get("status") == "available" and metric.get("value") is not None
        }
        for fold_index, model_ref in enumerate(model_bundles):
            model_dir = run_dir / model_ref
            try:
                model = create_model(model_name, task=config.task.kind)
                model = type(model).load(model_dir)
                references.append(log_local_model(
                    model,
                    artifact_path=f"{run_id}-{model_name}-fold-{fold_index}",
                    model_dir=model_dir,
                    tracking_uri=platform.paths.mlflow_tracking_uri,
                    experiment_name=platform.mlflow.experiment_name,
                    params={
                        "run_id": run_id,
                        "config_hash": result.ref.config_hash,
                        "snapshot_hash": result.ref.snapshot_hash,
                        "split_hash": result.ref.split_hash,
                        "task": config.task.kind,
                        "model": model_name,
                        "seed": config.models.seed,
                        "fold": fold_index,
                        "model_bundle": model_ref,
                    },
                    metrics=metrics,
                    idempotency_key=f"{run_id}:{model_name}:{fold_index}",
                ))
            except Exception as exc:
                errors.append(f"{model_name} fold {fold_index}: {exc!r}")
    sidecar = {
        "schema_version": "tracking-index.v1",
        "run_id": run_id,
        "status": "indexed" if not errors else "partial_failure",
        "references": references,
        "errors": errors,
    }
    if not errors:
        store.write_json(run_id, "tracking/index.json", sidecar)
    return sidecar


def evaluate_holdout(
    selected_run: RunResult | str,
    *,
    config: PipelineConfig | None = None,
    model_name: str | None = None,
    allocation_overrides: dict[str, Any] | None = None,
    backtest_overrides: dict[str, Any] | None = None,
) -> RunResult:
    """Evaluate an explicitly selected development run against its saved snapshot."""
    run_id = selected_run if isinstance(selected_run, str) else selected_run.ref.run_id
    result = load_run(run_id)
    if result.ref.parent_run_id is not None or not result.artifact_manifest.get("candidates"):
        raise ValueError("Holdout requires a completed, explicitly selected development run")
    original = PipelineConfig(**result.artifact_manifest["config"])
    config = config or original
    if allocation_overrides or backtest_overrides:
        values = config.model_dump(mode="python")
        values["allocation"] = {**values["allocation"], **(allocation_overrides or {})}
        values["backtest"] = {**values["backtest"], **(backtest_overrides or {})}
        config = PipelineConfig(**values)
    if config.resolved_config_hash != result.ref.config_hash and not (allocation_overrides or backtest_overrides):
        raise ValueError("Holdout configuration differs from the selected development run")
    if result.snapshot is None:
        raise ValueError("Selected run has no saved snapshot")
    store = ArtifactStore(get_platform_config().paths.artifacts_dir)
    selection_suffix = hashlib.sha256(json.dumps([model_name, allocation_overrides, backtest_overrides], sort_keys=True, default=str).encode()).hexdigest()[:12]
    child_id = f"{run_id}-holdout-{selection_suffix}"
    if (store.run_dir(child_id) / "manifest.json").exists():
        return load_run(child_id)
    existing_holdout_variants = list((store.root / "runs").glob(f"{run_id}-holdout-*"))
    exploratory_reuse = bool(existing_holdout_variants)
    dataset = prepare_dataset(config, snapshot=result.snapshot)
    if dataset.snapshot.snapshot_hash != result.ref.snapshot_hash or _split_hash(dataset) != result.ref.split_hash:
        raise ValueError("Saved snapshot or evaluation schedule changed")
    selected = model_name or next((item.get("selected_model") for item in result.artifact_manifest.get("candidates", []) if item.get("selected_model")), None)
    if selected is None:
        raise ValueError("Selected development candidate is missing")
    selected_candidate = next(
        (item for item in result.artifact_manifest.get("candidates", []) if item.get("model") == selected and item.get("status") == "completed"),
        None,
    )
    if selected_candidate is None:
        raise ValueError(f"Candidate '{selected}' is not a completed development candidate")
    model_name = selected
    selected_candidate = next(
        item for item in result.artifact_manifest.get("candidates", [])
        if item.get("model") == model_name and item.get("status") == "completed"
    )
    durations = [int(value) for value in selected_candidate.get("selected_durations", []) if int(value) > 0]
    holdout_duration = int(np.floor(np.median(durations) + 0.5)) if durations else None
    ledger = TrialLedger(get_platform_config().paths.artifacts_dir / "trials.sqlite")
    logical, attempt, _ = ledger.register_attempt(
        campaign=config.campaign.name,
        config_hash=config.resolved_config_hash,
        snapshot_hash=result.ref.snapshot_hash,
        evaluation_hash=result.ref.split_hash,
        settings={"evaluation": "holdout", "selected_run": run_id, "model": selected, "allocation": config.allocation.model_dump(mode="json"), "backtest": config.backtest.model_dump(mode="json")},
    )
    store.create_run(child_id)
    holdout_spec = FoldSpec(
        fold_id=-1,
        train_start=dataset.split_plan.development_start,
        train_end=dataset.split_plan.holdout_start,
        validation_start=dataset.split_plan.holdout_start,
        validation_end=dataset.split_plan.holdout_end,
    )
    try:
        bundle = prepare_fold(dataset, holdout_spec)
        model = train_fold(bundle, config, model_name=selected, duration=holdout_duration)
        model_dir = store.run_dir(child_id) / "models" / f"{model_name}-holdout"
        model.save(model_dir)
        model_ref = str(model_dir.relative_to(store.run_dir(child_id)))
        prediction = predict_fold(model, bundle, model_ref=model_ref, fold_ref="holdout", task=config.task.kind)
        metrics = evaluate_predictions((prediction,), (bundle,), task=config.task.kind)
        allocations, backtest = _strategy_result(dataset, (prediction,), config, period_name="holdout")
        holdout_result = persist_run(
            store=store, run_id=child_id, config=config, dataset=dataset,
            predictions=(prediction,), metrics=metrics, model_dirs=[model_ref],
            allocations=allocations, backtest=backtest, bundles=(bundle,),
            candidate_status=[{
                "selected_model": selected,
                "evaluation": "holdout",
                "selection_source": "explicit" if model_name else "development_default",
                "exploratory_holdout_reuse": exploratory_reuse,
                "allocation": config.allocation.model_dump(mode="json"),
                "backtest": config.backtest.model_dump(mode="json"),
            }],
            parent_run_id=run_id,
        )
        ledger.update_attempt(attempt, status="completed", result={"run_id": child_id, "logical_trial_id": logical})
        return holdout_result
    except Exception as exc:
        store.mark_failed(child_id, repr(exc), {"parent_run_id": run_id})
        ledger.update_attempt(attempt, status="failed", error=repr(exc))
        raise
    finally:
        ledger.close()


def evaluate_saved_predictions(
    run_id: str,
    allocation_overrides: dict[str, Any] | None = None,
    backtest_overrides: dict[str, Any] | None = None,
    *,
    model_name: str | None = None,
) -> RunResult:
    """Evaluate an immutable development strategy variant from saved predictions."""
    parent = load_run(run_id)
    if parent.ref.parent_run_id is not None or parent.snapshot is None:
        raise ValueError("Strategy comparison requires a saved development run")
    values = dict(parent.artifact_manifest["config"])
    values["allocation"] = {**values["allocation"], **(allocation_overrides or {})}
    values["backtest"] = {**values["backtest"], **(backtest_overrides or {})}
    config = PipelineConfig(**values)
    store = ArtifactStore(get_platform_config().paths.artifacts_dir)
    selected_predictions = parent.predictions if model_name is None else load_candidate_predictions(run_id, model_name)
    suffix = hashlib.sha256(json.dumps([model_name, allocation_overrides, backtest_overrides], sort_keys=True, default=str).encode()).hexdigest()[:12]
    child_id = f"{run_id}-child-{suffix}"
    if (store.run_dir(child_id) / "manifest.json").exists():
        return load_run(child_id)
    ledger = TrialLedger(get_platform_config().paths.artifacts_dir / "trials.sqlite")
    logical, attempt, _ = ledger.register_attempt(
        campaign=config.campaign.name, config_hash=config.resolved_config_hash,
        snapshot_hash=parent.ref.snapshot_hash, evaluation_hash=parent.ref.split_hash,
        settings={"evaluation": "development", "parent_prediction_run": run_id, "allocation": config.allocation.model_dump(mode="json"), "backtest": config.backtest.model_dump(mode="json")},
    )
    store.create_run(child_id)
    try:
        dataset = prepare_dataset(config, snapshot=parent.snapshot)
        if _split_hash(dataset) != parent.ref.split_hash:
            raise ValueError("Saved evaluation schedule changed")
        allocations, backtest = _strategy_result(dataset, selected_predictions, config, period_name="development")
        result = persist_run(
            store=store, run_id=child_id, config=config, dataset=dataset,
            predictions=selected_predictions, metrics=parent.metrics, model_dirs=[],
            allocations=allocations, backtest=backtest,
            parent_run_id=run_id, parent_model_ref=model_name or parent.ref.model_ref,
            candidate_status=[{"model": model_name or parent.ref.model_ref, "selection_source": "saved_development_predictions"}],
        )
        ledger.update_attempt(attempt, status="completed", result={"run_id": child_id, "logical_trial_id": logical})
        return result
    except Exception as exc:
        store.mark_failed(child_id, repr(exc), {"parent_run_id": run_id})
        ledger.update_attempt(attempt, status="failed", error=repr(exc))
        raise
    finally:
        ledger.close()
