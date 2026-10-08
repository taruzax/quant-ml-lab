"""Deterministic campaign planning and bounded experiment execution."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from multiprocessing import get_context
from typing import Any

import torch
from threadpoolctl import threadpool_limits

from lab.core.config import CampaignMatrixConfig, PipelineConfig, Timeframe, get_platform_config
from lab.core.contracts import CampaignPlan, CampaignRunEntry, ExperimentVariation
from lab.platform.artifacts import ArtifactStore
from lab.platform.campaign_store import CampaignStore
from lab.platform.market_store import SnapshotCatalog
from lab.core.schemas import CAUSAL_BASE_FEATURES, lagged_col, return_col, validate_feature_names
from lab.research.experiment import index_deferred_run, load_run, run_experiment


_MODEL_PARAMS = {
    "baseline": set(),
    "xgboost": {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"},
    "gru": {"hidden_size", "num_layers", "learning_rate", "epochs", "batch_size", "output_dropout", "recurrent_dropout"},
    "lstm": {"hidden_size", "num_layers", "learning_rate", "epochs", "batch_size", "output_dropout", "recurrent_dropout"},
}


def _variation_config(base: PipelineConfig, variation: ExperimentVariation) -> PipelineConfig:
    values = base.model_dump(mode="python")
    values["data"]["source"] = "snapshot"
    values["data"]["snapshot_id"] = variation.snapshot_id
    values["features"]["feature_columns"] = list(variation.feature_specification)
    for model_name, parameters in variation.parameter_overrides.items():
        if model_name not in _MODEL_PARAMS:
            raise ValueError(f"Unknown model parameter override target: {model_name}")
        unknown = set(parameters) - _MODEL_PARAMS[model_name]
        if unknown:
            raise ValueError(f"Unknown {model_name} parameters: {sorted(unknown)}")
        values["models"][model_name]["params"] = {**values["models"][model_name]["params"], **parameters}
    values["allocation"] = {**values["allocation"], **variation.allocation_overrides}
    values["backtest"] = {**values["backtest"], **variation.backtest_overrides}
    values["models"]["seed"] = variation.seed
    values["campaign"]["name"] = variation.campaign
    return PipelineConfig(**values)


def plan_campaign(base_config: PipelineConfig, matrix: CampaignMatrixConfig) -> CampaignPlan:
    snapshot_ids = matrix.snapshot_ids or ((base_config.data.snapshot_id,) if base_config.data.source == "snapshot" else ())
    if not snapshot_ids:
        raise ValueError("Campaign planning requires explicit saved snapshot_ids or data.source: snapshot")
    catalog = SnapshotCatalog(get_platform_config().paths.snapshot_dir, read_only=True)
    snapshots = {snapshot_id: catalog.load(snapshot_id) for snapshot_id in sorted(set(snapshot_ids))}
    incompatible = [
        snapshot_id for snapshot_id, snapshot in snapshots.items()
        if snapshot.calendar != base_config.data.calendar or snapshot.timeframe != base_config.data.timeframe
    ]
    if incompatible:
        raise ValueError(
            "Campaign snapshots must match the base pipeline calendar and timeframe; "
            f"incompatible snapshot IDs: {incompatible}"
        )
    available_features = tuple(
        name for name in CAUSAL_BASE_FEATURES
        if name not in {"hour_sin", "hour_cos"} or base_config.timeframe == Timeframe.H1
    ) + tuple(return_col(lag) for lag in base_config.return_lags) + tuple(
        lagged_col(lag, lookback)
        for lookback in base_config.lookback_periods
        for lag in base_config.return_lags
    )
    if matrix.feature_column_sets:
        feature_sets = matrix.feature_column_sets
    elif base_config.features.feature_columns:
        feature_sets = (tuple(base_config.features.feature_columns),)
    else:
        feature_sets = (validate_feature_names(available_features),)
    for snapshot_id, features in itertools.product(sorted(snapshots), feature_sets):
        try:
            validate_feature_names(features, set(available_features))
        except ValueError as exc:
            raise ValueError(f"Invalid campaign feature set for {snapshot_id}: {exc}") from exc
        forbidden = set(features) & {"ticker", "timestamp", "sector", "industry"}
        if forbidden:
            raise ValueError(f"Invalid campaign feature set for {snapshot_id}: identifiers are not features: {sorted(forbidden)}")
    model_overrides = matrix.model_parameter_overrides or ({},)
    seeds = matrix.seeds or (base_config.models.seed,)
    allocation_overrides = matrix.allocation_overrides or ({},)
    backtest_overrides = matrix.backtest_overrides or ({},)
    planned: dict[str, ExperimentVariation] = {}
    for snapshot_id, features, model_params, seed, allocation, backtest in itertools.product(
        sorted(snapshots), feature_sets, model_overrides, seeds, allocation_overrides, backtest_overrides
    ):
        provisional = ExperimentVariation(
            variation_id="pending",
            campaign=matrix.name,
            snapshot_id=snapshot_id,
            snapshot_hash=snapshots[snapshot_id].snapshot_hash,
            effective_config_hash="pending",
            feature_specification=tuple(features),
            seed=seed,
            parameter_overrides=dict(model_params),
            allocation_overrides=dict(allocation),
            backtest_overrides=dict(backtest),
            resource_request={"workers": matrix.worker_count, "device": base_config.models.device},
        )
        effective_config = _variation_config(base_config, provisional)
        effective_hash = effective_config.resolved_config_hash
        variation_identity = {
            "campaign": matrix.name,
            "snapshot_hash": snapshots[snapshot_id].snapshot_hash,
            "effective_config_hash": effective_hash,
        }
        variation_id = f"variation-{hashlib.sha256(json.dumps(variation_identity, sort_keys=True).encode()).hexdigest()[:24]}"
        variation = provisional.model_copy(update={"variation_id": variation_id, "effective_config_hash": effective_hash})
        planned[variation_id] = variation
    variations = tuple(planned[key] for key in sorted(planned))
    if len(variations) > matrix.max_planned_experiments:
        raise ValueError(f"Campaign plan has {len(variations)} unique experiments; maximum is {matrix.max_planned_experiments}")
    identity = {
        "campaign": matrix.name,
        "base_config_hash": base_config.resolved_config_hash,
        "variations": [item.model_dump(mode="json") for item in variations],
        "worker_count": matrix.worker_count,
        "max_planned_experiments": matrix.max_planned_experiments,
    }
    campaign_id = f"campaign-{hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]}"
    return CampaignPlan(
        campaign_id=campaign_id,
        campaign=matrix.name,
        base_config_hash=base_config.resolved_config_hash,
        variations=variations,
        max_planned_experiments=matrix.max_planned_experiments,
        worker_count=matrix.worker_count,
    )


def execute_campaign(
    base_config: PipelineConfig,
    plan: CampaignPlan,
    *,
    lease_seconds: float = 6 * 60 * 60,
) -> list[dict[str, Any]]:
    """Resume eligible variations under an expiring campaign-wide claim."""
    artifacts_root = get_platform_config().paths.artifacts_dir
    store = CampaignStore(artifacts_root)
    owner_id = f"campaign-owner-{uuid.uuid4().hex}"
    if not store.acquire_claim(plan.campaign_id, owner_id, lease_seconds=lease_seconds):
        store.close()
        raise RuntimeError(f"Campaign already has a live owner: {plan.campaign_id}")
    claim = store.claim(plan.campaign_id)
    fencing_generation = int(claim["fencing_generation"])
    stopped = threading.Event()
    lease_lost = threading.Event()
    heartbeat = threading.Thread(
        target=_renew_campaign_claim,
        args=(artifacts_root, plan.campaign_id, owner_id, fencing_generation, lease_seconds, stopped, lease_lost),
        daemon=True,
    )
    heartbeat.start()
    mutation_owner = {"owner_id": owner_id, "fencing_generation": fencing_generation}
    try:
        _require_campaign_claim(store, plan.campaign_id, owner_id, fencing_generation, lease_lost)
        store.write_plan(plan)
        existing = {entry.variation_id: entry for entry in store.entries(plan.campaign_id)}
        ready: list[ExperimentVariation] = []
        reconciled: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for variation in plan.variations:
            if lease_lost.is_set():
                raise RuntimeError(f"Campaign claim was lost: {plan.campaign_id}")
            current = existing.get(variation.variation_id)
            if current is not None and current.status == "completed":
                continue
            if current is None:
                current = CampaignRunEntry(
                    variation_id=variation.variation_id,
                    planned_identity=plan.campaign_id,
                    effective_config_hash=variation.effective_config_hash,
                    snapshot_hash=variation.snapshot_hash,
                    status="planned",
                    created_at=now,
                    updated_at=now,
                )
                _require_campaign_claim(store, plan.campaign_id, owner_id, fencing_generation, lease_lost)
                store.upsert(current, **mutation_owner)
            completed_run = _reconcile_completed_run(base_config, variation, artifacts_root)
            running = CampaignRunEntry(
                variation_id=variation.variation_id,
                planned_identity=plan.campaign_id,
                effective_config_hash=variation.effective_config_hash,
                snapshot_hash=variation.snapshot_hash,
                status="running",
                created_at=current.created_at,
                updated_at=datetime.now(timezone.utc),
                run_id=current.run_id,
                previous_status=current.status,
            )
            _require_campaign_claim(store, plan.campaign_id, owner_id, fencing_generation, lease_lost)
            store.upsert(running, **mutation_owner)
            if completed_run is not None:
                completed = CampaignRunEntry(
                    variation_id=variation.variation_id,
                    planned_identity=plan.campaign_id,
                    effective_config_hash=variation.effective_config_hash,
                    snapshot_hash=variation.snapshot_hash,
                    status="completed",
                    created_at=current.created_at,
                    updated_at=datetime.now(timezone.utc),
                    run_id=completed_run,
                    previous_status="running",
                )
                _require_campaign_claim(store, plan.campaign_id, owner_id, fencing_generation, lease_lost)
                store.upsert(completed, **mutation_owner)
                reconciled.append({"variation_id": variation.variation_id, "status": "completed", "run_id": completed_run, "reconciled": True})
            else:
                ready.append(variation)
        workers = min(plan.worker_count, len(ready)) if ready else 0
        if base_config.models.device in {"mps", "cuda"}:
            workers = min(workers, 1)
        results: list[dict[str, Any]] = list(reconciled)
        if workers <= 1:
            for variation in ready:
                _require_campaign_claim(store, plan.campaign_id, owner_id, fencing_generation, lease_lost)
                results.append(_execute_one(
                    base_config, variation, artifacts_root, plan.campaign_id, owner_id, fencing_generation,
                ))
        elif ready:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
                pending = iter(ready)
                futures = {}

                def submit_next() -> bool:
                    if lease_lost.is_set():
                        return False
                    try:
                        variation = next(pending)
                    except StopIteration:
                        return False
                    _require_campaign_claim(store, plan.campaign_id, owner_id, fencing_generation, lease_lost)
                    future = executor.submit(
                        _execute_one, base_config, variation, artifacts_root, plan.campaign_id,
                        owner_id, fencing_generation,
                    )
                    futures[future] = variation
                    return True

                for _ in range(workers):
                    if not submit_next():
                        break
                while futures:
                    if lease_lost.is_set():
                        for future in futures:
                            future.cancel()
                        raise RuntimeError(f"Campaign claim was lost: {plan.campaign_id}")
                    completed_futures, _ = wait(tuple(futures), timeout=0.1, return_when=FIRST_COMPLETED)
                    if not completed_futures:
                        continue
                    future = next(iter(completed_futures))
                    variation = futures[future]
                    del futures[future]
                    try:
                        results.append(future.result())
                    except BaseException as exc:
                        results.append({"variation_id": variation.variation_id, "status": "failed", "error": repr(exc)[:4000]})
                    submit_next()
        if lease_lost.is_set():
            raise RuntimeError(f"Campaign claim was lost: {plan.campaign_id}")
        by_id = {item.variation_id: item for item in plan.variations}
        for result in results:
            variation = by_id[result["variation_id"]]
            if not result.get("reconciled"):
                status = result["status"]
                _require_campaign_claim(store, plan.campaign_id, owner_id, fencing_generation, lease_lost)
                store.upsert(CampaignRunEntry(
                    variation_id=variation.variation_id,
                    planned_identity=plan.campaign_id,
                    effective_config_hash=variation.effective_config_hash,
                    snapshot_hash=variation.snapshot_hash,
                    status=status,
                    created_at=existing.get(variation.variation_id, CampaignRunEntry(
                        variation_id=variation.variation_id, planned_identity=plan.campaign_id,
                        effective_config_hash=variation.effective_config_hash, snapshot_hash=variation.snapshot_hash,
                        status="planned", created_at=now, updated_at=now,
                    )).created_at,
                    updated_at=datetime.now(timezone.utc),
                    run_id=result.get("run_id"),
                    error=result.get("error"),
                    previous_status="running",
                ), **mutation_owner)
            if result["status"] == "completed":
                try:
                    _require_campaign_claim(store, plan.campaign_id, owner_id, fencing_generation, lease_lost)
                    result["tracking"] = index_deferred_run(result["run_id"])
                except Exception as exc:
                    result["tracking"] = {"status": "failed", "error": repr(exc)[:1000]}
        _require_campaign_claim(store, plan.campaign_id, owner_id, fencing_generation, lease_lost)
        store.finalize(plan, **mutation_owner)
        return sorted(results, key=lambda item: item["variation_id"])
    finally:
        stopped.set()
        heartbeat.join(timeout=max(1.0, min(5.0, lease_seconds / 3)))
        store.release_claim(plan.campaign_id, owner_id, fencing_generation=fencing_generation)
        store.close()


def _renew_campaign_claim(
    artifacts_root: str | os.PathLike[str],
    campaign_id: str,
    owner_id: str,
    fencing_generation: int,
    lease_seconds: float,
    stopped: threading.Event,
    lease_lost: threading.Event,
) -> None:
    interval = max(0.1, lease_seconds / 3)
    while not stopped.wait(interval):
        store = None
        try:
            store = CampaignStore(artifacts_root)
            if not store.renew_claim(
                campaign_id, owner_id, lease_seconds=lease_seconds,
                fencing_generation=fencing_generation,
            ):
                lease_lost.set()
                return
        except Exception:
            lease_lost.set()
            return
        finally:
            if store is not None:
                store.close()


def _require_campaign_claim(
    store: CampaignStore, campaign_id: str, owner_id: str, fencing_generation: int,
    lease_lost: threading.Event,
) -> None:
    if lease_lost.is_set() or not store.assert_claim(campaign_id, owner_id, fencing_generation):
        lease_lost.set()
        raise RuntimeError(f"Campaign claim was lost: {campaign_id}")


def _reconcile_completed_run(
    base_config: PipelineConfig,
    variation: ExperimentVariation,
    artifacts_root: str | os.PathLike[str],
) -> str | None:
    expected_config = _variation_config(base_config, variation)
    runs_root = ArtifactStore(artifacts_root).root / "runs"
    if not runs_root.exists():
        return None
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        run_id = manifest_path.parent.name
        try:
            result = load_run(run_id, artifacts_dir=artifacts_root)
        except Exception:
            continue
        if result.ref.config_hash != expected_config.resolved_config_hash:
            continue
        manifest = result.artifact_manifest
        if variation.effective_config_hash != expected_config.resolved_config_hash:
            continue
        if result.ref.snapshot_hash != variation.snapshot_hash:
            continue
        if manifest.get("config", {}).get("campaign", {}).get("name") != variation.campaign:
            continue
        if result.ref.parent_run_id is not None or manifest.get("parent_run_id") is not None:
            continue
        if manifest.get("parent_model_ref") is not None:
            continue
        if any(
            manifest.get(marker) not in (None, False, "development")
            for marker in ("evaluation", "evaluation_type", "evaluation_mode", "holdout")
        ):
            continue
        if any(
            any(candidate.get(marker) not in (None, False, "development") for marker in ("evaluation", "evaluation_type", "holdout"))
            for candidate in manifest.get("candidates", []) if isinstance(candidate, dict)
        ):
            continue
        candidates = manifest.get("candidates", [])
        if not any(
            isinstance(candidate, dict)
            and candidate.get("status") == "completed"
            and candidate.get("model")
            for candidate in candidates
        ):
            continue
        return run_id
    return None


def _execute_one(
    base_config: PipelineConfig, variation: ExperimentVariation,
    artifacts_root: str | os.PathLike[str] | None = None, campaign_id: str | None = None,
    owner_id: str | None = None, fencing_generation: int | None = None,
) -> dict[str, Any]:
    if artifacts_root is not None and campaign_id is not None and owner_id is not None and fencing_generation is not None:
        store = CampaignStore(artifacts_root)
        try:
            if not store.assert_claim(campaign_id, owner_id, fencing_generation):
                return {"variation_id": variation.variation_id, "status": "cancelled", "error": "campaign claim lost before worker start"}
        finally:
            store.close()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        if "cannot set number of interop threads" not in str(exc).lower():
            raise
    started = datetime.now(timezone.utc).isoformat()
    try:
        with threadpool_limits(limits=1):
            result = run_experiment(_variation_config(base_config, variation), tracking_mode="deferred")
        return {"variation_id": variation.variation_id, "status": "completed", "run_id": result.ref.run_id, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat()}
    except Exception:
        return {"variation_id": variation.variation_id, "status": "failed", "error": traceback.format_exc()[-4000:], "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat()}
