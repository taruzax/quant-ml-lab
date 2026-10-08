"""Stable command-line adapters for local ingestion and research stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from yaml import safe_load

from lab.core.config import CampaignMatrixConfig, IngestionConfig, PipelineConfig, get_platform_config
from lab.platform.campaign_store import CampaignStore
from lab.platform.dataset_store import DatasetStore
from lab.platform.ingestion import IngestionService
from lab.platform.market_store import ProviderBatchStore, SnapshotCatalog
from lab.platform.report_store import ReportStore
from lab.research.campaign import execute_campaign, plan_campaign
from lab.research.experiment import (
    campaign_evidence,
    evaluate_holdout,
    evaluate_saved_predictions,
    index_deferred_run,
    load_run,
    run_experiment,
)
from lab.research.preprocessing import prepare_and_store_dataset
from lab.research.reporting import generate_run_report


def _emit(value: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, default=str, sort_keys=True))
    elif isinstance(value, dict):
        for key, item in value.items():
            print(f"{key}={item}")
    else:
        print(value)


def _config_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quant ML Lab research workflow")
    parser.add_argument("--json", action="store_true", dest="as_json")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("stream")
    ingest.add_argument("--config", type=Path, default=Path("config/ingestion.yaml"))

    snapshots = sub.add_parser("snapshot-list")
    snapshots.add_argument("--root", type=Path, default=get_platform_config().paths.snapshot_dir)
    snapshot = sub.add_parser("snapshot-inspect")
    snapshot.add_argument("snapshot_id")
    snapshot.add_argument("--root", type=Path, default=get_platform_config().paths.snapshot_dir)

    prepare = sub.add_parser("prepare")
    _config_option(prepare)
    prepare.add_argument("--store-root", type=Path, default=get_platform_config().paths.artifacts_dir)
    dataset_inspect = sub.add_parser("dataset-inspect")
    dataset_inspect.add_argument("dataset_id")
    dataset_inspect.add_argument("--store-root", type=Path, default=get_platform_config().paths.artifacts_dir)

    experiment = sub.add_parser("experiment")
    _config_option(experiment)
    experiment.add_argument("--dataset-id")

    campaign_plan = sub.add_parser("campaign-plan")
    campaign_plan.add_argument("--config", type=Path)
    campaign_plan.add_argument("--campaign", type=Path, required=True)
    campaign_run = sub.add_parser("campaign-run")
    campaign_run.add_argument("--config", type=Path)
    campaign_run.add_argument("--campaign", type=Path, required=True)
    campaign_status = sub.add_parser("campaign-status")
    campaign_status.add_argument("campaign_id")
    campaign_resume = sub.add_parser("campaign-resume")
    campaign_resume.add_argument("--config", type=Path)
    campaign_resume.add_argument("--campaign", type=Path, required=True)

    backtest = sub.add_parser("backtest")
    backtest.add_argument("run_id")
    backtest.add_argument("--model")
    backtest.add_argument("--allocation", type=Path)
    backtest.add_argument("--backtest-config", type=Path)

    report = sub.add_parser("report")
    report.add_argument("run_id")
    _config_option(report)
    report.add_argument("--output", type=Path, required=True)
    report_inspect = sub.add_parser("report-inspect")
    report_inspect.add_argument("report_id")
    report_inspect.add_argument("--store-root", type=Path, default=get_platform_config().paths.artifacts_dir)

    development = sub.add_parser("development")
    _config_option(development)
    development.add_argument("--store-dataset", action="store_true")

    holdout = sub.add_parser("holdout")
    holdout.add_argument("run_id")
    holdout.add_argument("--model", required=True)
    holdout.add_argument("--acknowledge-holdout", action="store_true")
    _config_option(holdout)
    return parser


def _load_overrides(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _campaign_plan(config_path: Path | None, matrix_path: Path):
    matrix = CampaignMatrixConfig.from_yaml(matrix_path)
    selected_config = config_path or matrix.base_pipeline_config
    if not selected_config.is_absolute() and not selected_config.exists():
        selected_config = matrix_path.parent / selected_config
    config = PipelineConfig.from_yaml(selected_config)
    return config, plan_campaign(config, matrix)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except Exception as exc:
        if args.as_json:
            print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}, sort_keys=True))
        else:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    as_json = args.as_json
    if args.command == "ingest":
        ingestion_config = IngestionConfig.from_yaml(args.config)
        stream = next((item for item in ingestion_config.streams if item.name == args.stream), None)
        if stream is None:
            raise KeyError(f"Unknown ingestion stream: {args.stream}")
        platform = get_platform_config()
        service = IngestionService(
            batch_store=ProviderBatchStore(platform.paths.snapshot_dir.parent / "market_batches"),
            catalog=SnapshotCatalog(platform.paths.snapshot_dir),
        )
        result = service.execute(stream)
        _emit(result.model_dump(mode="json"), as_json)
        return 0 if result.status != "failed" else 1
    if args.command == "snapshot-list":
        _emit({"snapshots": [item.model_dump(mode="json") for item in SnapshotCatalog(args.root).list()]}, as_json)
        return 0
    if args.command == "snapshot-inspect":
        catalog = SnapshotCatalog(args.root)
        entry = catalog.resolve(args.snapshot_id)
        bars = catalog.load(args.snapshot_id).bars
        _emit({**entry.model_dump(mode="json"), "rows": bars.height, "columns": bars.columns}, as_json)
        return 0
    if args.command == "prepare":
        config = PipelineConfig.from_yaml(args.config)
        reference, dataset = prepare_and_store_dataset(config, store_root=args.store_root)
        _emit({"dataset_id": reference.dataset_id, "snapshot_id": reference.snapshot_id, "snapshot_hash": reference.snapshot_hash, "rows": dataset.features.height, "bundle_path": reference.bundle_path}, as_json)
        return 0
    if args.command == "dataset-inspect":
        store = DatasetStore(args.store_root)
        reference = store.resolve(args.dataset_id)
        store.verify(reference)
        dataset = store.load(reference)
        _emit({"dataset_id": reference.dataset_id, "snapshot_id": reference.snapshot_id, "features": dataset.feature_specification, "feature_rows": dataset.features.height, "label_rows": dataset.labels.height, "split_hash": reference.split_hash, "bundle_path": reference.bundle_path}, as_json)
        return 0
    if args.command == "experiment":
        config = PipelineConfig.from_yaml(args.config)
        reference = DatasetStore(get_platform_config().paths.artifacts_dir).resolve(args.dataset_id) if args.dataset_id else None
        result = run_experiment(config, dataset_ref=reference)
        _emit({"run_id": result.ref.run_id, "snapshot_hash": result.ref.snapshot_hash, "config_hash": result.ref.config_hash, "artifact_directory": str(Path(get_platform_config().paths.artifacts_dir) / "runs" / result.ref.run_id)}, as_json)
        return 0
    if args.command in {"campaign-plan", "campaign-run", "campaign-resume"}:
        config, plan = _campaign_plan(args.config, args.campaign)
        if args.command == "campaign-plan":
            _emit({**plan.model_dump(mode="json"), "plan_json": plan.canonical_json(), "side_effects": False}, as_json)
        else:
            results = execute_campaign(config, plan)
            store = CampaignStore(get_platform_config().paths.artifacts_dir)
            campaign_result = store.finalize(plan)
            store.close()
            _emit({
                "campaign_id": plan.campaign_id,
                "results": results,
                "campaign_result": campaign_result.model_dump(mode="json") if campaign_result else None,
            }, as_json)
        return 0
    if args.command == "campaign-status":
        store = CampaignStore(get_platform_config().paths.artifacts_dir)
        entries = store.entries(args.campaign_id)
        store.close()
        _emit({"campaign_id": args.campaign_id, "entries": [item.model_dump(mode="json") for item in entries]}, as_json)
        return 0
    if args.command == "backtest":
        result = evaluate_saved_predictions(
            args.run_id,
            _load_overrides(args.allocation),
            _load_overrides(args.backtest_config),
            model_name=args.model,
        )
        _emit({"run_id": result.ref.run_id, "parent_run_id": result.ref.parent_run_id, "artifact_directory": str(Path(get_platform_config().paths.artifacts_dir) / "runs" / result.ref.run_id)}, as_json)
        return 0
    if args.command == "report":
        if args.output.exists():
            raise FileExistsError(f"Report output already exists; choose a new path to preserve prior reports: {args.output}")
        config = PipelineConfig.from_yaml(args.config)
        result = load_run(args.run_id)
        evidence = campaign_evidence(args.run_id)
        output = generate_run_report(args.output, result=result, config=config, evidence=evidence)
        run_dir = Path(get_platform_config().paths.artifacts_dir) / "runs" / args.run_id
        locator = ReportStore(get_platform_config().paths.artifacts_dir).write_locator(
            report_path=output,
            run_id=args.run_id,
            manifest_path=run_dir / "manifest.json",
            evidence=evidence.get("evidence_sidecar"),
            report_type="holdout" if result.ref.parent_run_id else "development",
        )
        _emit({"report_id": locator["report_id"], "report_path": str(output.resolve()), "locator_path": locator["locator_path"], "evidence_sidecar": evidence.get("evidence_sidecar")}, as_json)
        return 0
    if args.command == "report-inspect":
        _emit(ReportStore(args.store_root).load_locator(args.report_id), as_json)
        return 0
    if args.command == "development":
        config = PipelineConfig.from_yaml(args.config)
        if args.store_dataset:
            reference, dataset = prepare_and_store_dataset(config, store_root=get_platform_config().paths.artifacts_dir)
            result = run_experiment(config, dataset=dataset)
            dataset_id = reference.dataset_id
        else:
            result = run_experiment(config)
            dataset_id = None
        _emit({"run_id": result.ref.run_id, "dataset_id": dataset_id, "holdout_evaluated": False, "artifact_directory": str(Path(get_platform_config().paths.artifacts_dir) / "runs" / result.ref.run_id)}, as_json)
        return 0
    if args.command == "holdout":
        if not args.acknowledge_holdout:
            raise ValueError("Holdout command requires --acknowledge-holdout")
        parent = load_run(args.run_id)
        if parent.ref.parent_run_id is not None:
            raise ValueError("Holdout requires a completed development run ID")
        completed = {item["model"] for item in parent.artifact_manifest.get("candidates", []) if item.get("status") == "completed"}
        if args.model not in completed:
            raise ValueError(f"Selected model is not a completed candidate in the development run: {args.model}")
        result = evaluate_holdout(parent, config=PipelineConfig.from_yaml(args.config), model_name=args.model)
        _emit({"run_id": result.ref.run_id, "parent_run_id": result.ref.parent_run_id, "model": args.model}, as_json)
        return 0
    return 2
