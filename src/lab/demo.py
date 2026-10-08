from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from lab.core.config import PipelineConfig, Timeframe, get_platform_config
from lab.platform.synthetic_data import demo_metadata, generate_demo_bars
from lab.research.experiment import evaluate_holdout, run_experiment
from lab.research.reporting import generate_demo_report, write_demo_locator


def _selected_model(candidate_status: list[dict[str, Any]]) -> str:
    """Return the completed model selected by the development run."""
    selected = next(
        (item.get("selected_model") for item in candidate_status if item.get("selected_model")),
        None,
    )
    if not isinstance(selected, str):
        raise RuntimeError("Demo development run did not record a selected model")
    candidate = next(
        (
            item
            for item in candidate_status
            if item.get("model") == selected and item.get("status") == "completed"
        ),
        None,
    )
    if candidate is None:
        raise RuntimeError(f"Demo selected model is not a completed candidate: {selected}")
    return selected


def _require_backtest(result: Any, label: str) -> None:
    """Reject a result that cannot support saved return reporting."""
    if result.backtest is None or result.backtest.net_returns is None:
        raise RuntimeError(f"Demo {label} result has no saved net return series")


def run_demo(config_path: str | Path = "config/demo.yaml", *, timeframe: str | None = None) -> Path:
    """Run the finite bundled-data development and explicit holdout demonstration."""
    config = PipelineConfig.from_yaml(Path(config_path))
    if config.data.source != "bundled_demo":
        raise ValueError("The offline demo requires data.source: bundled_demo")
    if timeframe is not None:
        selected_timeframe = Timeframe(timeframe)
        if selected_timeframe == Timeframe.H1:
            demo_root = Path(get_platform_config().paths.artifacts_dir) / "demo"
            input_path = demo_root / "input-hourly.csv"
            metadata_path = demo_root / "input-hourly.yaml"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            generate_demo_bars(timeframe=selected_timeframe).write_csv(input_path)
            metadata_path.write_text(yaml.safe_dump(demo_metadata(timeframe=selected_timeframe), sort_keys=False))
            config = PipelineConfig.from_yaml(Path(config_path), overrides={"data": {"timeframe": "1h", "input_path": input_path, "metadata_path": metadata_path}})
    development = run_experiment(config)
    _require_backtest(development, "development")
    selected_model = _selected_model(development.artifact_manifest.get("candidates", []))
    holdout = evaluate_holdout(development, config=config, model_name=selected_model)
    _require_backtest(holdout, "holdout")
    report_root = Path(get_platform_config().paths.artifacts_dir) / "demo"
    report = generate_demo_report(
        report_root / f"{development.ref.run_id}.html",
        development=development,
        holdout=holdout,
        config=config,
        selected_model=selected_model,
    )
    locator = write_demo_locator(
        report_root / f"{development.ref.run_id}.locator.json",
        development=development,
        holdout=holdout,
        report_path=report,
        selected_model=selected_model,
        artifacts_root=Path(get_platform_config().paths.artifacts_dir),
    )
    print(f"development_run_id={development.ref.run_id}")
    print(f"holdout_run_id={holdout.ref.run_id}")
    print(f"selected_model={selected_model}")
    print(f"report_path={report.resolve()}")
    print(f"locator_path={locator.resolve()}")
    print(f"development_run_dir={Path(get_platform_config().paths.artifacts_dir, 'runs', development.ref.run_id).resolve()}")
    print(f"holdout_run_dir={Path(get_platform_config().paths.artifacts_dir, 'runs', holdout.ref.run_id).resolve()}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the finite offline research demo")
    parser.add_argument("--config", type=Path, default=Path("config/demo.yaml"))
    parser.add_argument("--timeframe", choices=[item.value for item in Timeframe])
    args = parser.parse_args(argv)
    run_demo(args.config, timeframe=args.timeframe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
