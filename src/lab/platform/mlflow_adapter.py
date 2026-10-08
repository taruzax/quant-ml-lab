"""Optional local MLflow adapter; model code remains MLflow-independent."""

import json
from pathlib import Path
from typing import Any

import mlflow
from lab.models.baseline import EmpiricalPriorBaseline, ZeroForecastBaseline
from lab.models.registry import create_model

try:
    from mlflow.pyfunc import PythonModel
except ImportError:  # pragma: no cover - exercised only in minimal installs
    PythonModel = object  # type: ignore[misc,assignment]


class LocalPythonModel(PythonModel):
    """Thin MLflow wrapper around a local model bundle."""

    def load_context(self, context: Any) -> None:
        model_dir = Path(context.artifacts["model_dir"])
        manifest = json.loads((model_dir / "manifest.json").read_text())
        model_name = manifest["model_name"]
        if model_name == "baseline":
            model_class = EmpiricalPriorBaseline if manifest["task"] == "classification" else ZeroForecastBaseline
            self.model = model_class.load(model_dir)
        else:
            self.model = create_model(model_name, task=manifest["task"])
            self.model = type(self.model).load(model_dir)

    def predict(self, context: Any, model_input: Any, params: dict[str, Any] | None = None):
        if getattr(self.model, "task", "regression") == "classification":
            return self.model.predict_proba(model_input)
        return self.model.predict(model_input)


def log_local_model(
    model: Any,
    *,
    artifact_path: str,
    model_dir: str | Path,
    tracking_uri: str,
    experiment_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    idempotency_key: str | None = None,
) -> dict[str, str]:
    """Record one exact local fold bundle and its metrics in local MLflow."""
    if tracking_uri.startswith("sqlite:///"):
        Path(tracking_uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    if idempotency_key is not None:
        existing = mlflow.tracking.MlflowClient().search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.quant_ml_tracking_key = '{idempotency_key}'",
            max_results=1,
        )
        if existing:
            run = existing[0]
            return {"run_id": run.info.run_id, "model_uri": f"runs:/{run.info.run_id}/{artifact_path}"}
    tags = {"quant_ml_tracking_key": idempotency_key} if idempotency_key is not None else None
    with mlflow.start_run(run_name=artifact_path, tags=tags) as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(metrics)
        info = mlflow.pyfunc.log_model(
            name=artifact_path,
            python_model=LocalPythonModel(),
            artifacts={"model_dir": str(Path(model_dir).resolve())},
        )
        return {"run_id": run.info.run_id, "model_uri": info.model_uri}
