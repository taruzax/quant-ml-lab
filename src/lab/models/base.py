from abc import ABC, abstractmethod
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from lab.core.contracts import WindowSet


class BaseModel(ABC):
    """Plain local model contract shared by every research adapter."""

    model_name = "base"

    def __init__(self, *, task: str = "regression") -> None:
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        self.task = task
        self.training_history: list[dict[str, float]] = []
        self.stopping_history: list[dict[str, float]] | None = None
        self.stopping_config: dict[str, Any] = {}
        self.stopping_diagnostics: dict[str, Any] = {}
        self.selected_duration: int | None = None
        self.feature_dim: int | None = None

    @abstractmethod
    def fit(
        self,
        X: Any,
        y: Any,
        stopping_data: tuple[Any, Any] | None = None,
        *,
        duration: int | None = None,
        stopping_patience: int = 5,
    ) -> "BaseModel":
        """Fit model parameters and return the fitted adapter."""
        ...

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Generate regression forecasts for input data."""
        ...

    def predict_proba(self, X: Any) -> np.ndarray:
        """Generate classification probabilities in [-1, 0, +1] order."""
        raise NotImplementedError(f"{type(self).__name__} does not implement classification")

    def save(self, directory: str | Path) -> Path:
        """Write a typed manifest and native local payload atomically enough for local runs."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        payload = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        payload_path = path / "model.pkl"
        temporary_payload = path / "model.pkl.tmp"
        temporary_manifest = path / "manifest.json.tmp"
        temporary_payload.write_bytes(payload)
        architecture = {
            key: getattr(self, key)
            for key in (
                "cell_type", "hidden_size", "num_layers", "input_dim", "output_dropout",
                "recurrent_dropout", "n_estimators", "max_depth", "learning_rate",
            )
            if hasattr(self, key)
        }
        class_mapping = getattr(self, "classes_seen", getattr(self, "classes", None))
        if isinstance(class_mapping, np.ndarray):
            class_mapping = class_mapping.tolist()
        manifest = {
            "schema_version": "model-manifest.v1",
            "model_name": self.model_name,
            "task": self.task,
            "feature_dim": self.feature_dim,
            "selected_duration": self.selected_duration,
            "stopping_config": self.stopping_config,
            "stopping_diagnostics": self.stopping_diagnostics,
            "architecture": architecture,
            "class_mapping": class_mapping,
            "seed": getattr(self, "seed", getattr(self, "random_state", None)),
            "device": getattr(self, "device", "cpu"),
            "device_resolution": getattr(self, "device_resolution", None),
            "runtime_metadata": getattr(self, "runtime_metadata", {}),
            "payload": payload_path.name,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        temporary_payload.replace(payload_path)
        temporary_manifest.replace(path / "manifest.json")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "BaseModel":
        """Load only a local trusted model payload and verify its manifest."""
        path = Path(directory)
        manifest_path = path / "manifest.json"
        payload_path = path / "model.pkl"
        if not manifest_path.exists() or not payload_path.exists():
            raise FileNotFoundError(f"Incomplete model bundle: {path}")
        manifest = json.loads(manifest_path.read_text())
        payload = payload_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest.get("payload_sha256"):
            raise ValueError("Model payload checksum mismatch")
        model = pickle.loads(payload)
        if not isinstance(model, cls):
            raise TypeError(f"Model bundle contains {type(model).__name__}, expected {cls.__name__}")
        if manifest.get("model_name") != model.model_name or manifest.get("task") != model.task:
            raise ValueError("Model manifest does not match the local payload")
        return model

    @staticmethod
    def validate_inputs(X: Any, y: Any | None = None) -> tuple[Any, np.ndarray | None]:
        if isinstance(X, WindowSet):
            if len(X) == 0:
                raise ValueError("Model features must contain at least one finite row")
            targets = None if y is None else np.asarray(y)
            if targets is not None:
                try:
                    finite_targets = np.isfinite(targets).all()
                except TypeError as exc:
                    raise ValueError("Model features and targets must be numeric") from exc
                if targets.ndim == 0 or targets.shape[0] != len(X) or not finite_targets:
                    raise ValueError("Targets must align with finite feature rows")
            return X, targets
        features = np.asarray(X)
        if features.ndim not in {2, 3}:
            raise ValueError("Model features must have shape [N, F] or [N, T, F]")
        try:
            finite_features = np.isfinite(features).all()
        except TypeError as exc:
            raise ValueError("Model features and targets must be numeric") from exc
        if features.shape[0] == 0 or not finite_features:
            raise ValueError("Model features must contain at least one finite row")
        targets = None if y is None else np.asarray(y)
        if targets is not None:
            try:
                finite_targets = np.isfinite(targets).all()
            except TypeError as exc:
                raise ValueError("Model features and targets must be numeric") from exc
            if targets.shape[0] != features.shape[0] or not finite_targets:
                raise ValueError("Targets must align with finite feature rows")
        return features, targets

    def validate_features(self, X: Any) -> Any:
        """Validate inference rows and preserve the fitted feature dimension."""
        features, _ = self.validate_inputs(X)
        if self.feature_dim is not None and features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, received {features.shape[-1]}")
        return features
