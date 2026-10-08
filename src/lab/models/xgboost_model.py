from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor

from lab.models.base import BaseModel
from lab.core.contracts import WindowSet
from lab.models.devices import resolve_device


class XGBoostModel(BaseModel):
    """XGBoost adapter for ordered regression and three-class windows."""

    model_name = "xgboost"
    classes = np.asarray([-1, 0, 1], dtype=np.int8)

    def __init__(
        self,
        *,
        task: str = "regression",
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int = 42,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(task=task)
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if not 0 < colsample_bytree <= 1:
            raise ValueError("colsample_bytree must be in (0, 1]")
        device = resolve_device(self.model_name, device).resolved_device

        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.random_state = int(random_state if seed is None else seed)
        self.device = device
        self.classes_seen = np.asarray([], dtype=np.int8)
        self.adapter_mode = "unfitted"
        self.window_len: int | None = None
        self._estimator: XGBClassifier | XGBRegressor | None = None

    def _prepare_features(self, X: Any, *, fitting: bool = False) -> np.ndarray:
        if isinstance(X, WindowSet):
            X = X.to_numpy()
        features = self.validate_features(X)
        if features.ndim == 3:
            if fitting:
                self.window_len = int(features.shape[1])
            elif self.window_len is not None and features.shape[1] != self.window_len:
                raise ValueError(f"Expected window length {self.window_len}, received {features.shape[1]}")
            flattened = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
        else:
            if fitting:
                self.window_len = None
            elif self.window_len is not None:
                raise ValueError("Expected three-dimensional windows for this fitted model")
            flattened = features
        return np.asarray(flattened, dtype=np.float32)

    def _estimator_params(self, *, n_estimators: int | None = None, stopping_patience: int | None = None) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators if n_estimators is None else n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "n_jobs": 1,
            "device": self.device,
            "verbosity": 0,
            **({"early_stopping_rounds": stopping_patience} if stopping_patience is not None else {}),
        }

    def fit(
        self,
        X: Any,
        y: Any,
        stopping_data: tuple[Any, Any] | None = None,
        *,
        duration: int | None = None,
        stopping_patience: int = 5,
    ) -> "XGBoostModel":
        features, targets = self.validate_inputs(X, y)
        assert targets is not None
        self.feature_dim = int(features.shape[-1])
        flattened = self._prepare_features(features, fitting=True)
        if duration is not None and duration <= 0:
            raise ValueError("duration must be positive")
        effective_estimators = self.n_estimators if duration is None else int(duration)
        if stopping_data is not None:
            if stopping_patience <= 0:
                raise ValueError("stopping_patience must be positive")
            stopping_features, stopping_targets = self.validate_inputs(*stopping_data)
            stopping_flattened = self._prepare_features(stopping_features)
        else:
            stopping_flattened = None
            stopping_targets = None
        self.stopping_history = None
        self.stopping_config = {}
        self.stopping_diagnostics = {}

        if self.task == "regression":
            estimator_params = self._estimator_params(
                n_estimators=effective_estimators,
                stopping_patience=stopping_patience if stopping_data is not None else None,
            )
            self._estimator = XGBRegressor(
                objective="reg:squarederror",
                **estimator_params,
            )
            train_targets = targets.reshape(-1).astype(np.float32)
            fit_kwargs = {"eval_set": [(flattened, train_targets), (stopping_flattened, stopping_targets.reshape(-1).astype(np.float32))], "verbose": False} if stopping_data is not None else {}
            with threadpool_limits(limits=1, user_api="openmp"):
                self._estimator.fit(flattened, train_targets, **fit_kwargs)
            self.classes_seen = np.asarray([], dtype=np.int8)
            self.adapter_mode = "regression"
        else:
            raw_labels = targets.reshape(-1)
            if not np.isfinite(raw_labels).all() or not np.equal(raw_labels, raw_labels.astype(int)).all():
                raise ValueError("Classification labels must be finite integers in {-1, 0, +1}")
            labels = raw_labels.astype(np.int8)
            if not np.isin(labels, self.classes).all():
                raise ValueError("Classification labels must be one of -1, 0, +1")
            self.classes_seen = np.asarray(sorted(set(labels.tolist())), dtype=np.int8)
            if self.classes_seen.size == 1:
                self._estimator = None
                self.adapter_mode = "constant_class"
            else:
                class_to_encoded = {int(label): index for index, label in enumerate(self.classes_seen)}
                encoded = np.asarray([class_to_encoded[int(label)] for label in labels], dtype=np.int8)
                stopping_encoded = None
                if stopping_data is not None:
                    raw_stopping = stopping_targets.reshape(-1)
                    if not np.isfinite(raw_stopping).all() or not np.equal(raw_stopping, raw_stopping.astype(int)).all():
                        raise ValueError("Classification stopping labels must be finite integers in {-1, 0, +1}")
                    stopping_labels = raw_stopping.astype(np.int8)
                    if not np.isin(stopping_labels, self.classes_seen).all():
                        raise ValueError("Stopping labels contain a class absent from training labels")
                    stopping_encoded = np.asarray([class_to_encoded[int(label)] for label in stopping_labels], dtype=np.int8)
                estimator_params = self._estimator_params(
                    n_estimators=effective_estimators,
                    stopping_patience=stopping_patience if stopping_data is not None else None,
                )
                self._estimator = XGBClassifier(
                    objective="multi:softprob",
                    num_class=int(self.classes_seen.size),
                    **estimator_params,
                )
                fit_kwargs = {"eval_set": [(flattened, encoded), (stopping_flattened, stopping_encoded)], "verbose": False} if stopping_data is not None else {}
                with threadpool_limits(limits=1, user_api="openmp"):
                    self._estimator.fit(flattened, encoded, **fit_kwargs)
                self.adapter_mode = "classification"

        if stopping_data is not None and self._estimator is not None:
            evaluation_result = self._estimator.evals_result()
            series = [next(iter(metrics.values()), []) for metrics in evaluation_result.values()]
            train_values = series[0] if series else []
            stop_values = series[-1] if series else []
            self.training_history = [{"round": float(index + 1), "loss": float(value)} for index, value in enumerate(train_values)]
            self.stopping_history = [{"round": float(index + 1), "loss": float(value)} for index, value in enumerate(stop_values)]
            best_iteration = getattr(self._estimator, "best_iteration", None)
            self.selected_duration = int(best_iteration + 1) if best_iteration is not None else effective_estimators
            self.stopping_config = {"patience": int(stopping_patience), "metric": "validation_loss"}
        else:
            self.selected_duration = effective_estimators
            self.training_history = [{
                "rows": float(features.shape[0]),
                "flattened_features": float(flattened.shape[1]),
                "estimators": float(self.selected_duration),
            }]
        if stopping_data is not None and self._estimator is None:
            self.stopping_history = []
            self.stopping_config = {"patience": int(stopping_patience), "metric": "validation_loss"}
        return self

    def _require_fitted(self) -> None:
        if self.adapter_mode == "unfitted":
            raise RuntimeError("XGBoostModel must be fitted before prediction")

    def predict(self, X: Any) -> np.ndarray:
        self._require_fitted()
        if self.task == "classification":
            probabilities = self.predict_proba(X)
            return self.classes[np.argmax(probabilities, axis=1)]
        if self._estimator is None:
            raise RuntimeError("Regression estimator is unavailable")
        with threadpool_limits(limits=1, user_api="openmp"):
            predictions = self._estimator.predict(self._prepare_features(X))
        return np.asarray(predictions, dtype=np.float64)

    def predict_proba(self, X: Any) -> np.ndarray:
        self._require_fitted()
        if self.task != "classification":
            raise ValueError("predict_proba is available only for classification")
        features = self._prepare_features(X)
        if self.adapter_mode == "constant_class":
            probabilities = np.zeros((features.shape[0], 3), dtype=np.float64)
            probabilities[:, np.flatnonzero(self.classes == self.classes_seen[0])[0]] = 1.0
            return probabilities
        if self._estimator is None:
            raise RuntimeError("Classification estimator is unavailable")

        with threadpool_limits(limits=1, user_api="openmp"):
            encoded_probabilities = np.asarray(self._estimator.predict_proba(features), dtype=np.float64)
        probabilities = np.zeros((features.shape[0], 3), dtype=np.float64)
        for encoded_index, label in enumerate(self.classes_seen):
            class_index = int(np.flatnonzero(self.classes == label)[0])
            probabilities[:, class_index] = encoded_probabilities[:, encoded_index]
        return probabilities

    def save(self, directory: str | Path) -> Path:
        """Persist the adapter while limiting native OpenMP calls."""
        with threadpool_limits(limits=1, user_api="openmp"):
            return super().save(directory)

    @classmethod
    def load(cls, directory: str | Path) -> "XGBoostModel":
        """Load a persisted adapter while limiting native OpenMP calls."""
        with threadpool_limits(limits=1, user_api="openmp"):
            model = super().load(directory)
        assert isinstance(model, cls)
        return model

    def copy(self) -> "XGBoostModel":
        """Copy the adapter and its native estimator under an OpenMP limit."""
        with threadpool_limits(limits=1, user_api="openmp"):
            return deepcopy(self)
