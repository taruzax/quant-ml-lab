from typing import Any

import numpy as np

from lab.models.base import BaseModel


class ZeroForecastBaseline(BaseModel):
    """Naive regression baseline predicting strictly zero return."""

    model_name = "baseline"

    def __init__(self, *, task: str = "regression") -> None:
        super().__init__(task=task)

    def fit(
        self,
        X: Any,
        y: Any,
        stopping_data: tuple[Any, Any] | None = None,
        *,
        duration: int | None = None,
        stopping_patience: int = 5,
    ) -> "ZeroForecastBaseline":
        """Validate the training shape without fitting parameters."""
        features, _ = self.validate_inputs(X, y)
        self.feature_dim = int(features.shape[-1])
        self.training_history = [{"rows": float(features.shape[0])}]
        self.selected_duration = 0
        return self

    def predict(self, X: Any = None, *, model_input: Any = None, context: Any = None) -> np.ndarray:
        """Return array of 0.0 values matching input length."""
        data = X if X is not None else model_input if model_input is not None else context
        if data is None:
            raise ValueError("ZeroForecastBaseline.predict requires model input")
        self.validate_features(data)
        n = len(data)
        return np.zeros(n, dtype=np.float64)


class EmpiricalPriorBaseline(BaseModel):
    """Classification baseline using the training class prior."""

    model_name = "baseline"
    classes = np.asarray([-1, 0, 1], dtype=np.int8)

    def __init__(self) -> None:
        super().__init__(task="classification")
        self.prior = np.zeros(3, dtype=np.float64)

    def fit(
        self,
        X: Any,
        y: Any,
        stopping_data: tuple[Any, Any] | None = None,
        *,
        duration: int | None = None,
        stopping_patience: int = 5,
    ) -> "EmpiricalPriorBaseline":
        features, targets = self.validate_inputs(X, y)
        assert targets is not None
        labels = targets.reshape(-1).astype(int)
        if not np.isin(labels, self.classes).all():
            raise ValueError("Classification labels must be one of -1, 0, +1")
        self.feature_dim = int(features.shape[-1])
        self.prior = np.asarray([(labels == label).mean() for label in self.classes], dtype=np.float64)
        self.training_history = [{"rows": float(features.shape[0])}]
        self.selected_duration = 0
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        features = self.validate_features(X)
        return np.repeat(self.prior[None, :], features.shape[0], axis=0)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]
