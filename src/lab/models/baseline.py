from typing import Any

import mlflow.pyfunc
import numpy as np

from lab.models.base import BaseModel


class ZeroForecastBaseline(mlflow.pyfunc.PythonModel, BaseModel):
    """Naive baseline predicting strictly zero return."""

    def fit(self, X: Any = None, y: Any = None) -> "ZeroForecastBaseline":
        """No-op training method for zero-forecast baseline."""
        return self

    def predict(self, context: Any = None, model_input: Any = None) -> np.ndarray:
        """Return array of 0.0 values matching input length."""
        data = model_input if model_input is not None else context
        n = len(data) if hasattr(data, "__len__") else 1
        return np.zeros(n, dtype=np.float64)
