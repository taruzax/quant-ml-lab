from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Abstract base model interface for time-series forecasting."""

    @abstractmethod
    def fit(self, X: Any = None, y: Any = None) -> "BaseModel":
        """Fit model parameters."""
        ...

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Generate return forecasts for input data."""
        ...
