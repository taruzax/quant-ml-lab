"""Lazy registry for the supported model names."""

from collections.abc import Callable
from typing import Any

from lab.models.base import BaseModel
from lab.models.baseline import EmpiricalPriorBaseline, ZeroForecastBaseline

MODEL_NAMES = ("baseline", "xgboost", "gru", "lstm")


class ModelUnavailableError(RuntimeError):
    """Raised when a disabled or not-yet-installed adapter is explicitly selected."""


def _baseline_factory(task: str, params: dict[str, Any]) -> BaseModel:
    if task == "classification":
        return EmpiricalPriorBaseline()
    return ZeroForecastBaseline(task="regression")


def _lazy_factory(name: str) -> Callable[[str, dict[str, Any]], BaseModel]:
    def factory(task: str, params: dict[str, Any]) -> BaseModel:
        try:
            if name == "xgboost":
                from lab.models.xgboost_model import XGBoostModel

                return XGBoostModel(task=task, **params)
            from lab.models.recurrent import GRUModel, LSTMModel

            model_type = GRUModel if name == "gru" else LSTMModel
            return model_type(task=task, **params)
        except ImportError as exc:
            raise ModelUnavailableError(f"Model adapter '{name}' is unavailable") from exc

    return factory


_FACTORIES: dict[str, Callable[[str, dict[str, Any]], BaseModel]] = {
    "baseline": _baseline_factory,
    "xgboost": _lazy_factory("xgboost"),
    "gru": _lazy_factory("gru"),
    "lstm": _lazy_factory("lstm"),
}


def create_model(name: str, *, task: str, params: dict[str, Any] | None = None) -> BaseModel:
    """Create one explicitly selected model without importing disabled adapters."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{name}'. Expected one of {MODEL_NAMES}")
    return _FACTORIES[name](task, dict(params or {}))


def registered_models() -> tuple[str, ...]:
    return MODEL_NAMES
