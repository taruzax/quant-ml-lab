"""Predictive and portfolio metric primitives with explicit unavailable states."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm, skew, kurtosis

from lab.core.contracts import MetricResult


def _array(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("Metric inputs must be non-empty and finite")
    return result


def regression_metrics(y_true: Any, y_pred: Any, *, zero_forecast: Any | None = None) -> dict[str, MetricResult]:
    actual = _array(y_true)
    predicted = _array(y_pred)
    if actual.shape != predicted.shape:
        raise ValueError("Metric inputs must have the same shape")
    error = predicted - actual
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    result = {
        "mse": MetricResult(name="mse", value=mse, status="available", inputs={"n": int(actual.size)}),
        "rmse": MetricResult(name="rmse", value=math.sqrt(mse), status="available", inputs={"n": int(actual.size)}),
        "mae": MetricResult(name="mae", value=mae, status="available", inputs={"n": int(actual.size)}),
    }
    baseline = np.zeros_like(actual) if zero_forecast is None else _array(zero_forecast)
    baseline_loss = float(np.mean((baseline - actual) ** 2))
    improvement = None if baseline_loss == 0.0 else 1.0 - mse / baseline_loss
    result["improvement_over_zero"] = MetricResult(
        name="improvement_over_zero",
        value=improvement,
        status="available" if improvement is not None and math.isfinite(improvement) else "unavailable",
        reason=None if improvement is not None else "zero baseline has zero loss",
        inputs={"baseline": "zero_forecast", "n": int(actual.size)},
    )
    return result


def classification_metrics(y_true: Any, probabilities: Any) -> dict[str, MetricResult]:
    actual = _array(y_true).astype(int)
    proba = np.asarray(probabilities, dtype=float)
    if proba.ndim != 2 or proba.shape != (actual.size, 3) or not np.isfinite(proba).all():
        raise ValueError("Classification probabilities must have shape [N, 3] and be finite")
    if (proba < 0).any() or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Classification probabilities must be nonnegative and sum to one")
    labels = np.asarray([-1, 0, 1])
    if not np.isin(actual, labels).all():
        raise ValueError("Classification labels must be -1, 0, or +1")
    indices = np.searchsorted(labels, actual)
    loss = float(-np.log(np.clip(proba[np.arange(actual.size), indices], 1e-15, 1.0)).mean())
    predicted = labels[np.argmax(proba, axis=1)]
    accuracy = float(np.mean(predicted == actual))
    return {
        "log_loss": MetricResult(name="log_loss", value=loss, status="available", inputs={"n": int(actual.size)}),
        "accuracy": MetricResult(name="accuracy", value=accuracy, status="available", inputs={"n": int(actual.size)}),
    }


def unavailable_metric(name: str, reason: str, **inputs: Any) -> MetricResult:
    return MetricResult(name=name, value=None, status="unavailable", reason=reason, inputs=inputs)


def sharpe_statistics(returns: Any, *, frequency: str = "daily", min_observations: int = 30) -> dict[str, Any]:
    """Return nonannualized Sharpe moments using sample standard deviation."""
    values = _array(returns)
    if values.size < min_observations:
        raise ValueError(f"At least {min_observations} returns are required")
    deviation = float(np.std(values, ddof=1))
    if deviation <= 0.0:
        raise ValueError("Sharpe statistics require non-constant returns")
    return {
        "sharpe": float(np.mean(values) / deviation),
        "observations": int(values.size),
        "skew": float(skew(values, bias=False)),
        "pearson_kurtosis": float(kurtosis(values, fisher=False, bias=False)),
        "frequency": frequency,
        "std_convention": "sample_ddof_1",
    }


def probabilistic_sharpe_ratio(
    returns: Any,
    *,
    threshold: float = 0.0,
    frequency: str = "daily",
    min_observations: int = 30,
) -> MetricResult:
    """Estimate PSR with the Bailey-López de Prado finite-sample denominator."""
    try:
        stats = sharpe_statistics(returns, frequency=frequency, min_observations=min_observations)
        sharpe = float(stats["sharpe"])
        denominator = 1.0 - stats["skew"] * sharpe + (stats["pearson_kurtosis"] - 1.0) * sharpe**2 / 4.0
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("PSR uncertainty denominator must be positive")
        z = (sharpe - threshold) * math.sqrt(stats["observations"] - 1) / math.sqrt(denominator)
        probability = float(norm.cdf(z))
        return MetricResult(
            name="psr", value=probability, status="available",
            inputs={**stats, "threshold": threshold, "uncertainty_denominator": denominator},
            conventions={"formula": "PSR", "returns": "net", "annualized": False},
        )
    except (TypeError, ValueError) as exc:
        return unavailable_metric("psr", str(exc), frequency=frequency, threshold=threshold)


def deflated_sharpe_ratio(
    returns: Any,
    *,
    effective_trial_count: float,
    trial_sharpe_variance: float,
    frequency: str = "daily",
    min_observations: int = 30,
) -> MetricResult:
    """Estimate DSR against the published expected-maximum Sharpe benchmark."""
    try:
        if effective_trial_count < 1.0 or not math.isfinite(effective_trial_count):
            raise ValueError("effective_trial_count must be finite and at least one")
        if trial_sharpe_variance < 0.0 or not math.isfinite(trial_sharpe_variance):
            raise ValueError("trial_sharpe_variance must be finite and nonnegative")
        stats = sharpe_statistics(returns, frequency=frequency, min_observations=min_observations)
        if effective_trial_count == 1.0:
            return unavailable_metric(
                "dsr", "DSR is not applicable for one trial; use PSR", **stats,
                effective_trial_count=effective_trial_count, trial_sharpe_variance=trial_sharpe_variance,
            )
        gamma = 0.5772156649015329
        variance_scale = math.sqrt(trial_sharpe_variance)
        benchmark = variance_scale * (
            (1.0 - gamma) * norm.ppf(1.0 - 1.0 / effective_trial_count)
            + gamma * norm.ppf(1.0 - 1.0 / (effective_trial_count * math.e))
        )
        result = probabilistic_sharpe_ratio(
            returns, threshold=float(benchmark), frequency=frequency, min_observations=min_observations,
        )
        result.name = "dsr"
        result.inputs.update({"benchmark": float(benchmark), "effective_trial_count": effective_trial_count, "trial_sharpe_variance": trial_sharpe_variance})
        result.conventions.update({"formula": "DSR", "benchmark": "expected_maximum_normal_quantile"})
        return result
    except (TypeError, ValueError) as exc:
        return unavailable_metric(
            "dsr", str(exc), frequency=frequency,
            effective_trial_count=effective_trial_count, trial_sharpe_variance=trial_sharpe_variance,
        )


def sharpe_evidence_metrics(
    returns: Any,
    *,
    trial_sharpes: list[float] | None = None,
    frequency: str = "daily",
    min_observations: int = 30,
) -> dict[str, MetricResult]:
    """Compute descriptive Sharpe, PSR and DSR from one frozen evidence set."""
    try:
        values = _array(returns)
    except (TypeError, ValueError) as exc:
        unavailable = unavailable_metric("insufficient_evidence", str(exc), frequency=frequency)
        return {"sharpe": unavailable_metric("sharpe", str(exc), frequency=frequency), "psr": unavailable_metric("psr", str(exc), frequency=frequency), "dsr": unavailable}
    psr = probabilistic_sharpe_ratio(values, frequency=frequency, min_observations=min_observations)
    sharpe = unavailable_metric("sharpe", "insufficient or degenerate returns", frequency=frequency)
    try:
        stats = sharpe_statistics(values, frequency=frequency, min_observations=min_observations)
        sharpe = MetricResult(name="sharpe", value=stats["sharpe"], status="available", inputs=stats, conventions={"annualized": False})
    except (TypeError, ValueError) as exc:
        sharpe = unavailable_metric("sharpe", str(exc), frequency=frequency)
    if trial_sharpes is None or len(trial_sharpes) <= 1:
        dsr = unavailable_metric("dsr", "trial evidence requires at least two complete comparable trials", frequency=frequency)
    else:
        dsr = deflated_sharpe_ratio(
            values,
            effective_trial_count=float(len(trial_sharpes)),
            trial_sharpe_variance=float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1)),
            frequency=frequency,
            min_observations=min_observations,
        )
    return {"sharpe": sharpe, "psr": psr, "dsr": dsr}
