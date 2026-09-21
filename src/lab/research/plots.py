"""Small plotting helpers used by the unexecuted research walkthrough."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_diagnostics(actual: Any, predicted: Any, *, title: str = "Predictions"):
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(np.asarray(actual).reshape(-1), label="actual")
    axis.plot(np.asarray(predicted).reshape(-1), label="predicted", alpha=0.8)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_equity(equity: Any, *, title: str = "Equity"):
    figure, axis = plt.subplots(figsize=(8, 4))
    values = equity["equity"] if hasattr(equity, "__getitem__") and not isinstance(equity, (list, tuple, np.ndarray)) else equity
    axis.plot(np.asarray(values).reshape(-1))
    axis.set_title(title)
    axis.set_ylabel("equity")
    figure.tight_layout()
    return figure, axis


def plot_drawdown(equity: Any, *, title: str = "Drawdown"):
    values = np.asarray(equity["equity"] if hasattr(equity, "__getitem__") and not isinstance(equity, (list, tuple, np.ndarray)) else equity, dtype=float).reshape(-1)
    running_max = np.maximum.accumulate(values)
    drawdown = values / running_max - 1.0
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(drawdown)
    axis.set_title(title)
    axis.set_ylabel("drawdown")
    figure.tight_layout()
    return figure, axis


def plot_positions(orders: Any, *, title: str = "Positions"):
    figure, axis = plt.subplots(figsize=(8, 4))
    if hasattr(orders, "group_by"):
        for ticker, group in orders.sort("timestamp").group_by("ticker", maintain_order=True):
            name = ticker[0] if isinstance(ticker, tuple) else ticker
            axis.plot(group.get_column("quantity").cum_sum().to_numpy(), label=name)
    axis.set_title(title)
    axis.legend() if axis.lines else None
    figure.tight_layout()
    return figure, axis


def plot_costs(orders: Any, *, title: str = "Transaction costs"):
    figure, axis = plt.subplots(figsize=(8, 4))
    if hasattr(orders, "get_column") and "fees" in orders.columns:
        axis.bar(np.arange(orders.height), orders.get_column("fees").to_numpy())
    axis.set_title(title)
    axis.set_ylabel("fees + slippage")
    figure.tight_layout()
    return figure, axis


def plot_fold_diagnostics(metrics: Any):
    figure, axis = plt.subplots(figsize=(8, 4))
    names = [metric.name for metric in metrics if metric.value is not None]
    values = [metric.value for metric in metrics if metric.value is not None]
    axis.bar(names, values)
    axis.set_title("Fold metrics")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    return figure, axis


def plot_trial_evidence(metrics: Any, *, title: str = "Statistical evidence"):
    """Plot available Sharpe evidence while leaving unavailable metrics visible."""
    figure, axis = plt.subplots(figsize=(8, 4))
    names = []
    values = []
    for name, metric in metrics.items():
        names.append(name)
        values.append(float(metric.value) if metric.value is not None else np.nan)
    axis.bar(names, np.nan_to_num(values, nan=0.0))
    for index, value in enumerate(values):
        if not np.isfinite(value):
            axis.text(index, 0.0, "unavailable", ha="center", va="bottom", rotation=90)
    axis.set_title(title)
    figure.tight_layout()
    return figure, axis


def plot_learning_curves(training_history: Any, stopping_history: Any | None = None):
    """Plot saved training and stopping curves without inventing missing history."""
    figure, axis = plt.subplots(figsize=(8, 4))
    if training_history:
        axis.plot([row.get("epoch", row.get("round")) for row in training_history], [row["loss"] for row in training_history], label="train")
    if stopping_history:
        axis.plot([row.get("epoch", row.get("round")) for row in stopping_history], [row["loss"] for row in stopping_history], label="stopping")
    axis.set_title("Learning curves")
    axis.set_xlabel("duration")
    axis.set_ylabel("loss")
    if axis.lines:
        axis.legend()
    figure.tight_layout()
    return figure, axis
