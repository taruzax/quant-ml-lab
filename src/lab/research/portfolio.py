"""Forecast-to-signal and forecast-to-budget policy stage."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from lab.core.config import AllocationConfig, BacktestConfig
from lab.core.contracts import AllocationFrame, MarketSnapshot, PredictionFrame, SampleKey
from lab.quant.constraints import apply_all_constraints, validate_budget
from lab.quant.covariance import estimate_ledoit_wolf
from lab.quant.hrp import hrp_pypfort


def predictions_to_signals(
    predictions: Any,
    *,
    task: str = "regression",
    dead_zone: float = 0.0,
    confidence: float | None = 0.5,
) -> np.ndarray:
    """Convert regression forecasts or ordered class probabilities to -1/0/+1."""
    values = np.asarray(predictions)
    if not np.isfinite(values).all():
        raise ValueError("Predictions must be finite")
    if task == "regression":
        if values.ndim != 1:
            raise ValueError("Regression predictions must be one-dimensional")
        return np.where(values > dead_zone, 1, np.where(values < -dead_zone, -1, 0)).astype(np.int8)
    if task != "classification" or values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Classification predictions must have shape [N, 3]")
    if (values < 0).any() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Classification predictions must be nonnegative probabilities summing to one")
    best = np.argmax(values, axis=1)
    ties = (values == values.max(axis=1, keepdims=True)).sum(axis=1) > 1
    signals = np.asarray([-1, 0, 1], dtype=np.int8)[best]
    signals[ties] = 0
    if confidence is not None:
        signals[values.max(axis=1) < confidence] = 0
    return signals


def _prediction_rows(prediction_period: Any) -> list[tuple[SampleKey, np.ndarray]]:
    if isinstance(prediction_period, PredictionFrame):
        return list(zip(prediction_period.keys, [row for row in prediction_period.predictions]))
    if isinstance(prediction_period, dict):
        return [
            (SampleKey(ticker=ticker, timestamp=timestamp), np.asarray(value))
            for (ticker, timestamp), value in prediction_period.items()
        ]
    rows: list[tuple[SampleKey, np.ndarray]] = []
    for item in prediction_period:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], SampleKey):
            rows.append((item[0], np.asarray(item[1])))
        else:
            raise TypeError("prediction_period must be a PredictionFrame or (SampleKey, prediction) pairs")
    return rows


def _next_open(bars: pl.DataFrame, decision_time: datetime) -> datetime:
    future = bars.filter(pl.col("timestamp") > decision_time).sort("timestamp")
    if future.is_empty():
        raise ValueError(f"No executable bar follows decision {decision_time}")
    return future["bar_open_time"][0] if "bar_open_time" in future.columns else future["timestamp"][0]


def _eligible_tickers(bars: pl.DataFrame, tickers: tuple[str, ...], decision_time: datetime, lookback: int) -> tuple[set[str], dict[str, str]]:
    history = bars.filter(pl.col("timestamp") <= decision_time).sort(["timestamp", "ticker"])
    pivot = history.pivot(index="timestamp", on="ticker", values="close", aggregate_function="first").sort("timestamp")
    eligibility: dict[str, str] = {}
    eligible: set[str] = set()
    for ticker in tickers:
        if ticker not in pivot.columns:
            eligibility[ticker] = "missing_history"
            continue
        prices = pivot.get_column(ticker).tail(lookback + 1).cast(pl.Float64).to_numpy()
        if len(prices) < lookback + 1 or not np.isfinite(prices).all():
            eligibility[ticker] = "insufficient_history"
            continue
        returns = prices[1:] / prices[:-1] - 1.0
        if not np.isfinite(returns).all() or np.var(returns) <= 0.0:
            eligibility[ticker] = "constant_or_invalid_returns"
            continue
        eligibility[ticker] = "eligible"
        eligible.add(ticker)
    return eligible, eligibility


def build_allocations(
    snapshot: MarketSnapshot,
    prediction_period: Any,
    allocation_config: AllocationConfig,
    *,
    task: str = "regression",
    backtest_config: BacktestConfig | None = None,
    config_hash: str = "unresolved",
) -> tuple[AllocationFrame, ...]:
    """Refresh raw-history budgets on observed bars and attach current signals."""
    rows = _prediction_rows(prediction_period)
    grouped: dict[datetime, list[tuple[SampleKey, np.ndarray]]] = {}
    for key, prediction in rows:
        grouped.setdefault(key.timestamp, []).append((key, prediction))
    if not grouped:
        return ()
    first_decision, last_decision = min(grouped), max(grouped)
    decision_times = snapshot.bars.filter(
        (pl.col("timestamp") >= first_decision) & (pl.col("timestamp") <= last_decision)
    ).get_column("timestamp").unique().sort().to_list()
    frames: list[AllocationFrame] = []
    ordered_tickers = tuple(sorted(snapshot.ticker_order))
    budgets: dict[str, float] = {}
    unconstrained: dict[str, float] = {}
    eligibility: dict[str, str] = {}
    exclusions: dict[str, str] = {}
    covariance_snapshot: dict[str, Any] | None = None
    last_refresh: datetime | None = None
    for decision_index, decision_time in enumerate(decision_times):
        entries = grouped.get(decision_time, [])
        if entries:
            raw_predictions = np.asarray([prediction for _, prediction in entries])
            signals = predictions_to_signals(
                raw_predictions if task == "classification" else raw_predictions.reshape(-1),
                task=task,
                dead_zone=0.0 if backtest_config is None else backtest_config.dead_zone,
                confidence=None if backtest_config is None else backtest_config.classification_confidence,
            )
        else:
            signals = np.asarray([], dtype=np.int8)
        if decision_index % allocation_config.rebalance_every_bars == 0:
            history_bars = max(252, allocation_config.lookback_bars) if allocation_config.method == "hrp" else allocation_config.lookback_bars
            eligible, eligibility = _eligible_tickers(snapshot.bars, ordered_tickers, decision_time, history_bars)
            selected = tuple(sorted(eligible))
            covariance_snapshot = None
            if allocation_config.method == "hrp" and len(selected) > 1:
                history = snapshot.bars.filter(
                    (pl.col("timestamp") <= decision_time) & pl.col("ticker").is_in(selected)
                ).sort("timestamp")
                prices = history.pivot(index="timestamp", on="ticker", values="close", aggregate_function="first").sort("timestamp")
                returns = prices.select(
                    [pl.col("timestamp"), *[pl.col(ticker).pct_change().alias(ticker) for ticker in selected]]
                ).tail(history_bars).drop_nulls()
                covariance, diagnostics = estimate_ledoit_wolf(
                    returns, min_observations=history_bars,
                )
                requested = hrp_pypfort(covariance, selected)
                covariance_snapshot = {"matrix": covariance.to_dict(as_series=False), "diagnostics": diagnostics.__dict__}
            elif selected:
                requested = {ticker: 1.0 / len(selected) for ticker in selected}
            else:
                requested = {}
            unconstrained = dict(sorted(requested.items()))
            budgets = apply_all_constraints(unconstrained, allocation_config)
            validate_budget(budgets, gross_limit=allocation_config.gross_limit)
            exclusions = {ticker: reason for ticker, reason in eligibility.items() if reason != "eligible"}
            last_refresh = decision_time
        try:
            valid_from = _next_open(snapshot.bars, decision_time)
        except ValueError:
            continue
        frames.append(
            AllocationFrame(
                decision_time=decision_time,
                valid_from=valid_from,
                ticker_budgets=budgets,
                unconstrained_budgets=unconstrained,
                eligibility=eligibility,
                exclusions=exclusions,
                config_hash=config_hash,
                cash_capacity=max(0.0, 1.0 - sum(budgets.values())),
                diagnostics={
                    "signals": {key.ticker: int(signal) for (key, _), signal in zip(entries, signals)},
                    "last_refresh": last_refresh.isoformat() if last_refresh is not None else None,
                    "covariance": covariance_snapshot,
                    "max_position_size": allocation_config.max_position_size,
                    "min_position_size": allocation_config.min_position_size,
                    "gross_limit": allocation_config.gross_limit,
                },
            )
        )
    return tuple(frames)
