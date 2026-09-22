"""VectorBT-backed event execution for regression and triple-barrier policies."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

try:
    import vectorbt as vbt
except ImportError:  # pragma: no cover - dependency is declared for normal installs
    vbt = None

from lab.core.config import BacktestConfig, TaskConfig
from lab.core.contracts import AllocationFrame, BacktestResult, MarketSnapshot, PredictionFrame


def _prediction_map(predictions: PredictionFrame | Any) -> dict[tuple[datetime, str], np.ndarray]:
    if isinstance(predictions, PredictionFrame):
        return {(key.timestamp, key.ticker): np.asarray(value) for key, value in zip(predictions.keys, predictions.predictions)}
    return {(key.timestamp, key.ticker): np.asarray(value) for key, value in predictions}


def _period_bounds(evaluation_period: Any) -> tuple[datetime, datetime, datetime | None]:
    if hasattr(evaluation_period, "first_decision"):
        return evaluation_period.first_decision, evaluation_period.last_decision, evaluation_period.final_liquidation_open
    if isinstance(evaluation_period, dict):
        return evaluation_period["first_decision"], evaluation_period["last_decision"], evaluation_period.get("final_liquidation_open")
    if isinstance(evaluation_period, tuple) and len(evaluation_period) >= 2:
        return evaluation_period[0], evaluation_period[1], evaluation_period[2] if len(evaluation_period) > 2 else None
    raise TypeError("evaluation_period must provide first_decision, last_decision, and optional final_liquidation_open")


def _price_matrix(snapshot: MarketSnapshot, start: datetime, end: datetime) -> tuple[pd.DataFrame, pl.DataFrame]:
    time_column = "bar_open_time" if "bar_open_time" in snapshot.bars.columns else "timestamp"
    bars = snapshot.bars.filter((pl.col(time_column) >= start) & (pl.col(time_column) <= end)).sort([time_column, "ticker"])
    if bars.is_empty():
        raise ValueError("Evaluation period contains no market bars")
    pandas = bars.select([time_column, "ticker", "open", "close"]).to_pandas()
    close = pandas.pivot(index=time_column, columns="ticker", values="close").sort_index().reindex(columns=sorted(snapshot.ticker_order))
    return close, bars


def _target_orders(
    close: pd.DataFrame,
    bars: pl.DataFrame,
    allocations: tuple[AllocationFrame, ...],
    prediction_map: dict[tuple[datetime, str], np.ndarray],
    *,
    task_config: TaskConfig,
    backtest_config: BacktestConfig,
    fee_rate: float,
    liquidation: datetime | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tickers = list(close.columns)
    quantities = {ticker: 0.0 for ticker in tickers}
    cash = float(backtest_config.initial_capital)
    order_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    active_events: dict[str, dict[str, Any]] = {}
    pending_exits: dict[str, str] = {}
    blocked_entries: set[str] = set()
    allocation_by_open = {allocation.valid_from: allocation for allocation in allocations}
    latest_policy: dict[str, Any] = {"max_position_size": 1.0, "min_position_size": 0.0, "gross_limit": 1.0}
    open_prices = {
        (row["bar_open_time"] if "bar_open_time" in row else row["timestamp"], row["ticker"]): float(row["open"])
        for row in bars.to_dicts()
    }
    last_known_price: dict[str, float] = {}
    for timestamp, prices in close.iterrows():
        close_marks = {ticker: float(prices[ticker]) for ticker in tickers if np.isfinite(prices[ticker])}
        allocation = allocation_by_open.get(timestamp.to_pydatetime())
        if allocation is not None:
            latest_policy.update({key: allocation.diagnostics[key] for key in latest_policy if key in allocation.diagnostics})
        execution_open = {
            ticker: price
            for ticker in tickers
            if (price := open_prices.get((timestamp.to_pydatetime(), ticker))) is not None
            and np.isfinite(price) and price > 0.0
        }
        missing_held = [ticker for ticker in tickers if quantities[ticker] != 0.0 and ticker not in execution_open]
        for ticker in missing_held:
            if ticker in last_known_price:
                execution_open[ticker] = last_known_price[ticker]
            else:
                raise ValueError(f"Held position {ticker} lacks both executable open and last known price at {timestamp}")

        equity = cash + sum(quantities[ticker] * execution_open[ticker] for ticker in tickers if ticker in execution_open)

        # Update last known price for the next bar
        for ticker, price in close_marks.items():
            last_known_price[ticker] = price
        if equity <= 0.0:
            raise ValueError("Portfolio equity is nonpositive at an executable open")
        gross_open = sum(abs(quantities[ticker] * execution_open[ticker]) for ticker in tickers if ticker in execution_open)
        needs_risk = gross_open > float(latest_policy["gross_limit"]) * equity + 1e-9 or any(
            abs(quantities[ticker] * execution_open[ticker]) > float(latest_policy["max_position_size"]) * equity + 1e-9
            for ticker in tickers if ticker in execution_open
        )
        # Terminal liquidation is an explicit event at the configured final open.
        is_liquidation = liquidation is not None and timestamp.to_pydatetime() == liquidation
        needs_regression_exit = task_config.kind == "regression" and any(value != 0.0 for value in quantities.values())
        if (allocation is not None or pending_exits or needs_regression_exit or needs_risk) and not is_liquidation:
            target_quantities: dict[str, float] = {}
            risk_reductions: set[str] = set()
            for ticker in tickers:
                if ticker not in execution_open:
                    continue
                if ticker in pending_exits or ticker in blocked_entries:
                    target_quantity = 0.0
                elif allocation is None:
                    if task_config.kind == "classification" and not needs_risk:
                        continue
                    target_quantity = quantities[ticker] if task_config.kind == "classification" else 0.0
                else:
                    raw = prediction_map.get((allocation.decision_time, ticker))
                    if task_config.kind == "classification" and ticker in active_events:
                        target_quantity = quantities[ticker]
                    elif raw is None:
                        target_quantity = quantities[ticker] if task_config.kind == "classification" else 0.0
                    elif task_config.kind == "classification":
                        active = np.argmax(raw) if raw.ndim else 1
                        direction = (-1, 0, 1)[int(active)]
                        target_quantity = direction * allocation.ticker_budgets.get(ticker, 0.0) * equity / execution_open[ticker]
                    else:
                        forecast = float(raw.reshape(-1)[0])
                        direction = 1 if forecast > backtest_config.dead_zone else -1 if forecast < -backtest_config.dead_zone else 0
                        target_quantity = direction * allocation.ticker_budgets.get(ticker, 0.0) * equity / execution_open[ticker]
                target_quantities[ticker] = target_quantity
            if needs_risk:
                max_value = float(latest_policy["max_position_size"]) * equity
                for ticker, target in tuple(target_quantities.items()):
                    current = quantities[ticker]
                    if current == 0.0 or np.sign(current) != np.sign(target):
                        continue
                    capped = np.sign(current) * min(abs(current), max_value / execution_open[ticker])
                    if abs(capped) < abs(target):
                        target_quantities[ticker] = capped
                        risk_reductions.add(ticker)
                gross_target = sum(abs(value * execution_open[ticker]) for ticker, value in target_quantities.items())
                if gross_target > float(latest_policy["gross_limit"]) * equity:
                    factor = float(latest_policy["gross_limit"]) * equity / gross_target
                    for ticker, target in tuple(target_quantities.items()):
                        if quantities[ticker] != 0.0 and np.sign(target) == np.sign(quantities[ticker]):
                            target_quantities[ticker] = target * factor
                            risk_reductions.add(ticker)
            existing_value = sum(
                min(abs(quantities[ticker]), abs(target)) * execution_open[ticker]
                for ticker, target in target_quantities.items()
                if quantities[ticker] != 0.0 and np.sign(quantities[ticker]) == np.sign(target)
            )
            new_value = sum(
                max(0.0, abs(target) - (abs(quantities[ticker]) if np.sign(quantities[ticker]) == np.sign(target) else 0.0)) * execution_open[ticker]
                for ticker, target in target_quantities.items()
            )
            capacity = max(0.0, float(latest_policy["gross_limit"]) * equity - existing_value)
            entry_scale = min(1.0, capacity / (new_value * (1.0 + fee_rate))) if new_value > 0.0 else 1.0
            for ticker, target in tuple(target_quantities.items()):
                current = quantities[ticker]
                if current == 0.0 or np.sign(current) != np.sign(target):
                    target_quantities[ticker] = target * entry_scale
                elif abs(target) > abs(current):
                    target_quantities[ticker] = current + (target - current) * entry_scale
                if current == 0.0 and 0.0 < abs(target_quantities[ticker] * execution_open[ticker]) / equity < float(latest_policy["min_position_size"]):
                    target_quantities[ticker] = 0.0
            ordered_orders = sorted(
                (
                    (target_quantities[ticker] - quantities[ticker], ticker, target_quantities[ticker])
                    for ticker in target_quantities
                ),
                key=lambda order: (order[0] >= 0.0, order[1]),
            )
            for delta, ticker, target_quantity in ordered_orders:
                if abs(delta) <= 1e-12:
                    continue
                execution_price = execution_open[ticker]
                if delta > 0.0:
                    cash_scale = max(abs(float(backtest_config.initial_capital)), abs(cash), 1.0)
                    cash_reserve = 8.0 * np.finfo(float).eps * cash_scale
                    affordable = max(0.0, cash - cash_reserve) / (execution_price * (1.0 + fee_rate))
                    delta = min(delta, affordable)
                    if delta <= 1e-12:
                        continue
                    target_quantity = quantities[ticker] + delta
                fee = abs(delta * execution_price) * fee_rate
                cash -= delta * execution_price + fee
                cash_tolerance = 1e-12 * max(abs(float(backtest_config.initial_capital)), abs(delta * execution_price), 1.0)
                if cash < -cash_tolerance:
                    raise ValueError(
                        f"Order planning produced materially negative cash at {timestamp}: "
                        f"ticker={ticker}, side={'buy' if delta > 0 else 'sell'}, "
                        f"quantity={delta:.12g}, cash={cash:.12g}, tolerance={cash_tolerance:.12g}"
                    )
                if cash < 0.0:
                    cash = 0.0
                previous_quantity = quantities[ticker]
                quantities[ticker] = target_quantity
                if ticker in pending_exits:
                    reason = pending_exits[ticker]
                elif ticker in risk_reductions:
                    reason = "risk_reduction"
                elif target_quantity == 0.0:
                    reason = "exit"
                elif previous_quantity == 0.0:
                    reason = "entry"
                elif np.sign(previous_quantity) != np.sign(target_quantity):
                    reason = "reversal"
                elif abs(target_quantity) < abs(previous_quantity):
                    reason = "reduction"
                else:
                    reason = "rebalance"
                order_rows.append(
                    {
                        "timestamp": timestamp.to_pydatetime(),
                        "ticker": ticker,
                        "quantity": delta,
                        "price": execution_price,
                        "side": "buy" if delta > 0 else "sell",
                        "reason": reason,
                        "fees": fee,
                    }
                )
                event_rows.append(
                    {
                        "timestamp": timestamp.to_pydatetime(),
                        "ticker": ticker,
                        "entry_quantity": target_quantity,
                        "entry_price": execution_price,
                        "exit_reason": reason,
                    }
                )
                if task_config.kind == "classification" and target_quantity != 0.0 and ticker not in active_events:
                    current_index = close.index.get_loc(timestamp)
                    history = close[ticker].iloc[:current_index] if ticker in close.columns else pd.Series(dtype=float)
                    volatility = float(history.pct_change().dropna().tail(task_config.triple_barrier.volatility_span).std())
                    active_events[ticker] = {
                        "entry_index": close.index.get_loc(timestamp),
                        "entry_price": execution_price,
                        "direction": 1 if target_quantity > 0.0 else -1,
                        "volatility": max(volatility if np.isfinite(volatility) else 0.0, task_config.triple_barrier.volatility_floor),
                    }
                if ticker in pending_exits:
                    active_events.pop(ticker, None)
                    blocked_entries.add(ticker)
            pending_exits.clear()
        if is_liquidation:
            for ticker in tickers:
                if quantities[ticker] != 0.0 and ticker not in execution_open:
                    raise ValueError(f"Terminal liquidation price unavailable for {ticker} at {timestamp}")
                if quantities[ticker] == 0.0:
                    continue
                delta = -quantities[ticker]
                execution_price = execution_open[ticker]
                fee = abs(delta * execution_price) * fee_rate
                cash -= delta * execution_price + fee
                cash_tolerance = 1e-12 * max(abs(float(backtest_config.initial_capital)), abs(delta * execution_price), 1.0)
                if cash < -cash_tolerance:
                    raise ValueError(
                        f"Terminal liquidation produced materially negative cash at {timestamp}: "
                        f"ticker={ticker}, side={'buy' if delta > 0 else 'sell'}, "
                        f"quantity={delta:.12g}, cash={cash:.12g}, tolerance={cash_tolerance:.12g}"
                    )
                if cash < 0.0:
                    cash = 0.0
                quantities[ticker] = 0.0
                order_rows.append(
                    {
                        "timestamp": timestamp.to_pydatetime(), "ticker": ticker, "quantity": delta,
                        "price": execution_price, "side": "buy" if delta > 0 else "sell",
                        "reason": "terminal_liquidation", "fees": fee,
                    }
                )
                event_rows.append(
                    {
                        "timestamp": timestamp.to_pydatetime(), "ticker": ticker, "entry_quantity": 0.0,
                        "entry_price": execution_price, "exit_reason": "terminal_liquidation",
                    }
                )
            break
        if task_config.kind == "classification":
            current_index = close.index.get_loc(timestamp)
            for ticker, event in tuple(active_events.items()):
                if ticker not in close_marks or current_index < event["entry_index"]:
                    continue
                signed_return = event["direction"] * (close_marks[ticker] / event["entry_price"] - 1.0)
                barrier = event["volatility"]
                if signed_return >= task_config.triple_barrier.profit_taking * barrier:
                    pending_exits[ticker] = "profit_barrier"
                elif signed_return <= -task_config.triple_barrier.stop_loss * barrier:
                    pending_exits[ticker] = "stop_barrier"
                elif current_index - event["entry_index"] >= task_config.triple_barrier.expiry_bars:
                    pending_exits[ticker] = "expiry"
        blocked_entries.clear()
    orders = pd.DataFrame(order_rows) if order_rows else pd.DataFrame(
        columns=["timestamp", "ticker", "quantity", "price", "side", "reason", "fees"]
    )
    return orders, event_rows


def _run_vectorbt(close: pd.DataFrame, orders: pd.DataFrame, config: BacktestConfig, *, fee_rate: float):
    if vbt is None:
        raise RuntimeError("VectorBT is required for simulate_strategy; install the declared project dependency")
    sizes = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    prices = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    call_seq = pd.DataFrame(
        np.tile(np.arange(len(close.columns), dtype=np.int64), (len(close.index), 1)),
        index=close.index, columns=close.columns,
    )
    for row in orders.itertuples(index=False):
        sizes.loc[row.timestamp, row.ticker] += float(row.quantity)
        prices.loc[row.timestamp, row.ticker] = float(row.price)
    for timestamp, group in orders.groupby("timestamp", sort=False):
        ordered = list(dict.fromkeys(group["ticker"].tolist()))
        ordered.extend(ticker for ticker in close.columns if ticker not in ordered)
        call_seq.loc[timestamp] = [close.columns.get_loc(ticker) for ticker in ordered]
    return vbt.Portfolio.from_orders(
        close,
        size=sizes,
        price=prices,
        size_type="amount",
        init_cash=float(config.initial_capital),
        fees=fee_rate,
        call_seq=call_seq,
        cash_sharing=True,
        group_by=True,
        allow_partial=False,
        raise_reject=True,
        freq="1D",
    )


def simulate_strategy(
    snapshot: MarketSnapshot,
    predictions: PredictionFrame | Any,
    allocations: tuple[AllocationFrame, ...] | list[AllocationFrame],
    task_config: TaskConfig,
    backtest_config: BacktestConfig,
    evaluation_period: Any,
) -> BacktestResult:
    """Execute forecast decisions at next opens and return VectorBT accounting outputs."""
    start, end, liquidation = _period_bounds(evaluation_period)
    close, bars = _price_matrix(snapshot, start, liquidation or end)
    prediction_map = _prediction_map(predictions)
    allocation_tuple = tuple(allocations)
    fee_rate = float(backtest_config.fees_bps + backtest_config.slippage_bps) / 10_000.0
    orders, event_rows = _target_orders(
        close,
        bars,
        allocation_tuple,
        prediction_map,
        task_config=task_config,
        backtest_config=backtest_config,
        fee_rate=fee_rate,
        liquidation=liquidation,
    )
    gross_orders, _ = _target_orders(
        close, bars, allocation_tuple, prediction_map,
        task_config=task_config, backtest_config=backtest_config,
        fee_rate=0.0, liquidation=liquidation,
    )
    gross_pf = _run_vectorbt(close, gross_orders, backtest_config, fee_rate=0.0)
    net_pf = _run_vectorbt(close, orders, backtest_config, fee_rate=fee_rate)
    actual_records = net_pf.orders.records_readable
    quantity_rtol = 1e-9
    quantity_atol = 1e-9
    notional_rtol = 1e-9
    notional_atol = 1e-8

    def normalized_timestamp(value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    requested_by_identity: dict[tuple[pd.Timestamp, str, str], list[int]] = {}
    for order_index, request in enumerate(orders.itertuples(index=False)):
        identity = (normalized_timestamp(request.timestamp), str(request.ticker), request.side.lower())
        requested_by_identity.setdefault(identity, []).append(order_index)
    matched_records: list[Any | None] = [None] * len(orders)
    for actual_index, actual in enumerate(actual_records.itertuples(index=False)):
        side = str(actual.Side).lower()
        identity = (normalized_timestamp(actual.Timestamp), str(actual.Column), side)
        matches = requested_by_identity.get(identity, [])
        if not matches:
            raise ValueError(
                f"VectorBT returned an unmatched order at order index={actual_index}: "
                f"timestamp={normalized_timestamp(actual.Timestamp)}, ticker={actual.Column}, "
                f"side={actual.Side}, requested quantity=0, executed quantity={float(actual.Size):.12g}, "
                f"absolute difference={abs(float(actual.Size)):.12g}, relative difference=inf, "
                f"requested notional=0, executed notional={abs(float(actual.Size) * float(actual.Price)):.12g}, "
                f"quantity tolerance=atol {quantity_atol:g} + rtol {quantity_rtol:g}"
            )
        matched_records[matches.pop(0)] = actual
    if len(actual_records) != len(orders) or any(actual is None for actual in matched_records):
        missing_index = next((index for index, actual in enumerate(matched_records) if actual is None), None)
        if missing_index is None:
            raise ValueError(f"VectorBT executed {len(actual_records)} orders for {len(orders)} requested orders")
        request = orders.iloc[missing_index]
        raise ValueError(
            f"VectorBT omitted order index={missing_index}, timestamp={request['timestamp']}, "
            f"ticker={request['ticker']}, side={request['side']}, requested quantity={request['quantity']:.12g}, "
            f"executed quantity=0, absolute difference={abs(float(request['quantity'])):.12g}, "
            f"relative difference=1, requested notional={abs(float(request['quantity'] * request['price'])):.12g}, "
            f"executed notional=0, quantity tolerance=atol {quantity_atol:g} + rtol {quantity_rtol:g}"
        )
    for order_index, (request, actual) in enumerate(zip(orders.itertuples(index=False), matched_records)):
        executed_quantity = float(actual.Size) * (1.0 if str(actual.Side).lower() == "buy" else -1.0)
        requested_notional = abs(float(request.quantity) * float(request.price))
        executed_notional = abs(float(actual.Size) * float(actual.Price))
        quantity_difference = abs(float(request.quantity) - executed_quantity)
        relative_quantity_difference = quantity_difference / max(abs(float(request.quantity)), quantity_atol)
        notional_difference = abs(requested_notional - executed_notional)
        quantity_matches = np.isclose(request.quantity, executed_quantity, rtol=quantity_rtol, atol=quantity_atol)
        notional_matches = np.isclose(requested_notional, executed_notional, rtol=notional_rtol, atol=notional_atol)
        if not quantity_matches or not notional_matches:
            mismatch = "quantity" if not quantity_matches else "notional"
            raise ValueError(
                f"VectorBT {mismatch} reconciliation failed for order index={order_index}, "
                f"timestamp={normalized_timestamp(request.timestamp)}, ticker={request.ticker}, side={request.side}: "
                f"requested quantity={request.quantity:.12g}, executed quantity={executed_quantity:.12g}, "
                f"absolute difference={quantity_difference:.12g}, relative difference={relative_quantity_difference:.12g}, "
                f"requested notional={requested_notional:.12g}, executed notional={executed_notional:.12g}, "
                f"quantity tolerance=atol {quantity_atol:g} + rtol {quantity_rtol:g}, "
                f"notional tolerance=atol {notional_atol:g} + rtol {notional_rtol:g}"
            )
    orders = orders.copy()
    if not orders.empty:
        orders["price"] = [float(actual.Price) for actual in matched_records]
        orders["fees"] = [float(actual.Fees) for actual in matched_records]
    values = pd.Series(net_pf.value(), index=close.index, name="equity")
    gross_values = pd.Series(gross_pf.value(), index=close.index, name="equity")
    equity = pl.DataFrame({"timestamp": list(values.index.to_pydatetime()), "equity": values.to_numpy(), "gross_equity": gross_values.to_numpy()})
    net_returns = values.pct_change().fillna(0.0).to_numpy()
    gross_returns = gross_values.pct_change().fillna(0.0).to_numpy()
    order_frame = pl.from_pandas(orders) if not orders.empty else pl.DataFrame(
        schema={"timestamp": pl.Datetime("us", "UTC"), "ticker": pl.String, "quantity": pl.Float64,
                "price": pl.Float64, "side": pl.String, "reason": pl.String, "fees": pl.Float64}
    )
    event_frame = pl.DataFrame(event_rows) if event_rows else pl.DataFrame(
        schema={"timestamp": pl.Datetime("us", "UTC"), "ticker": pl.String,
                "entry_quantity": pl.Float64, "entry_price": pl.Float64, "exit_reason": pl.String}
    )
    exposure = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if not orders.empty:
        for ticker in close.columns:
            changes = orders.loc[orders["ticker"] == ticker].groupby("timestamp")["quantity"].sum()
            exposure[ticker] = changes.reindex(close.index, fill_value=0.0).cumsum()
    exposure_frame = pl.from_pandas(exposure.reset_index(names="timestamp"))
    turnover = orders.groupby("timestamp")["quantity"].apply(lambda values: float(np.abs(values).sum())).reindex(close.index, fill_value=0.0) if not orders.empty else pd.Series(0.0, index=close.index)
    return BacktestResult(
        orders=order_frame,
        trade_events=event_frame,
        equity=equity,
        gross_returns=pl.Series("gross_returns", gross_returns),
        net_returns=pl.Series("net_returns", net_returns),
        exposure=exposure_frame,
        turnover=pl.Series("turnover", turnover.to_numpy()),
        period={"first_decision": start, "last_decision": end, "final_liquidation_open": liquidation},
        diagnostics={"engine": "vectorbt", "gross_final_equity": float(gross_values.iloc[-1]), "net_final_equity": float(values.iloc[-1])},
    )
