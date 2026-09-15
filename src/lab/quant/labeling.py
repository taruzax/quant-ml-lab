import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from lab.core.config import PipelineConfig, TaskConfig
from lab.core.contracts import LabelResult, MarketSnapshot
from lab.quant.barriers import barrier_prices, first_barrier_touch

logger = logging.getLogger(__name__)


def _empty_labels(kind: str) -> pl.DataFrame:
    columns: dict[str, pl.Series] = {
        "ticker": pl.Series([], dtype=pl.Utf8),
        "decision_time": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
        "entry_time": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
        "event_end": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
        "t1": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
    }
    if kind == "regression":
        columns.update(
            {
                "target": pl.Series([], dtype=pl.Float64),
                "target_1b_v2": pl.Series([], dtype=pl.Float64),
                "target_1b": pl.Series([], dtype=pl.Float64),
            }
        )
    else:
        columns.update(
            {
                "label": pl.Series([], dtype=pl.Int8),
                "target": pl.Series([], dtype=pl.Int8),
            }
        )
    return pl.DataFrame(columns)


def _empty_exclusions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": pl.Series([], dtype=pl.Utf8),
            "decision_time": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
            "reason": pl.Series([], dtype=pl.Utf8),
        }
    )


def _bar_time_column(bars: pl.DataFrame, name: str) -> str:
    if name in bars.columns:
        return name
    if "timestamp" in bars.columns:
        return "timestamp"
    raise ValueError("Bars require timestamp or canonical bar time columns")


def _causal_ewm_volatility(values: list[float], config: TaskConfig) -> list[float | None]:
    barrier = config.triple_barrier
    returns = [np.nan]
    returns.extend(values[i] / values[i - 1] - 1.0 for i in range(1, len(values)))
    series = pd.Series(returns, dtype="float64")
    volatility = series.ewm(
        span=barrier.volatility_span,
        adjust=barrier.ewm_adjust,
        min_periods=barrier.volatility_warmup,
    ).std(bias=barrier.ewm_bias)
    result: list[float | None] = []
    for value in volatility.to_list():
        if value is None or not np.isfinite(value):
            result.append(None)
        else:
            result.append(max(float(value), barrier.volatility_floor))
    return result


def compute_labels(snapshot: MarketSnapshot, task_config: TaskConfig) -> LabelResult:
    """Compute complete labels from raw bars without using feature columns.

    Regression uses the next two opens. Classification freezes volatility and
    percentage distances at the decision close, observes closes through the
    expiry bar, and resolves at the next open after the trigger or expiry.
    """
    bars = snapshot.bars.sort(["ticker", "timestamp"])
    time_column = _bar_time_column(bars, "timestamp")
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for ticker, group in bars.group_by("ticker", maintain_order=True):
        ticker_name = ticker[0] if isinstance(ticker, tuple) else ticker
        group = group.sort(time_column)
        rows = group.to_dicts()
        closes = [float(row["close"]) for row in rows]
        volatility = _causal_ewm_volatility(closes, task_config)
        for index, row in enumerate(rows):
            decision_time = row[time_column]
            if task_config.kind == "regression":
                if index + 2 >= len(rows):
                    exclusions.append({"ticker": ticker_name, "decision_time": decision_time, "reason": "tail"})
                    continue
                entry = rows[index + 1]
                finish = rows[index + 2]
                target = float(finish["open"]) / float(entry["open"]) - 1.0
                records.append(
                    {
                        "ticker": ticker_name,
                        "decision_time": decision_time,
                        "entry_time": entry.get("bar_open_time", entry[time_column]),
                        "event_end": finish.get("bar_close_time", finish[time_column]),
                        "t1": finish.get("bar_open_time", finish[time_column]),
                        "target": target,
                        "target_1b_v2": target,
                        "target_1b": target,
                    }
                )
                continue

            if volatility[index] is None:
                exclusions.append({"ticker": ticker_name, "decision_time": decision_time, "reason": "warmup"})
                continue
            expiry = task_config.triple_barrier.expiry_bars
            entry_index = index + 1
            expiry_index = index + expiry
            if expiry_index >= len(rows) or expiry_index + 1 >= len(rows):
                exclusions.append({"ticker": ticker_name, "decision_time": decision_time, "reason": "tail"})
                continue
            entry = rows[entry_index]
            entry_price = float(entry["open"])
            upper, lower = barrier_prices(entry_price, float(volatility[index]), task_config)
            touch_offset, direction = first_barrier_touch(
                [float(rows[offset]["close"]) for offset in range(entry_index, expiry_index + 1)], upper, lower
            )
            trigger_index = expiry_index if touch_offset is None else entry_index + touch_offset
            if touch_offset is not None and direction == 0:
                raise ValueError("Barrier touch must resolve to a non-zero direction")
            execution_index = trigger_index + 1
            if execution_index >= len(rows):
                exclusions.append({"ticker": ticker_name, "decision_time": decision_time, "reason": "tail"})
                continue
            trigger = rows[trigger_index]
            execution = rows[execution_index]
            records.append(
                {
                    "ticker": ticker_name,
                    "decision_time": decision_time,
                    "entry_time": entry.get("bar_open_time", entry[time_column]),
                    "event_end": trigger.get("bar_close_time", trigger[time_column]),
                    "t1": execution.get("bar_open_time", execution[time_column]),
                    "label": direction,
                    "target": direction,
                    "volatility": float(volatility[index]),
                    "upper_barrier": upper,
                    "lower_barrier": lower,
                }
            )

    labels = pl.DataFrame(records) if records else _empty_labels(task_config.kind)
    if records:
        labels = labels.sort(["ticker", "decision_time"])
    excluded_frame = pl.DataFrame(exclusions) if exclusions else _empty_exclusions()
    if exclusions:
        excluded_frame = excluded_frame.sort(["ticker", "decision_time"])
    target_name = "target_1b_v2" if task_config.kind == "regression" else "label"
    return LabelResult(
        labels=labels,
        exclusions=excluded_frame,
        target_name=target_name,
        conventions={
            "decision_time": "close timestamp",
            "entry_time": "next bar open",
            "event_end": "trigger or expiry close",
            "t1": "open after trigger or expiry",
            "classification_target": "barrier direction, not net P&L",
        },
    )


def calculate_volatility(
    df,
    span_bars: int = 100,
    vol_lookback_bars: int = 20,
    min_volatility: float = 1e-4,
    price_col: str = "close",
    group_col: str = "ticker",
):
    """Calculates exponentially weighted volatility"""
    returns_expr = (pl.col(price_col) / pl.col(price_col).shift(1).over(group_col)) - 1
    vol_expr = returns_expr.ewm_std(span=span_bars, ignore_nulls=True).over(group_col).fill_null(min_volatility)
    vol_floored = pl.when(vol_expr < min_volatility).then(pl.lit(min_volatility)).otherwise(vol_expr)
    vol_predictive = vol_floored.shift(1).over(group_col)
    return df.with_columns(vol_predictive.alias("volatility_t_minus_1"))


def triple_barrier_label(
    df, config: PipelineConfig, price_col: str = "close", date_col: str = "timestamp", group_col: str = "ticker"
):
    """Compute path-dependent Triple-Barrier labels and observation end times (t1)"""

    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Int8).alias("label"),
            pl.lit(None, dtype=pl.Datetime).alias("t1"),
            pl.lit(None, dtype=pl.Datetime).alias("t0"),
        )

    required_min_rows = config.vol_lookback_bars + config.expiry_bars
    valid_dfs = []

    for group_name, group_df in df.group_by(group_col, maintain_order=True):
        ticker_name = group_name[0] if isinstance(group_name, tuple) else group_name
        if group_df.height < required_min_rows:
            logger.warning(
                "Ticker '%s' has %d rows, minimum required is %d (vol_lookback %d + expiry %d). Skipping.",
                ticker_name,
                group_df.height,
                required_min_rows,
                config.vol_lookback_bars,
                config.expiry_bars,
            )
            continue
        valid_dfs.append(group_df)
    if not valid_dfs:
        logger.warning("No ticker groups satisfied minimum length requirement for labeling.")
        return df.clear().with_columns(
            pl.lit(None, dtype=pl.Int8).alias("label"),
            pl.lit(None, dtype=pl.Datetime).alias("t1"),
            pl.lit(None, dtype=pl.Datetime).alias("t0"),
        )
    processed_df = pl.concat(valid_dfs).sort([group_col, date_col])

    if "volatility" not in processed_df.columns:
        processed_df = calculate_volatility(
            processed_df,
            vol_lookback_bars=config.vol_lookback_bars,
            min_volatility=config.min_volatility,
            price_col=price_col,
            group_col=group_col,
        )

    processed_df = processed_df.with_columns(
        (pl.col(price_col) * (1.0 + config.profit_taking * pl.col("volatility"))).alias("upper_barrier"),
        (pl.col(price_col) * (1.0 - config.stop_loss * pl.col("volatility"))).alias("lower_barrier"),
        pl.col(date_col).alias("t0"),
    )

    expiry = config.expiry_bars
    label_expr = pl.lit(0, dtype=pl.Int8)
    t1_expr = pl.col(date_col).shift(-expiry).over(group_col)

    for k in range(expiry, 0, -1):
        future_close = pl.col(price_col).shift(-k).over(group_col)
        future_ts = pl.col(date_col).shift(-k).over(group_col)

        hit_upper = future_close >= pl.col("upper_barrier")
        hit_lower = future_close <= pl.col("lower_barrier")
        both_hit = hit_upper & hit_lower

        step_label = (
            pl.when(both_hit)
            .then(pl.when(future_close >= pl.col(price_col)).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(-1, dtype=pl.Int8)))
            .when(hit_upper)
            .then(pl.lit(1, dtype=pl.Int8))
            .when(hit_lower)
            .then(pl.lit(-1, dtype=pl.Int8))
            .otherwise(label_expr)
        )

        label_expr = step_label
        t1_expr = pl.when(hit_upper | hit_lower).then(future_ts).otherwise(t1_expr)

    result_df = processed_df.with_columns(
        label_expr.alias("label"),
        t1_expr.alias("t1"),
    )

    # Filter out trailing rows where vertical barrier extends beyond available future history
    return result_df.filter(pl.col("t1").is_not_null())
