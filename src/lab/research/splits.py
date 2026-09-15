from datetime import datetime, timedelta, timezone

import polars as pl

from lab.core.config import SplitsConfig
from lab.core.contracts import ExecutionPeriod, FoldSpec, MarketSnapshot, SplitPlan


def _as_utc(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _snapshot_bars(snapshot: MarketSnapshot | pl.DataFrame) -> pl.DataFrame:
    return snapshot.bars if isinstance(snapshot, MarketSnapshot) else snapshot


def _decision_timestamps(snapshot: MarketSnapshot | pl.DataFrame) -> tuple[datetime, ...]:
    bars = _snapshot_bars(snapshot)
    if "timestamp" not in bars.columns:
        raise ValueError("Split planning requires decision timestamp bars")
    values = bars.get_column("timestamp").unique().sort().to_list()
    if not values:
        raise ValueError("Cannot build a split plan from empty market data")
    return tuple(values)


def _boundary_end(timestamps: tuple[datetime, ...], index: int) -> datetime:
    if index < len(timestamps):
        return timestamps[index]
    return timestamps[-1] + timedelta(microseconds=1)


def _explicit_index(timestamps: tuple[datetime, ...], value: str | datetime, *, name: str) -> int:
    boundary = _as_utc(value)
    matching = [index for index, timestamp in enumerate(timestamps) if timestamp >= boundary]
    if not matching:
        raise ValueError(f"Explicit split boundary {name}={boundary.isoformat()} is after the snapshot")
    return matching[0]


def _execution_period(name: str, decision_times: tuple[datetime, ...], bars: pl.DataFrame) -> ExecutionPeriod:
    if not decision_times:
        raise ValueError(f"Execution period {name} has no usable decision timestamps")
    last_decision = decision_times[-1]
    future = bars.filter(pl.col("timestamp") > last_decision).sort("timestamp")
    if not future.is_empty() and "bar_open_time" in future.columns:
        liquidation_open = future["bar_open_time"][0]
    elif "bar_open_time" in bars.columns:
        liquidation_open = bars.filter(pl.col("timestamp") == last_decision)["bar_open_time"][0]
    else:
        liquidation_open = last_decision
    return ExecutionPeriod(
        name=name,
        first_decision=decision_times[0],
        last_decision=last_decision,
        final_liquidation_open=liquidation_open,
    )


def build_split_plan(snapshot: MarketSnapshot | pl.DataFrame, splits_config: SplitsConfig) -> SplitPlan:
    """Build deterministic chronological, half-open train/validation/holdout folds."""
    bars = _snapshot_bars(snapshot).sort(["ticker", "timestamp"])
    timestamps = _decision_timestamps(bars)
    n_timestamps = len(timestamps)
    holdout_count = max(1, int(n_timestamps * splits_config.holdout_fraction))
    if holdout_count >= n_timestamps:
        raise ValueError("Snapshot is too short to leave a non-empty development period")

    explicit = splits_config.explicit_boundaries or {}
    if explicit and "holdout_start" not in explicit:
        raise ValueError("Explicit split boundaries require holdout_start")
    holdout_start_index = (
        _explicit_index(timestamps, explicit["holdout_start"], name="holdout_start")
        if explicit
        else n_timestamps - holdout_count
    )
    development_start_index = (
        _explicit_index(timestamps, explicit.get("development_start", timestamps[0]), name="development_start")
        if explicit
        else 0
    )
    holdout_end = (
        _as_utc(explicit["holdout_end"])
        if "holdout_end" in explicit
        else _boundary_end(timestamps, n_timestamps)
    )
    if not 0 <= development_start_index < holdout_start_index < n_timestamps:
        raise ValueError("Explicit boundaries must leave non-empty development and holdout periods")

    development_times = timestamps[development_start_index:holdout_start_index]
    holdout_times = tuple(timestamp for timestamp in timestamps if timestamp >= timestamps[holdout_start_index] and timestamp < holdout_end)
    if not development_times or not holdout_times:
        raise ValueError("Split boundaries must produce non-empty development and holdout timestamps")

    train_count = int(len(development_times) * splits_config.development_train_fraction)
    if train_count <= 0 or train_count >= len(development_times):
        raise ValueError("Development split cannot produce both initial training and validation timestamps")
    validation_times = development_times[train_count:]
    if len(validation_times) < splits_config.n_folds:
        raise ValueError("Snapshot is too short for the requested non-empty validation folds")

    base_size, remainder = divmod(len(validation_times), splits_config.n_folds)
    folds: list[FoldSpec] = []
    start = 0
    for fold_id in range(splits_config.n_folds):
        width = base_size + (1 if fold_id < remainder else 0)
        validation_start_index = train_count + start
        validation_end_index = validation_start_index + width
        validation_start = development_times[validation_start_index]
        validation_end = _boundary_end(development_times, validation_end_index)
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_start=development_times[0],
                train_end=validation_start,
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )
        start += width

    development_period = _execution_period("development", development_times, bars)
    holdout_period = _execution_period("holdout", holdout_times, bars)
    return SplitPlan(
        decision_timestamps=timestamps,
        development_start=development_times[0],
        holdout_start=holdout_times[0],
        holdout_end=holdout_end,
        folds=tuple(folds),
        execution_periods=(development_period, holdout_period),
        embargo_bars=splits_config.embargo_bars,
    )


def period_timestamps(plan: SplitPlan, name: str) -> tuple[datetime, ...]:
    """Return decision timestamps in a named half-open execution period."""
    if name == "development":
        return tuple(timestamp for timestamp in plan.decision_timestamps if timestamp < plan.holdout_start)
    if name == "holdout":
        return tuple(
            timestamp
            for timestamp in plan.decision_timestamps
            if plan.holdout_start <= timestamp < plan.holdout_end
        )
    raise ValueError(f"Unknown execution period: {name}")
