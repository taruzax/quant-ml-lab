from datetime import datetime, timezone

import numpy as np
import polars as pl

from lab.core.contracts import SampleKey, SampleSet, WindowSet
from lab.core.schemas import validate_feature_names


_UTC_DATETIME = pl.Datetime(time_unit="us", time_zone="UTC")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Dataset timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _datetime_series(name: str, values: list[datetime]) -> pl.Series:
    return pl.Series(name, values, dtype=_UTC_DATETIME)


def _empty_candidates() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "decision_time": _UTC_DATETIME,
            "_block_id": pl.Int64,
            "_start": pl.Int64,
            "window_start": _UTC_DATETIME,
            "window_end": _UTC_DATETIME,
            "_candidate_order": pl.Int64,
        }
    )


def build_sample_set(
    frame: pl.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    sequence_len: int,
    *,
    labels: pl.DataFrame | None = None,
    allowed_decisions: set[tuple[str, datetime]] | None = None,
    target_name: str | None = None,
) -> SampleSet:
    """Build one lazy indexed sample set from continuous observable windows."""
    if sequence_len <= 0:
        raise ValueError("sequence_len must be positive")
    feature_columns = validate_feature_names(tuple(feature_columns), set(frame.columns))
    required = {"ticker", "timestamp", *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset frame is missing columns: {missing}")
    if labels is not None:
        if target_name is None:
            raise ValueError("target_name is required for supervised windows")
        if target_name not in labels.columns or "ticker" not in labels.columns or "decision_time" not in labels.columns:
            raise ValueError("Labels require ticker, decision_time, and target_name columns")
        if "t1" not in labels.columns:
            raise ValueError("Labels require t1; unresolved-event fallback is forbidden")
        if labels["t1"].null_count():
            raise ValueError("Labels with null t1 cannot become supervised samples")

    allowed = None if allowed_decisions is None else {
        (str(ticker), _utc(timestamp)) for ticker, timestamp in allowed_decisions
    }
    ordered = frame.sort(["ticker", "timestamp"])
    feature_blocks: list[np.ndarray] = []
    candidate_frames: list[pl.DataFrame] = []
    candidate_order = 0

    for block_id, (ticker, group) in enumerate(ordered.group_by("ticker", maintain_order=True)):
        ticker_name = str(ticker[0] if isinstance(ticker, tuple) else ticker)
        values = group.select(feature_columns).to_numpy().astype(np.float32, copy=False)
        feature_blocks.append(values)
        row_count = len(values)
        if row_count < sequence_len:
            continue

        timestamps = [_utc(value) for value in group["timestamp"].to_list()]
        starts = np.arange(row_count - sequence_len + 1, dtype=np.int64)
        ends = starts + sequence_len - 1
        finite_rows = np.isfinite(values).all(axis=1)
        bad_prefix = np.concatenate(([0], np.cumsum(~finite_rows, dtype=np.int64)))
        bad_counts = bad_prefix[ends + 1] - bad_prefix[starts]

        if "raw_bar_index" in group.columns:
            raw_indices = group["raw_bar_index"].to_numpy().astype(np.int64, copy=False)
        else:
            raw_indices = np.arange(row_count, dtype=np.int64)
        gaps = np.diff(raw_indices) != 1
        gap_prefix = np.concatenate(([0], np.cumsum(gaps, dtype=np.int64)))
        gap_counts = gap_prefix[ends] - gap_prefix[starts]
        eligible = (bad_counts == 0) & (gap_counts == 0)
        starts = starts[eligible]
        ends = ends[eligible]
        if not len(starts):
            continue

        candidate_frames.append(
            pl.DataFrame(
                {
                    "ticker": [ticker_name] * len(starts),
                    "decision_time": _datetime_series("decision_time", [timestamps[index] for index in ends]),
                    "_block_id": np.full(len(starts), block_id, dtype=np.int64),
                    "_start": starts,
                    "window_start": _datetime_series("window_start", [timestamps[index] for index in starts]),
                    "window_end": _datetime_series("window_end", [timestamps[index] for index in ends]),
                    "_candidate_order": np.arange(candidate_order, candidate_order + len(starts), dtype=np.int64),
                }
            )
        )
        candidate_order += len(starts)

    candidates = pl.concat(candidate_frames) if candidate_frames else _empty_candidates()
    if allowed is not None:
        allowed_frame = pl.DataFrame(
            {
                "ticker": pl.Series("ticker", [ticker for ticker, _ in allowed], dtype=pl.String),
                "decision_time": _datetime_series("decision_time", [timestamp for _, timestamp in allowed]),
            }
        ).unique(subset=["ticker", "decision_time"])
        candidates = candidates.join(allowed_frame, on=["ticker", "decision_time"], how="semi")

    if labels is not None:
        label_order = np.arange(labels.height, dtype=np.int64)
        label_frame = pl.DataFrame(
            {
                "ticker": labels["ticker"].cast(pl.String),
                "decision_time": _datetime_series(
                    "decision_time", [_utc(value) for value in labels["decision_time"].to_list()]
                ),
                target_name: labels[target_name],
                "event_end": labels["event_end"] if "event_end" in labels.columns else [None] * labels.height,
                "t1": labels["t1"],
                "_label_order": label_order,
            }
        ).sort("_label_order").unique(subset=["ticker", "decision_time"], keep="last", maintain_order=True)
        candidates = candidates.join(label_frame, on=["ticker", "decision_time"], how="inner")

    candidates = candidates.sort("_candidate_order")
    block_ids = candidates["_block_id"].to_numpy().astype(np.int64, copy=False)
    starts = candidates["_start"].to_numpy().astype(np.int64, copy=False)
    keys = tuple(
        SampleKey(ticker=ticker, timestamp=_utc(timestamp))
        for ticker, timestamp in zip(candidates["ticker"].to_list(), candidates["decision_time"].to_list())
    )
    window_start = tuple(_utc(value) for value in candidates["window_start"].to_list())
    window_end = tuple(_utc(value) for value in candidates["window_end"].to_list())
    targets = None
    metadata = None
    if labels is not None:
        targets = candidates[target_name].to_numpy().astype(np.float32, copy=False).reshape(-1, 1)
        metadata = tuple(
            {"decision_time": key.timestamp, "event_end": event_end, "t1": t1}
            for key, event_end, t1 in zip(keys, candidates["event_end"].to_list(), candidates["t1"].to_list())
        )

    windows = WindowSet(
        feature_blocks, block_ids, starts, sequence_len,
        feature_count=len(feature_columns), y=targets,
    )
    return SampleSet(
        X=windows,
        keys=keys,
        window_start=window_start,
        window_end=window_end,
        feature_order=feature_columns,
        y=targets,
        label_metadata=metadata,
    )
