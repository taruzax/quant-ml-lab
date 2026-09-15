from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from lab.core.contracts import SampleKey, SampleSet
from lab.core.schemas import validate_feature_names


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Dataset timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _allowed_keys(values: set[tuple[str, datetime]] | None) -> set[tuple[str, datetime]] | None:
    if values is None:
        return None
    return {(ticker, _utc(timestamp)) for ticker, timestamp in values}


def build_sample_set(
    frame: pl.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    sequence_len: int,
    *,
    labels: pl.DataFrame | None = None,
    allowed_decisions: set[tuple[str, datetime]] | None = None,
    target_name: str | None = None,
) -> SampleSet:
    """Build aligned supervised or target-free windows from observable rows."""
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

    allowed = _allowed_keys(allowed_decisions)
    label_map: dict[tuple[str, datetime], dict[str, Any]] = {}
    if labels is not None:
        for row in labels.to_dicts():
            label_map[(str(row["ticker"]), _utc(row["decision_time"]))] = row

    frame = frame.sort(["ticker", "timestamp"])
    X_rows: list[np.ndarray] = []
    keys: list[SampleKey] = []
    starts: list[datetime] = []
    ends: list[datetime] = []
    raw_indices: list[tuple[int, ...]] = []
    targets: list[list[float]] = []
    metadata: list[dict[str, Any]] = []

    for ticker, group in frame.group_by("ticker", maintain_order=True):
        ticker_name = str(ticker[0] if isinstance(ticker, tuple) else ticker)
        rows = group.to_dicts()
        values = group.select(feature_columns).to_numpy()
        timestamps = [_utc(row["timestamp"]) for row in rows]
        row_indices = [
            int(row.get("raw_bar_index", row_index))
            for row_index, row in enumerate(rows)
        ]
        for end_index in range(sequence_len - 1, len(rows)):
            start_index = end_index - sequence_len + 1
            window_key = (ticker_name, timestamps[end_index])
            if allowed is not None and window_key not in allowed:
                continue
            if any(
                row_indices[index] != row_indices[index - 1] + 1
                for index in range(start_index + 1, end_index + 1)
            ):
                continue
            window = values[start_index : end_index + 1].astype(np.float32)
            if not np.isfinite(window).all():
                continue
            label = label_map.get(window_key) if labels is not None else None
            if labels is not None and label is None:
                continue
            X_rows.append(window)
            keys.append(SampleKey(ticker=ticker_name, timestamp=timestamps[end_index]))
            starts.append(timestamps[start_index])
            ends.append(timestamps[end_index])
            raw_indices.append(tuple(row_indices[start_index : end_index + 1]))
            if label is not None:
                targets.append([float(label[target_name])])
                metadata.append(
                    {
                        "decision_time": timestamps[end_index],
                        "event_end": label.get("event_end"),
                        "t1": label["t1"],
                    }
                )

    array = np.stack(X_rows) if X_rows else np.empty((0, sequence_len, len(feature_columns)), dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32) if labels is not None else None
    return SampleSet(
        X=array,
        keys=tuple(keys),
        window_start=tuple(starts),
        window_end=tuple(ends),
        raw_row_indices=tuple(raw_indices),
        feature_order=feature_columns,
        y=target_array,
        label_metadata=tuple(metadata) if labels is not None else None,
    )
