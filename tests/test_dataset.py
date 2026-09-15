from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from lab.research.dataset import build_sample_set


def test_windows_include_final_valid_row_and_preserve_raw_indices():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(days=index) for index in range(5)]
    frame = pl.DataFrame(
        {
            "ticker": ["AAA"] * 5,
            "timestamp": timestamps,
            "raw_bar_index": [0, 1, 2, 3, 4],
            "close": np.arange(5, dtype=float),
        }
    )
    samples = build_sample_set(frame, ["close"], 3)
    assert samples.X.shape == (3, 3, 1)
    assert samples.raw_row_indices[-1] == (2, 3, 4)


def test_supervised_windows_require_explicit_t1():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "timestamp": [start + timedelta(days=index) for index in range(3)],
            "close": [1.0, 2.0, 3.0],
        }
    )
    labels = pl.DataFrame(
        {
            "ticker": ["AAA"],
            "decision_time": [start + timedelta(days=2)],
            "target": [1.0],
        }
    )
    try:
        build_sample_set(frame, ["close"], 3, labels=labels, target_name="target")
    except ValueError as exc:
        assert "t1" in str(exc)
    else:
        raise AssertionError("supervised windows accepted labels without t1")
