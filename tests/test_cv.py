from datetime import datetime, timezone

import polars as pl
import pytest

from lab.quant.cv import PurgedKFold, purge_by_information_interval, scoreable_labels


def test_purged_kfold_requires_t1():
    with pytest.raises(ValueError, match="explicit.*t1"):
        list(PurgedKFold(n_splits=2).split(pl.DataFrame({"timestamp": [1, 2, 3, 4]})))


def test_scoreable_labels_reports_unresolved_rows():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    labels = pl.DataFrame(
        {
            "decision_time": [start, start.replace(day=2)],
            "t1": [None, start.replace(day=3)],
        },
        schema={"decision_time": pl.Datetime(time_zone="UTC"), "t1": pl.Datetime(time_zone="UTC")},
    )
    scored, report = scoreable_labels(labels, start, start.replace(day=4))
    assert scored.height == 1
    assert report["unresolved_rows"] == 1


def test_information_interval_excludes_overlap():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    samples = pl.DataFrame(
        {
            "feature_start": [start, start.replace(day=4)],
            "t1": [start.replace(day=3), start.replace(day=5)],
        }
    )
    kept = purge_by_information_interval(samples, start.replace(day=3), start.replace(day=4))
    assert kept.height == 1
