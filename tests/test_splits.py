from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from lab.core.config import SplitsConfig
from lab.quant.cv import fold_training_labels
from lab.research.splits import build_split_plan


def _bars(count: int = 20) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(days=index) for index in range(count)]
    return pl.DataFrame(
        {
            "ticker": ["AAA", "BBB"] * (count // 2),
            "timestamp": timestamps,
            "bar_open_time": timestamps,
        }
    )


def test_split_plan_uses_unique_timestamps_and_latest_holdout():
    plan = build_split_plan(_bars(), SplitsConfig(n_folds=3))
    assert len(plan.decision_timestamps) == 20
    assert plan.holdout_start == plan.decision_timestamps[16]
    assert [fold.validation_start for fold in plan.folds] == sorted(
        fold.validation_start for fold in plan.folds
    )


def test_purge_removes_equality_boundary():
    labels = pl.DataFrame(
        {
            "decision_time": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "t1": [datetime(2024, 1, 3, tzinfo=timezone.utc)],
        }
    )
    assert fold_training_labels(
        labels,
        type(
            "Fold",
            (),
            {
                "train_start": labels["decision_time"][0],
                "train_end": labels["decision_time"][0] + timedelta(days=1),
                "validation_start": labels["t1"][0],
            },
        )(),
    ).is_empty()


def test_split_plan_rejects_short_validation_history():
    with pytest.raises(ValueError, match="too short"):
        build_split_plan(_bars(6), SplitsConfig(n_folds=3, development_train_fraction=0.8))
