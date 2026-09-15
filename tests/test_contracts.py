from datetime import datetime, timezone

import numpy as np
import pytest

from lab.core.contracts import PredictionFrame, SampleKey, SampleSet


def _keys(count: int) -> tuple[SampleKey, ...]:
    return tuple(SampleKey(ticker="AAA", timestamp=datetime(2024, 1, i + 1, tzinfo=timezone.utc)) for i in range(count))


def test_sample_set_rejects_future_features_and_preserves_keys():
    with pytest.raises(ValueError, match="Future-derived"):
        SampleSet(
            X=np.ones((2, 3, 1)),
            keys=_keys(2),
            window_start=tuple(key.timestamp for key in _keys(2)),
            window_end=tuple(key.timestamp for key in _keys(2)),
            raw_row_indices=((0, 1, 2), (1, 2, 3)),
            feature_order=("target_1b",),
        )


def test_prediction_frame_requires_three_probability_columns():
    with pytest.raises(ValueError, match=r"shape \[N, 3\]"):
        PredictionFrame(keys=_keys(2), model_ref="model", fold_ref="fold", predictions=np.ones((2, 2)))


def test_naive_sample_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        SampleKey(ticker="AAA", timestamp=datetime(2024, 1, 1))
