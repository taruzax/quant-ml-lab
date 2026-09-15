import math
from datetime import datetime, timedelta, timezone

import polars as pl

from lab.core.config import PipelineConfig
from lab.core.contracts import PreprocessingState
from lab.research.preprocessing import FoldPreprocessor, fit_preprocessor


def test_preprocessing_state_is_fit_only_on_train_history():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "timestamp": [start + timedelta(days=index) for index in range(4)],
            "value": [1.0, 2.0, 3.0, 10000.0],
        }
    )
    state = fit_preprocessor(frame, ["value"], start, start + timedelta(days=3), PipelineConfig())
    assert state.state.means["value"] < 10.0


def test_unknown_categories_are_encoded_without_refitting():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "timestamp": [start + timedelta(days=index) for index in range(3)],
            "category": ["known", "known", "new"],
        }
    )
    preprocessor = fit_preprocessor(frame, ["category"], start, start + timedelta(days=2), PipelineConfig())
    transformed = preprocessor.transform(frame)
    assert transformed["category"].to_list()[-1] == -1.0


def test_transform_executes_fitted_ffd_before_scaling():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "timestamp": [start + timedelta(days=index) for index in range(3)],
            "close": [1.0, 2.0, 4.0],
        }
    )
    preprocessor = FoldPreprocessor(
        PreprocessingState(
            feature_order=("close",),
            ffd_orders={"AAA": 1.0},
            ffd_features=("close",),
            ffd_threshold=0.001,
        )
    )
    transformed = preprocessor.transform(frame)
    assert transformed["close"][1] == math.log(2.0)
    assert transformed["close"][2] == math.log(2.0)
