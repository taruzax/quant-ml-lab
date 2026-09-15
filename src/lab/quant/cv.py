from collections.abc import Generator
from typing import cast

import numpy as np
import polars as pl
from sklearn.model_selection._split import _BaseKFold

from lab.core.contracts import FoldSpec


class PurgedKFold(_BaseKFold):
    """Purged and Embargoed Cross-Validation Splitter."""

    def __init__(
        self,
        n_splits: int = 5,
        t1: pl.Series | None = None,
        t0: pl.Series | None = None,
        embargo_bars: int = 0,
        pct_embargo: float = 0.0,
    ):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        if n_splits <= 1:
            raise ValueError(f"n_splits must be greater than 1, got {n_splits}")
        if embargo_bars < 0:
            raise ValueError(f"embargo_bars must be non-negative, got {embargo_bars}")
        if pct_embargo < 0.0:
            raise ValueError(f"pct_embargo must be non-negative, got {pct_embargo}")
        if pct_embargo >= 1.0:
            raise ValueError(f"pct_embargo must be less than 1.0, got {pct_embargo}")
        self.n_splits = n_splits
        self.t1 = t1
        self.t0 = t0
        self.embargo_bars = embargo_bars
        self.pct_embargo = pct_embargo

    @staticmethod
    def _as_polars(values, name: str) -> pl.Series:
        if not isinstance(values, pl.Series):
            raise TypeError(
                f"{name} must be a polars Series, got {type(values).__name__}. "
                "Wrap it with pl.Series(...) before passing it in."
            )
        return values.rename(name)

    @classmethod
    def _column(cls, X, candidates: tuple[str, ...]) -> pl.Series | None:
        if not isinstance(X, pl.DataFrame):
            return None
        for candidate in candidates:
            if candidate in X.columns:
                return cls._as_polars(X[candidate], candidate)
        return None

    def _resolve_series(
        self,
        explicit,
        X,
        candidates: tuple[str, ...],
        fallback_start: int,
        n_samples: int,
        name: str,
    ) -> pl.Series:
        """Resolve event times: explicit value -> DataFrame column -> positional fallback."""
        if explicit is not None:
            return self._as_polars(explicit, name)

        from_frame = self._column(X, candidates)
        if from_frame is not None:
            return from_frame

        return pl.int_range(fallback_start, fallback_start + n_samples, eager=True).rename(name)

    def _get_t1_series(self, X, n_samples: int) -> pl.Series:
        """Event end times, as a polars Series of length ``n_samples``."""
        if self.t1 is None and self._column(X, ("t1",)) is None:
            raise ValueError("PurgedKFold requires explicit non-null t1 event ends; positional fallback is forbidden")
        return self._resolve_series(self.t1, X, ("t1",), 1, n_samples, "t1")

    def _get_t0_series(self, X, n_samples: int) -> pl.Series:
        """Event start times, as a polars Series of length ``n_samples``."""
        return self._resolve_series(self.t0, X, ("t0", "timestamp"), 0, n_samples, "t0")

    def _validate_event_times(self, t0_series: pl.Series, t1_series: pl.Series, n_samples: int) -> None:
        for name, series in (("t1", t1_series), ("t0", t0_series)):
            if len(series) != n_samples:
                raise ValueError(f"Length of {name} ({len(series)}) does not match sample count ({n_samples})")
        if t1_series.null_count() or t0_series.null_count():
            raise ValueError("t0/t1 must not contain nulls; drop unresolved events before splitting.")
        if t0_series.dtype.is_temporal() != t1_series.dtype.is_temporal():
            raise ValueError(
                f"t0 dtype ({t0_series.dtype}) and t1 dtype ({t1_series.dtype}) are not comparable. "
                "Pass both as timestamps or both as positional/numeric values."
            )
        if not t0_series.is_sorted():
            raise ValueError("Samples must be sorted by t0 ascending before purged cross-validation.")

    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        if n_samples < self.n_splits:
            raise ValueError(f"Cannot have n_splits={self.n_splits} greater than n_samples={n_samples}")

        effective_embargo = self.embargo_bars
        if self.pct_embargo > 0.0:
            effective_embargo = max(effective_embargo, int(n_samples * self.pct_embargo))

        if n_samples / self.n_splits < effective_embargo and effective_embargo > 0:
            raise ValueError(
                f"Sample size {n_samples} is too short for {self.n_splits} splits with embargo_bars={effective_embargo}."
            )

        t1_series = self._get_t1_series(X, n_samples)
        t0_series = self._get_t0_series(X, n_samples)

        self._validate_event_times(t0_series, t1_series, n_samples)

        fold_bounds = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)

        for fold_idx in range(self.n_splits):
            test_start = int(fold_bounds[fold_idx])
            test_end = int(fold_bounds[fold_idx + 1])

            test_indices = np.arange(test_start, test_end)
            test_t0_start = t0_series[test_start]
            test_t1_max = t1_series.slice(test_start, test_end - test_start).max()

            if test_start > 0:
                keep_left = t1_series.slice(0, test_start) < test_t0_start
                left_indices = keep_left.arg_true().to_numpy()
            else:
                left_indices = np.empty(0, dtype=np.int64)

            footprint_end_idx = cast(int, t0_series.search_sorted(test_t1_max, side="right"))
            resume_idx = max(footprint_end_idx, test_end) + effective_embargo
            if resume_idx < n_samples:
                right_indices = np.arange(resume_idx, n_samples)
            else:
                right_indices = np.empty(0, dtype=np.int64)

            train_indices = np.concatenate((left_indices, right_indices)).astype(np.int64)
            yield train_indices, test_indices


def purge_training_labels(labels: pl.DataFrame, evaluation_start) -> pl.DataFrame:
    """Keep only labels whose information resolves strictly before evaluation."""
    if "t1" not in labels.columns:
        raise ValueError("Label frame must contain t1 for purging")
    if labels["t1"].null_count():
        raise ValueError("Unresolved labels with null t1 cannot enter a supervised split")
    if "decision_time" not in labels.columns:
        raise ValueError("Label frame must contain decision_time for purging")
    return labels.filter(pl.col("t1") < evaluation_start)


def scoreable_labels(labels: pl.DataFrame, evaluation_start, evaluation_end) -> tuple[pl.DataFrame, dict[str, int]]:
    """Return only predictions whose labels resolve inside the permitted period."""
    if "t1" not in labels.columns:
        raise ValueError("Label frame must contain t1 for scoring")
    unresolved = int(labels["t1"].null_count())
    resolved = labels.filter(pl.col("t1").is_not_null())
    scoreable = resolved.filter(
        (pl.col("decision_time") >= evaluation_start)
        & (pl.col("decision_time") < evaluation_end)
        & (pl.col("t1") >= evaluation_start)
        & (pl.col("t1") < evaluation_end)
    )
    return scoreable, {
        "input_rows": labels.height,
        "unresolved_rows": unresolved,
        "scoreable_rows": scoreable.height,
        "excluded_rows": labels.height - unresolved - scoreable.height,
    }


def information_interval(feature_start, label_end):
    """Return the full information interval used by a sequence sample."""
    if feature_start is None or label_end is None:
        raise ValueError("Information intervals require both feature start and label end")
    return feature_start, label_end


def purge_by_information_interval(
    samples: pl.DataFrame, evaluation_start, evaluation_end, *, start_col: str = "feature_start", end_col: str = "t1"
) -> pl.DataFrame:
    """Remove samples whose full feature/label information overlaps evaluation."""
    missing = {start_col, end_col} - set(samples.columns)
    if missing:
        raise ValueError(f"Sample information intervals are missing columns: {sorted(missing)}")
    if samples.select(pl.col(end_col).is_null().any()).item():
        raise ValueError("Information intervals cannot contain null label ends")
    return samples.filter((pl.col(end_col) < evaluation_start) | (pl.col(start_col) >= evaluation_end))


def fold_training_labels(labels: pl.DataFrame, fold: FoldSpec) -> pl.DataFrame:
    """Select chronological training labels and apply equality-safe purging."""
    candidates = labels.filter(
        (pl.col("decision_time") >= fold.train_start) & (pl.col("decision_time") < fold.train_end)
    )
    return purge_training_labels(candidates, fold.validation_start)
