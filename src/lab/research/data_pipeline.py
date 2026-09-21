"""Compatibility and public research preparation API."""

from lab.research.experiment import (
    evaluate_holdout,
    evaluate_predictions,
    evaluate_saved_predictions,
    load_run,
    persist_run,
    predict_fold,
    prepare_dataset,
    prepare_fold,
    run_experiment,
    train_fold,
)
from lab.research.portfolio import build_allocations

__all__ = [
    "build_allocations",
    "evaluate_holdout",
    "evaluate_predictions",
    "evaluate_saved_predictions",
    "load_run",
    "persist_run",
    "predict_fold",
    "prepare_dataset",
    "prepare_fold",
    "run_experiment",
    "train_fold",
]
