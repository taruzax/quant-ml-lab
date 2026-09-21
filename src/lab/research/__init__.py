"""Reusable research workflow components."""

__all__ = ["campaign_evidence", "evaluate_holdout", "load_candidate_predictions", "load_run", "prepare_dataset", "run_experiment"]


def __getattr__(name: str):
    if name in __all__:
        from lab.research import experiment

        return getattr(experiment, name)
    raise AttributeError(name)
