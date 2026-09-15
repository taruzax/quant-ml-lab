"""Compatibility exports for the relocated sample module."""

from lab.research.samples import InferenceTimeSeriesDataset, TimeSeriesDataset, collate_batch, create_dataloaders

__all__ = ["InferenceTimeSeriesDataset", "TimeSeriesDataset", "collate_batch", "create_dataloaders"]
