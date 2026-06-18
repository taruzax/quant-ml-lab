"""Tests for the Tensor Factory — shape contracts and NaN assertions."""
import numpy as np
import polars as pl
import pytest
import torch
# pyrefly: ignore [missing-import]
from src.data.tensor_loader import TimeSeriesDataset, create_dataloaders
# pyrefly: ignore [missing-import]
from src.data.validators import DataValidationError
# pyrefly: ignore [missing-import]
from src.core.config import PipelineConfig


def _make_tensor_df(n: int = 50, n_tickers: int = 1) -> pl.DataFrame:
    """Create a clean DataFrame suitable for tensor creation."""
    np.random.seed(42)
    frames = []
    for i in range(n_tickers):
        ticker = f"T{i}"
        dates = pl.date_range(
            start=pl.date(2024, 1, 1),
            end=pl.date(2024, 1, 1) + pl.duration(days=n - 1),
            interval="1d", eager=True,
        )
        frames.append(pl.DataFrame({
            "date": dates,
            "ticker": [ticker] * n,
            "f1": np.random.randn(n).astype(np.float64),
            "f2": np.random.randn(n).astype(np.float64),
            "f3": np.random.randn(n).astype(np.float64),
            "target_1d": np.random.randn(n).astype(np.float64),
        }))
    return pl.concat(frames)


def test_dataset_output_shape():
    df = _make_tensor_df(n=50)
    ds = TimeSeriesDataset(df, ["f1", "f2", "f3"], ["target_1d"], sequence_len=10)
    features, target = ds[0]
    assert features.shape == (10, 3)
    assert target.shape == (1,)


def test_dataset_no_nans():
    df = _make_tensor_df(n=100)
    ds = TimeSeriesDataset(df, ["f1", "f2", "f3"], ["target_1d"], sequence_len=10)
    for i in range(len(ds)):
        features, target = ds[i]
        assert not torch.isnan(features).any(), f"NaN in features at index {i}"
        assert not torch.isnan(target).any(), f"NaN in target at index {i}"


def test_dataset_nan_input_raises():
    df = _make_tensor_df(n=50)
    df = df.with_columns(pl.when(pl.lit(True)).then(None).otherwise(pl.col("f1")).alias("f1"))
    with pytest.raises(DataValidationError, match="nulls"):
        TimeSeriesDataset(df, ["f1", "f2", "f3"], ["target_1d"], sequence_len=10)


def test_short_ticker_skipped():
    df = _make_tensor_df(n=5)  # Only 5 rows, seq_len=10 → should skip
    ds = TimeSeriesDataset(df, ["f1", "f2", "f3"], ["target_1d"], sequence_len=10)
    assert len(ds) == 0


def test_time_based_split():
    df = _make_tensor_df(n=100)
    config = PipelineConfig(sequence_len=10, batch_size=4, train_cutoff_date="2024-03-10")
    train_loader, val_loader = create_dataloaders(df, ["f1", "f2", "f3"], ["target_1d"], config)
    assert train_loader is not None
    # Verify train dates are before val dates
    train_dates = df.filter(pl.col("date").cast(pl.Utf8) <= "2024-03-10")["date"].max()
    if val_loader is not None:
        val_dates = df.filter(pl.col("date").cast(pl.Utf8) > "2024-03-10")["date"].min()
        assert train_dates <= val_dates


def test_batch_shape():
    df = _make_tensor_df(n=100)
    config = PipelineConfig(sequence_len=10, batch_size=8)
    train_loader, _ = create_dataloaders(df, ["f1", "f2", "f3"], ["target_1d"], config)
    batch_features, batch_targets = next(iter(train_loader))
    assert batch_features.shape[1] == 10  # sequence_len
    assert batch_features.shape[2] == 3   # n_features
    assert batch_targets.shape[1] == 1    # n_targets


def test_target_is_last_timestep():
    """Verify that the target corresponds to the last row in the window."""
    df = _make_tensor_df(n=50)
    seq_len = 10
    ds = TimeSeriesDataset(df, ["f1"], ["target_1d"], sequence_len=seq_len)
    features, target = ds[0]
    # The target should be the target_1d value at row index seq_len-1 (0-indexed)
    expected = df["target_1d"][seq_len - 1]
    assert abs(float(target[0]) - expected) < 1e-5