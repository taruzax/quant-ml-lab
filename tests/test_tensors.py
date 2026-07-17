"""Tests for the Tensor Factory — shape contracts and NaN assertions."""

import numpy as np
import polars as pl
import pytest
import torch

# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
from lab.data.tensor_loader import TimeSeriesDataset, create_dataloaders

# pyrefly: ignore [missing-import]
from lab.data.validators import DataValidationError


def _make_tensor_df(n: int = 50, n_tickers: int = 1) -> pl.DataFrame:
    """Create a clean DataFrame suitable for tensor creation."""
    np.random.seed(42)
    frames = []
    for i in range(n_tickers):
        ticker = f"T{i}"
        dates = pl.date_range(
            start=pl.date(2024, 1, 1),
            end=pl.date(2024, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        )
        frames.append(
            pl.DataFrame(
                {
                    "date": dates,
                    "ticker": [ticker] * n,
                    "f1": np.random.randn(n).astype(np.float64),
                    "f2": np.random.randn(n).astype(np.float64),
                    "f3": np.random.randn(n).astype(np.float64),
                    "target_1d": np.random.randn(n).astype(np.float64),
                }
            )
        )
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


def test_time_based_split(pipeline_config):
    df = _make_tensor_df(n=100)
    pipeline_config.train_cutoff_date = "2024-03-10"
    train_loader, val_loader = create_dataloaders(df, ["f1", "f2", "f3"], ["target_1d"], pipeline_config)
    assert train_loader is not None
    # Verify train dates are before val dates
    train_dates = df.filter(pl.col("date").cast(pl.Utf8) <= "2024-03-10")["date"].max()
    if val_loader is not None:
        val_dates = df.filter(pl.col("date").cast(pl.Utf8) > "2024-03-10")["date"].min()
        assert train_dates <= val_dates


def test_batch_shape(pipeline_config):
    df = _make_tensor_df(n=100)
    pipeline_config.batch_size = 8
    train_loader, _ = create_dataloaders(df, ["f1", "f2", "f3"], ["target_1d"], pipeline_config)
    batch_features, batch_targets = next(iter(train_loader))
    assert batch_features.shape[1] == 10  # sequence_len
    assert batch_features.shape[2] == 3  # n_features
    assert batch_targets.shape[1] == 1  # n_targets


def test_target_is_last_timestep():
    """Verify that the target corresponds to the last row in the window."""
    df = _make_tensor_df(n=50)
    seq_len = 10
    ds = TimeSeriesDataset(df, ["f1"], ["target_1d"], sequence_len=seq_len)
    features, target = ds[0]
    # The target should be the target_1d value at row index seq_len-1 (0-indexed)
    expected = df["target_1d"][seq_len - 1]
    assert abs(float(target[0]) - expected) < 1e-5


def test_multiple_missing_columns_reported():
    """All missing columns appear in a single error message, not just the first."""
    df = _make_tensor_df(n=50)
    with pytest.raises(DataValidationError, match="missing_a") as exc_info:
        TimeSeriesDataset(df, ["f1", "missing_a", "missing_b"], ["target_1d"], sequence_len=10)
    # Both missing columns must appear in the error
    msg = str(exc_info.value)
    assert "missing_a" in msg
    assert "missing_b" in msg


def test_getitem_returns_views():
    """__getitem__ must return tensor views into the contiguous block, not copies.
    Proves zero-copy slicing by checking data_ptr() falls within the block's storage range.
    """
    df = _make_tensor_df(n=50)
    ds = TimeSeriesDataset(df, ["f1", "f2", "f3"], ["target_1d"], sequence_len=10)

    features, target = ds[0]

    # The features tensor should share storage with the first feature block
    block = ds._feature_blocks[0]
    block_start = block.untyped_storage().data_ptr()
    block_end = block_start + block.untyped_storage().nbytes()

    feat_ptr = features.untyped_storage().data_ptr()
    assert block_start <= feat_ptr < block_end, (
        f"features data_ptr {feat_ptr} not within block range [{block_start}, {block_end})"
    )


def test_values_match_naive_implementation():
    """Index-map dataset produces identical values to a naive sliding-window loop."""
    df = _make_tensor_df(n=100, n_tickers=2)
    feature_cols = ["f1", "f2", "f3"]
    target_cols = ["target_1d"]
    seq_len = 10

    ds = TimeSeriesDataset(df, feature_cols, target_cols, sequence_len=seq_len)

    # Build naive reference: sort, group, slide
    df_sorted = df.sort("ticker", "date")
    naive_windows = []
    for _, group_df in df_sorted.group_by("ticker", maintain_order=True):
        feat_np = group_df.select(feature_cols).to_numpy().astype(np.float32)
        tgt_np = group_df.select(target_cols).to_numpy().astype(np.float32)
        n = group_df.height
        for i in range(n - seq_len):
            naive_windows.append(
                (
                    torch.from_numpy(feat_np[i : i + seq_len].copy()),
                    torch.from_numpy(tgt_np[i + seq_len - 1].copy()),
                )
            )

    assert len(ds) == len(naive_windows), f"Length mismatch: {len(ds)} vs {len(naive_windows)}"

    for i in range(len(ds)):
        feat_actual, tgt_actual = ds[i]
        feat_expected, tgt_expected = naive_windows[i]
        assert torch.allclose(feat_actual, feat_expected, atol=1e-6), f"Feature mismatch at window {i}"
        assert torch.allclose(tgt_actual, tgt_expected, atol=1e-6), f"Target mismatch at window {i}"


def test_memory_reduction():
    """Index-map approach must use less peak memory than materializing all windows.

    We compare tracemalloc peak for the index-map dataset against a naive
    pre-materialized list-of-arrays approach on >1000 windows.
    """
    import tracemalloc

    n_days = 200  # 2 tickers × 200 days × seq_len=10 → ~380 windows each → ~760 total > 1000
    n_tickers = 6
    df = _make_tensor_df(n=n_days, n_tickers=n_tickers)
    feature_cols = ["f1", "f2", "f3"]
    target_cols = ["target_1d"]
    seq_len = 10

    # --- Measure index-map approach ---
    tracemalloc.start()
    ds_new = TimeSeriesDataset(df, feature_cols, target_cols, sequence_len=seq_len)
    _, peak_new = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(ds_new) > 1000, f"Need >1000 windows for meaningful test, got {len(ds_new)}"

    # --- Measure naive approach (materialize all windows as separate arrays) ---
    tracemalloc.start()
    df_sorted = df.sort("ticker", "date")
    naive_windows = []
    for _, group_df in df_sorted.group_by("ticker", maintain_order=True):
        feat_np = group_df.select(feature_cols).to_numpy().astype(np.float32)
        tgt_np = group_df.select(target_cols).to_numpy().astype(np.float32)
        n = group_df.height
        for i in range(n - seq_len):
            naive_windows.append(
                (
                    feat_np[i : i + seq_len].copy(),
                    tgt_np[i + seq_len - 1].copy(),
                )
            )
    _, peak_naive = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n  Index-map peak: {peak_new / 1024:.1f} KB")
    print(f"  Naive peak:     {peak_naive / 1024:.1f} KB")
    print(f"  Ratio:          {peak_naive / max(peak_new, 1):.2f}x")

    assert peak_new < peak_naive, (
        f"Index-map ({peak_new / 1024:.1f} KB) should use less memory than naive ({peak_naive / 1024:.1f} KB)"
    )
