import time
import tracemalloc
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
from torch.utils.data import DataLoader

from lab.research.dataset import build_sample_set


def make_bench_df(n_tickers: int = 17, n_periods: int = 2500, n_features: int = 60) -> pl.DataFrame:
    """Simulate grouped hourly features without prebuilding overlapping windows."""
    np.random.seed(42)
    frames = []
    for ticker_index in range(n_tickers):
        timestamps = [
            datetime(2015, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index)
            for index in range(n_periods)
        ]
        data = {"timestamp": timestamps, "ticker": [f"TICK_{ticker_index}"] * n_periods}
        for feature_index in range(n_features):
            data[f"f{feature_index}"] = np.random.randn(n_periods).astype(np.float32)
        frames.append(pl.DataFrame(data))
    return pl.concat(frames)


def run_benchmark():
    feature_cols = [f"f{index}" for index in range(60)]
    sequence_len = 60
    ticker_count = 100
    period_count = 2500

    frame_started = time.perf_counter()
    frame = make_bench_df(ticker_count, period_count, len(feature_cols))
    frame_seconds = time.perf_counter() - frame_started

    tracemalloc.start()
    construct_started = time.perf_counter()
    samples = build_sample_set(frame, feature_cols, sequence_len)
    construct_seconds = time.perf_counter() - construct_started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    iteration_started = time.perf_counter()
    for index in range(min(1000, len(samples.X))):
        sample = samples.X[index]
    iteration_seconds = time.perf_counter() - iteration_started

    loader = DataLoader(samples.X, batch_size=32, shuffle=False, drop_last=False)
    loader_started = time.perf_counter()
    batch_count = sum(1 for _ in loader)
    loader_seconds = time.perf_counter() - loader_started

    print("INDEXED WINDOW BENCHMARK SUMMARY")
    print(f"Tickers:       {ticker_count}")
    print(f"Periods:       {period_count}")
    print(f"Features:      {len(feature_cols)}")
    print(f"Sequence len:  {sequence_len}")
    print(f"Window count:  {len(samples.X)}")
    print(f"Window shape:  {sample.shape}")
    print(f"DataFrame:     {frame_seconds:.3f}s")
    print(f"Construction:  {construct_seconds:.3f}s")
    print(f"Peak memory:   {peak_bytes / 1024 / 1024:.1f} MB")
    print(f"Iteration/1k:  {iteration_seconds:.4f}s")
    print(f"DataLoader:    {loader_seconds:.3f}s ({batch_count} batches)")


if __name__ == "__main__":
    run_benchmark()
