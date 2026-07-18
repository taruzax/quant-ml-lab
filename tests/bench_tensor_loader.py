import os
import sys
import time
import tracemalloc
from datetime import datetime

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# pyrefly: ignore [missing-import]
from lab.data.tensor_loader import TimeSeriesDataset

# pyrefly: ignore [missing-import]
# from archive.archived_tensor_loader import TimeSeriesDataset


def make_bench_df(n_tickers: int = 17, n_periods: int = 2500, n_features: int = 60) -> pl.DataFrame:
    """Simulate production-scale data: 17 tickers × 2500 days × 60 features."""
    np.random.seed(42)
    frames = []
    for i in range(n_tickers):
        dates = pl.datetime_range(
            start=datetime(2015, 1, 1),
            end=datetime(2015, 1, 1) + pl.duration(hours=n_periods - 1),
            interval="1h",
            time_unit="ns",
            eager=True,
        )
        data = {"timestamp": dates, "ticker": [f"TICK_{i}"] * n_periods}
        for f in range(n_features):
            data[f"f{f}"] = np.random.randn(n_periods).astype(np.float64)
        data["target_1b"] = np.random.randn(n_periods).astype(np.float64)
        frames.append(pl.DataFrame(data))
    return pl.concat(frames)


def run_benchmark():
    print("=" * 60)
    print("TENSOR LOADER BENCHMARK SUMMARY")
    print("=" * 60)

    feature_cols = [f"f{i}" for i in range(60)]
    target_cols = ["target_1b"]
    seq_len = 60
    n_tickers = 100
    n_periods = 2500
    n_features = 60

    t0 = time.perf_counter()
    df = make_bench_df(n_tickers, n_periods, n_features)

    tracemalloc.start()
    t0 = time.perf_counter()

    ds = TimeSeriesDataset(df, feature_cols, target_cols, sequence_len=seq_len)

    construct_time = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # print(f"    Construction time: {construct_time:.3f}s")
    # print(f"    Dataset length:    {len(ds)} windows")
    # print(f"    Peak memory:       {peak_mem / 1024 / 1024:.1f} MB")
    # print(f"    Current memory:    {current_mem / 1024 / 1024:.1f} MB")

    t0 = time.perf_counter()
    for i in range(min(1000, len(ds))):
        feat, tgt = ds[i]
    iter_time = time.perf_counter() - t0

    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    t0 = time.perf_counter()
    n_batches = 0
    for batch_feat, batch_tgt in loader:
        n_batches += 1
    loader_time = time.perf_counter() - t0

    # --- Shape sanity ---
    feat, tgt = ds[0]
    print(f"Tickers:       {n_tickers}")
    print(f"Periods:       {n_periods}")
    print(f"Features:      {n_features}")
    print(f"Seq Length:    {seq_len}")
    print(f"Shape check: features={feat.shape}, target={tgt.shape}")
    print(f"DataFrame built in {time.perf_counter() - t0:.3f}s")
    print(f"Construction:  {construct_time:.3f}s")
    print(f"Peak memory:   {peak_mem / 1024 / 1024:.1f} MB")
    print(f"Iteration/1k:  {iter_time:.4f}s")
    print(f"DataLoader:    {loader_time:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()


"""
============================================================
TENSOR LOADER BENCHMARK SUMMARY BEFORE OPTIMIZATION
============================================================
Tickers:       100
Days:          2500
Features:      60
Seq Length:    60
Shape check: features=torch.Size([60, 60]), target=torch.Size([1])
DataFrame built in 2.109s
Construction:  0.424s
Peak memory:   129.1 MB
Iteration/1k:  0.0009s
DataLoader:    2.109s
============================================================
============================================================
TENSOR LOADER BENCHMARK SUMMARY AFTER OPTIMIZATION
============================================================
Tickers:       100
Days:          2500
Features:      60
Seq Length:    60
Shape check: features=torch.Size([60, 60]), target=torch.Size([1])
DataFrame built in 0.677s
Construction:  0.141s
Peak memory:   79.1 MB
Iteration/1k:  0.0011s
DataLoader:    0.677s
============================================================

"""
