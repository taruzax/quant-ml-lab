from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import polars as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# pyrefly: ignore [missing-import]
from lab.core.config import PipelineConfig

# pyrefly: ignore [missing-import]
from lab.data.cv import PurgedKFold

# pyrefly: ignore [missing-import]
from lab.data.validators import DataValidationError

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset that produces sliding windows from grouped time-series data"""

    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        sequence_len: int,
        group_col: str = "ticker",
        date_col: str = "timestamp",
    ):
        self.sequence_len = sequence_len
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        check_cols = feature_cols + target_cols
        missing = set(check_cols) - set(df.columns)
        if missing:
            raise DataValidationError(f"Columns not found in DataFrame: {sorted(missing)}")
        null_counts = df.select(pl.col(col).null_count().alias(col) for col in check_cols).row(0, named=True)
        cols_with_nulls = {col: n for col, n in null_counts.items() if n > 0}
        if cols_with_nulls:
            detail = ", ".join(f"'{c}': {n}" for c, n in cols_with_nulls.items())
            raise DataValidationError(f"Columns have nulls (drop before creating dataset): {detail}")

        if date_col not in df.columns and "date" in df.columns:
            date_col = "date"

        self._feature_blocks: list[torch.Tensor] = []
        self._target_blocks: list[torch.Tensor] = []
        self._date_blocks: list[np.ndarray] = []
        self._ticker_blocks: list[np.ndarray] = []
        self._index_map: list[tuple[int, int]] = []

        df = df.sort(group_col, date_col)

        for group_name, group_df in df.group_by(group_col, maintain_order=True):
            n_rows = group_df.height
            ticker_name = group_name[0] if isinstance(group_name, tuple) else group_name
            if n_rows < sequence_len + 1:
                logger.warning(
                    "Ticker '%s' has %d rows, need %d + 1. Skipping.",
                    ticker_name,
                    n_rows,
                    sequence_len,
                )
                continue

            features_np = group_df.select(feature_cols).to_numpy().astype(np.float32)
            targets_np = group_df.select(target_cols).to_numpy().astype(np.float32)
            dates_np = group_df[date_col].cast(pl.String).to_numpy()
            tickers_np = group_df[group_col].cast(pl.String).to_numpy()

            if np.isnan(features_np).any() or np.isnan(targets_np).any():
                raise DataValidationError(f"NaN detected in ticker '{ticker_name}' after numpy conversion.")

            block_idx = len(self._feature_blocks)
            self._feature_blocks.append(torch.from_numpy(features_np))
            self._target_blocks.append(torch.from_numpy(targets_np))
            self._date_blocks.append(dates_np)
            self._ticker_blocks.append(tickers_np)

            n_windows = n_rows - sequence_len
            for i in range(n_windows):
                self._index_map.append((block_idx, i))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx):
        block_idx, start = self._index_map[idx]
        target_idx = start + self.sequence_len - 1
        features = self._feature_blocks[block_idx][start : start + self.sequence_len]
        target = self._target_blocks[block_idx][target_idx]
        timestamp = self._date_blocks[block_idx][target_idx]
        ticker = self._ticker_blocks[block_idx][target_idx]
        return features, target, timestamp, ticker


def create_dataloaders(
    df,
    feature_cols,
    target_cols,
    config: PipelineConfig,
    date_col: str = "timestamp",
    t1_col: str = "t1",
    cv_mode: bool | None = None,
    n_splits: int | None = None,
    embargo_bars: int | None = None,
) -> tuple[DataLoader, DataLoader | None] | list[tuple[DataLoader, DataLoader]]:
    """Create train/val DataLoaders with time-based or Purged K-Fold CV split."""
    if date_col not in df.columns:
        date_col = "date" if "date" in df.columns else "timestamp"

    def _make_loader(dataset: TimeSeriesDataset | ConcatDataset | None) -> DataLoader | None:
        return (
            DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
            if dataset is not None and len(dataset) > 0
            else None
        )

    def _make_ds(subset_df: pl.DataFrame | None) -> TimeSeriesDataset | None:
        if subset_df is None or subset_df.is_empty():
            return None
        return TimeSeriesDataset(subset_df, feature_cols, target_cols, config.sequence_len, date_col=date_col)

    # 1. Standardize null-cleaning across both modes
    clean_cols = [c for c in feature_cols + target_cols + [t1_col, date_col] if c in df.columns]
    df = df.drop_nulls(subset=clean_cols)

    # 2. Standard Single Split Mode (cv_mode=False)
    if not (cv_mode if cv_mode is not None else config.cv_mode):
        if config.train_cutoff_date is not None:
            cutoff = datetime.fromisoformat(config.train_cutoff_date)
            if len(config.train_cutoff_date) == 10:
                cutoff = cutoff.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            all_dates = df[date_col].unique().sort()
            cutoff = all_dates[int(all_dates.len() * 0.8)]

        train_df = df.filter(pl.col(date_col) <= cutoff)
        val_df = df.filter(pl.col(date_col) > cutoff)

        if train_df.is_empty():
            raise DataValidationError("Training set is empty after split.")
        if val_df.is_empty():
            logger.warning("Validation set is empty — all data used for training.")

        train_loader = _make_loader(_make_ds(train_df))
        if train_loader is None:
            raise DataValidationError("Training set has no valid continuous sequences.")
        val_loader = _make_loader(_make_ds(val_df))
        return train_loader, val_loader

    # 3. Purged K-Fold Cross-Validation Mode (cv_mode=True)
    timeline_df = (
        df.group_by(date_col).agg(pl.col(t1_col).max()).sort(date_col)
        if t1_col in df.columns
        else df.select(pl.col(date_col).unique()).with_columns(pl.col(date_col).alias(t1_col)).sort(date_col)
    )

    splitter = PurgedKFold(
        n_splits=n_splits if n_splits is not None else config.n_splits,
        t1=timeline_df[t1_col],
        t0=timeline_df[date_col],
        embargo_bars=embargo_bars if embargo_bars is not None else config.embargo_bars,
    )

    fold_loaders: list[tuple[DataLoader, DataLoader]] = []
    for fold_idx, (train_ts_indices, val_ts_indices) in enumerate(splitter.split(timeline_df)):
        val_timestamps = timeline_df[date_col].gather(val_ts_indices)
        val_start, val_end = val_timestamps.min(), val_timestamps.max()

        # Build contiguous Left and Right blocks (loop eliminates duplicate code)
        train_ts = timeline_df[date_col].gather(train_ts_indices)
        train_datasets: list[TimeSeriesDataset] = []
        for block_ts in (train_ts.filter(train_ts < val_start), train_ts.filter(train_ts > val_end)):
            if len(block_ts) > config.sequence_len:
                block_ds = _make_ds(df.filter(pl.col(date_col).is_in(block_ts)))
                if block_ds is not None and len(block_ds) > 0:
                    train_datasets.append(block_ds)

        if not train_datasets:
            raise DataValidationError(f"Fold {fold_idx} training set has no valid continuous sequences.")

        # O(1) Positional Warmup Window for validation coverage
        warmup_idx = max(0, int(val_ts_indices[0]) - (config.sequence_len - 1))
        val_full_ts = timeline_df[date_col].gather(np.arange(warmup_idx, int(val_ts_indices[-1]) + 1))
        val_ds = _make_ds(df.filter(pl.col(date_col).is_in(val_full_ts)))

        combined_train = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        train_loader = _make_loader(combined_train)
        val_loader = _make_loader(val_ds)
        if train_loader is None:
            raise DataValidationError(f"Fold {fold_idx} training set has no valid continuous sequences.")
        if val_loader is None:
            raise DataValidationError(f"Fold {fold_idx} validation set has no valid continuous sequences.")
        fold_loaders.append((train_loader, val_loader))

    return fold_loaders
