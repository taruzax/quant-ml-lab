from __future__ import annotations

import logging
from datetime import date
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

# pyrefly: ignore [missing-import]
from src.core.config import PipelineConfig
# pyrefly: ignore [missing-import]
from src.data.validators import DataValidationError

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset that produces sliding windows from grouped time-series data"""

    def __init__(
        self, 
        df,
        feature_cols,
        target_cols,
        sequence_len,
        group_col : str = "ticker",
        date_col: str = "date",
    ):
        self.sequence_len = sequence_len
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        check_cols = feature_cols + target_cols
        missing = set(check_cols) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Columns not found in DataFrame: {sorted(missing)}"
            )
        null_counts = df.select(
            pl.col(col).null_count().alias(col) for col in check_cols
        ).row(0, named = True)
        cols_with_nulls = {col: n for col, n in null_counts.items() if n>0}
        if cols_with_nulls:
            detail = ", ".join(f"'{c}': {n}" for c, n in cols_with_nulls.items())
            raise DataValidationError(
                f"Columns have nulls (drop before creating dataset): {detail}"
            )
        
        self._windows: list[tuple[np.ndarray, np.ndarray]] = []
        df = df.sort(group_col,date_col)
        for group_name, group_df in df.group_by(group_col, maintain_order=True):
            n_rows = group_df.height
            ticker_name = group_name[0] if isinstance(group_name, tuple) else group_name
            if n_rows < sequence_len + 1:   
                logger.warning(
                    "Ticker '%s' has %d rows, need %d + 1. Skipping.",
                    ticker_name, n_rows, sequence_len,
                )
                continue

            features_np = group_df.select(feature_cols).to_numpy().astype(np.float32)
            targets_np = group_df.select(target_cols).to_numpy().astype(np.float32)
            
            if np.isnan(features_np).any() or np.isnan(targets_np).any():
                raise DataValidationError(
                    f"NaN detected in ticker '{ticker_name}' after numpy conversion."
                )
            for i in range(n_rows - sequence_len):
                feature_window = features_np[i:i+sequence_len]
                target_vector = targets_np[i+sequence_len-1]
                self._windows.append((feature_window, target_vector))
    
    def __len__(self):
        return len(self._windows)
    
    def __getitem__(self, idx):
        features, targets = self._windows[idx]
        return(
            torch.from_numpy(features),
            torch.from_numpy(targets),
        )


def create_dataloaders(df, feature_cols, target_cols, config:PipelineConfig, date_col: str = "date",):
    """Create train/val DataLoaders with time-based split.
    Returns: (train_loader, val_loader)
    """
    if config.train_cutoff_date is not None:
        cutoff = date.fromisoformat(config.train_cutoff_date)
        train_df = df.filter(pl.col(date_col)<=cutoff)
        val_df = df.filter(pl.col(date_col) > cutoff)
    else:
        all_dates = df[date_col].unique().sort()
        n_dates = all_dates.len()
        cutoff_idx = int(n_dates*0.8)
        cutoff_date = all_dates[cutoff_idx]
        train_df = df.filter(pl.col(date_col)<= cutoff_date)
        val_df = df.filter(pl.col(date_col) > cutoff_date)

    if train_df.is_empty():
        raise DataValidationError("Training set is empty after split.")

    if val_df.is_empty():
        logger.warning("Validation set is empty — all data used for training.")


    train_dataset = TimeSeriesDataset(
        train_df, feature_cols, target_cols, config.sequence_len
    )
    val_dataset = TimeSeriesDataset(
        val_df, feature_cols, target_cols, config.sequence_len
    ) if not val_df.is_empty() else None


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # preserve order in timeseries
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    ) if val_dataset is not None and len(val_dataset) > 0 else None

    return train_loader, val_loader