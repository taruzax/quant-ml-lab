"""Hierarchical risk parity allocation over validated raw-return covariance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from pypfopt import HRPOpt

from lab.quant.covariance import covariance_array, estimate_ledoit_wolf


def hrp_custom(cov_matrix: Any, tickers: list[str] | tuple[str, ...]) -> dict[str, float]:
    """Explicitly mark the experimental custom branch as unsupported."""
    raise NotImplementedError("custom HRP is not supported; use the PyPortfolioOpt adapter")


def hrp_pypfort(cov_matrix: Any, tickers: list[str] | tuple[str, ...]) -> dict[str, float]:
    """Compute HRP weights without cleaning, rounding, or covariance denoising."""
    matrix, inferred_tickers = covariance_array(cov_matrix)
    names = tuple(tickers)
    if len(names) != matrix.shape[0] or len(set(names)) != len(names):
        raise ValueError("Ticker labels must be unique and match covariance dimensions")
    if isinstance(cov_matrix, pl.DataFrame) and names != inferred_tickers:
        raise ValueError("Ticker order differs from labeled covariance order")
    covariance = pd.DataFrame(matrix, index=names, columns=names)
    weights = HRPOpt(returns=None, cov_matrix=covariance).optimize()
    if set(weights) != set(names):
        raise ValueError("HRP result does not contain exactly the requested tickers")
    result = {ticker: float(weights[ticker]) for ticker in names}
    if not np.isfinite(list(result.values())).all() or any(value < 0.0 for value in result.values()):
        raise ValueError("HRP weights must be finite and nonnegative")
    if not np.isclose(sum(result.values()), 1.0, atol=1e-8):
        raise ValueError("HRP weights must sum to one")
    return result


def hrp_pipe(
    returns_df: pl.DataFrame,
    custom: bool = False,
    *,
    min_observations: int = 252,
) -> dict[str, float]:
    """Estimate covariance from raw returns and calculate HRP weights."""
    covariance, diagnostics = estimate_ledoit_wolf(
        returns_df,
        min_observations=min_observations,
    )
    if custom:
        return hrp_custom(covariance, diagnostics.tickers)
    return hrp_pypfort(covariance, diagnostics.tickers)
