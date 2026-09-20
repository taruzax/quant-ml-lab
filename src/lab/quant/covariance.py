"""Validated covariance estimation for portfolio allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class CovarianceDiagnostics:
    tickers: tuple[str, ...]
    observations: int
    assets: int
    shrinkage: float
    estimator: str = "ledoit_wolf"
    source: str = "raw_synchronized_returns"


def _as_return_matrix(returns: Any, tickers: tuple[str, ...] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(returns, pl.DataFrame):
        columns = [column for column in returns.columns if column not in {"timestamp", "date"}]
        if tickers is not None:
            columns = list(tickers)
        matrix = returns.select(columns).to_numpy()
        names = tuple(columns)
    elif hasattr(returns, "columns") and hasattr(returns, "to_numpy"):
        columns = [str(column) for column in returns.columns if str(column) not in {"timestamp", "date"}]
        if tickers is not None:
            columns = list(tickers)
        matrix = returns.loc[:, columns].to_numpy()
        names = tuple(columns)
    else:
        matrix = np.asarray(returns, dtype=float)
        width = matrix.shape[1] if matrix.ndim == 2 else 0
        names = tuple(tickers or (f"asset_{index}" for index in range(width)))
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Returns must be a non-empty two-dimensional matrix")
    if len(names) != matrix.shape[1]:
        raise ValueError("Ticker labels must match the return matrix columns")
    if not np.isfinite(matrix).all():
        raise ValueError("Returns contain non-finite values")
    return matrix, names


def estimate_ledoit_wolf(
    returns: Any,
    *,
    tickers: tuple[str, ...] | None = None,
    min_observations: int = 252,
) -> tuple[pl.DataFrame, CovarianceDiagnostics]:
    """Estimate validated Ledoit-Wolf covariance from synchronized returns."""
    matrix, names = _as_return_matrix(returns, tickers)
    if matrix.shape[0] < min_observations:
        raise ValueError(
            f"At least {min_observations} synchronized return observations are required; "
            f"received {matrix.shape[0]}"
        )
    estimator = LedoitWolf().fit(matrix)
    covariance_array_value = np.asarray(estimator.covariance_, dtype=float)
    covariance_array_value = (covariance_array_value + covariance_array_value.T) / 2.0
    if (np.diag(covariance_array_value) <= 0.0).any() or np.linalg.eigvalsh(covariance_array_value).min() < -1e-10:
        raise ValueError("Ledoit-Wolf covariance is not numerically positive semidefinite")
    covariance = pl.DataFrame(covariance_array_value, schema=list(names))
    covariance = covariance.with_columns(pl.Series("ticker", names)).select(["ticker", *names])
    diagnostics = CovarianceDiagnostics(
        tickers=names,
        observations=matrix.shape[0],
        assets=matrix.shape[1],
        shrinkage=float(estimator.shrinkage_),
    )
    return covariance, diagnostics


def covariance_array(covariance: pl.DataFrame | np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract a symmetric numeric covariance matrix and its labels."""
    if isinstance(covariance, pl.DataFrame):
        if "ticker" in covariance.columns:
            names = tuple(covariance["ticker"].cast(pl.String).to_list())
            matrix = covariance.select(names).to_numpy()
        else:
            names = tuple(covariance.columns)
            matrix = covariance.to_numpy()
    else:
        matrix = np.asarray(covariance, dtype=float)
        names = tuple(f"asset_{index}" for index in range(matrix.shape[0]))
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("Covariance must be a finite square matrix")
    return (matrix + matrix.T) / 2.0, names


def led_wo_shrinkage(returns: Any) -> tuple[np.ndarray, float]:
    """Backward-compatible array API for the relocated helper."""
    matrix, _ = _as_return_matrix(returns)
    estimator = LedoitWolf().fit(matrix)
    return estimator.covariance_, float(estimator.shrinkage_)


def denoise_cov(cov: np.ndarray, n_observarions: int, method: str | None = None) -> np.ndarray:
    """Legacy denoising utility; not used by the supported HRP route."""
    if n_observarions <= 0:
        raise ValueError("n_observarions must be positive")
    matrix = np.asarray(cov, dtype=float)
    q = matrix.shape[0] / n_observarions
    eigvals, eigvecs = np.linalg.eigh((matrix + matrix.T) / 2.0)
    idx = np.flip(np.argsort(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    lambda_plus = np.median(eigvals) * (1 + np.sqrt(q)) ** 2
    noise_mask = eigvals < lambda_plus
    if noise_mask.any():
        eigvals[noise_mask] = eigvals[noise_mask].mean()
    return (eigvecs @ np.diag(np.maximum(eigvals, 0.0)) @ eigvecs.T).real


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    matrix = np.asarray(cov, dtype=float)
    std = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    std = np.where(std == 0, 1e-10, std)
    corr = np.clip(matrix / np.outer(std, std), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr
