
from scipy.optimize import minimize_scalar
import polars as pl
from statsmodels.tsa.stattools import adfuller
import numpy as np
import warnings
from typing import NamedTuple
import logging
logger = logging.getLogger(__name__)

class AdfResult(NamedTuple):
    statistic: float
    p_value: float
    is_stationary: bool

def run_adf_test(series: np.ndarray, significance: float = 0.05) -> AdfResult:
    """Wrapper around statsmodels adfuller with a clean return type.
    Returns AdfResult with test statistic, p-value, and boolean.
    """
    stat, p_val, *_ = adfuller(series, maxlag=1, regression="c", autolag="AIC")
    return AdfResult(statistic=stat, p_value=p_val, is_stationary=p_val < significance)

def get_weights_ffd(d: float, threshold: float, max_lags: int = 1000) -> list[float]:
    """Migrated from transform.py"""
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold or k >= max_lags:
            break
        w.append(w_k)
        k += 1
    return w


def frac_diff_polars(df: pl.DataFrame, col_name: str, d: float, threshold: float = 0.001) -> pl.DataFrame:
    """Migrated from transform.py"""
    weights = get_weights_ffd(d, threshold)
    expr = weights[0] * pl.col(col_name)

    for i, weight in enumerate(weights[1:], start=1):
        expr = expr + weight * pl.col(col_name).shift(i).over("ticker")
        
    return df.with_columns(expr.alias(f"{col_name}_frac_diff"))


def find_min_d_grid(df_ticker: pl.DataFrame, col_name: str = 'close', max_d: float = 1.0, threshold: float = 0.01, min_d: float = 0.1, significance: float = 0.05):
    """
    Find the minimum d that makes the series stationary.

    min_d: floor for d — skips d=0 (raw series) which can pass ADF by luck
           on finite windows without being truly stationary. Default 0.1.
    significance: ADF p-value threshold.
    """
    d_values = [d for d in np.linspace(0, max_d, 11) if d >= min_d]

    for d in d_values:
        df_diff = frac_diff_polars(df_ticker, col_name, d, threshold)
        diff_col_name = f"{col_name}_frac_diff"
        series_np = df_diff.get_column(diff_col_name).drop_nulls().to_numpy()

        if len(series_np) < 50:
            continue

        result = run_adf_test(series_np, significance)
        if result.is_stationary:
            return d

    return max_d

def compute_memory_corr(original, differenced):
    """Pearson correlation between original and FFD series"""
    mask = np.isfinite(original) & np.isfinite(differenced)
    if mask.sum()<2:
        return 0.0
    corr_matrix = np.corrcoef(original[mask], differenced[mask])
    return float(corr_matrix[0, 1])

def find_min_d(df_ticker, col_name: str = 'close', max_d: float = 1.0, threshold: float = 0.01, min_d: float = 0.1, significance: float = 0.05):
    """Find the minimum d that makes the series stationary.

    Primary method: scipy.optimize.minimize_scalar with bounded method.
    Fallback: 11-point grid search (original algorithm).
    """
    def adf_p_value(d: float):
        df_dff = frac_diff_polars(df_ticker, col_name, d, threshold)
        series_np = df_dff.get_column(f"{col_name}_frac_diff").drop_nulls().to_numpy()
        if len(series_np)<50:
            return 1
        result = run_adf_test(series_np, significance)
        return result.p_value
    try:
        opt_result = minimize_scalar(
            adf_p_value, 
            bounds = (min_d, max_d),
            method = "bounded",
            options = {'xatol':0.01, 'maxiter':50}
        )
        if opt_result.success and opt_result.fun < significance:
            d_candidate = opt_result.x
            for d_test in np.linspace(min_d, d_candidate, 10):
                if adf_p_value(d_test)<significance:
                    return float(round(d_test, 3))
            return float(round(d_candidate, 3))
    except Exception as e:
        logger.warning("scipy optimizer failed (%s), falling back to grid search.", e)
    return find_min_d_grid(df_ticker, col_name, max_d, threshold, min_d, significance)


def find_global_d(df: pl.DataFrame, col_name: str = 'log_close', reference_ticker: str | None = None, coverage_threshold: float = 0.8, max_d: float = 1.0, threshold: float = 0.01, min_d: float = 0.1, significance: float = 0.05) -> float:
    if reference_ticker is not None:
        df_ticker = df.filter(pl.col("ticker") == reference_ticker)
        return float(find_min_d(df_ticker, col_name=col_name, max_d=max_d, min_d=min_d, threshold=threshold, significance=significance))
        
    tickers = df.get_column("ticker").unique().to_list()
    optimal_ds = []
    
    for ticker in tickers:
        df_ticker = df.filter(pl.col("ticker") == ticker)
        d = find_min_d(df_ticker, col_name=col_name, max_d=max_d, min_d=min_d, threshold=threshold, significance=significance)
        optimal_ds.append(d)
        
    if not optimal_ds:
        return max_d
        
    optimal_ds.sort()
    idx = int(len(optimal_ds) * coverage_threshold)
    if idx >= len(optimal_ds):
        idx = len(optimal_ds) - 1
        
    return float(optimal_ds[idx])

