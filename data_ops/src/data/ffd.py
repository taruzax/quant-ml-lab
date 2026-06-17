from IPython.core import logger
from scipy.optimize import minimize_scalar
from torchgen.api.cpp import return_type
import polars as pl
import polars_talib as plta
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


def find_global_d(df: pl.DataFrame, col_name: str = 'log_close', reference_ticker: str = None, coverage_threshold: float = 0.8, max_d: float = 1.0, threshold: float = 0.01, min_d: float = 0.1, significance: float = 0.05) -> float:
    if reference_ticker is not None:
        df_ticker = df.filter(pl.col("ticker") == reference_ticker)
        return find_min_d(df_ticker, col_name=col_name, max_d=max_d, min_d=min_d, threshold=threshold, significance=significance)
        
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
        
    return optimal_ds[idx]


# def calculate_dollar_volume(df: pl.DataFrame) -> pl.DataFrame:
    
#     return (
#         df.with_columns(dollar_vol = pl.col("close") * pl.col("volume"))
#         .with_columns(
#             dollar_vol_1m = pl.col("dollar_vol")
#             .rolling_mean(window_size=21)
#             .over("ticker")
#         )
#         .with_columns(
#             dollar_vol_rank = pl.col("dollar_vol_1m")
#             .rank(descending=True)
#             .over("date")
#         )
#     )

# def calculate_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
#     return df.with_columns(
#         pl.col("close").ta.ema(5).over("ticker").alias("ema5"),
#         pl.col("close").ta.macd(12, 26, 9).over("ticker").struct.field("macd"),
#         pl.col("close").ta.macd(12, 26, 9).over("ticker").struct.field("macdsignal"),
#         pl.col("open").ta.cdl2crows(
#             pl.col("high"), pl.col("low"), pl.col("close")
#         ).over("ticker").alias("cdl2crows"),
#         pl.col("close").ta.wclprice("high", "low").over("ticker").alias("wclprice"),
#     )

# def calculate_returns(df: pl.DataFrame):
#     lags = [1, 5, 10, 21, 42, 63]
#     q = 0.001
#     exprs = []
    
#     for lag in lags:
#         ret_col = f"return_{lag}d"
#         #standard formula for percentage return current price/previous price - 1
#         raw_return = (pl.col("close")/pl.col("close").shift(lag).over("ticker")) - 1
#         clipped_return = raw_return.clip(raw_return.quantile(q),
#         raw_return.quantile(1-q))
#         normalize_return = clipped_return.add(1).pow(1/lag).sub(1).alias(ret_col)
#         exprs.append(normalize_return)

#     df = df.with_columns(exprs)

#     #creating back in time machine to look at how these returns looked before
#     shift_exprs = []
#     shift_time = [1,2,3,4,5]
#     shift_lag = [1,5,10,21]
#     for t in shift_time:
#         for lag in shift_lag:
#             current_target = f"return_{lag}d"
#             shift_exprs.append(
#                 pl.col(ret_col).shift(t * lag).over('ticker').alias(f"{current_target}_lag{t}")
#             )

#     for t in [1, 5, 10, 21]:
#         shift_exprs.append(
#             pl.col(f"return_{t}d").shift(-t).over("ticker").alias(f"target_{t}d")
#         )

#     return df.with_columns(shift_exprs)


# def create_final_features(df: pl.DataFrame) -> pl.DataFrame:
#     df = df.with_columns(
#         year = pl.col("date").dt.year(),
#         month = pl.col("date").dt.month()
#     )
    
#     # One-hot encoding
#     df = df.to_dummies(["year", "month", "sector"], drop_first=True)
#     return df.drop("industry")






# def apply_features(df: pl.DataFrame, global_d: float = None, training_cutoff_date: str = None, threshold: float = 0.001) -> pl.DataFrame:
#     """
#     Applies all feature engineering transformations sequentially.
#     """
#     df = calculate_dollar_volume(df)
#     df = calculate_technical_indicators(df)
#     df = calculate_returns(df)
#     df = df.with_columns(pl.col('close').log().alias('log_close'))
    
#     if global_d is None:
#         if training_cutoff_date is None:
#             warnings.warn("No training_cutoff_date provided! FFD optimization will use the entire dataset, introducing look-ahead bias.")
#             df_train = df
#         else:
#             df_train = df.filter(pl.col("date").cast(pl.Utf8) <= str(training_cutoff_date))
            
#         global_d = find_global_d(df_train, col_name='log_close', threshold=threshold)
#         print(f"Optimal global d computed: {global_d}")
        
#     df = frac_diff_polars(df, 'log_close', global_d, threshold=threshold)
#     df = create_final_features(df)
    
#     print("Feature engineering is complete")
#     return df.drop_nulls()
