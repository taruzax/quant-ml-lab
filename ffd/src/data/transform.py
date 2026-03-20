import polars as pl
import polars_talib as plta
from statsmodels.tsa.stattools import adfuller
import numpy as np


def calculate_dollar_volume(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(dollar_vol = pl.col("close") * pl.col("volume"))
        .with_columns(
            dollar_vol_1m = pl.col("dollar_vol")
            .rolling_mean(window_size=21)
            .over("ticker")
        )
        .with_columns(
            dollar_vol_rank = pl.col("dollar_vol_1m")
            .rank(descending=True)
            .over("date")
        )
    )

def calculate_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("close").ta.ema(5).over("ticker").alias("ema5"),
        pl.col("close").ta.macd(12, 26, 9).over("ticker").struct.field("macd"),
        pl.col("close").ta.macd(12, 26, 9).over("ticker").struct.field("macdsignal"),
        pl.col("open").ta.cdl2crows(
            pl.col("high"), pl.col("low"), pl.col("close")
        ).over("ticker").alias("cdl2crows"),
        pl.col("close").ta.wclprice("high", "low").over("ticker").alias("wclprice"),
    )

def calculate_returns(df: pl.DataFrame):
    lags = [1, 5, 10, 21, 42, 63]
    q = 0.001
    exprs = []
    
    for lag in lags:
        ret_col = f"return_{lag}d"
        #standard formula for percentage return current price/previous price - 1
        raw_return = (pl.col("close")/pl.col("close").shift(lag).over("ticker")) - 1
        clipped_return = raw_return.clip(raw_return.quantile(q),
        raw_return.quantile(1-q))
        normalize_return = clipped_return.add(1).pow(1/lag).sub(1).alias(ret_col)
        exprs.append(normalize_return)

    df = df.with_columns(exprs)

    #creating back in time machine to look at how these returns looked before
    shift_exprs = []
    shift_time = [1,2,3,4,5]
    shift_lag = [1,5,10,21]
    for t in shift_time:
        for lag in shift_lag:
            current_target = f"return_{lag}d"
            shift_exprs.append(
                pl.col(ret_col).shift(t * lag).over('ticker').alias(f"{current_target}_lag{t}")
            )

    for t in [1, 5, 10, 21]:
        shift_exprs.append(
            pl.col(f"return_{t}d").shift(-t).over("ticker").alias(f"target_{t}d")
        )

    return df.with_columns(shift_exprs)


def create_final_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        year = pl.col("date").dt.year(),
        month = pl.col("date").dt.month()
    )
    
    # One-hot encoding
    df = df.to_dummies(["year", "month", "sector"], drop_first=True)
    return df.drop("industry")


def get_weights_ffd(d: float, threshold: float, max_lags: int = 1000) -> list[float]:
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
    
    weights = get_weights_ffd(d, threshold)
    expr = weights[0] * pl.col(col_name)

    for i, weight in enumerate(weights[1:], start=1):
        expr = expr + weight * pl.col(col_name).shift(i).over("ticker")
        
    return df.with_columns(expr.alias(f"{col_name}_frac_diff"))


def find_min_ffd(df_ticker: pl.DataFrame, col_name: str = 'close', max_d: float = 1.0, threshold: float = 0.01, min_d: float = 0.1, significance: float = 0.05):
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

        if len(series_np) < 20:
            continue

        adf_result = adfuller(series_np, maxlag=1, regression='c', autolag=None)
        p_val = adf_result[1]
        if p_val < significance:
            return d

    return max_d

def apply_dynamic_frac_diff(df: pl.DataFrame, col_name: str = 'close', threshold: float = 0.01) -> pl.DataFrame:
    processed_dfs = []
    
    tickers = df.get_column("ticker").unique().to_list()
    
    for ticker in tickers:
        
        df_ticker = df.filter(pl.col("ticker") == ticker)
        
        optimal_d = find_min_ffd(df_ticker, col_name, threshold=threshold)
        
        df_ticker_diffed = frac_diff_polars(df_ticker, col_name, optimal_d, threshold)
        
        processed_dfs.append(df_ticker_diffed)
        
    return pl.concat(processed_dfs)


def apply_features(df: pl.DataFrame, d_value: float = 0.4, threshold: float = 0.001) -> pl.DataFrame:
    """
    Applies all feature engineering transformations sequentially.
    """
    df = calculate_dollar_volume(df)
    df = calculate_technical_indicators(df)
    df = calculate_returns(df)
    df = df.with_columns(pl.col('close').log().alias('log_close'))
    df = apply_dynamic_frac_diff(df, 'log_close', threshold)
    df = create_final_features(df)
    
    print("Feature engineering is complete")
    return df.drop_nulls()
