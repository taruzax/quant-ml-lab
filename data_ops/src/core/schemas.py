import polars as pl

REQUIRED_OHLCV_COLUMNS: list[str] = ["date", "ticker", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS: list[str] = ["open", "high", "low", "close"]
CATEGORICAL_COLUMNS: list[str] = ["sector", "industry"]

REQUIRED_DTYPES: dict[str, pl.DataType] = {
    "date": pl.Date,
    "ticker": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}
DOLLAR_VOL_COLUMNS: list[str] = ["dollar_vol", "dollar_vol_1m", "dollar_vol_rank"]
TECHNICAL_COLUMNS: list[str] = ["ema5", "macd", "macdsignal", "cdl2crows", "wclprice"]

def return_col(lag: int): 
    return f"return_{lag}d"

def lagged_col(lag: int, lookback: int): 
    return f"return_{lag}d_lag{lookback}"

def target_col(horizon: int): 
    return f"target_{horizon}d"