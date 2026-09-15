import polars as pl

REQUIRED_OHLCV_COLUMNS: list[str] = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
CANONICAL_BAR_COLUMNS: list[str] = [
    "ticker",
    "raw_bar_index",
    "bar_open_time",
    "bar_close_time",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "session_id",
]
PRICE_COLUMNS: list[str] = ["open", "high", "low", "close"]
CATEGORICAL_COLUMNS: list[str] = ["sector", "industry"]
FUTURE_DERIVED_COLUMNS: set[str] = {
    "target",
    "target_1b",
    "t1",
    "entry_time",
    "event_end",
    "barrier_hit",
    "upper_barrier",
    "lower_barrier",
}

REQUIRED_DTYPES: dict[str, type[pl.DataType]] = {
    "timestamp": pl.Datetime,
    "ticker": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

DOLLAR_VOL_COLUMNS: list[str] = ["dollar_vol", "dollar_vol_1m", "dollar_vol_rank"]
TECHNICAL_COLUMNS: list[str] = ["ema5", "macd", "macdsignal", "cdl2crows", "wclprice"]


def return_col(lag: int) -> str:
    return f"return_{lag}b"


def lagged_col(lag: int, lookback: int) -> str:
    return f"return_{lag}b_lag{lookback}"


def target_col(horizon: int) -> str:
    return f"target_{horizon}b"


def validate_feature_names(feature_names: list[str] | tuple[str, ...], available_columns: set[str] | None = None) -> tuple[str, ...]:
    names = tuple(feature_names)
    if not names:
        raise ValueError("At least one feature must be selected explicitly")
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique")
    forbidden = [name for name in names if name in FUTURE_DERIVED_COLUMNS or name.startswith("target_")]
    if forbidden:
        raise ValueError(f"Future-derived columns cannot be model features: {forbidden}")
    if available_columns is not None:
        unknown = sorted(set(names) - available_columns)
        if unknown:
            raise ValueError(f"Unknown feature columns: {unknown}")
    return names
