import numpy as np
import polars as pl
import polars_talib as plta

# pyrefly: ignore [missing-import]
from lab.core.config import PipelineConfig, Timeframe

# pyrefly: ignore [missing-import]
from lab.core.schemas import (
    CAUSAL_BASE_FEATURES,
    FUTURE_DERIVED_COLUMNS,
    lagged_col,
    return_col,
    target_col,
    validate_feature_names,
)


def calculate_dollar_volume(df: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Migrated from transform.py"""
    return (
        df.with_columns(dollar_vol=pl.col("close") * pl.col("volume"))
        .with_columns(dollar_vol_1m=pl.col("dollar_vol").rolling_mean(window_size=config.rolling_month_bars).over("ticker"))
        .with_columns(dollar_vol_rank=pl.col("dollar_vol_1m").rank(descending=True).over("timestamp"))
    )


def calculate_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Migrated from transform.py"""
    return df.with_columns(
        plta.ema(pl.col("close"), timeperiod=5).over("ticker").alias("ema5"),
        plta.macd(pl.col("close"), fastperiod=12, slowperiod=26, signalperiod=9).over("ticker").struct.field("macd"),
        plta.cdl2crows(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")).over("ticker").alias("cdl2crows"),
        plta.wclprice(pl.col("high"), pl.col("low"), pl.col("close")).over("ticker").alias("wclprice"),
    )


def calculate_returns(
    df: pl.DataFrame,
    lags: list[int] | None = None,
    clip_quantile: float | None = None,
):
    """
    For each lag in lags, computes:
      raw_return = close / close.shift(lag) - 1
      normalized = (raw_return + 1)^(1/lag) - 1

    Clipping is deliberately not performed here. Quantile thresholds are a
    fold-fitted preprocessing state, and fitting them on the full frame would
    let future observations alter historical features.

    Migrated from transform.py
    """

    if lags is None:
        lags = [1, 5, 10, 21, 42, 63]
    exprs = []
    for lag in lags:
        col_name = return_col(lag)
        raw_return = (pl.col("close") / pl.col("close").shift(lag).over("ticker")) - 1
        normalized = raw_return.add(1).pow(1 / lag).sub(1).alias(col_name)
        exprs.append(normalized)

    return df.with_columns(exprs)

    # #creating back in time machine to look at how these returns looked before
    # shift_exprs = []
    # shift_time = [1,2,3,4,5]
    # shift_lag = [1,5,10,21]
    # for t in shift_time:
    #     for lag in shift_lag:
    #         current_target = f"return_{lag}d"
    #         shift_exprs.append(
    #             pl.col(current_target).shift(t * lag).over('ticker').alias(f"{current_target}_lag{t}")
    #         )

    # for t in [1, 5, 10, 21]:
    #     shift_exprs.append(
    #         pl.col(f"return_{t}d").shift(-t).over("ticker").alias(f"target_{t}d")
    #     )

    # return df.with_columns(shift_exprs)


def calculate_lagged_features(
    df: pl.DataFrame,
    return_lags: list[int] | None = None,
    lookback_periods: list[int] | None = None,
) -> pl.DataFrame:
    """
    For each (lag, lookback), creates:
      return_{lag}d_lag{lookback} = return_{lag}d shifted by (lookback * lag)

    Migrated from transform.py
    """
    if return_lags is None:
        return_lags = [1, 5, 10, 21]
    if lookback_periods is None:
        lookback_periods = [1, 2, 3, 4, 5]

    shift_exprs = []
    for t in lookback_periods:
        for lag in return_lags:
            source_col = return_col(lag)
            alias_name = lagged_col(lag, t)
            shift_exprs.append(pl.col(source_col).shift(t * lag).over("ticker").alias(alias_name))

    return df.with_columns(shift_exprs)


def calculate_forward_targets(
    df: pl.DataFrame,
    horizons: list[int] | None = None,
) -> pl.DataFrame:
    """
    Create forward-looking return targets:
     target_{h}d = return_{h}d shifted by -h (i.e., the return h periods ahead).

    Migrated from transform.py
    """
    if horizons is None:
        horizons = [1, 5, 10, 21]

    shift_exprs = []
    for h in horizons:
        shift_exprs.append(pl.col(return_col(h)).shift(-h).over("ticker").alias(target_col(h)))

    return df.with_columns(shift_exprs)


def create_sector_dummies(df: pl.DataFrame) -> pl.DataFrame:
    """One-hot encode sector columns, then drop industry.

    Migrated from transform.py lines
    """
    if "sector" not in df.columns:
        raise ValueError(f"Column 'sector' not found. Cannot create sector dummies. Available columns: {df.columns}")

    dummy_cols = []
    for col in ["sector"]:
        if col in df.columns:
            dummy_cols.append(col)

    df = df.to_dummies(dummy_cols, drop_first=True)

    if "industry" in df.columns:
        df = df.drop("industry")

    return df


# def create_time_features(df: pl.DataFrame) -> pl.DataFrame:
#     """Extract year and month from the date column.

#     Migrated from transform.py
#     """
#     return df.with_columns(
#         year=pl.col("date").dt.year(),
#         month=pl.col("date").dt.month(),
#     )


def create_time_cycles(df: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Uses cyclical encoding for time instead of one-hot encode"""
    df = df.with_columns(
        year_scaled=(pl.col("timestamp").dt.year() - 2020),
        month_sin=(2 * np.pi * pl.col("timestamp").dt.month() / 12).sin(),
        month_cos=(2 * np.pi * pl.col("timestamp").dt.month() / 12).cos(),
        weekday_sin=(2 * np.pi * pl.col("timestamp").dt.weekday() / 7).sin(),
        weekday_cos=(2 * np.pi * pl.col("timestamp").dt.weekday() / 7).cos(),
    )

    if config.timeframe == Timeframe.H1:
        df = df.with_columns(
            hour_sin=(2 * np.pi * pl.col("timestamp").dt.hour() / 24).sin(),
            hour_cos=(2 * np.pi * pl.col("timestamp").dt.hour() / 24).cos(),
        )

    return df


def apply_all_features(df: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Build only unfitted, causal feature candidates.

    Forward labels, quantile clipping, categorical vocabularies, and FFD
    selection are fold operations and intentionally do not happen here.
    """
    if "ticker" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("Feature construction requires ticker and timestamp columns")
    future_columns = [
        column
        for column in df.columns
        if column in FUTURE_DERIVED_COLUMNS or column.startswith("target_")
    ]
    if future_columns:
        df = df.drop(future_columns)
    df = df.sort(["ticker", "timestamp"])
    df = calculate_dollar_volume(df, config)
    df = calculate_technical_indicators(df)
    df = calculate_returns(df, config.return_lags)
    df = calculate_lagged_features(df, config.return_lags, config.lookback_periods)
    df = create_time_cycles(df, config)
    return df


def feature_specification(df: pl.DataFrame, config: PipelineConfig) -> tuple[str, ...]:
    """Return the explicit causal feature order for a frame.

    The default is a named allow-list, never ``all numeric columns``. A
    configured list is validated against the same future-derived deny-list.
    """
    requested = config.features.feature_columns
    if requested is not None:
        names = validate_feature_names(requested, set(df.columns))
    else:
        candidates = list(CAUSAL_BASE_FEATURES)
        candidates.extend(return_col(lag) for lag in config.return_lags)
        candidates.extend(
            lagged_col(lag, lookback)
            for lookback in config.lookback_periods
            for lag in config.return_lags
        )
        names = tuple(name for name in candidates if name in df.columns)
        names = validate_feature_names(names, set(df.columns))
    forbidden = [name for name in names if name in {"timestamp", "ticker", "sector", "industry"}]
    if forbidden:
        raise ValueError(f"Identifiers and raw categorical columns cannot be model features: {forbidden}")
    return names


def select_features(df: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Select the configured causal feature columns without fitting state."""
    return df.select(["ticker", "timestamp", *feature_specification(df, config)])
