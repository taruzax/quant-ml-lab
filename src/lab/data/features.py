import numpy as np
import polars as pl
import polars_talib as plta

# pyrefly: ignore [missing-import]
from lab.core.config import PipelineConfig

# pyrefly: ignore [missing-import]
from lab.core.schemas import lagged_col, return_col, target_col


def calculate_dollar_volume(df: pl.DataFrame) -> pl.DataFrame:
    """Migrated from transform.py"""
    return (
        df.with_columns(dollar_vol=pl.col("close") * pl.col("volume"))
        .with_columns(dollar_vol_1m=pl.col("dollar_vol").rolling_mean(window_size=21).over("ticker"))
        .with_columns(dollar_vol_rank=pl.col("dollar_vol_1m").rank(descending=True).over("date"))
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
    clip_quantile: float = 0.001,
):
    """
    For each lag in lags, computes:
      raw_return = close / close.shift(lag) - 1
      clipped = clip to [quantile(q), quantile(1-q)]
      normalized = (clipped + 1)^(1/lag) - 1

    Migrated from transform.py
    """

    if lags is None:
        lags = [1, 5, 10, 21, 42, 63]
    exprs = []
    for lag in lags:
        col_name = return_col(lag)
        raw_return = (pl.col("close") / pl.col("close").shift(lag).over("ticker")) - 1
        clipped_return = raw_return.clip(
            raw_return.quantile(clip_quantile),
            raw_return.quantile(1 - clip_quantile),
        )
        normalized = clipped_return.add(1).pow(1 / lag).sub(1).alias(col_name)
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


def create_time_cycles(df: pl.DataFrame) -> pl.DataFrame:
    """Uses cyclical encoding for time instead of one-hot encode"""
    df = df.with_columns(
        year_scaled=(pl.col("date").dt.year() - 2020),
        month_sin=(2 * np.pi * pl.col("date").dt.month() / 12).sin(),
        month_cos=(2 * np.pi * pl.col("date").dt.month() / 12).cos(),
        weekday_sin=(2 * np.pi * pl.col("date").dt.weekday() / 7).sin(),
        weekday_cos=(2 * np.pi * pl.col("date").dt.weekday() / 7).cos(),
        # hour_sin = (2 * np.pi * pl.col("date").dt.hour() / 24).sin(),
        # hour_cos = (2 * np.pi * pl.col("date").dt.hour() / 24).cos()
    )

    return df


def apply_all_features(df: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Applies all feature transformations in sequence. Does NOT apply FFD — that's separate."""
    df = calculate_dollar_volume(df)
    df = calculate_technical_indicators(df)
    df = calculate_returns(df, config.return_lags, config.clip_quantile)
    df = calculate_lagged_features(df, config.target_horizons, config.lookback_periods)
    df = calculate_forward_targets(df, config.target_horizons)
    df = create_time_cycles(df)
    df = create_sector_dummies(df)
    return df
