import logging

import polars as pl

from lab.core.config import PipelineConfig

logger = logging.getLogger(__name__)


def calculate_volatility(
    df,
    span_bars: int = 100,
    vol_lookback_bars: int = 20,
    min_volatility: float = 1e-4,
    price_col: str = "close",
    group_col: str = "ticker",
):
    """Calculates exponentially weighted volatility"""
    returns_expr = (pl.col(price_col) / pl.col(price_col).shift(1).over(group_col)) - 1
    vol_expr = returns_expr.ewm_std(span=span_bars, ignore_nulls=True).over(group_col).fill_null(min_volatility)
    vol_floored = pl.when(vol_expr < min_volatility).then(pl.lit(min_volatility)).otherwise(vol_expr)
    vol_predictive = vol_floored.shift(1).over(group_col)
    return df.with_columns(vol_predictive.alias("volatility_t_minus_1"))


def triple_barrier_label(
    df, config: PipelineConfig, price_col: str = "close", date_col: str = "timestamp", group_col: str = "ticker"
):
    """Compute path-dependent Triple-Barrier labels and observation end times (t1)"""

    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Int8).alias("label"),
            pl.lit(None, dtype=pl.Datetime).alias("t1"),
            pl.lit(None, dtype=pl.Datetime).alias("t0"),
        )

    required_min_rows = config.vol_lookback_bars + config.expiry_bars
    valid_dfs = []

    for group_name, group_df in df.group_by(group_col, maintain_order=True):
        ticker_name = group_name[0] if isinstance(group_name, tuple) else group_name
        if group_df.height < required_min_rows:
            logger.warning(
                "Ticker '%s' has %d rows, minimum required is %d (vol_lookback %d + expiry %d). Skipping.",
                ticker_name,
                group_df.height,
                required_min_rows,
                config.vol_lookback_bars,
                config.expiry_bars,
            )
            continue
        valid_dfs.append(group_df)
    if not valid_dfs:
        logger.warning("No ticker groups satisfied minimum length requirement for labeling.")
        return df.clear().with_columns(
            pl.lit(None, dtype=pl.Int8).alias("label"),
            pl.lit(None, dtype=pl.Datetime).alias("t1"),
            pl.lit(None, dtype=pl.Datetime).alias("t0"),
        )
    processed_df = pl.concat(valid_dfs).sort([group_col, date_col])

    if "volatility" not in processed_df.columns:
        processed_df = calculate_volatility(
            processed_df,
            vol_lookback_bars=config.vol_lookback_bars,
            min_volatility=config.min_volatility,
            price_col=price_col,
            group_col=group_col,
        )

    processed_df = processed_df.with_columns(
        (pl.col(price_col) * (1.0 + config.profit_taking * pl.col("volatility"))).alias("upper_barrier"),
        (pl.col(price_col) * (1.0 - config.stop_loss * pl.col("volatility"))).alias("lower_barrier"),
        pl.col(date_col).alias("t0"),
    )

    expiry = config.expiry_bars
    label_expr = pl.lit(0, dtype=pl.Int8)
    t1_expr = pl.col(date_col).shift(-expiry).over(group_col)

    for k in range(expiry, 0, -1):
        future_close = pl.col(price_col).shift(-k).over(group_col)
        future_ts = pl.col(date_col).shift(-k).over(group_col)

        hit_upper = future_close >= pl.col("upper_barrier")
        hit_lower = future_close <= pl.col("lower_barrier")
        both_hit = hit_upper & hit_lower

        step_label = (
            pl.when(both_hit)
            .then(pl.when(future_close >= pl.col(price_col)).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(-1, dtype=pl.Int8)))
            .when(hit_upper)
            .then(pl.lit(1, dtype=pl.Int8))
            .when(hit_lower)
            .then(pl.lit(-1, dtype=pl.Int8))
            .otherwise(label_expr)
        )

        label_expr = step_label
        t1_expr = pl.when(hit_upper | hit_lower).then(future_ts).otherwise(t1_expr)

    result_df = processed_df.with_columns(
        label_expr.alias("label"),
        t1_expr.alias("t1"),
    )

    # Filter out trailing rows where vertical barrier extends beyond available future history
    return result_df.filter(pl.col("t1").is_not_null())
