from datetime import timedelta

import pandera.polars as pa
import polars as pl
from dagster import AssetCheckResult, asset_check

from lab.core.schemas import PRICE_COLUMNS
from lab.defs.assets import features, ffd_features, raw_ohlcv, tensors, validated_data
from lab.defs.resources import PipelineConfigResource

RawOHLCVSchema = pa.DataFrameSchema(
    {
        "timestamp": pa.Column(pl.Datetime, nullable=False),
        "ticker": pa.Column(str, nullable=False),
        "open": pa.Column(float, nullable=False),
        "high": pa.Column(float, nullable=False),
        "low": pa.Column(float, nullable=False),
        "close": pa.Column(float, nullable=False),
        "volume": pa.Column(float, nullable=False),
    },
    strict=False,
)

ValidatedDataSchema = RawOHLCVSchema

TensorsSchema = pa.DataFrameSchema(
    {
        "timeframe": pa.Column(str, nullable=False),
        "n_rows": pa.Column(int, pa.Check.ge(0), nullable=False),
        "n_windows": pa.Column(int, pa.Check.ge(0), nullable=False),
        "n_features": pa.Column(int, pa.Check.gt(0), nullable=False),
        "n_targets": pa.Column(int, pa.Check.gt(0), nullable=False),
        "sequence_len": pa.Column(int, pa.Check.gt(0), nullable=False),
    },
    strict=False,
)


def _schema_result(schema: pa.DataFrameSchema, df: pl.DataFrame) -> AssetCheckResult:
    try:
        schema.validate(df, lazy=True)
    except Exception as exc:
        return AssetCheckResult(passed=False, metadata={"error": str(exc)[:1000]})
    return AssetCheckResult(passed=True)


def _gap_check(df: pl.DataFrame, tolerance: timedelta) -> AssetCheckResult:
    if df.is_empty():
        return AssetCheckResult(passed=True)

    trading_bars = df.sort(["ticker", "timestamp"]).with_columns(
        pl.col("timestamp").dt.weekday().alias("weekday"),
        pl.col("timestamp").diff().over("ticker").alias("_gap"),
    )

    intraday_gaps = trading_bars.filter(pl.col("weekday") < 5)
    max_gap = intraday_gaps.select(pl.col("_gap").max()).item()

    if max_gap is None:
        return AssetCheckResult(passed=True)

    passed = max_gap <= tolerance
    return AssetCheckResult(
        passed=passed,
        metadata={"max_gap": str(max_gap), "tolerance": str(tolerance)},
    )


def _finite_columns_check(df: pl.DataFrame) -> AssetCheckResult:
    if df.is_empty():
        return AssetCheckResult(passed=True)

    numeric_cols = [col for col, dtype in df.schema.items() if dtype in {pl.Float32, pl.Float64}]

    checks = []
    for col in numeric_cols:
        checks.append((pl.col(col).is_finite() | pl.col(col).is_null() | pl.col(col).is_nan()).all().alias(col))

    if not checks:
        return AssetCheckResult(passed=False, metadata={"error": "No columns available for finite check"})

    row = df.select(checks).row(0, named=True)
    failed = [col for col, passed in row.items() if not passed]
    return AssetCheckResult(passed=not failed, metadata={"failed_columns": failed})


@asset_check(asset=raw_ohlcv)
def raw_ohlcv_schema_check(raw_ohlcv: pl.DataFrame, config_py: PipelineConfigResource) -> AssetCheckResult:
    schema_result = _schema_result(RawOHLCVSchema, raw_ohlcv)
    if not schema_result.passed:
        return schema_result
    return _gap_check(raw_ohlcv, config_py.to_pipeline_config().gap_tolerance)


@asset_check(asset=validated_data)
def validated_data_schema_check(validated_data: pl.DataFrame) -> AssetCheckResult:
    schema_result = _schema_result(ValidatedDataSchema, validated_data)
    if not schema_result.passed:
        return schema_result

    bad_prices = validated_data.filter(pl.any_horizontal([pl.col(col) <= 0 for col in PRICE_COLUMNS]))
    return AssetCheckResult(passed=bad_prices.is_empty(), metadata={"bad_price_rows": bad_prices.height})


@asset_check(asset=features)
def features_finite_check(features: pl.DataFrame) -> AssetCheckResult:
    return _finite_columns_check(features)


@asset_check(asset=ffd_features)
def ffd_features_finite_check(ffd_features: pl.DataFrame) -> AssetCheckResult:
    return _finite_columns_check(ffd_features)


@asset_check(asset=tensors)
def tensors_manifest_check(tensors: pl.DataFrame) -> AssetCheckResult:
    return _schema_result(TensorsSchema, tensors)
