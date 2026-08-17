from datetime import datetime, timedelta
from unittest.mock import patch

import dagster as dg
import numpy as np
import polars as pl
from dagster import MultiPartitionKey

from lab.core.config import Timeframe
from lab.defs.assets import features, ffd_features, raw_ohlcv, tensors, validated_data
from lab.defs.checks import (
    features_finite_check,
    ffd_features_finite_check,
    raw_ohlcv_schema_check,
    tensors_manifest_check,
    validated_data_schema_check,
)
from lab.defs.resources import PipelineConfigResource

# We want to test the full graph, including the checks.
ASSETS = [raw_ohlcv, validated_data, features, ffd_features, tensors]
CHECKS = [
    raw_ohlcv_schema_check,
    validated_data_schema_check,
    features_finite_check,
    ffd_features_finite_check,
    tensors_manifest_check,
]


def test_smoke_asset_graph(tmp_path):
    """
    Smoke test that validates the pipeline logic from raw_ohlcv to tensors.
    We test this by directly calling the asset functions to bypass the
    MultiPartitionMapping lookback IO issue that occurs during `materialize()`.
    """
    n_rows = 140
    rng = np.random.default_rng(42)
    close = np.exp(np.cumsum(rng.normal(0, 0.01, n_rows))) * 150
    synthetic_df = pl.DataFrame(
        {
            "timestamp": pl.Series(
                [datetime(2025, 1, 1, 10, 0) + timedelta(hours=i) for i in range(n_rows)], dtype=pl.Datetime("ns")
            ),
            "ticker": ["AAPL"] * n_rows,
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": rng.uniform(1_000_000, 2_000_000, n_rows),
            "sector": ["Technology"] * n_rows,
            "industry": ["Software"] * n_rows,
        }
    )

    # 1. Raw OHLCV
    config = PipelineConfigResource(timeframe=Timeframe.H1.value, sequence_len=10)

    # Mock the context for partition_window
    context = dg.build_asset_context(partition_key=MultiPartitionKey({"ticker": "AAPL", "time": "2025-01-01"}))

    with patch("lab.defs.assets.load_market_data", return_value=synthetic_df):
        df_raw = raw_ohlcv(context, config)

    assert df_raw.height == n_rows

    # Run the check
    chk_raw = raw_ohlcv_schema_check(df_raw, config)
    if not chk_raw.passed:
        print("SCHEMA ERROR:", chk_raw.metadata["error"].text)
    assert chk_raw.passed

    # 2. Validated Data
    df_val = validated_data(df_raw, config)
    assert df_val.height == n_rows

    chk_val = validated_data_schema_check(df_val)
    assert chk_val.passed

    # 3. Features
    # The features asset expects a dict due to MultiPartitionMapping
    df_features = features(context, {"2025-01-01": df_val}, config)
    assert df_features.height > 0

    chk_feat = features_finite_check(df_features)
    assert chk_feat.passed

    # 4. FFD Features
    df_ffd = ffd_features(context, {"2025-01-01": df_features}, config)
    assert df_ffd.height > 0

    chk_ffd = ffd_features_finite_check(df_ffd)
    assert chk_ffd.passed

    # 5. Tensors
    context_tensors = dg.build_asset_context(partition_key=MultiPartitionKey({"ticker": "AAPL", "time": "2025-01-01"}))
    res_tensors = tensors(context_tensors, df_ffd, config)

    # Extract the dataframe from the MaterializeResult
    manifest_df = res_tensors.value
    assert manifest_df.height == 1

    chk_tensors = tensors_manifest_check(manifest_df)
    assert chk_tensors.passed

    print("Smoke test successfully processed synthetic data through all assets and checks.")
