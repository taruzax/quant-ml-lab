from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import polars as pl
import yaml
from dagster import DagsterInstance, MultiPartitionKey, materialize
from dagster_polars import PolarsParquetIOManager

from lab.core.config import PIPELINE_CONFIG_PATH, PipelineConfig, Timeframe
from lab.platform.dagster.assets import (
    features,
    ffd_features,
    raw_ohlcv,
    tensors,
    validated_data,
)
from lab.platform.dagster.resources import PipelineConfigResource

ASSETS = [raw_ohlcv, validated_data, features, ffd_features, tensors]


def test_partition_key_parsing():
    key = MultiPartitionKey({"ticker": "AAPL", "time": "2024-01-02"})
    assert key.keys_by_dimension["ticker"] == "AAPL"
    assert key.keys_by_dimension["time"] == "2024-01-02"


def test_partition_isolation(tmp_path):
    instance = DagsterInstance.ephemeral()
    instance.add_dynamic_partitions("tickers", ["AAPL", "MSFT"])
    n_rows = 50
    rng = np.random.default_rng(42)
    close = np.exp(np.cumsum(rng.normal(0, 0.01, n_rows))) * 150
    synthetic_df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2, 10, 0) + timedelta(hours=i) for i in range(n_rows)],
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
    with patch("lab.platform.dagster.assets.load_market_data", return_value=synthetic_df):
        result = materialize(
            [raw_ohlcv],
            partition_key=MultiPartitionKey({"ticker": "AAPL", "time": "2024-01-02"}),
            instance=instance,
            resources={
                "config_py": PipelineConfigResource(timeframe="1h"),
                "io_manager": PolarsParquetIOManager(base_dir=str(tmp_path / "dagster")),
            },
        )
        assert result.success


def test_pipeline_yaml_matches_pipeline_config_fields():
    raw = yaml.safe_load(PIPELINE_CONFIG_PATH.read_text())
    yaml_fields = set()
    for section_values in raw.values():
        yaml_fields.update(section_values)

    assert yaml_fields == set(PipelineConfig.model_fields)


# tests/test_dagster_assets.py


def test_asset_materialization_synthetic(tmp_path):
    """
    Tests that the asset graph can execute and materialize end-to-end on synthetic data.
    """
    n_rows = 140
    rng = np.random.default_rng(42)
    close = np.exp(np.cumsum(rng.normal(0, 0.01, n_rows))) * 150
    synthetic_df = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, 1, 10, 0) + timedelta(hours=i) for i in range(n_rows)],
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

    instance = DagsterInstance.ephemeral()
    instance.add_dynamic_partitions("tickers", ["AAPL"])

    def mock_load_input(context):
        if context.asset_key.path[0] in ["validated_data", "features"]:
            return {"2024-01-02": synthetic_df}
        return synthetic_df

    with (
        patch("lab.platform.dagster.assets.load_market_data", return_value=synthetic_df),
        patch("dagster_polars.PolarsParquetIOManager.load_input", side_effect=mock_load_input),
    ):
        result = materialize(
            partition_key=MultiPartitionKey({"ticker": "AAPL", "time": "2024-01-02"}),
            assets=ASSETS,
            instance=instance,
            resources={
                "config_py": PipelineConfigResource(
                    timeframe=Timeframe.H1.value,
                    sequence_len=10,
                ),
                "io_manager": PolarsParquetIOManager(base_dir=str(tmp_path / "dagster")),
            },
        )

        assert result.success
        materialized_keys = [event.asset_key.path[0] for event in result.get_asset_materialization_events()]
        expected_assets = {"raw_ohlcv", "validated_data", "features", "ffd_features", "tensors"}

        for asset in expected_assets:
            assert asset in materialized_keys
