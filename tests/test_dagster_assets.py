from datetime import datetime, timedelta
from unittest.mock import patch

import dagster as dg
import numpy as np
import polars as pl
import yaml
from dagster import materialize

from lab.core.config import PIPELINE_CONFIG_PATH, PipelineConfig, Timeframe
from lab.defs.assets import (
    features,
    ffd_features,
    raw_ohlcv,
    tensors,
    validated_data,
)
from lab.defs.resources import PipelineConfigResource

ASSETS = [raw_ohlcv, validated_data, features, ffd_features, tensors]


def test_pipeline_yaml_matches_pipeline_config_fields():
    raw = yaml.safe_load(PIPELINE_CONFIG_PATH.read_text())
    yaml_fields = set()
    for section_values in raw.values():
        yaml_fields.update(section_values)

    assert yaml_fields == set(PipelineConfig.model_fields)


def test_asset_graph_resolves():
    """
    Tests that Dagster can successfully resolve the dependency graph
    """
    defs = dg.Definitions(
        assets=ASSETS,
        resources={
            "config_py": PipelineConfigResource(),
        },
    )
    job = defs.get_implicit_global_asset_job_def()
    assert job is not None
    assert len(defs.resolve_asset_graph().get_all_asset_keys()) == 5


def test_asset_materialization_synthetic():
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

    with patch("lab.defs.assets.load_market_data", return_value=synthetic_df):
        result = materialize(
            assets=ASSETS,
            resources={
                "config_py": PipelineConfigResource(
                    timeframe=Timeframe.H1.value,
                    sequence_len=10,
                ),
            },
        )

        assert result.success
        materialized_keys = [event.asset_key.path[0] for event in result.get_asset_materialization_events()]
        expected_assets = {"raw_ohlcv", "validated_data", "features", "ffd_features", "tensors"}

        for asset in expected_assets:
            assert asset in materialized_keys
