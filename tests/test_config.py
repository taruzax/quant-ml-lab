from pathlib import Path

import pytest

from lab.core.config import PipelineConfig


def test_nested_config_round_trips_with_hash():
    config = PipelineConfig.from_yaml(Path("config/pipeline.yaml"))
    restored = PipelineConfig.model_validate(config.model_dump())
    assert restored.model_dump() == config.model_dump()
    assert restored.model_hash() == config.model_hash()
    assert config.models.xgboost.params["max_depth"] == 3
    assert config.allocation.lookback_bars == 252


def test_legacy_sections_translate_with_flat_compatibility_accessors():
    with pytest.warns(UserWarning):
        config = PipelineConfig(
            data={"timeframe": "1d"},
            returns={"target_horizons": [1]},
            labeling={"vol_lookback_bars": 12},
            risk={"max_position_size": 0.1},
            cv={"n_splits": 2},
        )
    assert config.timeframe.value == "1d"
    assert config.target_horizons == [1]
    assert config.vol_lookback_bars == 12
    assert config.max_position_size == 0.1
    assert config.n_splits == 2


def test_conflicting_legacy_and_canonical_settings_raise():
    with pytest.raises(ValueError, match="Conflicting settings"):
        PipelineConfig(
            allocation={"max_position_size": 0.2},
            risk={"max_position_size": 0.1},
        )


def test_unknown_nested_keys_are_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(models={"xgboost": {"enabled": True, "misspelled": 1}})
