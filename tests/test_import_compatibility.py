from importlib import import_module


def test_relocated_financial_modules_preserve_legacy_exports():
    pairs = (
        ("lab.data.features", "lab.quant.features", "apply_all_features"),
        ("lab.data.ffd", "lab.quant.ffd", "frac_diff_polars"),
        ("lab.data.labeling", "lab.quant.labeling", "triple_barrier_label"),
        ("lab.data.cv", "lab.quant.cv", "PurgedKFold"),
        ("lab.data.validators", "lab.quant.validators", "run_all_validations"),
        ("lab.risk_engine.constraints", "lab.quant.constraints", "apply_all_constraints"),
        ("lab.risk_engine.covariance", "lab.quant.covariance", "led_wo_shrinkage"),
        ("lab.risk_engine.hrp", "lab.quant.hrp", "hrp_pipe"),
    )
    for legacy_name, current_name, symbol in pairs:
        legacy = import_module(legacy_name)
        current = import_module(current_name)
        assert getattr(legacy, symbol) is getattr(current, symbol)


def test_relocated_platform_and_research_modules_preserve_legacy_exports():
    loader = import_module("lab.data.loader")
    data_access = import_module("lab.platform.data_access")
    tensors = import_module("lab.data.tensor_loader")
    samples = import_module("lab.research.samples")
    assert loader.load_market_data is data_access.load_market_data
    assert tensors.TimeSeriesDataset is samples.TimeSeriesDataset


def test_dagster_has_one_compatibility_definitions_object():
    current = import_module("lab.platform.dagster")
    legacy = import_module("lab.defs")
    root = import_module("lab.definitions")
    assert legacy.defs is current.defs
    assert root.defs is current.defs
