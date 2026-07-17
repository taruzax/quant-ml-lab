import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller

# pyrefly: ignore [missing-import]
import lab.data.ffd as ffd_module

# pyrefly: ignore [missing-import]
from lab.data.ffd import (
    compute_memory_corr,
    find_min_d,
    find_min_d_grid,
    frac_diff_polars,
    get_weights_ffd,
    run_adf_test,
)

# ─────────────────────────────────────────────
# 1. WEIGHT CORRECTNESS
# ─────────────────────────────────────────────


def test_weights_d0():
    """d=0 → weight[0] == 1.0 and no further weights cross threshold."""
    w = get_weights_ffd(d=0.0, threshold=1e-4)
    assert w[0] == 1.0, f"Expected w[0]=1.0, got {w[0]}"
    assert len(w) == 1, f"d=0 should produce 1 weight, got {len(w)}"
    print(f"  [PASS] d=0: weights = {w}")


def test_weights_d1():
    """d=1 → standard differencing: w[0]=1, w[1]=-1, nothing more above threshold."""
    w = get_weights_ffd(d=1.0, threshold=1e-5)
    assert abs(w[0] - 1.0) < 1e-9, f"w[0] expected 1.0, got {w[0]}"
    assert abs(w[1] - (-1.0)) < 1e-9, f"w[1] expected -1.0, got {w[1]}"
    assert len(w) == 2, f"d=1 should produce 2 weights above threshold, got {len(w)}"
    print(f"  [PASS] d=1: weights = {w}")


def test_weights_fractional_decay():
    """For 0 < d < 1, first weight is 1, rest are negative and monotonically decreasing in absolute value."""
    w = get_weights_ffd(d=0.4, threshold=1e-5)
    assert w[0] == 1.0
    # All weights after k=0 should be negative for d in (0,1)
    assert all(wi < 0 for wi in w[1:]), "All weights after k=0 must be negative for d in (0,1)"
    # Absolute values should decay as k increases
    abs_w = [abs(wi) for wi in w[1:]]
    assert abs_w == sorted(abs_w, reverse=True), "Absolute weights must be monotonically decaying"
    print(f"  [PASS] d=0.4: {len(w)} weights, last={w[-1]:.6f}")


def test_weights_threshold_cutoff():
    """No weight in returned list should have |w| < threshold."""
    threshold = 1e-3
    w = get_weights_ffd(d=0.4, threshold=threshold)
    for i, wi in enumerate(w):
        assert abs(wi) >= threshold or i == 0, f"Weight at k={i} is {wi} which is below threshold {threshold}"
    print(f"  [PASS] All {len(w)} weights are above threshold {threshold}")


# ─────────────────────────────────────────────
# 2. FRAC DIFF POLARS OUTPUT SHAPE
# ─────────────────────────────────────────────


def test_frac_diff_output_columns(single_ticker_df):
    """frac_diff_polars should add a '{col}_frac_diff' column."""
    df = single_ticker_df
    result = frac_diff_polars(df, col_name="close", d=0.4, threshold=1e-3)
    assert "close_frac_diff" in result.columns, "Missing 'close_frac_diff' column"
    print(f"  [PASS] Column 'close_frac_diff' present. Shape: {result.shape}")


def test_frac_diff_length_preserved(single_ticker_df):
    """Output DataFrame must have the same number of rows as input."""
    df = single_ticker_df
    result = frac_diff_polars(df, col_name="close", d=0.4, threshold=1e-3)
    assert result.shape[0] == df.shape[0], f"Row count changed: {df.shape[0]} → {result.shape[0]}"
    print(f"  [PASS] Row count preserved: {result.shape[0]}")


def test_frac_diff_d0_equals_original(single_ticker_df):
    """With d=0, the fractionally differenced series should equal the original."""
    df = single_ticker_df
    result = frac_diff_polars(df, col_name="close", d=0.0, threshold=1e-4)
    original = df["close"].to_numpy()
    diffed = result["close_frac_diff"].to_numpy()
    np.testing.assert_allclose(original, diffed, rtol=1e-6, err_msg="d=0 should leave the series unchanged")
    print("  [PASS] d=0 leaves series unchanged")


def test_frac_diff_no_cross_ticker_contamination(multi_ticker_df):
    """Shifts must respect ticker boundaries — ticker B's values must not bleed into ticker A."""
    combined = multi_ticker_df
    result = frac_diff_polars(combined, col_name="close", d=0.4, threshold=1e-3)

    for ticker in ["AAA", "BBB"]:
        solo_df = combined.filter(pl.col("ticker") == ticker)
        solo_res = frac_diff_polars(solo_df, col_name="close", d=0.4, threshold=1e-3)

        combined_vals = result.filter(pl.col("ticker") == ticker)["close_frac_diff"].drop_nulls().to_numpy()
        solo_vals = solo_res["close_frac_diff"].drop_nulls().to_numpy()
        np.testing.assert_allclose(
            combined_vals, solo_vals, rtol=1e-6, err_msg=f"Cross-ticker contamination detected for {ticker}"
        )
    print("  [PASS] No cross-ticker contamination")


# ─────────────────────────────────────────────
# 3. ADF STATIONARITY CHECK
# ─────────────────────────────────────────────


def test_adf_raw_prices_nonstationary():
    """Raw random-walk prices should fail the ADF test (p > 0.05)."""
    np.random.seed(0)
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))
    _, p_val, *_ = adfuller(prices, maxlag=1, regression="c", autolag=None)
    assert p_val > 0.05, f"Expected raw prices to be non-stationary, got p={p_val:.4f}"
    print(f"  [PASS] Raw prices non-stationary: p={p_val:.4f}")


def test_adf_fully_differenced_stationary():
    """Integer differencing (d=1) should produce a stationary series (p < 0.05)."""
    np.random.seed(0)
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))
    returns = np.diff(prices)
    _, p_val, *_ = adfuller(returns, maxlag=1, regression="c", autolag=None)
    assert p_val < 0.05, f"Expected returns to be stationary, got p={p_val:.4f}"
    print(f"  [PASS] Integer-differenced series stationary: p={p_val:.4f}")


def test_frac_diff_achieves_stationarity(single_ticker_df):
    """Applying FFD with an optimal d should produce a stationary series."""
    df = single_ticker_df
    df = df.with_columns(pl.col("close").log().alias("log_close"))

    optimal_d = find_min_d(df, col_name="log_close", threshold=0.01)
    print(f"  optimal_d found = {optimal_d:.2f}")

    result = frac_diff_polars(df, col_name="log_close", d=optimal_d, threshold=0.01)
    series = result["log_close_frac_diff"].drop_nulls().to_numpy()

    _, p_val, *_ = adfuller(series, maxlag=1, regression="c", autolag=None)
    assert p_val < 0.05, f"FFD series not stationary after find_min_ffd: d={optimal_d:.2f}, p={p_val:.4f}"
    print(f"  [PASS] FFD series stationary: d={optimal_d:.2f}, p={p_val:.4f}")


def test_find_min_ffd_returns_minimal_d(single_ticker_df):
    """find_min_ffd should return d well below 1.0 for a standard random walk."""
    df = single_ticker_df
    df = df.with_columns(pl.col("close").log().alias("log_close"))
    optimal_d = find_min_d(df, col_name="log_close", threshold=0.01)
    assert optimal_d < 1.0, f"Expected d < 1.0, got {optimal_d}"
    print(f"  [PASS] Minimal d = {optimal_d:.2f} (below 1.0)")


def test_run_adf_test_wrapper():
    """run_adf_test should return an AdfResult namedtuple with correct fields."""
    np.random.seed(42)
    stationary_series = np.random.normal(0, 1, 500)
    result = run_adf_test(stationary_series)
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")
    assert hasattr(result, "is_stationary")
    assert result.is_stationary is True  # white noise is stationary


def test_scipy_optimizer_matches_grid(single_ticker_df):
    """Scipy optimizer and grid search should produce d values within ±0.05."""
    df = single_ticker_df
    df = df.with_columns(pl.col("close").log().alias("log_close"))

    d_scipy = find_min_d(df, col_name="log_close", threshold=0.01)
    d_grid = find_min_d_grid(df, col_name="log_close", threshold=0.01)

    assert abs(d_scipy - d_grid) <= 0.15, f"Scipy d={d_scipy:.3f} vs grid d={d_grid:.3f}, diff={abs(d_scipy - d_grid):.3f}"


def test_memory_correlation_high_for_low_d(single_ticker_df):
    """With d=0.1, correlation with original should be > 0.8 (high memory preserved)."""
    df = single_ticker_df
    df = df.with_columns(pl.col("close").log().alias("log_close"))
    result = frac_diff_polars(df, "log_close", d=0.1, threshold=0.01)

    original = df["log_close"].to_numpy()
    differenced = result["log_close_frac_diff"].to_numpy()

    # Align on non-null
    mask = np.isfinite(differenced)
    corr = compute_memory_corr(original[mask], differenced[mask])
    assert corr > 0.8, f"Expected correlation > 0.8 for d=0.1, got {corr:.4f}"


def test_memory_correlation_low_for_high_d(single_ticker_df):
    """With d=1.0, correlation with original should be < 0.5 (memory destroyed)."""
    df = single_ticker_df
    df = df.with_columns(pl.col("close").log().alias("log_close"))
    result = frac_diff_polars(df, "log_close", d=1.0, threshold=0.01)

    original = df["log_close"].to_numpy()
    differenced = result["log_close_frac_diff"].to_numpy()

    mask = np.isfinite(differenced)
    corr = compute_memory_corr(original[mask], differenced[mask])
    assert corr < 0.5, f"Expected correlation < 0.5 for d=1.0, got {corr:.4f}"


def test_find_min_d_fallback(monkeypatch, single_ticker_df):
    """If scipy raises, find_min_d should fall back to grid search."""
    df = single_ticker_df
    df = df.with_columns(pl.col("close").log().alias("log_close"))

    # Mock scipy to raise
    def mock_minimize(*args, **kwargs):
        raise RuntimeError("Mocked scipy failure")

    monkeypatch.setattr(ffd_module, "minimize_scalar", mock_minimize)

    # Should still return a valid d via grid search fallback
    d = find_min_d(df, col_name="log_close", threshold=0.01)
    assert 0.1 <= d <= 1.0, f"Fallback returned invalid d={d}"
