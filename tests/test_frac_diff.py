"""
Tests for fractional differencing implementation in ffd/src/data/transform.py

Run with:
    python -m pytest tests/test_frac_diff.py -v
or for a quick stdout-friendly run:
    python tests/test_frac_diff.py
"""
import sys
import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, ".")
from ffd_adf.src.data.transform import get_weights_ffd, frac_diff_polars, find_min_ffd


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
    assert abs(w[0] - 1.0) < 1e-9,  f"w[0] expected 1.0, got {w[0]}"
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

def _make_ticker_df(n=300, ticker="TEST") -> pl.DataFrame:
    """Create a synthetic price DataFrame for one ticker."""
    np.random.seed(42)
    # Random walk for price (non-stationary)
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    dates = pl.date_range(
        start=pl.date(2024, 1, 1),
        end=pl.date(2024, 1, 1) + pl.duration(days=n - 1),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame({"date": dates, "ticker": ticker, "close": prices})


def test_frac_diff_output_columns():
    """frac_diff_polars should add a '{col}_frac_diff' column."""
    df = _make_ticker_df()
    result = frac_diff_polars(df, col_name="close", d=0.4, threshold=1e-3)
    assert "close_frac_diff" in result.columns, "Missing 'close_frac_diff' column"
    print(f"  [PASS] Column 'close_frac_diff' present. Shape: {result.shape}")


def test_frac_diff_length_preserved():
    """Output DataFrame must have the same number of rows as input."""
    df = _make_ticker_df(n=200)
    result = frac_diff_polars(df, col_name="close", d=0.4, threshold=1e-3)
    assert result.shape[0] == df.shape[0], (
        f"Row count changed: {df.shape[0]} → {result.shape[0]}"
    )
    print(f"  [PASS] Row count preserved: {result.shape[0]}")


def test_frac_diff_d0_equals_original():
    """With d=0, the fractionally differenced series should equal the original."""
    df = _make_ticker_df(n=100)
    result = frac_diff_polars(df, col_name="close", d=0.0, threshold=1e-4)
    original = df["close"].to_numpy()
    diffed   = result["close_frac_diff"].to_numpy()
    np.testing.assert_allclose(original, diffed, rtol=1e-6,
        err_msg="d=0 should leave the series unchanged")
    print(f"  [PASS] d=0 leaves series unchanged")


def test_frac_diff_no_cross_ticker_contamination():
    """Shifts must respect ticker boundaries — ticker B's values must not bleed into ticker A."""
    df_a = _make_ticker_df(n=100, ticker="AAA")
    df_b = _make_ticker_df(n=100, ticker="BBB")
    combined = pl.concat([df_a, df_b])
    result = frac_diff_polars(combined, col_name="close", d=0.4, threshold=1e-3)

    for ticker in ["AAA", "BBB"]:
        solo_df  = _make_ticker_df(n=100, ticker=ticker)
        solo_res = frac_diff_polars(solo_df, col_name="close", d=0.4, threshold=1e-3)

        combined_vals = (
            result.filter(pl.col("ticker") == ticker)["close_frac_diff"]
            .drop_nulls().to_numpy()
        )
        solo_vals = solo_res["close_frac_diff"].drop_nulls().to_numpy()
        np.testing.assert_allclose(combined_vals, solo_vals, rtol=1e-6,
            err_msg=f"Cross-ticker contamination detected for {ticker}")
    print(f"  [PASS] No cross-ticker contamination")


# ─────────────────────────────────────────────
# 3. ADF STATIONARITY CHECK
# ─────────────────────────────────────────────

def test_adf_raw_prices_nonstationary():
    """Raw random-walk prices should fail the ADF test (p > 0.05)."""
    np.random.seed(0)
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))
    _, p_val, *_ = adfuller(prices, maxlag=1, regression='c', autolag=None)
    assert p_val > 0.05, f"Expected raw prices to be non-stationary, got p={p_val:.4f}"
    print(f"  [PASS] Raw prices non-stationary: p={p_val:.4f}")


def test_adf_fully_differenced_stationary():
    """Integer differencing (d=1) should produce a stationary series (p < 0.05)."""
    np.random.seed(0)
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))
    returns = np.diff(prices)
    _, p_val, *_ = adfuller(returns, maxlag=1, regression='c', autolag=None)
    assert p_val < 0.05, f"Expected returns to be stationary, got p={p_val:.4f}"
    print(f"  [PASS] Integer-differenced series stationary: p={p_val:.4f}")


def test_frac_diff_achieves_stationarity():
    """Applying FFD with an optimal d should produce a stationary series."""
    df = _make_ticker_df(n=400)
    df = df.with_columns(pl.col("close").log().alias("log_close"))

    optimal_d = find_min_ffd(df, col_name="log_close", threshold=0.01)
    print(f"  optimal_d found = {optimal_d:.2f}")

    result = frac_diff_polars(df, col_name="log_close", d=optimal_d, threshold=0.01)
    series = result["log_close_frac_diff"].drop_nulls().to_numpy()

    _, p_val, *_ = adfuller(series, maxlag=1, regression='c', autolag=None)
    assert p_val < 0.05, (
        f"FFD series not stationary after find_min_ffd: d={optimal_d:.2f}, p={p_val:.4f}"
    )
    print(f"  [PASS] FFD series stationary: d={optimal_d:.2f}, p={p_val:.4f}")


def test_find_min_ffd_returns_minimal_d():
    """find_min_ffd should return d well below 1.0 for a standard random walk."""
    df = _make_ticker_df(n=400)
    df = df.with_columns(pl.col("close").log().alias("log_close"))
    optimal_d = find_min_ffd(df, col_name="log_close", threshold=0.01)
    assert optimal_d < 1.0, f"Expected d < 1.0, got {optimal_d}"
    print(f"  [PASS] Minimal d = {optimal_d:.2f} (below 1.0)")


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # Weight tests
        test_weights_d0,
        test_weights_d1,
        test_weights_fractional_decay,
        test_weights_threshold_cutoff,
        # Shape tests
        test_frac_diff_output_columns,
        test_frac_diff_length_preserved,
        test_frac_diff_d0_equals_original,
        test_frac_diff_no_cross_ticker_contamination,
        # ADF tests
        test_adf_raw_prices_nonstationary,
        test_adf_fully_differenced_stationary,
        test_frac_diff_achieves_stationarity,
        test_find_min_ffd_returns_minimal_d,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        print(f"\n{'─'*50}")
        print(f"Running: {test_fn.__name__}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{'═'*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)
