"""Compatibility exports for the relocated FFD module."""

from lab.quant.ffd import (
    AdfResult,
    compute_memory_corr,
    find_global_d,
    find_min_d,
    find_min_d_grid,
    frac_diff_polars,
    get_weights_ffd,
    run_adf_test,
)

__all__ = [
    "AdfResult",
    "compute_memory_corr",
    "find_global_d",
    "find_min_d",
    "find_min_d_grid",
    "frac_diff_polars",
    "get_weights_ffd",
    "run_adf_test",
]
