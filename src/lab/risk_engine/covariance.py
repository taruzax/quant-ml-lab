"""Compatibility exports for the relocated covariance module."""

from lab.quant.covariance import cov_to_corr, denoise_cov, led_wo_shrinkage

__all__ = ["cov_to_corr", "denoise_cov", "led_wo_shrinkage"]
