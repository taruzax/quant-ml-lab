import numpy as np
import pandas as pd
import polars as pl
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from pypfopt import HRPOpt
from src.risk_engine.covariance import cov_to_corr, led_wo_shrinkage, denoise_cov

def hrp_custom(cov_matrix, tickers):
    """Custom polars native HRP"""
    pass

def hrp_pypfort(cov_matrix, tickers):
    """Non optimized, non custom HRP implementation from PyPortfolioOpt package
        Inputs:
            - cov matrix as an input (denoised or normal)
            - list of assets names
    """
    denoise_cov_df = pd.DataFrame(cov_matrix,index=tickers, columns=tickers )
    hrp = HRPOpt(returns=None, cov_matrix=denoised_cov_df)
    hrp.optimize()
    return dict(hrp.clean_weights())

def hrp_pipe(returns_df, custom: bool = False):
    wide_df = returns_df.select(pl.exclude('date'))
    tickers = wide_df.columns
    returns_matrix = wide_df.to_numpy()
    n_observations = returns_matrix.shape[0]

    cov_matrix, _ = led_wo_shrinkage(returns_matrix)
    denoised_matrix = denoise_cov(cov_matrix, n_observations)

    if custom:
        return hrp_custom(denoised_matrix, tickers)
    else:
        return hrp_pypfort(denoised_matrix, tickers)
    

