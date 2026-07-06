import numpy as np
from sklearn.covariance import LedoitWolf

def led_wo_shrinkage(returns):
    """Apply ledoit wolf shrinkage to estimate a stable covariance matrix"""
    lw = LedoitWolf()
    lw.fit(returns)

    return lw.covariance_, lw.shrinkage_


def denoise_cov(cov, n_observarions, method):
    """Denoise a covariance matrix using Marchenko-Pastur random matrix theory"""

    n_assets = cov.shape[0]
    q = n_assets / n_observarions

    #extract eigvalues and eigvectors and sort descending 
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.flip(np.argsort(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    sigma_sq = np.median(eigvals)
    lambdaplus = sigma_sq*(1+np.sqrt(q))**2

    noise_mask = eigvals<lambdaplus

    if noise_mask.any():
        noise_avg = eigvals[noise_mask].mean()
        eigvals[noise_mask] = noise_avg
    
    eigvals = np.maximum(eigvals, 0.0)


    #reconstruct covariance
    denoised = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    denoised = (denoised+denoised.T)/2
    return denoised


def cov_to_corr(cov):
    std = np.sqrt(np.diag(cov))
    std = np.where(std ==0, 1e-10, std)
    corr = cov/np.outer(std, std)

    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr