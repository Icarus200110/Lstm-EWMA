"""
Residual computation for Poisson-distributed count data.
"""

import numpy as np


def compute_pearson_residuals(y, mu, eps=1e-8):
    """
    Pearson residuals: sensitive to scale changes in count data.
    r = (y - mu) / sqrt(mu)
    """
    mu_np = np.clip(mu.astype(np.float64), eps, 1e10)
    y_np = y.astype(np.float64)
    return (y_np - mu_np) / np.sqrt(mu_np)


def compute_deviance_residuals(y, mu, eps=1e-8):
    """
    Deviance residuals: more stable residual definition.
    d = sign(y - mu) * sqrt(2 * [y * log(y/mu) - (y - mu)])
    """
    y_np = y.astype(np.float64)
    mu_np = np.clip(mu.astype(np.float64), eps, 1e10)

    deviance = np.zeros_like(y_np)
    for i in range(len(y_np)):
        yi, mui = y_np[i], mu_np[i]
        if yi == 0:
            term = 2 * mui
        else:
            term = 2 * (yi * np.log(yi / mui) - (yi - mui))
        deviance[i] = np.sign(yi - mui) * np.sqrt(np.abs(term))

    return deviance


def compute_standardized_residuals(y, mu, eps=1e-8):
    """
    Standardized residuals: Pearson residuals with rolling standardization.
    Variance approximately 1, suitable for EWMA monitoring.
    """
    pearson = compute_pearson_residuals(y, mu, eps)
    standardized = np.zeros_like(pearson)
    for i in range(len(pearson)):
        if i < 30:
            window = pearson[:i + 1]
        else:
            window = pearson[:i]
        if len(window) > 1:
            standardized[i] = (pearson[i] -
                               np.mean(window)) / (np.std(window) + eps)
        else:
            standardized[i] = pearson[i]
    return standardized


def compute_log_residuals(y, mu, eps=1e-8):
    """
    Log residuals: computed on logarithmic scale.
    More sensitive to multiplicative shifts in count data.
    """
    y_np = np.clip(y.astype(np.float64), eps, 1e10)
    mu_np = np.clip(mu.astype(np.float64), eps, 1e10)
    return np.log(y_np) - np.log(mu_np)
