"""Relative Gaussian noise models and log-likelihoods.

The reference model uses independent errors across frequencies. An exponential
correlation model is provided for sensitivity analysis because measurements
derived from one acquisition can have correlated errors.
"""
from __future__ import annotations

import numpy as np


def add_relative_noise(y, sigma_rel, rng):
    y = np.asarray(y, float)
    return y * (1.0 + sigma_rel * rng.standard_normal(y.shape))


def gaussian_loglik(obs, pred, sigma_rel):
    """Independent relative Gaussian errors."""
    obs = np.asarray(obs, float); pred = np.asarray(pred, float)
    s = sigma_rel * np.abs(pred)
    s = np.where(s > 0, s, 1e-30)
    r = (obs - pred) / s
    return float(-0.5 * np.sum(r ** 2) - np.sum(np.log(s)))


def correlated_loglik(obs, pred, sigma_rel, rho):
    """Exponentially correlated errors across the frequency index.

    Provided so that the independence assumption can be tested rather than
    assumed: a smooth estimator applied to one acquisition produces errors that
    are correlated across frequency, which reduces the effective number of
    independent observations.
    """
    obs = np.asarray(obs, float); pred = np.asarray(pred, float)
    if not (0 <= rho < 1):
        raise ValueError("rho must satisfy 0 <= rho < 1")
    n = len(obs)
    s = sigma_rel * np.abs(pred)
    s = np.where(s > 0, s, 1e-30)
    i = np.arange(n)
    C = rho ** np.abs(i[:, None] - i[None, :]) * np.outer(s, s)
    L = np.linalg.cholesky(C + 1e-18 * np.eye(n))
    z = np.linalg.solve(L, obs - pred)
    return float(-0.5 * z @ z - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi))
