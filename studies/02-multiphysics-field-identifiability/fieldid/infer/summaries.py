"""Closed-form posterior summaries.

Every reported quantity is either a free parameter or a product of free
parameters, so in the logarithmic coordinates the package already uses it is a
*linear* function of the parameter vector.  Under the Gaussian approximation of
the posterior its own distribution is therefore Gaussian in closed form, and the
median, the interval and the contraction follow without drawing a single sample.

This removes the Monte Carlo noise that otherwise makes a contraction of exactly
one come out as 1.003, and it makes the coverage study affordable.
"""
from __future__ import annotations
import numpy as np

from ..params import Z90
from . import diagnostics as DG

#: how each reported quantity is built from free parameters, in logarithms
DERIVED = {"S_v": {"phi": 1.0, "C_v": 1.0}}


def coefficients(inv, name):
    """Row vector a with log(quantity) = a . x + constant, in transformed
    coordinates.  Returns None when the quantity is not a function of the free
    parameters."""
    a = np.zeros(len(inv.free))
    if name in inv.free:
        a[inv.space.index(name)] = 1.0
        return a
    rule = DERIVED.get(name)
    if rule is None or not all(n in inv.free for n in rule):
        return None
    for n, w in rule.items():
        a[inv.space.index(n)] = w
    return a


def _moments(inv, a, mean_z, cov_z):
    """(location, scale) of the transformed quantity under N(mean_z, cov_z)."""
    _, s = DG.standardize(inv)
    b = a * s
    loc = float(np.dot(a, DG.z_to_x(inv, mean_z)))
    var = float(b @ cov_z @ b)
    return loc, float(np.sqrt(max(var, 0.0)))


def prior_covariance(inv):
    return np.linalg.inv(DG.prior_precision(inv))


def summarize_gaussian(inv, lap, names, truth=None):
    """Closed-form posterior summary for each requested quantity."""
    truth = truth or inv.truth_physical()
    cov_pri = prior_covariance(inv)
    mean_pri = np.zeros(len(inv.free))
    out = {}
    for n in names:
        a = coefficients(inv, n)
        if a is None:
            continue
        loc, sd = _moments(inv, a, lap["mean"], lap["cov"])
        _, sd_p = _moments(inv, a, mean_pri, cov_pri)
        entry = inv.space.entry(n) if n in inv.free else None
        log_scale = entry is None or entry.transform == "log"
        if entry is not None and entry.kind == "uniform":
            # a uniform marginal is not Gaussian; its 90 percent width is exact
            sd_p = 0.9 * (entry.to_x(entry.hi) - entry.to_x(entry.lo)) / (2 * Z90)
        med = np.exp(loc) if log_scale else loc
        lo = np.exp(loc - Z90 * sd) if log_scale else loc - Z90 * sd
        hi = np.exp(loc + Z90 * sd) if log_scale else loc + Z90 * sd
        out[n] = dict(median=float(med), lo=float(lo), hi=float(hi),
                      width_post=float(2 * Z90 * sd),
                      width_prior=float(2 * Z90 * sd_p),
                      contraction=float(sd / sd_p) if sd_p > 0 else float("nan"),
                      truth=float(truth[n]),
                      bias_log=float(loc - (np.log(truth[n]) if log_scale
                                            else truth[n])),
                      covered=bool(lo <= truth[n] <= hi))
    return out
