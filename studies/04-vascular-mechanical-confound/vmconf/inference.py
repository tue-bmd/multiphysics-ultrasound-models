"""Posterior inference over (mu, eta, s) and the comparison arms.

Why a grid rather than MCMC.  Three parameters, bounded, with a cheap forward
model once the vascular response is tabulated (mech.VascularResponse). A dense
grid gives deterministic normalization and marginals, so the arm comparison does
not depend on sampler convergence.

The arms, and what distinguishes them
-------------------------------------
swe        SWE only.  Estimates (mu, eta, s_swe): the mechanical data alone
           cannot separate a vascular from a matrix change, so s appears here
           as a mechanical nuisance.
ceus       CEUS only.  Estimates s_ceus from the AUC; the matrix
           parameters stay at their priors, since transport does not see them.
           No contrast-kinetic estimator is involved - see vmconf/ceus.py.
joint_free Both datasets, no coupling: separate s_swe and s_ceus. Sharing one s
           would itself impose the cross-modality relation, so the CEUS radius scale is
           marginalized independently in this control.
joint_shared Both datasets with the coupling: one radius scale generates both,
             with stated uncertainty in the relation.
joint_wrong The coupling applied with a deliberate offset, to test whether an
           incorrect relation produces a narrow but displaced posterior.
perfect_model Radius scale fixed at the truth.  A diagnostic upper benchmark,
              not an evidence-matched comparator.
"""
from __future__ import annotations

import numpy as np

from . import likelihood as LK


class Grid3:
    def __init__(self, mu, eta, s):
        self.mu, self.eta, self.s = mu, eta, s
        self.shape = (len(mu), len(eta), len(s))

    def marginal(self, logp, axis):
        p = np.exp(logp - logp.max())
        p /= p.sum()
        keep = [0, 1, 2]
        keep.remove(axis)
        return p.sum(axis=tuple(keep))

    def summary(self, logp, axis, level=0.90):
        x = (self.mu, self.eta, self.s)[axis]
        m = self.marginal(logp, axis)
        med = _quantile(x, m, 0.5)
        lo = _quantile(x, m, 0.5 - level / 2)
        hi = _quantile(x, m, 0.5 + level / 2)
        return dict(median=float(med), lo=float(lo), hi=float(hi),
                    width=float(hi - lo), mean=float(np.sum(x * m)))


def _quantile(x, w, q):
    """Discrete quantile evaluated at bin centres.

    Using the plain cumulative sum puts the quantile at the bin edge, so a
    marginal concentrated on one grid point returns the midpoint between that
    point and its neighbour.  Subtracting half the local weight places the
    cumulative at the centre of each bin, which returns the point itself.
    """
    w = np.asarray(w, float)
    tot = w.sum()
    if tot <= 0:
        return float(np.nan)
    w = w / tot
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q, c, x))


def swe_predict(resp, omega, mu, eta, s, dG, rho=1050.0, with_alpha=False):
    """Phase velocity (and attenuation) from the precomputed response."""
    from . import mech
    gm = mech.kelvin_voigt(omega, mu, eta)
    gv = resp(omega, mu, s, dG)
    c, a = mech.shear_wave(gm + gv, omega, rho)
    return np.concatenate([c, a]) if with_alpha else c


class SwePredictor:
    """Precomputed SWE values on the whole (mu, eta, s) grid.

    These simulated values do not depend on the data, so they are built once and reused
    for every noise realization and every arm.
    """

    def __init__(self, grid, resp, omega, dG, rho=1050.0, with_alpha=False):
        mu = np.asarray(grid.mu)[:, None, None, None]
        eta = np.asarray(grid.eta)[None, :, None, None]
        ss = np.asarray(grid.s)[None, None, :, None]
        om = np.asarray(omega)[None, None, None, :]
        a = (resp.mu_ref / mu) * ss ** -4.0                  # (nmu,1,ns,1)
        x = np.clip(om * a, resp.x[0], resp.x[-1])
        gv = dG * (np.interp(x, resp.x, resp.re) + 1j * np.interp(x, resp.x, resp.im))
        g = (mu + 1j * om * eta) + gv
        k = om * np.sqrt(rho / g)
        k = np.where(k.real < 0, -k, k)
        c = om / k.real
        self.pred = np.concatenate([c, -np.minimum(k.imag, 0.0)], axis=3) \
            if with_alpha else c
        self.shape = grid.shape

    def loglik(self, obs, sigma_rel):
        """Vectorised independent relative Gaussian log-likelihood."""
        s = sigma_rel * np.abs(self.pred)
        s = np.where(s > 0, s, 1e-30)
        r = (np.asarray(obs)[None, None, None, :] - self.pred) / s
        return -0.5 * np.sum(r ** 2, axis=3) - np.sum(np.log(s), axis=3)


def log_posterior(grid, arm, data, swe_pred, sigma_swe, sigma_ceus,
                  ceus_interp=None, s_true=None, relation_sd=0.0,
                  relation_offset=0.0):
    """Log posterior on the (mu, eta, s) grid for one arm.

    `data` holds the noisy observables actually available to the arm.
    """
    ns = grid.shape[2]
    if arm in ("swe", "joint_free", "joint_shared", "joint_wrong", "perfect_model"):
        lp = swe_pred.loglik(data["swe"], sigma_swe)
    else:
        lp = np.zeros(grid.shape)

    # CEUS term: depends on s only
    if arm in ("ceus", "joint_free", "joint_shared", "joint_wrong"):
        A = np.atleast_1d(ceus_interp(grid.s))
        ll = np.array([LK.gaussian_loglik(data["ceus"], np.array([A[k]]),
                                          sigma_ceus) for k in range(ns)])
        if arm == "joint_free":
            # separate s per modality: the CEUS likelihood is marginalised over
            # its own s, so it contributes a constant and cannot constrain the
            # mechanical s. This preserves the uncoupled control.
            lp = lp + float(_logsumexp(ll) - np.log(ns))
        else:
            if arm == "joint_wrong":
                # the coupling relation is displaced: the CEUS likelihood is
                # evaluated at a shifted radius scale
                A2 = np.atleast_1d(ceus_interp(np.clip(grid.s + relation_offset, 1e-6, None)))
                ll = np.array([LK.gaussian_loglik(data["ceus"], np.array([A2[k]]),
                                                  sigma_ceus) for k in range(ns)])
            if relation_sd > 0:
                ll = _smooth_in_s(ll, grid.s, relation_sd)
            lp = lp + ll[None, None, :]

    if arm == "perfect_model":
        if s_true is None:
            raise ValueError("perfect_model arm requires s_true")
        k0 = int(np.argmin(np.abs(grid.s - s_true)))
        mask = np.full(grid.shape, -np.inf)
        mask[:, :, k0] = 0.0
        lp = lp + mask
    return lp


def _logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))


def _smooth_in_s(ll, s, sd):
    """Carry uncertainty in the coupling relation by convolving the CEUS
    likelihood along s.  A coupling known only to within `sd` cannot pin the
    radius scale more tightly than that."""
    w = np.exp(-0.5 * ((s[:, None] - s[None, :]) / sd) ** 2)
    w /= w.sum(axis=1, keepdims=True)
    p = np.exp(ll - ll.max())
    return np.log(np.maximum(w @ p, 1e-300)) + ll.max()
