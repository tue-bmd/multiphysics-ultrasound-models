"""Data-generating models richer than the model used for inversion.

Used only in the model-discrepancy experiment.  Each departure is a physically
motivated feature that the reduced model does not contain:

swe    the push has a lateral extent instead of being a line source.  The
       medium is linear and invariant under translation along the observation
       line, so an extended push is an exact superposition of line sources and
       the response is the line-source response integrated against the push
       profile.  No approximation is involved.

ceus   two transport pathways instead of one, with different drift and
       dispersion, delivering one tracer amount between them.  This is the
       simplest departure from a single convection-dispersion description and is
       the one the vascular-network work makes likely.

relax  a distribution of consolidation coefficients instead of one, obtained by
       mixing over a lognormal spread of permeability.  Since the permeability
       enters the relaxation only through the consolidation coefficient, the
       mixture over the full response is identical to the mixture over the
       series, so no new algebra is needed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from . import mechanics as ME
from . import transport as TP


@dataclass
class Richer:
    push_width: float = 0.75e-3    # standard deviation of the lateral push [m]
    n_push: int = 21
    slow_fraction: float = 0.25    # tracer share of the slow pathway
    slow_v: float = 0.35           # its drift, as a fraction of v
    slow_D: float = 2.5            # its dispersion, as a multiple of D
    k_spread: float = 0.6          # lognormal spread of permeability
    n_k: int = 7

    # ------------------------------------------------------------------ SWE --
    def swe(self, cfg, p):
        if self.push_width <= 0:
            return ME.shear_displacement(cfg.positions(), cfg.times(), p,
                                         cfg.f_band, cfg.geometry)
        w = self.push_width
        off = np.linspace(-3 * w, 3 * w, self.n_push)
        wt = np.exp(-0.5 * (off / w) ** 2)
        wt /= wt.sum()
        x = cfg.positions()
        # distance from each source element to each observation point
        d = np.abs(x[:, None] - off[None, :]).ravel()
        d = np.clip(d, 1e-6, None)
        u = ME.shear_displacement(d, cfg.times(), p, cfg.f_band, cfg.geometry)
        u = u.reshape(len(x), len(off), -1)
        return np.einsum("xot,o->xt", u, wt)

    # ----------------------------------------------------------------- CEUS --
    def ceus(self, cfg, p):
        t = cfg.times()
        r = cfg.positions()
        f = self.slow_fraction
        fast = dict(p); fast["A"] = p["A"] * (1 - f)
        slow = dict(p); slow["A"] = p["A"] * f
        slow["vmag"] = p["vmag"] * self.slow_v
        slow["D"] = p["D"] * self.slow_D
        return TP.occupancy(r, t, fast) + TP.occupancy(r, t, slow)

    # ---------------------------------------------------------------- relax --
    def relax(self, cfg, p):
        t = cfg.times()
        if self.k_spread <= 0:
            return ME.relax_displacement(t, p)
        # Gauss-Hermite nodes over log k
        nodes, wts = np.polynomial.hermite_e.hermegauss(self.n_k)
        wts = wts / wts.sum()
        out = np.zeros_like(t)
        for nd, wt in zip(nodes, wts):
            q = dict(p)
            q["k"] = p["k"] * np.exp(self.k_spread * nd)
            out += wt * ME.relax_displacement(t, q)
        return out
