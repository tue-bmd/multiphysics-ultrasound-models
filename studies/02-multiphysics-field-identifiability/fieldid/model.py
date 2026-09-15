"""The inverse problem: free parameters, windows, prior, likelihood.

An ``Inversion`` binds a free-parameter space, a coupling level, an acquisition
and a set of observation windows.  Windows are concatenated into one observation
vector with a known diagonal noise covariance.  Nothing in this module knows
about ultrasound: observations are physical fields plus additive noise.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from . import coupling as CP
from .acquisition import Acquisition
from .constants import FIXED
from .params import CATALOGUE, Space, TRUTH
from .models import mechanics as ME
from .models import transport as TP
from .compat import trapezoid

WINDOWS = ("swe", "ceus", "relax")

#: which free parameters each window can possibly depend on
WINDOW_PARAMS = {
    "swe": ("mu", "eta_s", "phi", "k", "F0", "Tp"),
    "ceus": ("phi", "D", "vmag", "vth", "vaz", "A", "t0"),
    "relax": ("M", "S_v", "C_v", "k", "P", "L", "mu"),
}


@dataclass
class Inversion:
    free: tuple
    level: str = "independent"
    windows: tuple = ("swe", "ceus")
    acq: Acquisition = field(default_factory=Acquisition)
    truth: dict = field(default_factory=lambda: dict(TRUTH))
    corr: CP.CorrelatedPrior = None
    generator: object = None        # optional richer forward model for E7
    overrides: dict = field(default_factory=dict)   # prior overrides by name

    def __post_init__(self):
        self.free = tuple(self.free)
        self.windows = tuple(self.windows)
        for w in self.windows:
            if w not in WINDOWS:
                raise ValueError("unknown window %r" % w)
        if self.level not in CP.LEVELS:
            raise ValueError("unknown coupling level %r" % self.level)
        self.space = Space(list(self.free), dict(self.overrides))
        if self.level == "network" and self.corr is None:
            raise ValueError("the network level needs a CorrelatedPrior")
        self._sigma = None
        self._peak = None

    # ---------------------------------------------------------------- maps --
    def physical(self, x):
        """Full physical parameter dictionary at transformed vector ``x``."""
        p = dict(self.truth)
        p.update(self.space.to_dict(np.asarray(x, float)))
        p = CP.apply(self.level, p)
        p.update(FIXED)
        return p

    def truth_physical(self):
        return self.physical(self.space.x0(self.truth))

    # ------------------------------------------------------------- forward --
    def _window(self, name, p, gen=None):
        gen = gen if gen is not None else self.generator
        a = self.acq
        if name == "swe":
            if gen is not None and hasattr(gen, "swe"):
                u = gen.swe(a.swe, p)
            else:
                u = ME.shear_displacement(a.swe.positions(), a.swe.times(), p,
                                          a.swe.f_band, a.swe.geometry)
            return u.ravel()
        if name == "ceus":
            t = a.ceus.times()
            if gen is not None and hasattr(gen, "ceus"):
                b = gen.ceus(a.ceus, p)
            else:
                b = TP.occupancy(a.ceus.positions(), t, p,
                                 delivery=a.ceus.delivery)
            if a.ceus.normalized:
                area = trapezoid(b, t, axis=-1)
                b = b / np.where(area > 0, area, 1.0)[:, None]
            return b.ravel()
        if name == "relax":
            t = a.relax.times()
            if gen is not None and hasattr(gen, "relax"):
                return gen.relax(a.relax, p).ravel()
            return ME.relax_displacement(t, p).ravel()
        raise ValueError(name)

    def predict(self, x, gen=None):
        p = self.physical(x)
        return np.concatenate([self._window(w, p, gen) for w in self.windows])

    def window_slices(self):
        out, i = {}, 0
        p = self.truth_physical()
        for w in self.windows:
            n = self._window(w, p).size
            out[w] = slice(i, i + n)
            i += n
        return out

    # --------------------------------------------------------------- noise --
    def sigma(self):
        """Diagonal noise standard deviations, fixed at the truth so that the
        likelihood is Gaussian with known variance."""
        if self._sigma is not None:
            return self._sigma
        a = self.acq
        p = self.truth_physical()
        parts = []
        for w in self.windows:
            y = self._window(w, p)
            if w == "swe":
                parts.append(np.full(y.size, a.swe.sigma))
            elif w == "ceus":
                if a.ceus.normalized:
                    # first-order propagation of the absolute noise through the
                    # unit-area normalization; the neglected area term is
                    # O(1 / sqrt(n_t)) and is checked in the tests
                    t = a.ceus.times()
                    b = TP.occupancy(a.ceus.positions(), t, p,
                                     delivery=a.ceus.delivery)
                    area = trapezoid(b, t, axis=-1)
                    s_abs = a.ceus.sigma_rel * b.max()
                    parts.append(np.repeat(s_abs / np.where(area > 0, area, 1.0),
                                           len(t)))
                else:
                    parts.append(np.full(y.size, a.ceus.sigma_rel * max(y.max(), 1e-300)))
            else:
                parts.append(np.full(y.size, a.relax.sigma))
        self._sigma = np.concatenate(parts)
        return self._sigma

    def simulate(self, rng, x=None, gen=None):
        """One noisy observation vector."""
        x = self.space.x0(self.truth) if x is None else x
        return self.predict(x, gen) + rng.normal(0.0, self.sigma())

    # ----------------------------------------------------------- posterior --
    def log_prior(self, x):
        x = np.asarray(x, float)
        if self.corr is None:
            return self.space.log_prior(x)
        sub = [n for n in self.corr.names if n in self.free]
        if len(sub) != len(self.corr.names):
            return self.space.log_prior(x)
        d = self.space.to_dict(x)
        rest = sum(self.space.entry(n).log_pdf(xi)
                   for n, xi in zip(self.free, x) if n not in self.corr.names)
        return float(rest + self.corr.log_pdf(d, self.space.entry))

    def log_like(self, x, y):
        r = (y - self.predict(x)) / self.sigma()
        if not np.all(np.isfinite(r)):
            return -np.inf
        return float(-0.5 * np.dot(r, r))

    def log_post(self, x, y):
        lp = self.log_prior(x)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_like(x, y)

    # ------------------------------------------------------------ reporting -
    def quantity(self, x, name):
        """Any physical quantity, free or derived, at ``x``."""
        return self.physical(x)[name]
