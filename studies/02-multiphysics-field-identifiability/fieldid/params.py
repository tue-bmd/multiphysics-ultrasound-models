"""Parameters, transforms and marginal priors.

Every positive quantity is carried in natural logarithm, so that sensitivities,
Jacobian null spaces and Fisher eigenvectors are dimensionless and read directly
as products of powers of the parameters.  Angles and arrival time are carried
linearly.

A marginal prior is specified by a nominal value and a 90 percent interval.  For
a log parameter this gives a lognormal with log-mean log(nominal) and log-sd
(log hi - log lo) / (2 * 1.6449); for a linear parameter a normal with the same
construction.  Uniform priors are available for angles.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

Z90 = 1.6448536269514722          # standard normal 95th percentile

#: What the ``role`` of an entry means.  The distinction that matters is where
#: the information came from, because the profile levels are built on it.
#:
#:   free        estimated from the observations, with a broad catalogue prior
#:   nuisance    estimated, not reported, and not measured elsewhere
#:   calibrated  **measured independently** of the imaging observations.  This
#:               is data from an auxiliary experiment and would sit in a
#:               likelihood if that experiment were written out.
#:   matched     **not measured at all.**  A marginal deliberately set equal to
#:               the one another configuration induces, so that two arms differ
#:               only in the structure under test.  It is prior information.
#:               Kept distinct from ``calibrated`` so that a matched control is
#:               never read as an auxiliary measurement.
#:   fixed       held at a constant, with its consequence screened in E0
#:   derived     computed from others by a coupling relation, never sampled
ROLES = ("free", "nuisance", "calibrated", "matched", "fixed", "derived")


@dataclass(frozen=True)
class Entry:
    name: str
    transform: str                # "log" or "lin"
    nominal: float
    lo: float                     # 5th percentile of the marginal prior
    hi: float                     # 95th percentile
    role: str                     # see ROLES
    kind: str = "normal"          # normal | uniform  (in the transformed space)

    # ---- transform -------------------------------------------------------
    def to_x(self, value):
        return np.log(value) if self.transform == "log" else value

    def to_value(self, x):
        return np.exp(x) if self.transform == "log" else x

    # ---- marginal prior in transformed space -----------------------------
    @property
    def mean(self):
        """Centre of the prior in transformed space.

        This is the midpoint of the stated interval, not the nominal value.
        The nominal value is the synthetic truth; placing the prior median
        exactly on it in every case would flatter the coverage study."""
        return 0.5 * (self.to_x(self.lo) + self.to_x(self.hi))

    @property
    def sd(self):
        return (self.to_x(self.hi) - self.to_x(self.lo)) / (2 * Z90)

    #: width of the soft wall of a uniform prior, as a fraction of its interval
    WALL = 0.02

    def log_pdf(self, x):
        """Log density up to a constant.

        A uniform prior is given a soft wall rather than a hard edge: flat
        inside the interval and quadratic outside, over a width of two percent
        of the interval.  A hard edge makes the objective minus infinity outside
        the box, which leaves the optimizer without a gradient and produces
        undefined differences; the soft wall keeps the objective finite and
        differentiable everywhere while changing nothing inside the interval."""
        if self.kind == "uniform":
            a, b = self.to_x(self.lo), self.to_x(self.hi)
            w = self.WALL * (b - a)
            if x < a:
                return -0.5 * ((a - x) / w) ** 2
            if x > b:
                return -0.5 * ((x - b) / w) ** 2
            return 0.0
        return -0.5 * ((x - self.mean) / self.sd) ** 2

    def sample(self, rng, size=None):
        if self.kind == "uniform":
            return rng.uniform(self.to_x(self.lo), self.to_x(self.hi), size)
        return rng.normal(self.mean, self.sd, size)


def _e(name, tr, nom, lo, hi, role, kind="normal"):
    if role not in ROLES:
        raise ValueError("unknown role %r" % role)
    return Entry(name, tr, nom, lo, hi, role, kind)


#: The full catalogue.  ``S_v`` is present with the marginal that the product
#: ``phi * C_v`` induces, so that the independent configuration and the coupled
#: configuration have matched marginal priors and differ only in correlation.
CATALOGUE = {e.name: e for e in [
    # --- tissue parameters ------------------------------------------------
    _e("mu",    "log", 3.0e3,   1.0e3,  1.0e4,  "free"),
    _e("eta_s", "log", 1.5,     0.1,    10.0,   "free"),
    _e("phi",   "log", 0.028,   0.005,  0.10,   "free"),
    _e("k",     "log", 3.5e-13, 1.0e-14, 1.0e-11, "free"),
    _e("D",     "log", 1.0e-6,  1.0e-7, 1.0e-5, "free"),
    _e("vmag",  "log", 1.0488088481701515e-3, 1.0e-4, 1.0e-2, "free"),
    _e("vth",   "lin", 0.30627736916966936, 0.0, np.pi, "free", "uniform"),
    _e("vaz",   "lin", 0.32175055439664224, -np.pi, np.pi, "free", "uniform"),
    _e("C_v",   "log", 1.0e-3,  1.0e-4, 1.0e-2, "free"),
    # induced marginal of phi * C_v: log-mean log(0.028 * 1e-3), log-sd the
    # quadrature sum of the two.  Computed in ``induced_Sv`` below and patched in.
    _e("S_v",   "log", 2.8e-5,  1.0e-6, 7.9e-4, "free"),
    _e("M",     "log", 9.6e4,   1.0e4,  1.0e7,  "free"),
    # --- nuisance ---------------------------------------------------------
    # calibrated so that the peak shear displacement over the observation
    # aperture is 10 micrometres, the order of a clinical push
    _e("F0",    "log", 4.513e-4, 4.5e-5, 4.5e-3, "nuisance"),
    _e("Tp",    "log", 2.0e-4,  5.0e-5, 8.0e-4, "nuisance"),
    _e("A",     "log", 1.0,     0.1,    10.0,   "nuisance"),
    _e("t0",    "lin", 4.0,     0.0,    15.0,   "nuisance", "uniform"),
    # --- calibrated -------------------------------------------------------
    _e("P",     "log", 500.0,   500.0 / 1.1, 500.0 * 1.1, "calibrated"),
    _e("L",     "log", 5.0e-3,  5.0e-3 / 1.1, 5.0e-3 * 1.1, "calibrated"),
]}


def induced_Sv():
    """Marginal of ``S_v = phi * C_v`` implied by the marginals of the factors."""
    a, b = CATALOGUE["phi"], CATALOGUE["C_v"]
    m = a.mean + b.mean
    s = float(np.hypot(a.sd, b.sd))
    return m, s


def _patch_Sv():
    """Give S_v the marginal that the product phi * C_v induces, keeping its
    nominal value at the product of the two nominal values."""
    m, s = induced_Sv()
    e = CATALOGUE["S_v"]
    CATALOGUE["S_v"] = Entry(e.name, e.transform,
                             CATALOGUE["phi"].nominal * CATALOGUE["C_v"].nominal,
                             float(np.exp(m - Z90 * s)), float(np.exp(m + Z90 * s)),
                             e.role, e.kind)


_patch_Sv()


def variant(name, lo=None, hi=None, nominal=None, role=None, kind=None):
    """A copy of a catalogue entry with a different prior interval or role.

    Used to move a quantity between the calibrated and nuisance roles without
    editing the catalogue, so that an experiment can ask what happens when, for
    example, the applied compressive stress is not measured."""
    e = CATALOGUE[name]
    if role is not None and role not in ROLES:
        raise ValueError("unknown role %r" % role)
    return Entry(e.name, e.transform,
                 e.nominal if nominal is None else nominal,
                 e.lo if lo is None else lo,
                 e.hi if hi is None else hi,
                 e.role if role is None else role,
                 e.kind if kind is None else kind)


@dataclass
class Space:
    """An ordered set of free parameters, with optional prior overrides."""
    names: list
    overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        for n in self.names:
            if n not in CATALOGUE:
                raise KeyError("unknown parameter %r" % n)
        for n in self.overrides:
            if n not in CATALOGUE:
                raise KeyError("unknown parameter %r" % n)

    def entry(self, name):
        return self.overrides.get(name, CATALOGUE[name])

    @property
    def entries(self):
        return [self.entry(n) for n in self.names]

    def __len__(self):
        return len(self.names)

    def x0(self, truth=None):
        """Transformed vector at the nominal values, or at ``truth``."""
        t = truth or {}
        return np.array([self.entry(n).to_x(t.get(n, self.entry(n).nominal))
                         for n in self.names], float)

    def to_dict(self, x):
        return {n: self.entry(n).to_value(xi) for n, xi in zip(self.names, x)}

    def log_prior(self, x):
        return float(sum(self.entry(n).log_pdf(xi) for n, xi in zip(self.names, x)))

    def prior_sd(self):
        return np.array([self.entry(n).sd for n in self.names], float)

    def prior_mean(self):
        return np.array([self.entry(n).mean for n in self.names], float)

    def sample_prior(self, rng, n):
        return np.column_stack([self.entry(nm).sample(rng, n) for nm in self.names])

    def index(self, name):
        return self.names.index(name)


#: The synthetic truth used throughout unless a config overrides it.
TRUTH = {n: e.nominal for n, e in CATALOGUE.items()}
TRUTH["S_v"] = TRUTH["phi"] * TRUTH["C_v"]
