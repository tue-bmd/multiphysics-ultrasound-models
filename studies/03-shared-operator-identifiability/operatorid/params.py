"""Parameters, transforms and marginal priors.

Every positive quantity is carried in natural logarithm, so that sensitivities,
Jacobian null spaces and Fisher eigenvectors are dimensionless and read directly
as products of powers.  Positions and times are carried linearly.

A marginal prior is a nominal value and a 90 percent interval: for a log
parameter a lognormal with log-mean the interval midpoint and log-sd the
half-width over 1.6449, for a linear parameter a normal built the same way.
The prior median is therefore the midpoint of the stated interval and **not**
the synthetic truth, which would flatter the coverage study.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

Z90 = 1.6448536269514722          # standard normal 95th percentile

#: What the ``role`` of an entry means.  The distinction is where the
#: information came from, because the profile levels are built on it.
#:
#:   free        estimated from the observations, with a broad prior
#:   nuisance    estimated, not reported as a result, not measured elsewhere
#:   operator    a nuisance that acts inside the observation operator rather
#:               than as an additive error.  Reported separately because the
#:               whole study is about one of them
#:   calibrated  measured independently of these observations
#:   fixed       held at a constant, with its consequence screened
ROLES = ("free", "nuisance", "operator", "calibrated", "fixed")


@dataclass(frozen=True)
class Entry:
    name: str
    transform: str                # "log" or "lin"
    nominal: float
    lo: float                     # 5th percentile of the marginal prior
    hi: float                     # 95th percentile
    role: str
    kind: str = "normal"          # normal | uniform (in the transformed space)

    def to_x(self, value):
        return np.log(value) if self.transform == "log" else value

    def to_value(self, x):
        return np.exp(x) if self.transform == "log" else x

    @property
    def mean(self):
        return 0.5 * (self.to_x(self.lo) + self.to_x(self.hi))

    @property
    def sd(self):
        return (self.to_x(self.hi) - self.to_x(self.lo)) / (2 * Z90)

    #: width of the soft wall of a uniform prior, as a fraction of its interval
    WALL = 0.02

    def log_pdf(self, x):
        """Log density up to a constant.  A uniform prior gets a soft wall:
        flat inside, quadratic outside over two percent of the interval, so the
        objective stays differentiable without changing anything inside."""
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


#: The catalogue.
#:
#: ``w0`` is the only quantity any two modalities have in common.  The tissue
#: parameters are deliberately **not** shared between modalities: this study
#: isolates operator sharing, and the physical coupling between tissue
#: parameters is the subject of the separate identifiability package.  With no
#: tissue coupling in either arm of the main comparison, any difference between
#: the arms is attributable to the operator and to nothing else.
CATALOGUE = {e.name: e for e in [
    # --- the shared operator parameter -----------------------------------
    _e("w0",     "log", 0.35e-3, 0.12e-3, 1.00e-3, "operator"),

    # --- B-mode: anatomy and its uncalibrated intensity -------------------
    _e("x_c",    "lin", 12.0e-3, 9.0e-3, 15.0e-3, "free"),
    _e("R",      "log",  5.0e-3, 2.0e-3, 12.0e-3, "free"),
    _e("sig_e",  "log",  0.6e-3, 0.10e-3, 4.0e-3, "free"),
    # B-mode intensity is not absolutely calibrated; these two carry the gain
    # rather than a separate gain factor, which would be redundant with them
    _e("b0",     "log",  1.0,    0.3,    3.0,    "nuisance"),
    _e("db",     "log",  0.5,    0.1,    2.0,    "nuisance"),

    # --- SWE ---------------------------------------------------------------
    _e("c_s",    "log",  2.0,    1.0,    5.0,    "free"),
    _e("alpha0", "log", 40.0,    5.0,  400.0,    "free"),
    _e("a_S",    "log", 10.0e-6, 1.0e-6, 1.0e-4, "nuisance"),

    # --- CEUS --------------------------------------------------------------
    _e("v",      "log",  1.0e-3, 0.3e-3, 4.0e-3, "free"),
    _e("D",      "log",  1.0e-7, 1.0e-8, 1.0e-6, "free"),
    _e("a_C",    "log",  1.0,    0.2,    5.0,    "nuisance"),
    _e("t0",     "lin",  1.0,    0.0,    4.0,    "nuisance", "uniform"),

    # --- sequence-specific operator corrections ----------------------------
    # Everything the frequency ratio does not capture: aperture, focusing,
    # apodization, excitation, receive processing, and for contrast the
    # nonlinear bubble response and its pulse sequence.  Nominally one.  The
    # ``shared_calibrated`` arm estimates them under these calibration priors
    # instead of assuming them; the other arms hold them at one, which is an
    # assumption and is labelled as one.
    _e("gamma_swe",  "log", 1.0, 1.0 / 1.10, 1.10, "calibrated"),
    _e("gamma_ceus", "log", 1.0, 1.0 / 1.10, 1.10, "calibrated"),

    # --- the coordinate scale, used only by the sensitivity arm ------------
    # An uncertain reconstructed-to-true length scale.  It is an exact symmetry
    # direction of the whole model (see METHODS.md) and is held at
    # one in the main experiment by *defining* lengths as reconstructed lengths,
    # not by fixing an uncertain quantity at its truth.  The sensitivity arm
    # frees it under this calibrated prior, equally in every comparison arm.
    _e("scale",  "log",  1.0,    1.0 / 1.02, 1.02, "calibrated"),
]}

#: The quantities reported as results.  ``sig_e`` is the one the mechanism is
#: expected to reach: it is exactly unidentifiable from B-mode alone.
REPORT = ("sig_e", "R", "x_c", "c_s", "alpha0", "v", "D", "w0")

#: Per-modality point-spread widths, as free parameters of the independent arm.
PSF_NAMES = {"bmode": "w_bmode", "swe": "w_swe", "ceus": "w_ceus"}


def psf_variants():
    """Independent per-window point-spread widths whose marginal priors match
    what the shared parameter induces.

    Sharing introduces a *relationship* between the three widths.  It must not
    also introduce a different marginal, or the comparison would confound the
    relationship with a tighter prior.

    The map ``w0 -> w_m`` is monotone in every window, so the 5th and 95th
    percentiles of ``w_m`` are the images of those of ``w0``, exactly.  For
    B-mode and contrast the map is linear and the induced marginal is lognormal.
    For the shear window it is not: the tracking kernel adds in quadrature, so
    the induced marginal is only approximately lognormal and this reproduces its
    90 percent interval exactly and its shape approximately.  That approximation
    affects the matched control's shear width, which no reported quantity
    depends on, and it is recorded here rather than left implicit."""
    from .constants import psf_widths
    e = CATALOGUE["w0"]
    lo, hi, nom = psf_widths(e.lo), psf_widths(e.hi), psf_widths(e.nominal)
    out = {}
    for m in PSF_NAMES:
        out[PSF_NAMES[m]] = Entry(PSF_NAMES[m], "log", float(nom[m]),
                                  float(lo[m]), float(hi[m]), "operator",
                                  e.kind)
    return out


def variant(name, lo=None, hi=None, nominal=None, role=None, kind=None,
            catalogue=None):
    """A copy of an entry with a different interval or role."""
    cat = catalogue or CATALOGUE
    e = cat[name]
    if role is not None and role not in ROLES:
        raise ValueError("unknown role %r" % role)
    return Entry(e.name, e.transform,
                 e.nominal if nominal is None else nominal,
                 e.lo if lo is None else lo,
                 e.hi if hi is None else hi,
                 e.role if role is None else role,
                 e.kind if kind is None else kind)


def full_catalogue():
    """The catalogue plus the three independent point-spread widths."""
    out = dict(CATALOGUE)
    out.update(psf_variants())
    return out


@dataclass
class Space:
    """An ordered set of free parameters, with optional prior overrides."""
    names: list
    overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        cat = full_catalogue()
        for n in list(self.names) + list(self.overrides):
            if n not in cat:
                raise KeyError("unknown parameter %r" % n)
        self._cat = cat

    def entry(self, name):
        return self.overrides.get(name, self._cat[name])

    @property
    def entries(self):
        return [self.entry(n) for n in self.names]

    def __len__(self):
        return len(self.names)

    def index(self, name):
        return self.names.index(name)

    def x0(self, truth=None):
        t = truth or {}
        return np.array([self.entry(n).to_x(t.get(n, self.entry(n).nominal))
                         for n in self.names], float)

    def to_dict(self, x):
        return {n: self.entry(n).to_value(xi) for n, xi in zip(self.names, x)}

    def log_prior(self, x):
        return float(sum(self.entry(n).log_pdf(xi)
                         for n, xi in zip(self.names, x)))

    def prior_sd(self):
        return np.array([self.entry(n).sd for n in self.names], float)

    def prior_mean(self):
        return np.array([self.entry(n).mean for n in self.names], float)

    def sample_prior(self, rng, n):
        return np.column_stack([self.entry(nm).sample(rng, n)
                                for nm in self.names])


#: The synthetic truth.
TRUTH = {n: e.nominal for n, e in full_catalogue().items()}
