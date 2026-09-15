"""The inverse problem: free parameters, windows, operator sharing, likelihood.

Three sharing levels, and they differ in one thing only:

    independent   each window carries its own point-spread width, with the
                  marginal prior the shared parameter would induce for it
    shared        one width w0, entering window m as r_m w0
    oracle        the width is held at the generating value.  This is a
                  benchmark, not a candidate method: it uses knowledge no
                  experiment supplies

Everything else, the tissue model, the data, the noise model, the tissue priors
and the nuisance priors, is identical across the three.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .acquisition import Acquisition
from .constants import GAMMA_NOMINAL, SCALE, psf_widths
from .models import bmode as BM, ceus as CE, swe as SW
from .params import PSF_NAMES, Space, TRUTH

WINDOWS = ("bmode", "swe", "ceus")

#: How each arm treats the operator.
#:
#:   independent        three unrelated widths
#:   shared             one aperture component, sequence corrections held at one
#:   shared_calibrated  one aperture component, sequence corrections estimated
#:                      under calibration priors.  The arm closest to an applied setting:
#:                      the aperture is genuinely common, the rest of the
#:                      sequence-specific operator is measured imperfectly
#:   oracle             the generating operator, handed to the fit.  A benchmark
LEVELS = ("independent", "shared", "shared_calibrated", "oracle")

#: free parameters each window contributes, beyond the point spread
WINDOW_FREE = {
    "bmode": ("x_c", "R", "sig_e", "b0", "db"),
    "swe": ("c_s", "alpha0", "a_S"),
    "ceus": ("v", "D", "a_C", "t0"),
}

#: the sequence-specific corrections, by the window each belongs to
GAMMA_NAMES = {"swe": "gamma_swe", "ceus": "gamma_ceus"}


def level_free(level):
    """The operator parameters a sharing level makes free."""
    if level == "shared":
        return ("w0",)
    if level == "shared_calibrated":
        return ("w0",) + tuple(GAMMA_NAMES[m] for m in ("swe", "ceus"))
    if level == "independent":
        return tuple(PSF_NAMES[m] for m in WINDOWS)
    if level == "oracle":
        return ()
    raise ValueError("unknown level %r" % level)


@dataclass
class Inversion:
    free: tuple
    level: str = "shared"
    windows: tuple = WINDOWS
    acq: Acquisition = field(default_factory=Acquisition)
    truth: dict = field(default_factory=lambda: dict(TRUTH))
    overrides: dict = field(default_factory=dict)
    #: sequence-specific corrections this object uses when they are not free
    #: parameters.  The generating object is built with the *true* ones; every
    #: inversion arm except the oracle is built with the nominal ones, and the
    #: incorrect-sharing control is exactly the case where the two differ.
    gamma: dict = field(default_factory=lambda: dict(GAMMA_NOMINAL))
    #: noise standard deviations supplied from outside.  Every arm of a case is
    #: given the noise of the generating model, so that no arm's noise depends
    #: on what that arm assumes about the operator.
    sigma_fixed: object = None

    def __post_init__(self):
        self.free = tuple(self.free)
        self.windows = tuple(w for w in WINDOWS if w in self.windows)
        if self.level not in LEVELS:
            raise ValueError("unknown level %r" % self.level)
        for w in self.windows:
            if w not in WINDOWS:
                raise ValueError("unknown window %r" % w)
        self.space = Space(list(self.free), dict(self.overrides))
        self._sigma = None

    # ------------------------------------------------------------- mapping --
    def physical(self, x):
        """Full physical parameter dictionary at transformed vector ``x``."""
        p = dict(self.truth)
        p.update(self.space.to_dict(np.asarray(x, float)))
        return p

    def truth_physical(self):
        return self.physical(self.space.x0(self.truth))

    def width(self, p, window):
        """The point-spread width this configuration uses in one window [m].

        ``independent``        the window's own free width
        ``shared``             from the free aperture component, corrections at
                               their nominal values
        ``shared_calibrated``  from the free aperture component and the free
                               corrections
        ``oracle``             from the generating aperture component and the
                               generating corrections
        """
        if self.level == "independent":
            return p[PSF_NAMES[window]]
        if self.level == "oracle":
            return float(psf_widths(self.truth["w0"], self.gamma)[window])
        g = dict(self.gamma)
        if self.level == "shared_calibrated":
            for m, n in GAMMA_NAMES.items():
                if n in p:
                    g[m] = p[n]
        return float(psf_widths(p["w0"], g)[window])

    def scale(self, p):
        return p.get("scale", SCALE)

    # ------------------------------------------------------------- forward --
    def _window(self, name, p):
        a, s = self.acq, self.scale(p)
        w = self.width(p, name)
        if name == "bmode":
            return BM.profile(a.bmode.positions(), p, w, s).ravel()
        if name == "swe":
            return SW.displacement(a.swe.positions(), a.swe.times(), p, w,
                                   a.swe.f0, a.swe.bandwidth, s).ravel()
        if name == "ceus":
            return CE.frames(a.ceus.positions(), a.ceus.times(), p, w, s).ravel()
        raise ValueError(name)

    def predict(self, x):
        p = self.physical(x)
        return np.concatenate([self._window(w, p) for w in self.windows])

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
        """Diagonal noise standard deviations.

        Fixed at the truth so that the likelihood is Gaussian with known
        variance.  When ``sigma_fixed`` is supplied it is used unchanged: every
        arm of a case is given the noise of the **generating** model, because
        the contrast noise is a fraction of the contrast peak and the peak
        depends on the operator.  Letting each arm compute its own would make
        the noise depend on what that arm assumes, which is not a difference
        between analyses of the same data."""
        if self.sigma_fixed is not None:
            return np.asarray(self.sigma_fixed, float)
        if self._sigma is not None:
            return self._sigma
        a, p = self.acq, self.truth_physical()
        parts = []
        for w in self.windows:
            y = self._window(w, p)
            if w == "bmode":
                parts.append(np.full(y.size, a.bmode.sigma))
            elif w == "swe":
                parts.append(np.full(y.size, a.swe.sigma))
            else:
                parts.append(np.full(y.size, a.ceus.sigma_rel * max(y.max(), 1e-300)))
        self._sigma = np.concatenate(parts)
        return self._sigma

    def simulate(self, rng, x=None):
        """One noisy observation vector from *this* object.

        Used only by the generating object of `experiments.generate`; the arms
        never call it, because every arm of a case must analyze the same data
        rather than each generating its own."""
        x = self.space.x0(self.truth) if x is None else x
        return self.predict(x) + rng.normal(0.0, self.sigma())

    # ----------------------------------------------------------- posterior --
    def log_prior(self, x):
        return self.space.log_prior(np.asarray(x, float))

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

    def quantity(self, x, name):
        """Any physical quantity at ``x``, including the derived widths.

        ``w0`` is asked of every arm so that the three can be put on one axis,
        and what it means differs by arm.  The decision is made on the sharing
        level, not on whether the name happens to be in the parameter
        dictionary: the synthetic truth carries a value for every quantity, so
        looking it up there would silently return the generating value for an
        arm that never estimates it."""
        p = self.physical(x)
        if name in ("w_bmode", "w_swe", "w_ceus"):
            return self.width(p, name[2:])
        if name == "w0":
            if self.level in ("shared", "shared_calibrated"):
                return p["w0"]
            if self.level == "independent":
                # the B-mode width *is* the aperture component in this model,
                # so the independent arm's own B-mode width is what it knows
                # about w0.  Its contrast width is reported separately, and the
                # two together are the point of section 3
                return p[PSF_NAMES["bmode"]]
            return self.truth["w0"]         # oracle: held, not estimated
        return p[name]
