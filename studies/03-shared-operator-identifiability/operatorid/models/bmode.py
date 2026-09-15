"""B-mode: an echogenicity profile through a lesion.

Ideal field.  A lesion of half-width ``R`` centered at ``x_c``, whose boundary
is not a step but has an intrinsic width ``sig_e``: the transition from lesion
to background is a Gaussian-smoothed edge.  That width is the quantity of
interest.  It is an anatomical property, the sharpness of the margin, and a
pushing margin and an infiltrative one differ in it.

    f_B(x) = b0 + db * [ Phi((x - x_c + R)/sig_e) - Phi((x - x_c - R)/sig_e) ]

Observation.  The reconstructed image is the ideal profile convolved with the
system point spread, taken as Gaussian of width ``w``.  A Gaussian convolved
with a Gaussian is a Gaussian, so the observed profile has exactly the same
form with

    sig_obs = sqrt(sig_e^2 + w^2).

**This is the exact degeneracy the study is built on.**  From a B-mode profile
alone, ``sig_e`` and ``w`` enter only through that quadrature sum: no amount of
signal to noise separates them, because they are the same function of the data.
The intrinsic margin width is structurally unidentifiable from B-mode alone.
It becomes identifiable exactly to the extent that something else determines
``w``.

What this is not.  The profile is an echogenicity template.  There is no
speckle, no scattering statistics, no envelope detection and no log compression;
`METHODS.md` says what each of those would add.  ``b0`` and ``db``
are in arbitrary intensity units and carry the system gain, which is why no
separate gain parameter appears: it would be redundant with ``db``.
"""
from __future__ import annotations
import numpy as np
from scipy.special import erf


def _phi(z):
    """Standard normal cumulative distribution."""
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def observed_width(sig_e, w):
    """sqrt(sig_e^2 + w^2): the only combination a B-mode profile determines."""
    return np.sqrt(np.asarray(sig_e, float) ** 2 + np.asarray(w, float) ** 2)


def profile(x, p, w, scale=1.0):
    """Reconstructed echogenicity profile [arbitrary intensity units].

    ``scale`` maps true lengths to reconstructed ones; the point spread is a
    property of the imaging system and is already in reconstructed length, so
    it is not scaled.  The main experiment uses ``scale = 1`` by defining
    lengths as reconstructed lengths (`METHODS.md`)."""
    x = np.asarray(x, float)
    s = observed_width(scale * p["sig_e"], w)
    left = (x - scale * (p["x_c"] - p["R"])) / s
    right = (x - scale * (p["x_c"] + p["R"])) / s
    return p["b0"] + p["db"] * (_phi(left) - _phi(right))


def ideal(x, p, scale=1.0):
    """The same profile with no point spread at all."""
    return profile(x, p, 0.0, scale)
