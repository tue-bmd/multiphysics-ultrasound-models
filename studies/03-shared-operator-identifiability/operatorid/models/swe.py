"""SWE: a broadband shear pulse tracked along the lateral axis.

Ideal field.  A shear pulse launched at the origin, propagating in the positive
direction, written in the frequency domain with a Gaussian source spectrum:

    U(x, omega) = a_S S(omega) exp(-alpha(omega) x) exp(-i omega x / c_s)
    alpha(omega) = alpha0 (|omega| / omega0)^Y

with ``S`` centered at ``omega0``.  The convention is ``exp(+ i omega t)``, so
an outgoing wave in the positive direction is ``exp(- i k x)`` with
``k = omega / c_s``.  A single shear speed, no dispersion in the wave speed:
the frequency dependence is entirely in the attenuation, whose exponent ``Y``
is held at one and screened.

Observation.  Displacement is tracked on the same beams as the anatomy, so it
is smoothed in space by a point spread of width ``w``.  A spatial Gaussian
convolution multiplies the spatial spectrum by a Gaussian in wavenumber:

    U_obs(x, omega) = U(x, omega) exp( - (omega w / c_s)^2 / 2 ).

**The blur and the attenuation are separable in principle, and the reason
matters.**  Both suppress high frequencies, but the attenuation term grows with
propagation distance while the blur term does not depend on ``x`` at all.  One
observation position cannot separate them; a set of positions can, because only
one of the two changes across them.  At a single frequency the blur is instead
degenerate with the push amplitude ``a_S``, which is why a broadband pulse is
used: the blur imposes a specific Gaussian shape across frequency that a
constant amplitude cannot imitate.

The blur factor depends on ``c_s`` as well, through ``k = omega / c_s``.  That
is not an artifact of the parameterization: a fixed spatial kernel corresponds
to a different fraction of a wavelength in a faster medium.

What this is not.  There is no acoustic radiation force calculation, no
displacement-tracking estimator, no jitter model and no elastic boundary.  The
spatial convolution is a reduced stand-in for what tracking on finite beams does
to the recovered displacement field; `METHODS.md` states the gap.
"""
from __future__ import annotations
import numpy as np

from .. import constants as _C

#: read through the module so that the screening can vary it
Y_ATTEN = _C.Y_ATTEN


def _spectrum(omega, omega0, bandwidth_rad):
    """Gaussian source spectrum, centered at omega0."""
    return np.exp(-((omega - omega0) ** 2) / (2.0 * bandwidth_rad ** 2))


def blur_factor(omega, c_s, w):
    """exp(-(omega w / c_s)^2 / 2): a distance-independent low pass."""
    k = np.asarray(omega, float) / c_s
    return np.exp(-0.5 * (k * np.asarray(w, float)) ** 2)


def displacement(x, t, p, w, f0, bandwidth_hz, scale=1.0):
    """Tracked displacement u(x, t), shape (len(x), len(t)) [m].

    ``scale`` maps true lengths to reconstructed ones: a wave observed on a
    stretched grid appears to travel faster, so the shear speed enters as
    ``scale * c_s``.  The point spread is already a reconstructed length."""
    x = np.atleast_1d(np.asarray(x, float))
    t = np.asarray(t, float)
    n = t.size
    dt = float(t[1] - t[0])
    om = 2.0 * np.pi * np.fft.rfftfreq(n, dt)
    om0 = 2.0 * np.pi * f0
    c = scale * p["c_s"]
    s = _spectrum(om, om0, 2.0 * np.pi * bandwidth_hz)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(om > 0, np.abs(om) / om0, 0.0)
    atten_per_m = p["alpha0"] * ratio ** Y_ATTEN
    lp = blur_factor(om, c, w)
    # x enters only through the attenuation and the phase, so the transfer is
    # built once per position
    u = np.empty((x.size, n))
    for i, xi in enumerate(x):
        h = (p["a_S"] * s * lp
             * np.exp(-atten_per_m * xi)
             * np.exp(-1j * om * xi / c))
        u[i] = np.fft.irfft(h, n=n)
    return u


def ideal(x, t, p, f0, bandwidth_hz, scale=1.0):
    """The same field with no point spread."""
    return displacement(x, t, p, 0.0, f0, bandwidth_hz, scale)
