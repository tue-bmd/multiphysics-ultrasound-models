"""CEUS: a contrast bolus carried by advection and dispersion.

Ideal field.  A one-dimensional bolus released at the origin at time ``t0``,
carried at effective drift ``v`` and spread by effective dispersion ``D``:

    c(x, t) = a_C / sqrt(4 pi D dt) * exp( -(x - v dt)^2 / (4 D dt) ),  dt = t - t0

``v`` and ``D`` are effective transport coefficients of the reduced model, not
derived from a vascular network; the network is the subject of a separate
package and nothing here depends on it.

Observation.  The reconstructed image is the ideal field convolved in space
with the system point spread, Gaussian of width ``w``.  A Gaussian convolved
with a Gaussian is a Gaussian, so the observed profile is the same form with

    var_obs(t) = 2 D (t - t0) + w^2.

**This is where the shared width becomes identifiable.**  Dispersion and blur
both widen the bolus, but they widen it differently in time: the dispersive part
grows linearly with elapsed time while the point spread contributes a constant.
A single frame sees only their sum and cannot separate them; a sequence of
frames sees a straight line whose slope is ``2 D`` and whose intercept is
``w^2``.  The intercept is the information this study transfers.

That also fixes the information-removal control: restricting the contrast
window to one frame must remove the separation, and with it any benefit from
sharing.  This is a prediction of the algebra above, not an empirical hope, and
it is asserted as a test.

What this is not.  There is no microbubble acoustics, no nonlinear pulse
sequence, no attenuation of the contrast signal with depth and no bubble
destruction.  ``a_C`` is in arbitrary units and carries the injected dose and
the system gain together.
"""
from __future__ import annotations
import numpy as np


def observed_variance(D, dt, w):
    """2 D dt + w^2: the observed spatial variance of the bolus [m^2]."""
    return 2.0 * np.asarray(D, float) * np.asarray(dt, float) + np.asarray(w, float) ** 2


def frames(x, t, p, w, scale=1.0):
    """Reconstructed contrast frames, shape (len(t), len(x)) [arbitrary units].

    Frames before ``t0`` are identically zero: the bolus has not arrived.
    ``scale`` maps true lengths to reconstructed ones, so it multiplies the
    drift and squares into the dispersion; the point spread is already a
    reconstructed length."""
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    dt = t - p["t0"]
    out = np.zeros((t.size, x.size))
    live = dt > 0
    if not np.any(live):
        return out
    var = observed_variance(scale ** 2 * p["D"], dt[live], w)
    mu = scale * p["v"] * dt[live]
    z = (x[None, :] - mu[:, None]) ** 2 / (2.0 * var[:, None])
    out[live] = p["a_C"] / np.sqrt(2.0 * np.pi * var[:, None]) * np.exp(-z)
    return out


def ideal(x, t, p, scale=1.0):
    """The same frames with no point spread."""
    return frames(x, t, p, 0.0, scale)
