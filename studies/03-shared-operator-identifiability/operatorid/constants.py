"""Quantities held fixed, and how the shared aperture reaches each sequence.

Everything here is a property of the acquisition that the study treats as known
or as calibrated.  `screening.py` reports what a ten percent error in each would
cost, so that "held fixed" is never confused with "free of consequence".
"""
from __future__ import annotations
import numpy as np

#: The spatial coordinate is the **reconstructed** one: lengths are read off the
#: beamformed grid.  Converting them to true physical lengths needs the scale
#: calibration of `METHODS.md`, which is an uncertain measurement
#: and is offered as a separate sensitivity arm, never assumed away.
SCALE = 1.0

#: Transmit frequencies of the three sequences [Hz].
TRANSMIT_HZ = {"bmode": 7.0e6, "swe": 5.0e6, "ceus": 2.8e6}

#: Correlation kernel length of the displacement tracker [m].  A fixed length,
#: not a fraction of the point spread, which is why it enters in quadrature and
#: makes the shear width a nonlinear function of the aperture component.
TRACK_KERNEL_M = 0.40e-3

#: Nominal values of the sequence-specific corrections.  One per sequence that
#: has one; see `psf_widths`.
GAMMA_NOMINAL = {"swe": 1.0, "ceus": 1.0}


def psf_widths(w0, gamma=None, track_kernel=None):
    """Point-spread width of each window, from the shared aperture component.

        w_B = w0
        w_C = gamma_C (f_B / f_C) w0
        w_S = sqrt( (gamma_S (f_B / f_S) w0)^2 + w_track^2 )

    ``w0`` is the aperture-limited resolution at the B-mode frequency and is the
    only quantity the three sequences have in common.  What each does with it is
    different, and is written here rather than assumed:

    * a focused aperture resolves in proportion to the wavelength, so a sequence
      transmitting at a lower frequency has a proportionally wider point spread;
    * displacement tracking additionally correlates over a fixed kernel length,
      which adds in quadrature and does **not** scale with the aperture.  The
      shear width is therefore a nonlinear function of ``w0``: at the nominal
      0.35 mm it is 1.807 times it, but the factor falls toward 1.4 for a wide
      aperture and rises without bound for a narrow one.

    **The frequency scaling is a nominal diffraction argument, not a
    measurement.**  A real cross-sequence point spread also depends on the
    aperture used, the focusing and apodization, the excitation, the receive
    processing, and for contrast on the nonlinear bubble response and the pulse
    sequence built around it.  The ``gamma_m`` carry everything the frequency
    ratio does not: they are the sequence-specific corrections, nominally one,
    and the ``shared_calibrated`` arm estimates them under calibration priors
    rather than assuming them.  See `METHODS.md`.
    """
    g = dict(GAMMA_NOMINAL)
    g.update(gamma or {})
    k = TRACK_KERNEL_M if track_kernel is None else track_kernel
    f = TRANSMIT_HZ
    w0 = np.asarray(w0, float)
    diff_swe = g["swe"] * (f["bmode"] / f["swe"]) * w0
    return {"bmode": w0,
            "swe": np.sqrt(diff_swe ** 2 + k ** 2),
            "ceus": g["ceus"] * (f["bmode"] / f["ceus"]) * w0}


def nominal_ratio(window, w0=0.35e-3):
    """The width over ``w0`` at a stated aperture, for reporting only.

    Exactly constant for B-mode and contrast; for the shear window it is a local
    linearization of `psf_widths` and is quoted with the ``w0`` it belongs to.
    """
    return float(psf_widths(w0)[window] / w0)


#: Shear attenuation power law exponent: alpha(omega) = alpha0 (omega/omega0)^Y.
Y_ATTEN = 1.0

UNITS = {
    "w0": "m", "x_c": "m", "R": "m", "sig_e": "m", "b0": "1", "db": "1",
    "c_s": "m/s", "alpha0": "Np/m", "a_S": "m", "v": "m/s", "D": "m^2/s",
    "a_C": "1", "t0": "s", "scale": "1",
    "gamma_swe": "1", "gamma_ceus": "1",
    "w_bmode": "m", "w_swe": "m", "w_ceus": "m",
}

FIXED = {"Y_atten": Y_ATTEN, "scale": SCALE, "track_kernel_m": TRACK_KERNEL_M}
