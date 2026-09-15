"""What the study holds fixed, and what that costs.

"Held fixed" is not the same as "free of consequence".  For each quantity the
model treats as known, this reports how far the prediction moves, in noise
standard deviations, when it is wrong by ten percent.  A large number does not
mean the study is wrong; it means the conclusion is conditional on that quantity
and the condition is stated rather than hidden.

Also reported: the nondimensional groups that say what regime the three windows
are in, and the convergence of the finite-difference Jacobian every diagnostic
depends on.
"""
from __future__ import annotations
import numpy as np

from .acquisition import Acquisition
from .constants import (GAMMA_NOMINAL, TRANSMIT_HZ, TRACK_KERNEL_M,
                        Y_ATTEN, nominal_ratio, psf_widths)
from .infer import diagnostics as DG
from .model import Inversion, WINDOW_FREE, level_free
from .models import bmode as BM, ceus as CE


def _full(level="shared"):
    free = sum([WINDOW_FREE[w] for w in ("bmode", "swe", "ceus")], ())
    return Inversion(free=free + level_free(level), level=level)


def groups():
    """Nondimensional groups at the synthetic truth."""
    inv = _full()
    p, a = inv.truth_physical(), inv.acq
    w0 = p["w0"]
    out = {}

    # --- B-mode -----------------------------------------------------------
    W = psf_widths(w0)
    out["blur over margin width, w_B / sig_e"] = float(W["bmode"]) / p["sig_e"]
    out["margin width over half-width, sig_e / R"] = p["sig_e"] / p["R"]
    out["quadrature sum over margin, sig_obs / sig_e"] = float(
        BM.observed_width(p["sig_e"], W["bmode"]) / p["sig_e"])
    out["profile samples per observed margin"] = float(
        BM.observed_width(p["sig_e"], W["bmode"])
        / ((a.bmode.x1 - a.bmode.x0) / (a.bmode.n_x - 1)))

    # --- SWE --------------------------------------------------------------
    lam = p["c_s"] / a.swe.f0
    out["shear wavelength (mm)"] = lam * 1e3
    out["aperture in wavelengths"] = (a.swe.x1 - a.swe.x0) / lam
    out["blur in wavelengths, w_S / lambda"] = float(W["swe"]) / lam
    out["blur attenuation at f0, exp(-(k w)^2/2)"] = float(
        np.exp(-0.5 * (2 * np.pi * a.swe.f0 / p["c_s"] * float(W["swe"])) ** 2))
    out["medium attenuation over the aperture"] = float(
        np.exp(-p["alpha0"] * (a.swe.x1 - a.swe.x0)))

    # --- CEUS -------------------------------------------------------------
    t1, t2 = a.ceus.t_first - p["t0"], a.ceus.t_last - p["t0"]
    wc = float(W["ceus"])
    out["blur share of variance, first frame"] = float(
        wc ** 2 / CE.observed_variance(p["D"], t1, wc))
    out["blur share of variance, last frame"] = float(
        wc ** 2 / CE.observed_variance(p["D"], t2, wc))
    out["Peclet over the window, v L / D"] = float(
        p["v"] * (a.ceus.x1 - a.ceus.x0) / p["D"])
    out["transit fraction, v T / L"] = float(
        p["v"] * t2 / (a.ceus.x1 - a.ceus.x0))
    return {k: float(v) for k, v in out.items()}


def fixed_screening(rel=0.10):
    """Prediction movement, in noise standard deviations, per ten percent error
    in each quantity the model holds fixed."""
    inv = _full()
    p0 = inv.truth_physical()
    y0, sig = inv.predict(inv.space.x0(inv.truth)), inv.sigma()
    out = {}

    # the attenuation power-law exponent
    import operatorid.constants as C
    saved = C.Y_ATTEN
    try:
        from operatorid.models import swe as SW
        SW.Y_ATTEN = saved * (1 + rel)
        y = inv.predict(inv.space.x0(inv.truth))
        out["Y_atten"] = float(np.linalg.norm((y - y0) / sig))
    finally:
        SW.Y_ATTEN = saved

    # each sequence-specific correction, one at a time
    for m in GAMMA_NOMINAL:
        g = dict(GAMMA_NOMINAL)
        g[m] = g[m] * (1 + rel)
        inv_g = Inversion(free=inv.free, level="shared", gamma=g)
        y = inv_g.predict(inv_g.space.x0(inv_g.truth))
        out["gamma_%s" % m] = float(np.linalg.norm((y - y0) / sig))
    # and the tracking kernel, which is the nonlinear part of the shear map
    from . import constants as _C
    saved_k = _C.TRACK_KERNEL_M
    try:
        _C.TRACK_KERNEL_M = saved_k * (1 + rel)
        y = inv.predict(inv.space.x0(inv.truth))
        out["track_kernel"] = float(np.linalg.norm((y - y0) / sig))
    finally:
        _C.TRACK_KERNEL_M = saved_k

    # the coordinate scale
    q = dict(inv.truth)
    q["scale"] = q.get("scale", 1.0) * (1 + rel)
    inv2 = Inversion(free=inv.free, level=inv.level, truth=q)
    y = inv2.predict(inv2.space.x0(q))
    out["scale"] = float(np.linalg.norm((y - y0) / sig))
    return out


def width_map(w0_values=(0.15e-3, 0.35e-3, 0.80e-3)):
    """The width of each window against the aperture component.

    The shear map is nonlinear, so its ratio to ``w0`` is not a constant and is
    reported at several apertures rather than quoted once.  This is the check
    that the implementation matches the documented expression."""
    out = {}
    for w0 in w0_values:
        W = psf_widths(w0)
        out["%.2f mm" % (w0 * 1e3)] = {
            m: dict(width_mm=float(W[m] * 1e3), ratio=float(W[m] / w0))
            for m in ("bmode", "swe", "ceus")}
    return out


def run():
    inv = _full()
    return dict(
        groups=groups(),
        fixed_screening=fixed_screening(),
        width_map=width_map(),
        transmit_hz=dict(TRANSMIT_HZ),
        track_kernel_m=TRACK_KERNEL_M,
        y_atten=Y_ATTEN,
        jacobian_step_convergence=DG.jacobian_convergence(inv),
        window_sensitivity=DG.window_sensitivity(inv),
        structural_rank={k: v for k, v in
                         [("rank", DG.structural_rank(inv)["rank"]),
                          ("n_par", DG.structural_rank(inv)["n_par"])]},
    )
