"""Experiment E0: nondimensional groups and sensitivity screening.

Screening covers the free parameters, the nuisance quantities and the
quantities held fixed.  A fixed quantity is reported by how far the prediction
moves, in units of the observation noise, when it is changed by ten percent.
That is the number that justifies fixing it; nothing is fixed silently.
"""
from __future__ import annotations
import numpy as np

from .acquisition import Acquisition
from .constants import FIXED
from .model import Inversion
from .models import mechanics as ME
from .infer import diagnostics as DG


def groups(p, f_probe=200.0, x_ap=10.0e-3, z_ceus=6.0e-3, t_acq=90.0,
           t_relax=30.0):
    """The nondimensional groups of METHODS.md, evaluated at ``p``."""
    om = 2 * np.pi * f_probe
    re = ME.rho_eff(np.array([om]), p["phi"], p["k"])
    ks = ME.shear_wavenumber(np.array([om]), p["mu"], p["eta_s"], re)[0]
    lam = 2 * np.pi / ks.real
    om_c = p["phi"] * p["eta_b"] / (p["rho_f"] * p["k"] * p["alpha_inf"])
    cv = ME.consolidation_coefficient(p)
    return {
        "loss_tangent": float(om * p["eta_s"] / p["mu"]),
        "biot_frequency_ratio": float(om / om_c),
        "biot_inertial_coupling": float(p["rho_f"] ** 2 * om * p["k"]
                                        / (p["rho"] * p["eta_b"])),
        "rho_eff_over_rho": float(abs(re[0]) / p["rho"]),
        "shear_wavelength_mm": float(lam * 1e3),
        "aperture_in_wavelengths": float(x_ap / lam),
        "aperture_attenuation": float(-ks.imag * x_ap),
        "peclet": float(p["vmag"] * z_ceus / p["D"]),
        "transit_fraction": float(p["vmag"] * t_acq / z_ceus),
        "storage_ratio": float(p["M"] * p["S_v"] / p["alpha"] ** 2),
        "relaxing_fraction": float(ME.relaxing_fraction(p)),
        "consolidation_coefficient_m2_s": float(cv),
        "consolidation_number": float(cv * t_relax / p["L"] ** 2),
        "consolidation_time_s": float(4 * p["L"] ** 2 / (np.pi ** 2 * cv)),
    }


def screen_fixed(inv, rel=0.10):
    """Movement of the prediction, in noise units, per ten percent change of
    each quantity held fixed."""
    x0 = inv.space.x0(inv.truth)
    base = inv.predict(x0)
    s = inv.sigma()
    out = {}
    for name, val in FIXED.items():
        t = dict(inv.truth)
        t[name] = val * (1 + rel)
        alt = Inversion(free=inv.free, level=inv.level, windows=inv.windows,
                        acq=inv.acq, truth=t, corr=inv.corr,
                        overrides=inv.overrides)
        # the constant must survive Inversion.physical, which overwrites with FIXED
        alt._force = {name: val * (1 + rel)}
        y = _predict_with(alt, x0, {name: val * (1 + rel)})
        out[name] = float(np.linalg.norm((y - base) / s))
    return out


def _predict_with(inv, x, force):
    p = inv.physical(x)
    p.update(force)
    return np.concatenate([inv._window(w, p) for w in inv.windows])


def run(seed=0):
    """E0 for the three windows separately and for the current protocol."""
    acq = Acquisition()
    acq.relax.enabled = True
    full = Inversion(free=("mu", "eta_s", "phi", "k", "D", "vmag", "vth", "vaz",
                           "S_v", "M", "F0", "Tp", "A", "t0", "P"),
                     windows=("swe", "ceus", "relax"), acq=acq)
    p = full.truth_physical()
    out = dict(groups=groups(p),
               truth={k: float(v) for k, v in p.items()
                      if isinstance(v, (int, float))},
               sensitivity=DG.sensitivity_table(full),
               window_sensitivity=DG.window_sensitivity(full),
               fixed_screening=screen_fixed(full),
               jacobian_step_convergence=DG.jacobian_convergence(
                   Inversion(free=("mu", "eta_s", "phi", "k"), windows=("swe",))),
               )
    sp = DG.spectrum(full)
    out["eigenvalues"] = sp["eigenvalues"].tolist()
    out["contraction"] = sp["contraction"].tolist()
    out["eig_labels_log"] = sp["labels_log"]
    sr = DG.structural_rank(full)
    out["rank"] = sr["rank"]
    out["n_par"] = sr["n_par"]
    out["null_labels_log"] = sr["null_labels_log"]
    return out
