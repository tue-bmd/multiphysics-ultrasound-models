"""Fixed physical quantities, with units.

Nothing here is silently assumed: every entry is screened in experiment E0 and
the screening result is reported, so that "fixed" means "checked and fixed", not
"ignored".
"""
from __future__ import annotations

FIXED = {
    "rho": 1050.0,        # tissue bulk density                       [kg m^-3]
    "rho_f": 1060.0,      # blood density                             [kg m^-3]
    "eta_b": 3.5e-3,      # blood viscosity                           [Pa s]
    "alpha_inf": 1.0,     # inertial tortuosity of the vascular space [-]
    "alpha": 1.0,         # Biot effective-stress coefficient         [-]
}

UNITS = {
    "mu": "Pa", "eta_s": "Pa s", "phi": "-", "k": "m^2", "D": "m^2 s^-1",
    "vmag": "m s^-1", "vth": "rad", "vaz": "rad", "C_v": "Pa^-1", "S_v": "Pa^-1",
    "M": "Pa", "F0": "N m^-1", "Tp": "s", "A": "arb", "t0": "s",
    "P": "Pa", "L": "m",
    "rho": "kg m^-3", "rho_f": "kg m^-3", "eta_b": "Pa s",
    "alpha_inf": "-", "alpha": "-",
}
