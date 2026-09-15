"""Mechanical physical fields.

Fourier convention: f(t) = (1/2pi) int f_hat(omega) exp(+ i omega t) d omega, so
d/dt -> + i omega, G*(omega) = mu + i omega eta_s, and an outgoing wave in the
+ x direction is exp(- i k x) with Im(k) <= 0.

Two windows:

M1  the shear wave of the current SWE protocol.  The divergence-free part of the
    displacement carries no pore pressure and couples to the vascular fluid only
    through inertial drag, which enters as an effective density.  All of the
    sensitivity of this window to the permeability lives in that one term, and it
    is computed rather than assumed.

M2  consolidation relaxation after a step compression, used only as a
    contingency observation.  Confined slab, drained at z = 0, sealed at z = L.
    See METHODS.md: M is an effective constrained
    modulus with lateral relief, not the confined modulus of a nearly
    incompressible skeleton.
"""
from __future__ import annotations
import numpy as np
from scipy.special import hankel2

from ..constants import FIXED


# ---------------------------------------------------------------- M1 --------

def rho_eff(omega, phi, k, rho=None, rho_f=None, eta_b=None, alpha_inf=None):
    """Biot effective density for the shear wave [kg m^-3].

    rho_tilde = alpha_inf rho_f / phi - i eta_b / (omega k),
    rho_eff   = rho - rho_f^2 / rho_tilde.

    At omega = 0 the drag term diverges and rho_eff = rho exactly; the same
    limit is reached as k -> 0 or eta_b -> infinity.
    """
    rho = FIXED["rho"] if rho is None else rho
    rho_f = FIXED["rho_f"] if rho_f is None else rho_f
    eta_b = FIXED["eta_b"] if eta_b is None else eta_b
    alpha_inf = FIXED["alpha_inf"] if alpha_inf is None else alpha_inf
    omega = np.asarray(omega, float)
    out = np.full(omega.shape, rho, dtype=complex)
    nz = omega != 0
    if k <= 0 or eta_b == np.inf:
        return out
    rt = alpha_inf * rho_f / phi - 1j * eta_b / (omega[nz] * k)
    out[nz] = rho - rho_f ** 2 / rt
    return out


def shear_wavenumber(omega, mu, eta_s, re):
    """k_s = omega sqrt(rho_eff / G*), branch with Im(k_s) <= 0 [rad m^-1]."""
    g = mu + 1j * np.asarray(omega, float) * eta_s
    ks = np.asarray(omega, float) * np.sqrt(re / g)
    return np.where(ks.imag > 0, -ks, ks)


def band_response(f, f_band, order=4):
    """Zero-phase Butterworth-magnitude band-pass of the displacement record.

    Shear-wave processing band-passes the tracked displacement, and the model
    has to contain that filter rather than a hard spectral mask: a hard mask has
    a discontinuity, so the time-domain result depends on how finely the
    spectrum happens to be sampled and never converges as the transform is
    padded.  This taper is smooth and is a fixed function of frequency, applied
    identically to the prediction and to the synthetic observation, so its
    pass-band ripple (half a percent one octave inside each corner, for the
    fourth order used here) is part of the acquisition and not an error."""
    f = np.asarray(f, float)
    f1, f2 = f_band
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(f > 0, (f / f1) ** (2 * order), 0.0)
    hp = r1 / (1.0 + r1)
    lp = 1.0 / (1.0 + (f / f2) ** (2 * order))
    return hp * lp


def _source_spectrum(omega, F0, Tp):
    """Spectrum of a push of amplitude F0 and duration Tp: F0 sinc(omega Tp / 2)."""
    return F0 * np.sinc(omega * Tp / (2 * np.pi))


def shear_transfer(omega, x, p, geometry="cylindrical"):
    """Green function of the shear window, displacement per unit source, at
    lateral offsets ``x`` [m] and angular frequencies ``omega`` [rad s^-1].

    Returns an array of shape (len(x), len(omega)) [m N^-1 m].
    """
    omega = np.asarray(omega, float)
    x = np.atleast_1d(np.asarray(x, float))
    re = rho_eff(omega, p["phi"], p["k"],
                 p.get("rho"), p.get("rho_f"), p.get("eta_b"), p.get("alpha_inf"))
    ks = shear_wavenumber(omega, p["mu"], p["eta_s"], re)
    g = p["mu"] + 1j * omega * p["eta_s"]
    z = ks[None, :] * x[:, None]
    out = np.zeros(z.shape, complex)
    ok = np.abs(z) > 0
    if geometry == "cylindrical":
        out[ok] = (1j / (4 * np.broadcast_to(g[None, :], z.shape)[ok])) * hankel2(0, z[ok])
    elif geometry == "plane":
        kk = np.broadcast_to(ks[None, :], z.shape)[ok]
        out[ok] = np.exp(-1j * z[ok]) / (2j * kk * np.broadcast_to(g[None, :], z.shape)[ok])
    else:
        raise ValueError("unknown geometry %r" % geometry)
    return out


def shear_displacement(x, t, p, f_band=(40.0, 800.0), geometry="cylindrical",
                       pad=16):
    """Transverse displacement u_z(x, t) [m] of the shear window.

    The zero-frequency bin is removed and the response is restricted to
    ``f_band``: the static response of an unbounded medium to a line force is not
    defined, and the SWE observation is band limited in practice.  The transform
    is padded by ``pad`` times the record length; band limiting spreads the
    response in time, so the padding has to outlast that spread rather than only
    the wave transit.  Convergence in ``pad`` is checked in the tests.
    """
    t = np.asarray(t, float)
    n = len(t)
    dt = float(t[1] - t[0])
    npad = int(pad * n)
    f = np.fft.rfftfreq(npad, dt)
    om = 2 * np.pi * f
    band = band_response(f, f_band)
    # the Hankel function is the cost of this model, so it is evaluated only
    # where the acquisition filter passes something; the discarded bins are
    # below 1e-9 of the pass band and contribute nothing to the record
    keep = band > 1e-9
    weight = _source_spectrum(om[keep], p["F0"], p["Tp"]) * band[keep]
    u = np.zeros((len(np.atleast_1d(x)), len(f)), complex)
    u[:, keep] = shear_transfer(om[keep], x, p, geometry) * weight[None, :]
    return np.fft.irfft(u, n=npad, axis=1)[:, :n] / dt


# ---------------------------------------------------------------- M2 --------

def storage_total(p):
    """S_tot = S_v + alpha^2 / M  [Pa^-1]."""
    al = p.get("alpha", FIXED["alpha"])
    return p["S_v"] + al ** 2 / p["M"]


def consolidation_coefficient(p):
    """c_v = k / (eta_b S_tot)  [m^2 s^-1]."""
    return p["k"] / (p.get("eta_b", FIXED["eta_b"]) * storage_total(p))


def relaxing_fraction(p):
    """R = alpha^2 / (alpha^2 + M S_v), the time-dependent share of the
    compaction.  Depends on the storage ratio only."""
    al = p.get("alpha", FIXED["alpha"])
    return al ** 2 / (al ** 2 + p["M"] * p["S_v"])


def relax_displacement(t, p, n_terms=400):
    """Surface compaction u(t) [m] of the confined slab after a step of
    compressive total stress P applied at t = 0.

    u(t) = ( - P L + alpha int_0^L p dz ) / M, with the Terzaghi series

        p(z, t) = sum_{n odd} (4 p_u / (n pi)) sin(n pi z / 2L) exp(- n^2 tau),
        tau = pi^2 c_v t / (4 L^2),    p_u = alpha P / (M S_v + alpha^2),

    so that int_0^L p dz = (8 p_u L / pi^2) sum_{n odd} exp(- n^2 tau) / n^2.
    At t = 0 the sum is pi^2 / 8 and u = L eps_u; as t -> infinity u = L eps_d.
    """
    t = np.asarray(t, float)
    al = p.get("alpha", FIXED["alpha"])
    pu = al * p["P"] / (p["M"] * p["S_v"] + al ** 2)
    cv = consolidation_coefficient(p)
    tau = np.pi ** 2 * cv * np.clip(t, 0.0, None) / (4 * p["L"] ** 2)
    n = np.arange(1, 2 * n_terms, 2, dtype=float)
    s = np.sum(np.exp(-np.outer(tau, n ** 2)) / n ** 2, axis=1)
    intp = 8 * pu * p["L"] / np.pi ** 2 * s
    return (-p["P"] * p["L"] + al * intp) / p["M"]


def relax_limits(p):
    """(u(0), u(infinity)) [m], in closed form."""
    al = p.get("alpha", FIXED["alpha"])
    eps_u = -p["P"] * p["S_v"] / (p["M"] * p["S_v"] + al ** 2)
    eps_d = -p["P"] / p["M"]
    return eps_u * p["L"], eps_d * p["L"]
