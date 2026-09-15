"""Mechanical model: limiting cases, analytic identities, convergence."""
import numpy as np
import pytest

from fieldid.constants import FIXED
from fieldid.models import mechanics as ME
from fieldid.params import TRUTH


def truth():
    p = dict(TRUTH)
    p.update(FIXED)
    return p


# ------------------------------------------------------------ effective density

def test_rho_eff_zero_frequency_is_bulk_density():
    p = truth()
    re = ME.rho_eff(np.array([0.0]), p["phi"], p["k"])
    assert re[0].real == pytest.approx(FIXED["rho"], rel=0, abs=0)
    assert re[0].imag == 0.0


def test_rho_eff_zero_permeability_is_bulk_density():
    p = truth()
    re = ME.rho_eff(np.array([1000.0]), p["phi"], 0.0)
    assert np.allclose(re, FIXED["rho"])


def test_rho_eff_infinite_blood_viscosity_is_bulk_density():
    p = truth()
    re = ME.rho_eff(np.array([1000.0]), p["phi"], p["k"], eta_b=np.inf)
    assert np.allclose(re, FIXED["rho"])


def test_rho_eff_departure_matches_the_inertial_coupling_group():
    """|rho_eff - rho| / rho equals rho_f^2 omega k / (rho eta_b) to first
    order in the viscous-dominated regime."""
    p = truth()
    om = 2 * np.pi * 200.0
    re = ME.rho_eff(np.array([om]), p["phi"], p["k"])[0]
    predicted = FIXED["rho_f"] ** 2 * om * p["k"] / (FIXED["rho"] * FIXED["eta_b"])
    assert abs(re - FIXED["rho"]) / FIXED["rho"] == pytest.approx(predicted, rel=2e-3)


# ------------------------------------------------------------------ wavenumber

def test_shear_wavenumber_branch_is_decaying():
    p = truth()
    om = 2 * np.pi * np.array([50.0, 200.0, 800.0])
    re = ME.rho_eff(om, p["phi"], p["k"])
    ks = ME.shear_wavenumber(om, p["mu"], p["eta_s"], re)
    assert np.all(ks.imag <= 0)
    assert np.all(ks.real > 0)


def test_lossless_phase_speed_is_sqrt_mu_over_rho():
    p = truth()
    p["eta_s"] = 0.0
    p["k"] = 0.0
    om = np.array([2 * np.pi * 200.0])
    re = ME.rho_eff(om, p["phi"], p["k"])
    ks = ME.shear_wavenumber(om, p["mu"], p["eta_s"], re)
    c = om / ks.real
    assert c[0] == pytest.approx(np.sqrt(p["mu"] / FIXED["rho"]), rel=1e-12)
    assert abs(ks.imag[0]) < 1e-12


# ------------------------------------------------------------- Green functions

def test_cylindrical_approaches_plane_wave_in_the_far_field():
    """H_0^(2)(z) ~ sqrt(2 / pi z) exp(-i(z - pi/4)); the ratio of the
    cylindrical to the plane transfer function must approach that asymptote."""
    p = truth()
    om = np.array([2 * np.pi * 300.0])
    x = np.array([0.2])                       # far field: many wavelengths
    cyl = ME.shear_transfer(om, x, p, "cylindrical")[0, 0]
    pla = ME.shear_transfer(om, x, p, "plane")[0, 0]
    re = ME.rho_eff(om, p["phi"], p["k"])
    ks = ME.shear_wavenumber(om, p["mu"], p["eta_s"], re)[0]
    z = ks * x[0]
    expect = -(ks / 2) * np.sqrt(2 / (np.pi * z)) * np.exp(1j * np.pi / 4)
    assert cyl / pla == pytest.approx(expect, rel=2e-2)


def test_displacement_is_linear_in_push_amplitude():
    p = truth()
    t = np.arange(160) / 8e3
    x = np.linspace(2e-3, 12e-3, 5)
    u1 = ME.shear_displacement(x, t, p)
    q = dict(p); q["F0"] = 3.0 * p["F0"]
    u3 = ME.shear_displacement(x, t, q)
    assert np.allclose(u3, 3.0 * u1, rtol=1e-12, atol=0)


def test_displacement_is_causal_and_arrives_later_further_out():
    p = truth()
    t = np.arange(320) / 8e3
    x = np.array([3e-3, 9e-3])
    u = ME.shear_displacement(x, t, p)
    peak = [t[np.argmax(np.abs(u[i]))] for i in range(2)]
    assert peak[1] > peak[0]
    speed = (x[1] - x[0]) / (peak[1] - peak[0])
    assert 0.5 * np.sqrt(p["mu"] / FIXED["rho"]) < speed < 2.0 * np.sqrt(p["mu"] / FIXED["rho"])


def test_padding_converges():
    """Band limiting spreads the impulse response in time, so the record has to
    be padded well beyond the wave transit.  The difference between successive
    paddings must fall, and at the default padding it must be far below the
    displacement noise of the acquisition."""
    p = truth()
    t = np.arange(160) / 8e3
    x = np.array([5e-3])
    us = {q: ME.shear_displacement(x, t, p, pad=q) for q in (2, 4, 8, 16, 48)}
    d = [np.abs(us[q] - us[48]).max() for q in (2, 4, 8, 16)]
    assert d[0] > d[1] > d[2] > d[3]
    assert d[3] < 0.01 * 2.0e-7          # default padding, against the noise


# ------------------------------------------------------------------ relaxation

def test_relaxation_endpoints_match_closed_form():
    p = truth()
    u0, ui = ME.relax_limits(p)
    t = np.array([0.0, 1e6])
    u = ME.relax_displacement(t, p, n_terms=4000)
    assert u[0] == pytest.approx(u0, rel=1e-3)
    assert u[1] == pytest.approx(ui, rel=1e-12)


def test_relaxing_fraction_matches_the_storage_ratio():
    p = truth()
    u0, ui = ME.relax_limits(p)
    assert 1 - u0 / ui == pytest.approx(ME.relaxing_fraction(p), rel=1e-12)
    assert ME.relaxing_fraction(p) == pytest.approx(
        1.0 / (1.0 + p["M"] * p["S_v"]), rel=1e-12)


def test_relaxation_is_monotone():
    p = truth()
    t = np.linspace(0, 40, 400)
    u = ME.relax_displacement(t, p)
    assert np.all(np.diff(u) < 1e-15)


def test_relaxation_series_truncation_converges():
    p = truth()
    t = np.linspace(0.0, 20.0, 50)
    ref = ME.relax_displacement(t, p, n_terms=8000)
    errs = [np.abs(ME.relax_displacement(t, p, n_terms=n) - ref).max()
            for n in (50, 200, 800)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 1e-3 * np.abs(ref).max()


def test_relaxation_matches_a_finite_difference_solve():
    """Crank-Nicolson on dp/dt = c_v p_zz with the same conditions, refined in
    space and time, must approach the series solution."""
    p = truth()
    cv = ME.consolidation_coefficient(p)
    L, al = p["L"], p["alpha"]
    pu = al * p["P"] / (p["M"] * p["S_v"] + al ** 2)
    t_end = 12.0
    prev = None
    errs = []
    for nz, nt in ((40, 200), (80, 800), (160, 3200)):
        dz, dt = L / nz, t_end / nt
        z = np.linspace(0, L, nz + 1)
        pp = np.full(nz + 1, pu)
        pp[0] = 0.0
        lam = cv * dt / dz ** 2
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        main = np.full(nz + 1, 1 + lam)
        off = np.full(nz, -lam / 2)
        A = sp.diags([off, main, off], [-1, 0, 1], format="lil")
        B = sp.diags([-off, np.full(nz + 1, 1 - lam), -off], [-1, 0, 1], format="lil")
        A[0, :] = 0; A[0, 0] = 1; B[0, :] = 0          # drained
        A[nz, nz - 1] = -lam; B[nz, nz - 1] = lam      # sealed, mirror node
        A = A.tocsc(); B = B.tocsr()
        lu = spl.splu(A)
        out = []
        for i in range(nt + 1):
            out.append(np.trapezoid(pp, z) if hasattr(np, "trapezoid")
                       else np.trapz(pp, z))
            pp = lu.solve(B @ pp)
            pp[0] = 0.0
        ts = np.arange(nt + 1) * dt
        u_fd = (-p["P"] * L + al * np.array(out)) / p["M"]
        u_an = ME.relax_displacement(ts, p, n_terms=4000)
        # skip the first steps, where the initial discontinuity at z = 0 is not
        # resolved by any finite grid
        m = ts > 0.5
        errs.append(np.abs(u_fd[m] - u_an[m]).max() / abs(u_an[-1]))
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 2e-3


def test_band_response_is_flat_in_the_pass_band_and_rejects_outside():
    f = np.array([1.0, 40.0, 200.0, 400.0, 800.0, 4000.0])
    h = ME.band_response(f, (40.0, 800.0))
    assert h[2] > 0.999 and h[3] > 0.99       # flat between the corners
    assert h[1] == pytest.approx(0.5, rel=1e-9)
    assert h[4] == pytest.approx(0.5, rel=1e-9)
    assert h[0] < 1e-6 and h[5] < 1e-3
    assert ME.band_response(np.array([0.0]), (40.0, 800.0))[0] == 0.0
