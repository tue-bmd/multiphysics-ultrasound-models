import numpy as np

from vmconf import mech


def test_vascular_modulus_limits():
    tau = np.logspace(-4, 0, 40)
    w = np.ones_like(tau) / len(tau)
    dG = 700.0
    lo = mech.vascular_modulus(w, tau, [1e-8], dG)
    hi = mech.vascular_modulus(w, tau, [1e12], dG)
    assert abs(lo[0]) < 1e-3 * dG                 # relaxed: vessels carry no shear
    assert abs(hi[0].real - dG) < 1e-3 * dG       # locked: full increment
    assert hi[0].imag < 1e-3 * dG


def test_vascular_modulus_is_passive():
    """Storage and loss moduli must both be non-negative at every frequency."""
    tau = np.logspace(-4, 0, 40)
    rng = np.random.default_rng(0)
    w = rng.random(len(tau))
    om = mech.band()
    g = mech.vascular_modulus(w, tau, om, 500.0)
    assert np.all(g.real >= -1e-9)
    assert np.all(g.imag >= -1e-9)


def test_shear_wave_elastic_limit():
    mu, rho = 3000.0, 1050.0
    om = mech.band()
    c, a = mech.shear_wave(np.full(len(om), mu + 0j), om, rho)
    assert np.allclose(c, np.sqrt(mu / rho))
    assert np.allclose(a, 0.0, atol=1e-9)


def test_shear_wave_kelvin_voigt_closed_form():
    mu, eta, rho = 2000.0, 1.5, 1050.0
    om = mech.band()
    g = mech.kelvin_voigt(om, mu, eta)
    c, a = mech.shear_wave(g, om, rho)
    # closed form: c = sqrt(2(mu^2+(om eta)^2) / (rho (mu + sqrt(mu^2+(om eta)^2))))
    m = np.sqrt(mu ** 2 + (om * eta) ** 2)
    c_ref = np.sqrt(2 * m ** 2 / (rho * (mu + m)))
    assert np.allclose(c, c_ref, rtol=1e-10)
    assert np.all(a > 0)            # a lossy medium must attenuate


def test_kv_dispersion_increases_with_frequency():
    om = mech.band()
    c, _ = mech.shear_wave(mech.kelvin_voigt(om, 2000.0, 1.5), om)
    assert np.all(np.diff(c) > 0)


def test_fixed_tissue_element_constriction_increases_storage_modulus():
    """At fixed amplitude, shifting tau by s^-4 increases Maxwell storage."""
    om = mech.band()
    tau = np.logspace(-5, -2, 40)
    w = np.ones_like(tau) / len(tau)
    g0 = mech.vascular_modulus(w, tau, om, 500.0)
    g1 = mech.vascular_modulus(w, tau * 0.8 ** -4, om, 500.0)
    assert np.all(g1.real > g0.real)


def test_series_recovers_matrix_when_vascular_vanishes():
    """The zero-vascular limit must return the matrix, not zero."""
    om = mech.band()
    gm = mech.kelvin_voigt(om, 2000.0, 1.0)
    zero = np.zeros(len(om), complex)
    got = mech.combine(gm, zero, "series", phase_fraction=0.035)
    assert np.allclose(got, gm, rtol=1e-12)


def test_series_differs_from_parallel():
    tau = np.logspace(-4, -1, 30)
    w = np.ones_like(tau) / len(tau)
    om = mech.band()
    gm = mech.kelvin_voigt(om, 2000.0, 1.0)
    gv = mech.vascular_modulus(w, tau, om, 400.0)
    assert not np.allclose(mech.combine(gm, gv, "parallel"),
                           mech.combine(gm, gv, "series", phase_fraction=0.035))


def test_single_maxwell_gives_half_slope():
    """A slope near 0.5 carries no information about the vessel tree.

    One Maxwell element centred in the band gives exactly 0.5 on a symmetric
    log grid, so the fitted exponent alone cannot validate a branching origin;
    only the curvature distinguishes them.
    """
    om = mech.band()
    tau_c = 1.0 / np.sqrt(om[0] * om[-1])
    d = mech.spectrum_curvature(np.array([1.0]), np.array([tau_c]), om)
    assert abs(d["slope"] - 0.5) < 1e-6
    assert d["is_power_law"] is False


def test_constriction_scales_times_as_s_to_the_minus_four():
    r = np.array([100e-6]); L = np.array([1e-3]); v = np.array([1.0])
    _, t1 = mech.segment_times(r, L, v, 2000.0, s=1.0)
    _, t2 = mech.segment_times(r, L, v, 2000.0, s=0.8)
    assert abs(t2[0] / t1[0] - 0.8 ** -4) < 1e-12


def test_poiseuille_validity_flags_large_vessels():
    r = np.array([25e-6, 250e-6]); w = np.array([0.5, 0.5])
    frac = mech.poiseuille_validity(r, w, [2 * np.pi * 500.0])
    assert 0.0 < frac[0] <= 1.0


def test_springpot_is_power_law():
    om = mech.band()
    g = mech.springpot(om, 2000.0, 0.15)
    r = np.abs(g)
    slope = np.polyfit(np.log(om), np.log(r), 1)[0]
    assert abs(slope - 0.15) < 1e-6
