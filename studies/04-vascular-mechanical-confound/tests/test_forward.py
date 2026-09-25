import numpy as np

from vmconf import forward, mech, netops


def test_spectrum_normalised_and_binned():
    r = np.array([50e-6, 100e-6, 200e-6])
    L = np.array([1e-3, 2e-3, 4e-3])
    v = np.array([1.0, 2.0, 3.0])
    w, tau = mech.microchannel_spectrum(r, L, v, 2000.0)
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.all(np.diff(tau) > 0)


def test_tau_scales_as_inverse_square_of_radius():
    r = np.array([100e-6]); L = np.array([1e-3]); v = np.array([1.0])
    w1, t1 = mech.microchannel_spectrum(r, L, v, 2000.0)
    w2, t2 = mech.microchannel_spectrum(0.5 * r, L, v, 2000.0)
    tau1 = t1[np.argmax(w1)]; tau2 = t2[np.argmax(w2)]
    assert abs(tau2 / tau1 - 4.0) < 0.2        # within one log bin


def test_tau_inversely_proportional_to_matrix_modulus():
    r = np.array([100e-6]); L = np.array([1e-3]); v = np.array([1.0])
    w1, t1 = mech.microchannel_spectrum(r, L, v, 1000.0)
    w2, t2 = mech.microchannel_spectrum(r, L, v, 4000.0)
    assert abs((t2[np.argmax(w2)] / t1[np.argmax(w1)]) - 0.25) < 0.02


def test_summary_locates_band():
    tau = np.array([1e-5, 1e-3, 1e-1])
    w = np.array([1.0, 1.0, 1.0]) / 3
    s = mech.spectrum_summary(w, tau)
    assert s["below"] == 1 / 3 and s["inband"] == 1 / 3 and s["above"] == 1 / 3


def test_analytic_constriction_scaling():
    """The analytic intervention applies s^2 to volume and s^-4 to time."""
    from porovasc.geometry import network as N
    net = N.build(N.Params(n_feeders=2, d_term_gland=80e-6, d_term_rve=60e-6,
                           rve_centres=((0.0, -5e-3, 0.0),), rve_half=3e-3, seed=3))
    c = (0.0, -5e-3, 0.0); h = 3e-3
    s = 0.8
    vol, phi = forward.region_weights(net, c, h)
    w0, tau0 = mech.segment_times(net.r, net.Lpath, vol, 2000.0)
    w1, tau1 = mech.segment_times(net.r, net.Lpath, vol, 2000.0, s=s)
    out = forward.swe_curve(net, c, h, 2000.0, 1.0, 0.2, s=s, dG=300.0)
    assert np.allclose(w1, w0)
    assert np.allclose(tau1, tau0 * s ** -4)
    assert abs(out["phi"] / phi - s ** 2) < 1e-12


def test_flow_routes_are_not_equivalent_and_the_gap_is_measured():
    """The two constriction routes differ; the size of the difference is
    reported rather than assumed away."""
    from porovasc.geometry import network as N
    net = N.build(N.Params(n_feeders=2, d_term_gland=80e-6, d_term_rve=60e-6,
                           rve_centres=((0.0, -5e-3, 0.0),), rve_half=3e-3, seed=3))
    d = netops.flow_agreement(net, 0.8)
    assert d["rel_L2_flow"] > 1e-3          # they are genuinely different
    assert abs(d["bed_conductance_ratio"] - 0.8 ** 3) < 1e-12


def test_constriction_validation_rejects_bad_cases():
    import numpy as _np
    from porovasc.geometry import network as N
    net = N.build(N.Params(n_feeders=2, d_term_gland=80e-6, d_term_rve=60e-6,
                           rve_centres=((0.0, -5e-3, 0.0),), rve_half=3e-3, seed=3))
    assert abs(netops.assert_constriction_valid(net, netops.scaled_copy(net, 0.8)) - 0.8) < 1e-9
    try:
        netops.assert_constriction_valid(net, netops.scaled_copy(net, 1.2))
    except AssertionError:
        pass
    else:
        raise AssertionError("dilation was accepted as a constriction")
    bad = netops.scaled_copy(net, 0.8); bad.r = bad.r * _np.linspace(1.0, 1.1, len(bad.r))
    try:
        netops.assert_constriction_valid(net, bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("non-uniform scaling was accepted")


def test_share_is_defined_for_the_unconstricted_network():
    """At s = 1 the realized share must equal the requested one."""
    from porovasc.geometry import network as N
    net = N.build(N.Params(n_feeders=2, d_term_gland=80e-6, d_term_rve=60e-6,
                           rve_centres=((0.0, -5e-3, 0.0),), rve_half=3e-3, seed=3))
    c = (0.0, -5e-3, 0.0); h = 3e-3
    eta, f_ref, want = 1.0, 200.0, 0.2
    r = forward.swe_curve(net, c, h, 2000.0, eta, want, s=1.0)
    om = 2 * np.pi * f_ref
    gv = mech.vascular_modulus(r["spectrum"], r["tau"], [om], r["dG"])[0]
    got = gv.imag / (gv.imag + om * eta)
    assert abs(got - want) < 1e-6


def test_scaled_copy_preserves_structure():
    from porovasc.geometry import network as N
    net = N.build(N.Params(n_feeders=2, d_term_gland=80e-6, d_term_rve=60e-6,
                           rve_centres=((0.0, -5e-3, 0.0),), rve_half=3e-3, seed=3))
    net_s = netops.scaled_copy(net, 0.8)
    netops.assert_constriction_valid(net, net_s)
    assert abs(netops.lumen_fraction_ratio(0.8) - 0.64) < 1e-12
