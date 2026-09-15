"""Fast analytic tests of the numerical kernels (no gland build; about 2 s).

Every case here has a closed-form answer, so a failure points at the code and
not at the tissue model.  The slow invariant tests on a whole gland are in
test_invariants.py.
"""
import numpy as np
import pytest
from types import SimpleNamespace

from porovasc.geometry.network import clip_segment_cube, cube_fraction, clip_segment_sphere
from porovasc.physics import transport as T
from porovasc.homogenise import darcy as D


# ----------------------------------------------------------------- clipping
def test_clip_cube_cases():
    p0 = np.array([[-5., 0, 0], [0.2, 0, 0], [10., 10, 10], [-1., 0, 0]])
    p1 = np.array([[5., 0, 0], [0.4, 0, 0], [11., 11, 11], [-1., 1, 0]])
    hit, t0, t1 = clip_segment_cube(p0, p1, [0, 0, 0], 1.0)
    assert list(hit) == [True, True, False, False]        # last one is a tangency
    assert np.allclose(t0[:2], [0.4, 0.0]) and np.allclose(t1[:2], [0.6, 1.0])
    assert np.allclose(cube_fraction(p0, p1, [0, 0, 0], 1.0)[:3], [0.2, 1.0, 0.0])


def test_clip_sphere():
    hit, t0, t1 = clip_segment_sphere(np.array([[-5., 0, 0]]), np.array([[5., 0, 0]]), [0, 0, 0], 2.0)
    assert hit[0] and np.isclose(t0[0], 0.3) and np.isclose(t1[0], 0.7)


# -------------------------------------------------------------- deposition
@pytest.mark.parametrize("a,b", [(0.0, 0.01), (0.03, 0.0302), (0.0, 1.0), (0.007, 0.033)])
def test_box_deposition_preserves_mass_and_mean(a, b):
    t, _ = T.time_grid()
    g = np.zeros(len(t)); T._box_bins(g, a, b, 1.0)
    mass = g.sum() * T.DT
    mean = (t * g).sum() * T.DT / mass
    assert np.isclose(mass, b - a, rtol=1e-12)
    assert np.isclose(mean, 0.5 * (a + b), atol=1e-12)


def test_delta_deposition_preserves_mass_and_position():
    t, _ = T.time_grid()
    g = np.zeros(len(t)); T._delta_bins(g, 0.0317, 2.0)
    assert np.isclose(g.sum() * T.DT, 2.0, rtol=1e-12)
    assert np.isclose((t * g).sum() * T.DT / (g.sum() * T.DT), 0.0317, atol=1e-12)


# ---------------------------------------------------------------- transport
@pytest.mark.parametrize("pois", [False, True])
def test_piece_partition_and_residence(pois):
    t, w = T.time_grid()
    L, v, d = 5e-3, 5e-3, 100e-6
    for kw in ({}, {"t": t}):                       # analytic and binned forms
        whole = T.piece_occupancy(w, 0.0, L, v, d, pois, **kw)
        cuts = np.linspace(0, L, 7)
        parts = sum(T.piece_occupancy(w, a, b, v, d, pois, **kw) for a, b in zip(cuts[:-1], cuts[1:]))
        assert np.abs(whole - parts).max() < 1e-12
        assert np.isclose(whole[0].real, T.piece_residence(0.0, L, v, d, pois), rtol=1e-10)


def test_capped_poiseuille_residence():
    rc = T.rho_cap(100e-6)
    assert np.isclose(T.piece_residence(0, 1e-3, 1e-3, 100e-6, True), 1e-3 / (1e-3 * (2 - rc ** 2)))


def test_degenerate_profile_is_centreline():
    u, p = T._profile(1e-3, T.D_BUBBLE)             # rho_c -> 0
    assert np.isclose(u[0], 2e-3) and np.isclose(p.sum(), 1.0)
    assert np.isclose(T.piece_residence(0, 1e-3, 1e-3, T.D_BUBBLE, True), 0.5)


def test_two_positions_on_one_vessel_are_separated_in_time():
    """A 10 mm vessel at 1 mm/s, sampled by two cubes of full side 1 mm centred
    at 0.5 and 9.5 mm: the first moments must differ by 9 s."""
    t, w = T.time_grid()
    A = np.fft.rfft(T.aif(t)) * T.DT
    L, v, d = 10e-3, 1e-3, 200e-6
    p0 = np.array([[0., 0, 0]]); p1 = np.array([[L, 0, 0]])
    def m1(centre):
        hit, a, b = clip_segment_cube(p0, p1, [centre, 0, 0], 0.5e-3)
        occ = T.piece_occupancy(w, a[0] * L, b[0] * L, v, d, False, t=t)
        y = np.fft.irfft(A * occ, n=len(t)) / T.DT
        return (t * y).sum() / y.sum()
    assert abs((m1(9.5e-3) - m1(0.5e-3)) - 9.0) < 0.09        # within 1 %


def test_gamma_bed_moments():
    """The lumped bed kernel must have the prescribed mean and coefficient of
    variation, checked on the time-domain kernel that the model uses."""
    t, w = T.time_grid()
    mean, cv = 1.0, 0.5
    g = np.fft.irfft(T._h_gamma(w, mean, cv), n=len(t)) / T.DT
    mass = g.sum() * T.DT
    m1 = (t * g).sum() * T.DT / mass
    var = ((t - m1) ** 2 * g).sum() * T.DT / mass
    assert np.isclose(mass, 1.0, rtol=1e-6)
    assert np.isclose(m1, mean, rtol=1e-3)
    assert np.isclose(np.sqrt(var) / m1, cv, rtol=1e-2)


def test_box_variance_inflation_is_phase_dependent():
    """Linear splitting adds f(1-f) DT^2 to a point mass at phase f, so a box is
    inflated by up to DT^2/4 and by DT^2/6 when it covers a whole cell."""
    t, _ = T.time_grid()
    def var(a, b):
        g = np.zeros(len(t)); T._box_bins(g, a, b, 1.0)
        mass = g.sum() * T.DT; m1 = (t * g).sum() * T.DT / mass
        return ((t - m1) ** 2 * g).sum() * T.DT / mass
    full = var(0.0, T.DT)                       # exact variance DT^2/12
    assert np.isclose(full - T.DT ** 2 / 12, T.DT ** 2 / 6, rtol=1e-6)
    narrow = var(0.5 * T.DT - 1e-6, 0.5 * T.DT + 1e-6)
    assert narrow < T.DT ** 2 / 4 + 1e-12


def test_travel_time_bound_detects_wrap():
    """A record shorter than the latest possible arrival must be flagged."""
    net = SimpleNamespace(Lpath=np.array([2e-3]), d=np.array([100e-6]))
    ok = SimpleNamespace(t_arr=np.array([2.4]), v=np.array([5e-3]))
    slow = SimpleNamespace(t_arr=np.array([10.0]), v=np.array([5e-5]))
    assert T.travel_time_bound(net, ok, True) < T.T_END
    assert T.travel_time_bound(net, slow, True) > T.T_END


# --------------------------------------------------------------- permeability
def _one_vessel(p0, p1, r, starts=None):
    p0 = np.array(p0, float); p1 = np.array(p1, float); r = np.array(r, float)
    L = np.linalg.norm(p1 - p0, axis=1)
    net = SimpleNamespace(p0=p0, p1=p1, r=r, d=2 * r, Lpath=L.copy(), Lchord=L.copy(), n=len(r),
                          params=SimpleNamespace(d_term_rve=20e-6))
    fl = SimpleNamespace(start=np.arange(len(r)) - 1 if starts is None else np.array(starts),
                         bed_a=np.array([], int), bed_v=np.array([], int),
                         bed_r=np.array([]), bed_w=np.array([]), G_bed0=0.0)
    return net, fl


H, R = 0.5e-3, 50e-6
K_TUBE = np.pi * R ** 4 / (8 * (2 * H) ** 2)     # a single tube through the cube


def test_darcy_tube_crossing_with_both_ends_outside():
    net, fl = _one_vessel([[-2e-3, 0, 0]], [[2e-3, 0, 0]], [R])
    k, _, _ = D.face_to_face(net, fl, [0, 0, 0], H)
    assert np.isclose(k[0], K_TUBE, rtol=1e-10) and k[1] == 0 and k[2] == 0


def test_darcy_tube_with_endpoints_exactly_on_the_faces():
    net, fl = _one_vessel([[-H, 0, 0]], [[H, 0, 0]], [R])
    k, _, _ = D.face_to_face(net, fl, [0, 0, 0], H)
    assert np.isclose(k[0], K_TUBE, rtol=1e-10)


def test_darcy_series_pieces_equal_one_tube():
    net, fl = _one_vessel([[-2e-3, 0, 0], [0, 0, 0]], [[0, 0, 0], [2e-3, 0, 0]], [R, R], starts=[-1, 0])
    k, _, _ = D.face_to_face(net, fl, [0, 0, 0], H)
    assert np.isclose(k[0], K_TUBE, rtol=1e-10)


def test_darcy_oblique_tube():
    net, fl = _one_vessel([[-2e-3, -2e-3, 0]], [[2e-3, 2e-3, 0]], [R])
    k, _, _ = D.face_to_face(net, fl, [0, 0, 0], H)
    assert np.isclose(k[0], K_TUBE * np.cos(np.pi / 4), rtol=1e-10)


def test_darcy_dead_end_and_disconnected_give_zero():
    net, fl = _one_vessel([[-H, 0, 0]], [[0.2e-3, 0, 0]], [R])
    k, _, _ = D.face_to_face(net, fl, [0, 0, 0], H)
    assert k[0] == 0
    net, fl = _one_vessel([[-2e-3, 0, 0], [0.2e-3, 0, 0]], [[-0.2e-3, 0, 0], [2e-3, 0, 0]], [R, R],
                          starts=[-1, -1])
    k, _, _ = D.face_to_face(net, fl, [0, 0, 0], H)
    assert np.all(k == 0)


def test_causality_rules_and_transfer_functions_are_wired():
    """On a series path the three causality rules accept the downstream voxels
    and both transfer functions return finite estimates; an unknown rule is
    refused."""
    import numpy as np, pytest
    from porovasc.physics import transport as TR
    t = np.arange(0, 40.0, TR.DT); dz = 1e-3; v, D = 3e-3, 2e-6
    w = 2 * np.pi * np.fft.rfftfreq(len(t), TR.DT)
    tin = np.exp(-0.5 * ((np.log(np.maximum(t, 1e-9)) - np.log(8.0)) / 0.4) ** 2)
    outs = [np.fft.irfft(np.fft.rfft(tin) * TR.T_new(w, dz * s, v, D), n=len(t)) for s in (0.9, 1.0, 1.1, 1.0)]
    for rule in TR.CAUSAL_RULES:
        for tf in ("new", "old"):
            ve, De, n, r2 = TR.identify_shell(t, tin, outs, dz, tf=tf, rule=rule)
            assert n == 4 and np.isfinite(ve) and np.isfinite(De), (rule, tf)
            if tf == "new":
                assert abs(np.log(ve / v)) < 0.1
    with pytest.raises(ValueError):
        TR.identify_shell(t, tin, outs, dz, rule="peak")


def test_arrival_rules_order_as_expected():
    import numpy as np
    from porovasc.physics import transport as TR
    t = np.arange(0, 20.0, 0.01); y = np.exp(-0.5 * ((t - 8) / 1.5) ** 2); y /= y.sum() * 0.01
    a10, a20, am = (TR._arrival(t, y, r) for r in ("front10", "front20", "moment"))
    assert a10 < a20 < am and abs(am - 8.0) < 0.02


def test_drainage_rest_state_is_steady_after_radius_rescaling():
    """With rescaled radii the rest state must be recomputed; otherwise the
    relaxation drifts and never reaches its equilibrium."""
    import numpy as np
    from porovasc.geometry import network as N
    from porovasc.physics import flow as F, drainage as D
    net = N.build(N.Params(seed=3, d_term_gland=120e-6, d_term_rve=100e-6))
    fl = F.solve(net)
    c = np.zeros(3)
    d0 = D.solve(net, fl, c, 4e-3, dP_ext=100.0, H=5e3)
    d2 = D.solve(net, fl, c, 4e-3, dP_ext=100.0, H=5e3, radius_scale=1.2)
    assert abs(d0.f[-1] - 1) < 2e-3 and abs(d2.f[-1] - 1) < 2e-3
    assert np.isfinite(d2.tau_rc) and abs(d2.tau_rc / d0.tau_rc - 1 / 1.44) < 0.1


def test_drainage_observable_is_the_compressed_region_only():
    """dV(t) is summed over the segments with a part inside the sphere; the
    uncompressed segments store blood during the transient and must not enter.
    f still reaches 1 exactly, because Pext is zero outside the region, and the
    region-only time constant grows with the radius on this small gland."""
    import numpy as np
    from porovasc.geometry import network as N
    from porovasc.physics import flow as F, drainage as D
    net = N.build(N.Params(seed=3, d_term_gland=120e-6, d_term_rve=100e-6))
    fl = F.solve(net)
    taus = []
    for R in (1.5e-3, 3e-3, 6e-3):
        d = D.solve(net, fl, np.zeros(3), R, dP_ext=100.0, H=5e3)
        assert abs(d.f[-1] - 1) < 2e-3
        assert d.in_region.sum() < net.n                 # the region is a proper subset
        assert np.isclose(d.dV_inf, -np.sum(np.pi * net.r ** 2 * net.Lpath / 5e3 * 100.0 * d.frac))
        taus.append(d.tau_rc)
    assert taus[0] < taus[1] < taus[2]


def test_drainage_handles_a_compression_sphere_with_no_segments_inside():
    """A small R_comp in a sparse pocket of the gland can miss the vasculature
    entirely: dV and dV_inf are both exactly zero, so f = 0/0.  This must be
    reported as an undefined relaxation (phi = 0, tau_rc and the spectrum NaN),
    not crash the NNLS spectrum fit with a NaN input."""
    import numpy as np
    from porovasc.geometry import network as N
    from porovasc.physics import flow as F, drainage as D
    net = N.build(N.Params(seed=3, d_term_gland=120e-6, d_term_rve=100e-6))
    fl = F.solve(net)
    far = np.array([0.0, 0.0, 0.0])
    far[np.argmax(N.GLAND_SEMI)] = N.GLAND_SEMI[np.argmax(N.GLAND_SEMI)] * 5   # well outside the gland
    d = D.solve(net, fl, far, R_comp=1e-6, dP_ext=100.0, H=5e3)
    assert d.dV_inf == 0.0 and d.phi == 0.0 and d.V_blood == 0.0
    assert np.isnan(d.tau_rc) and np.all(np.isnan(d.f)) and np.all(np.isnan(d.spectrum))
    # the ordinary case is unaffected, including a small sphere that captures
    # only a few segments: dV_inf = -V_blood * dP_ext / H exactly (sign
    # negative by the code's convention, however small V_blood is), which is
    # a well-scaled real number, not floating-point noise, and f still
    # reaches 1 because dV(t) and dV_inf carry the same sign asymptotically
    for R in (3e-3, 4e-3, 8e-3):
        d0 = D.solve(net, fl, np.zeros(3), R_comp=R, dP_ext=100.0, H=5e3)
        assert d0.in_region.any() and d0.dV_inf != 0.0
        assert np.isclose(d0.dV_inf, -d0.V_blood * 100.0 / 5e3)
        assert np.isfinite(d0.tau_rc) and abs(d0.f[-1] - 1) < 2e-3
