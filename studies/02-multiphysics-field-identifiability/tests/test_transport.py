"""Contrast transport: conservation, moments, limits, PDE residual."""
import numpy as np
import pytest

from fieldid.compat import trapezoid
from fieldid.constants import FIXED
from fieldid.models import transport as TP
from fieldid.params import TRUTH


def truth():
    p = dict(TRUTH)
    p.update(FIXED)
    return p


def test_drift_vector_reproduces_the_nominal_components():
    p = truth()
    v = TP.drift_vector(p["vmag"], p["vth"], p["vaz"])
    assert v == pytest.approx([1.0e-3, 0.3e-3, 0.1e-3], rel=1e-12)
    assert np.linalg.norm(v) == pytest.approx(p["vmag"], rel=1e-12)


def test_mass_is_conserved_on_a_fine_grid():
    """Integral of the occupancy over space is phi A at every time.  The grid
    follows the packet, since the point of the test is conservation and not the
    truncation of a fixed window."""
    p = truth()
    v = TP.drift_vector(p["vmag"], p["vth"], p["vaz"])
    d = 0.6e-3
    ax = (np.arange(72) - 35.5) * d
    g = np.meshgrid(ax, ax, ax, indexing="ij")
    base = np.column_stack([a.ravel() for a in g])
    for tau in (2.0, 6.0, 12.0):
        r = base + v * tau
        b = TP.occupancy(r, np.array([p["t0"] + tau]), p)
        assert TP.total_mass(b, (d, d, d))[0] == pytest.approx(
            p["phi"] * p["A"], rel=3e-3)


def test_occupancy_is_zero_before_arrival():
    p = truth()
    t = np.array([p["t0"] - 1.0, p["t0"], p["t0"] + 1.0])
    b = TP.occupancy(np.array([[6e-3, 0.0, 0.0]]), t, p)
    assert b[0, 0] == 0.0 and b[0, 1] == 0.0 and b[0, 2] > 0.0


def test_occupancy_scales_with_the_product_phi_A_only():
    """The defining degeneracy: phi and A enter only through their product."""
    p = truth()
    t = np.linspace(p["t0"] + 0.5, p["t0"] + 60, 200)
    r = np.array([[6e-3, 0.0, 0.0]])
    a = TP.occupancy(r, t, p)
    q = dict(p); q["phi"] = p["phi"] * 3.0; q["A"] = p["A"] / 3.0
    assert np.allclose(TP.occupancy(r, t, q), a, rtol=1e-13, atol=0)


def test_normalization_removes_phi_and_A():
    p = truth()
    t = np.linspace(p["t0"] + 0.5, p["t0"] + 80, 400)
    r = np.array([[6e-3, 0.0, 0.0], [8e-3, 2e-3, 0.0]])
    a = TP.normalize(TP.occupancy(r, t, p), t)
    q = dict(p); q["phi"] = p["phi"] * 7.0; q["A"] = p["A"] * 0.2
    b = TP.normalize(TP.occupancy(r, t, q), t)
    assert np.allclose(a, b, rtol=1e-12, atol=0)


def test_small_dispersion_concentrates_at_the_ballistic_arrival():
    p = truth()
    r = np.array([[6e-3, 1.8e-3, 0.6e-3]])         # on the drift direction
    v = TP.drift_vector(p["vmag"], p["vth"], p["vaz"])
    t_bal = p["t0"] + np.dot(r[0], v) / np.dot(v, v)
    t = np.linspace(p["t0"] + 0.01, p["t0"] + 30, 20000)
    err = []
    for D in (1e-6, 1e-7, 1e-8, 1e-9):
        q = dict(p); q["D"] = D
        b = TP.occupancy(r, t, q)[0]
        err.append(abs(t[np.argmax(b)] - t_bal))
    assert err[0] > err[1] > err[2] > err[3]
    assert err[-1] < 0.05


def test_solution_satisfies_the_transport_equation():
    """Finite-difference residual of d b/dt + v.grad b - D lap b = 0 must fall
    at second order as the grid is refined."""
    p = truth()
    v = TP.drift_vector(p["vmag"], p["vth"], p["vaz"])
    t0 = p["t0"] + 8.0
    r0 = np.array([6e-3, 1.0e-3, 0.5e-3])
    errs = []
    for h, dt in ((4e-4, 0.10), (2e-4, 0.05), (1e-4, 0.025)):
        pts, idx = [], {}
        for key, off in (("c", (0, 0, 0)), ("xp", (1, 0, 0)), ("xm", (-1, 0, 0)),
                         ("yp", (0, 1, 0)), ("ym", (0, -1, 0)),
                         ("zp", (0, 0, 1)), ("zm", (0, 0, -1))):
            idx[key] = len(pts)
            pts.append(r0 + h * np.array(off, float))
        pts = np.array(pts)
        ts = np.array([t0 - dt, t0, t0 + dt])
        b = TP.occupancy(pts, ts, p)
        c = b[idx["c"]]
        dbdt = (c[2] - c[0]) / (2 * dt)
        grad = np.array([(b[idx[a]][1] - b[idx[m]][1]) / (2 * h)
                         for a, m in (("xp", "xm"), ("yp", "ym"), ("zp", "zm"))])
        lap = sum((b[idx[a]][1] + b[idx[m]][1] - 2 * c[1]) / h ** 2
                  for a, m in (("xp", "xm"), ("yp", "ym"), ("zp", "zm")))
        res = dbdt + np.dot(v, grad) - p["D"] * lap
        errs.append(abs(res) / abs(c[1]))
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 2e-3


def test_first_moment_grows_with_offset_along_the_drift():
    p = truth()
    v = TP.drift_vector(p["vmag"], p["vth"], p["vaz"])
    u = v / np.linalg.norm(v)
    t = np.linspace(p["t0"] + 0.02, p["t0"] + 200, 8000)
    r = np.array([4e-3 * u, 8e-3 * u])
    _, m1, _ = TP.moments(TP.occupancy(r, t, p), t)
    assert m1[1] > m1[0]


def test_normalized_curves_lose_the_drift_direction():
    """Under unit-area normalization the occupancy depends on position only
    through the distance from the source.

    b carries exp(r.v / 2D) as a factor that does not depend on time, so it
    cancels in the ratio and the normalized curve is a function of |r|, |v|, D
    and t_0 alone.  Normalization therefore removes the direction of the drift
    as exactly as it removes phi and A."""
    p = truth()
    t = np.linspace(p["t0"] + 0.5, p["t0"] + 90, 400)
    R = 6.0e-3
    r = np.array([[R, 0.0, 0.0], [0.0, R, 0.0], [-R / np.sqrt(2), 0.0, R / np.sqrt(2)]])
    n = TP.normalize(TP.occupancy(r, t, p), t)
    assert np.allclose(n[0], n[1], rtol=1e-12, atol=0)
    assert np.allclose(n[0], n[2], rtol=1e-12, atol=0)
    # and the absolute curves are not the same, so the information was there
    b = TP.occupancy(r, t, p)
    assert not np.allclose(b[0], b[1], rtol=1e-6)
