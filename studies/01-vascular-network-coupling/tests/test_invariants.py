"""Invariant tests: each module's checker must pass on a small gland.

    pytest -q                 (about 2 min: one whole-gland build, shared by all tests)
    pytest -q -m "not slow"   (runs the fast analytic suite in test_analytic.py)

The checkers print their evidence; run with -s to see it.
"""
import pytest

pytestmark = pytest.mark.slow      # every test here builds a gland

import io, contextlib
import numpy as np
import pytest

from porovasc.geometry import network as N
from porovasc.physics import flow as F, drainage as DR, transport as TR
from porovasc.homogenise import darcy as DA
from porovasc.debug import check_geometry as CG, check_flow as CF, check_drainage as CD, check_transport as CT


@pytest.fixture(scope="module")
def gland():
    prm = N.Params(seed=100, rve_centres=((-5e-3, -8e-3, 0.0),))
    net = N.build(prm)
    fl = F.solve(net)
    return prm, net, fl


@pytest.mark.slow
def test_geometry_invariants(gland):
    prm, net, fl = gland
    with contextlib.redirect_stdout(io.StringIO()):
        assert CG.check(net)


@pytest.mark.slow
def test_flow_invariants(gland):
    prm, net, fl = gland
    with contextlib.redirect_stdout(io.StringIO()):
        assert CF.check(net, fl)
    # perfusion is an OUTPUT of the pressure-driven solve, not an input
    Q = np.abs(fl.Q[net.gen == 0]).sum() * 6e7               # mL/min
    mass_g = 4 / 3 * np.pi * np.prod(N.GLAND_SEMI) * 1e6 * 1.05
    assert 10 < 100 * Q / mass_g < 40                         # mL/min/100 g


@pytest.mark.slow
def test_drainage_closure_identity(gland):
    prm, net, fl = gland
    c = np.asarray(prm.rve_centres[0])
    dr = DR.solve(net, fl, c, R_comp=3e-3, dP_ext=100.0, H=5e3)
    # fractional relaxation / applied strain = vascular volume fraction (derived on the network)
    assert abs(-dr.dV_inf / dr.V_tissue / (100.0 / 5e3) / dr.phi - 1) < 1e-4
    assert 1e-3 < dr.tau_rc < 0.2
    with contextlib.redirect_stdout(io.StringIO()):
        ok, _ = CD.check(net, fl, c, 3e-3)         # scaling in H and in radius
    assert ok


@pytest.mark.slow
def test_transport_central_volume(gland):
    """TIC area = sum over segments of (partial-volume weight x Q x mean residence)
    + bed volume seen by the voxel (each bed node's Q x tau_cap times the fraction
    of its R_LAT territory inside the voxel).  Plug flow: visible = geometric volume."""
    prm, net, fl = gland
    c = np.asarray(prm.rve_centres[0]); h = 0.75e-3
    w = N.cube_fraction(net.p0, net.p1, c, h); idx = np.where(w > 0)[0]
    qn = {}
    for e in range(len(fl.bed_a)): qn[int(fl.bed_a[e])] = qn.get(int(fl.bed_a[e]), 0.0) + abs(fl.Q_bed[e])
    bn = np.array(sorted(qn)); wb = TR._bed_overlap(net.p1[bn], c, h, F.R_LAT)
    bed = sum(wb[j] * qn[int(bn[j])] for j in range(len(bn))) * TR.TAU_CAP
    for pois in (False, True):
        tr = TR.propagate(net, fl, [(c, h)], poiseuille=pois)
        vis = sum(w[i] * tr.V_vis.get(i, 0.0) for i in idx)
        area = tr.tic[0].sum() * TR.DT
        assert abs(area / (vis + bed) - 1) < 0.02
        if not pois:
            geo = (w * np.pi * net.r ** 2 * net.Lpath)[idx].sum()
            assert abs(vis / geo - 1) < 1e-6


@pytest.mark.slow
def test_darcy_measurement(gland):
    prm, net, fl = gland
    c = np.asarray(prm.rve_centres[0])
    da = DA.measure(net, fl, c, prm.rve_half)
    assert np.all(da.k_face >= 0) and 0 < da.phi < 0.2
    assert da.c_k_bundle > 32                                   # 32 <T^2>, T > 1
