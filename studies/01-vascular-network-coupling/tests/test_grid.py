"""The dense grid and the acquisition operations applied to it."""
import numpy as np
import pytest

from porovasc import grid as G
from porovasc.physics import transport as TR


def _toy(n=9, nf=6, dx=0.75e-3, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.random((n, n, n, nf))
    return G.Grid(C=C, t=np.arange(nf) * 1.0, origin=np.zeros(3), dx=dx,
                  frame_dt=1.0, vox_half=dx / 2)


def test_margin_covers_the_kernel_and_the_derivative_support():
    """The fit at one voxel reaches the kernel radius plus the truncation of
    the Gaussian derivative, and the margin has to hold both."""
    dx = 0.75e-3
    assert G.margin_voxels(1.5e-3, dx, 7) == int(np.ceil(4 * 1.5e-3 / dx)) + 3
    assert G.margin_voxels(1.5e-3, dx, 7) == 11
    assert G.margin_voxels(0.75e-3, dx, 7) == 7
    assert G.margin_voxels(1.5e-3, dx, 11) > G.margin_voxels(1.5e-3, dx, 7)
    assert G.margin_voxels(3.0e-3, dx, 7) > G.margin_voxels(1.5e-3, dx, 7)


def test_block_holds_the_region_of_interest_with_its_margin():
    n, origin, m = G.grid_for(np.zeros(3), 3e-3, 0.75e-3, 1.5e-3, 7)
    assert m == 11 and n == 8 + 2 * 11 + 1
    # the block is centred, so the inner cube sits inside it with margin m
    g = G.Grid(C=np.zeros((n, n, n, 2)), t=np.zeros(2), origin=origin, dx=0.75e-3,
               frame_dt=1.0, vox_half=0.375e-3)
    idx, off = g.index_of(np.zeros(3))
    assert off < 1e-12 and g.contains(idx, margin=m)


def test_frames_average_rather_than_subsample():
    """A frame integrates; subsampling would alias whatever happens between
    frames.  A curve that alternates between two values must therefore reduce
    to their mean, not to one of them."""
    t, per, nf = G.frame_times(4.0)
    assert per == int(round(4.0 / TR.DT)) and nf * per <= len(TR.time_grid()[0])
    alt = np.tile([0.0, 2.0], len(TR.time_grid()[0]) // 2)
    red = G._reduce(alt, per, nf)
    assert np.allclose(red, 1.0)


def test_frame_times_reject_a_frame_shorter_than_the_internal_step():
    with pytest.raises(ValueError):
        G.frame_times(TR.DT / 2)


def test_rebinning_averages_whole_frames_and_conserves_the_mean():
    g = _toy(nf=12)
    r = G.rebin(g, 3.0)
    assert r.C.shape[3] == 4 and r.frame_dt == 3.0
    assert np.allclose(r.C[..., 0], g.C[..., 0:3].mean(axis=3))
    assert np.isclose(r.C.mean(), g.C.mean())
    assert G.rebin(g, 1.0) is g                       # a no-op returns the same grid


def test_rebinning_refuses_a_non_multiple():
    with pytest.raises(ValueError):
        G.rebin(_toy(), 1.5)


def test_psf_conserves_bubbles_and_smooths():
    """The point spread function redistributes the signal, it does not create
    or destroy it, and a wider one smooths more.  Conservation is exact for a
    source well inside the block; over the whole block it holds only to the
    accuracy of the edge rule, which extends the tissue rather than truncating
    it."""
    g = _toy(n=21, nf=3)
    rough = lambda A: np.abs(np.diff(A, axis=0)).mean()
    a = G.apply_psf(g, 1.1e-3)
    b = G.apply_psf(g, 2.0e-3)
    assert np.isclose(a.C.sum(), g.C.sum(), rtol=1e-4)
    assert rough(b.C) < rough(a.C) < rough(g.C)
    assert G.apply_psf(g, 0) is g                     # no blur is one of the settings
    # a single bubble in the middle stays one bubble, and spreads
    pt = G.Grid(C=np.zeros((21, 21, 21, 1)), t=np.zeros(1), origin=np.zeros(3),
                dx=g.dx, frame_dt=1.0, vox_half=g.vox_half)
    pt.C[10, 10, 10, 0] = 1.0
    sm = G.apply_psf(pt, 1.1e-3)
    assert np.isclose(sm.C.sum(), 1.0, rtol=1e-9)
    assert sm.C[10, 10, 10, 0] < 0.5 and sm.C[11, 10, 10, 0] > 0.05


def test_sampling_is_exact_on_grid_nodes_and_linear_between_them():
    g = _toy(n=9, nf=4)
    node = g.position_of((4, 4, 4))
    assert np.allclose(G.sample_at(g, [node])[0], g.C[4, 4, 4, :])
    mid = (g.position_of((4, 4, 4)) + g.position_of((5, 4, 4))) / 2
    assert np.allclose(G.sample_at(g, [mid])[0], 0.5 * (g.C[4, 4, 4, :] + g.C[5, 4, 4, :]))


def test_sampling_outside_the_block_raises_rather_than_clamping():
    g = _toy(n=9)
    with pytest.raises(ValueError):
        G.sample_at(g, [g.origin - g.dx])
    with pytest.raises(ValueError):
        G.sample_at(g, [g.origin + 9 * g.dx])


@pytest.mark.slow
def test_grid_curves_equal_a_direct_propagation():
    """The chunked, frame-averaged grid must agree with propagating the same
    voxels one at a time: chunking and reduction must not change the physics."""
    from porovasc.geometry import network as N
    from porovasc.physics import flow as F
    from porovasc.config import BASE, apply_globals
    apply_globals(dict(BASE))
    c = np.array([-5e-3, -8e-3, 0.0])
    net = N.build(N.Params(seed=100, d_term_gland=60e-6, d_term_rve=40e-6))
    fl = F.solve(net)
    g = G.propagate_grid(net, fl, c, 1.5e-3, sigma_x=0.75e-3, s=7, frame_dt=4.0, chunk=500)
    peak = g.C.max(axis=3)
    bright = np.dstack(np.unravel_index(np.argsort(peak.ravel())[::-1][:5], peak.shape))[0]
    ks = [(g.position_of(tuple(b)), g.vox_half) for b in bright]
    tr = TR.propagate(net, fl, ks, poiseuille=False)
    _, per, nf = G.frame_times(4.0)
    for q, b in enumerate(bright):
        ref = G._reduce(tr.tic[q], per, nf)
        assert ref.max() > 0
        assert np.allclose(g.C[tuple(b)], ref, rtol=1e-12, atol=0)


def test_psf_can_be_anisotropic():
    """A real three-dimensional contrast point spread function is not
    isotropic: axial, lateral and elevational extents differ by a factor of a
    few.  An anisotropic blur must smooth each axis by its own amount, which
    matters because the grid estimator fits a tensor and would otherwise
    attribute the instrument's anisotropy to the tissue."""
    g = _toy(n=25, nf=2)
    pt = G.Grid(C=np.zeros((25, 25, 25, 1)), t=np.zeros(1), origin=np.zeros(3),
                dx=g.dx, frame_dt=1.0, vox_half=g.vox_half)
    pt.C[12, 12, 12, 0] = 1.0
    sm = G.apply_psf(pt, (0.8e-3, 1.1e-3, 2.5e-3))
    assert np.isclose(sm.C.sum(), 1.0, rtol=1e-9)
    # the second moment along each axis must follow the requested widths
    ax = np.arange(25) * g.dx
    mom = []
    for axis in (0, 1, 2):
        prof = sm.C[..., 0].sum(axis=tuple(a for a in (0, 1, 2) if a != axis))
        m = (prof * ax).sum() / prof.sum()
        mom.append(np.sqrt((prof * (ax - m) ** 2).sum() / prof.sum()))
    assert mom[0] < mom[1] < mom[2]
    # and each matches its own sigma, the voxel adding its own width
    for got, fwhm in zip(mom, (0.8e-3, 1.1e-3, 2.5e-3)):
        assert abs(got - fwhm / 2.35482) < 0.15e-3
    # one number still means isotropic
    iso = G.apply_psf(pt, 1.1e-3)
    assert np.allclose(iso.C, G.apply_psf(pt, (1.1e-3,) * 3).C)
