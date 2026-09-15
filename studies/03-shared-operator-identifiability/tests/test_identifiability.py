"""The identifiability statements the result rests on.

If a change to the forward model would invalidate one of these, it fails here
first.
"""
import numpy as np
import pytest

from operatorid import Acquisition, Inversion
from operatorid.infer import diagnostics as DG
from operatorid.model import WINDOW_FREE, level_free
from operatorid.params import PSF_NAMES, psf_variants


def _bmode_alone():
    return Inversion(free=WINDOW_FREE["bmode"] + ("w_bmode",),
                     level="independent", windows=("bmode",),
                     overrides=psf_variants())


def _ceus_alone(n_frames_used=0):
    a = Acquisition()
    a.ceus.n_frames_used = n_frames_used
    return Inversion(free=WINDOW_FREE["ceus"] + ("w_ceus",),
                     level="independent", windows=("ceus",), acq=a,
                     overrides=psf_variants())


def test_the_margin_width_is_structurally_unidentifiable_from_bmode_alone():
    """An exact null direction, and it is the quadrature relation.

    The analytic null of sqrt(sig_e^2 + w^2) = constant, in logarithms, has
    d(log sig_e) / d(log w) = -w^2 / sig_e^2."""
    inv = _bmode_alone()
    sr = DG.structural_rank(inv)
    assert sr["rank"] == sr["n_par"] - 1
    v = sr["null_space_log"][:, 0]
    i, j = inv.space.index("sig_e"), inv.space.index("w_bmode")
    p = inv.truth_physical()
    want = -(p["w0"] ** 2) / (p["sig_e"] ** 2)
    assert (v[i] / v[j]) == pytest.approx(want, rel=0.02)
    others = [k for k in range(len(v)) if k not in (i, j)]
    assert np.abs(v[others]).max() < 0.02


def test_the_contrast_window_separates_the_blur_from_the_dispersion():
    inv = _ceus_alone()
    sr = DG.structural_rank(inv)
    assert sr["rank"] == sr["n_par"]           # full rank with every frame


def test_one_contrast_frame_cannot_separate_them():
    """The information-removal control is a prediction of the algebra: with one
    frame the observed variance is a single number, 2 D dt + w^2."""
    inv = _ceus_alone(n_frames_used=1)
    sr = DG.structural_rank(inv)
    assert sr["rank"] == sr["n_par"] - 1
    v = sr["null_space_log"][:, 0]
    i, j = inv.space.index("D"), inv.space.index("w_ceus")
    assert abs(v[i]) > 0.2 and abs(v[j]) > 0.2
    assert v[i] * v[j] < 0                     # they trade off
    others = [k for k in range(len(v)) if k not in (i, j)]
    assert np.abs(v[others]).max() < 0.05


def test_the_contrast_window_supplies_most_of_the_shared_width():
    """Which measurement supplies the operator information, and by how much."""
    free = sum([WINDOW_FREE[w] for w in ("bmode", "swe", "ceus")], ())
    inv = Inversion(free=free + ("w0",), level="shared")
    s = DG.window_sensitivity(inv)
    assert s["ceus"]["w0"] > 5 * s["swe"]["w0"]
    assert s["ceus"]["w0"] > 5 * s["bmode"]["w0"]


def test_the_joint_model_has_no_exact_null_direction():
    free = sum([WINDOW_FREE[w] for w in ("bmode", "swe", "ceus")], ())
    for level in ("independent", "shared"):
        ov = psf_variants() if level == "independent" else {}
        inv = Inversion(free=free + level_free(level), level=level,
                        overrides=ov)
        sr = DG.structural_rank(inv)
        if level == "shared":
            assert sr["rank"] == sr["n_par"], level
        else:
            # the independent arm keeps the B-mode degeneracy, and only that one
            assert sr["rank"] == sr["n_par"] - 1, level


def _rescale(p, a):
    """The coordinate-scale transformation of METHODS.md.

    Lengths and speeds in true units divide by ``a`` while the scale multiplies
    by it, so every reconstructed quantity is unchanged.  The dispersion carries
    two powers because it is a length squared over time.  The point spread does
    **not** appear: it is already a reconstructed length.
    """
    q = dict(p)
    q["scale"] = p["scale"] * a
    for n in ("x_c", "R", "sig_e", "v", "c_s"):
        q[n] = p[n] / a
    q["D"] = p["D"] / a ** 2
    return q


def test_an_uncertain_scale_is_an_exact_symmetry_of_every_window():
    """The reason the coordinate-scale mechanism was rejected as the shared
    effect.  Asserted as an identity rather than as a numerical rank: the
    symmetry is exact in the equations, while a finite-difference Jacobian
    resolves it only to about one part in ten million."""
    free = sum([WINDOW_FREE[w] for w in ("bmode", "swe", "ceus")], ())
    inv = Inversion(free=free + ("w0", "scale"), level="shared")
    p0 = inv.truth_physical()
    sl, sig = inv.window_slices(), inv.sigma()
    for a in (1.02, 1.10, 1.50):
        q = _rescale(p0, a)
        for w in ("bmode", "swe", "ceus"):
            r = (inv._window(w, q) - inv._window(w, p0)) / sig[sl[w]]
            assert float((r ** 2).sum()) < 1e-16, (a, w)


def test_the_scale_symmetry_does_not_touch_the_shared_point_spread():
    """Which is why it is orthogonal to the question this study asks: the
    transformation above leaves every point-spread width exactly alone."""
    free = sum([WINDOW_FREE[w] for w in ("bmode", "swe", "ceus")], ())
    inv = Inversion(free=free + ("w0", "scale"), level="shared")
    p0 = inv.truth_physical()
    q = _rescale(p0, 1.50)
    assert q["w0"] == p0["w0"]
    for m in ("bmode", "swe", "ceus"):
        assert inv.width(q, m) == inv.width(p0, m)


def test_the_scale_direction_is_the_least_informed_one_when_it_is_free():
    """And it is recognizable in the spectrum: freeing the scale adds a
    direction far below every other, which is the symmetry showing through the
    finite-difference floor."""
    free = sum([WINDOW_FREE[w] for w in ("bmode", "swe", "ceus")], ())
    inv = Inversion(free=free + ("w0", "scale"), level="shared")
    s = DG.structural_rank(inv)["singular_values"]
    assert s[-1] / s.max() < 1e-5
    assert s[-2] / s[-1] > 1e4                 # and it is isolated
    _, _, vt = np.linalg.svd(DG.scaled_jacobian(inv))
    v = vt[-1]
    assert abs(v[inv.space.index("scale")]) > 0.9   # it is the scale direction
    assert abs(v[inv.space.index("w0")]) < 0.02     # not the point spread
