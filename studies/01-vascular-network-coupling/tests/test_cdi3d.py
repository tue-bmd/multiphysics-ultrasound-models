"""The 3D convective-dispersion estimator against cases with a known answer.

Three kinds of test, in order of how much they isolate:

1. A superposition of periodic Fourier modes of the CONCENTRATION field (not
   of any acoustic field: no ultrasound propagation is simulated anywhere in
   this package).  Such a superposition solves the equation exactly for a
   given tensor and velocity and is periodic in space, so that with a circular
   convolution there is no boundary error and no truncation error at all.
   Differentiating on the whole record and fitting on its interior removes the
   temporal boundary error too.  What is left is the discretisation of the
   Gaussian derivatives, so the estimator must recover the coefficients to
   better than a percent.  This is the test of the implementation.

2. The two analytic fields used for the same purpose by the source (R. R.
   Wildeboer, PhD thesis, TU/e, 2019, chapter 7, equations 7.16 and 7.17): a
   rigidly translating Gaussian blob, and the diffusion Green's function.
   These are not periodic and are observed through a finite window, so the
   tolerances are looser; they test the estimator as it would be used.

3. Properties of the estimator that the comparison with the shell estimator is
   about: that the answer depends on the derivative scale, that the ridge term
   shrinks, and that an undersampled recording is not silently wrong but
   visibly so.

All lengths are in metres and all times in seconds, as everywhere in this
package, so a voxel of 0.75 mm is 0.75e-3 and a dispersion of 1 mm^2/s is 1e-6.
"""
import numpy as np
import pytest

from porovasc.physics import cdi3d


DX = 0.75e-3           # voxel spacing, as in the clinical acquisition


# --------------------------------------------------------------- exact fields
def fourier_modes(D, v, n=24, nt=48, dt=0.5, dx=DX, seed=0,
                modes=((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
                       (0, 1, 1), (1, 1, 1), (2, 1, 0), (1, 2, 1))):
    """A superposition of periodic Fourier modes of the concentration, each of
    which solves

        dC/dt = div(D grad C) - v . grad C

    exactly: for C = exp(-(k.D k) t) cos(k.r - (v.k) t + phase) the three terms
    match term by term.  The wave vectors are on the grid of the periodic box,
    so the field is exactly periodic in space.  Several directions are needed
    because one mode alone cannot determine nine coefficients.  These are a
    test pattern for the fitting code; they are not acoustic waves."""
    D = np.asarray(D, float)
    if D.ndim == 0:
        D = float(D) * np.eye(3)
    v = np.asarray(v, float)
    L = n * dx
    x = np.arange(n) * dx
    t = np.arange(nt) * dt
    X, Y, Z, T = np.meshgrid(x, x, x, t, indexing="ij")
    rng = np.random.default_rng(seed)
    C = np.zeros_like(X)
    for m in modes:
        k = 2 * np.pi * np.asarray(m, float) / L
        decay = float(k @ D @ k)
        C += rng.uniform(0.5, 1.5) * np.exp(-decay * T) * \
            np.cos(k[0] * X + k[1] * Y + k[2] * Z - float(v @ k) * T + rng.uniform(0, 2 * np.pi))
    return C


def translating_blob(v_y, gamma=8e-6, C0=400.0, n=47, nt=48, dt=0.5, dx=DX):
    """Equation 7.16: a Gaussian blob translating along y, centred on the
    volume at the middle frame.  Solves the equation with D = 0 and
    v = (0, v_y, 0)."""
    x = (np.arange(n) - (n - 1) / 2) * dx
    t = (np.arange(nt) - (nt - 1) / 2) * dt
    X, Y, Z, T = np.meshgrid(x, x, x, t, indexing="ij")
    return C0 * np.exp(-((X ** 2 + (Y - v_y * T) ** 2 + Z ** 2) / gamma))


def diffusing_source(D, C0=5e8, n=41, nt=48, dt=0.5, dx=DX, t0=10.0):
    """Equation 7.17: the Green's function of the diffusion equation, which
    solves the equation with v = 0 and an isotropic D."""
    x = (np.arange(n) - (n - 1) / 2) * dx
    t = t0 + np.arange(nt) * dt
    X, Y, Z, T = np.meshgrid(x, x, x, t, indexing="ij")
    r2 = X ** 2 + Y ** 2 + Z ** 2
    return C0 / (4 * np.pi * D * T) ** 1.5 * np.exp(-r2 / (4 * D * T))


def _centre(C, **kw):
    """Fit at the centre of the volume.  By default on the interior of the
    record: the temporal convolution reaches about four sigma_t beyond each
    frame it differentiates, so the ends of any finite recording are biased
    (see test_fitting_across_the_ends_of_the_record_biases_the_estimate)."""
    n = C.shape[0]
    c = (n // 2, n // 2, n // 2)
    kw.setdefault("s", 7)
    kw.setdefault("mode", ("reflect", "nearest"))
    nt = C.shape[3]
    kw.setdefault("t_slice", slice(nt // 3, 2 * nt // 3))
    out = cdi3d.fit_volume(C, DX, kw.pop("dt", 0.5), kw.pop("sigma_x", 1.5e-3),
                           kw.pop("sigma_t", 2.0), centres=[c], **kw)
    assert c in out, "the kernel should fit at the centre of the volume"
    return out[c]


# ------------------------------------------------- 1. exact, no boundary error
def test_fourier_modes_recover_an_isotropic_tensor_and_a_velocity():
    """With a periodic field, a circular convolution in space and the fit
    taken on the interior of the record, only the discretisation of the
    Gaussian derivatives is left: the coefficients must come back."""
    D, v = 1.0e-6, np.array([0.3e-3, -0.5e-3, 0.2e-3])
    C = fourier_modes(D, v)
    f = _centre(C, dt=0.5, sigma_x=1.5e-3, sigma_t=2.0, l0=0.0,
                mode=("wrap", "nearest"), t_slice=slice(16, 32))
    assert abs(f.D_CD / D - 1) < 0.01
    assert np.allclose(f.v, v, rtol=0.01, atol=1e-6)
    assert f.residual < 1e-3


def test_fourier_modes_recover_an_anisotropic_tensor():
    """The estimator fits a full tensor, so an anisotropic one must come back
    element by element, not only through its trace."""
    D = np.array([[1.2e-6, 0.3e-6, 0.0],
                  [0.3e-6, 0.6e-6, -0.2e-6],
                  [0.0, -0.2e-6, 0.9e-6]])
    v = np.array([0.4e-3, 0.0, -0.2e-3])
    C = fourier_modes(D, v)
    f = _centre(C, dt=0.5, sigma_x=1.5e-3, sigma_t=2.0, l0=0.0,
                mode=("wrap", "nearest"), t_slice=slice(16, 32))
    assert np.allclose(f.D, D, rtol=0.02, atol=2e-8)
    assert np.allclose(f.v, v, rtol=0.02, atol=1e-6)
    # and the reported scalar is the mean of the diagonal, not of everything
    assert np.isclose(f.D_CD, np.trace(D) / 3, rtol=0.02)


def test_the_trace_hides_anisotropy_by_construction():
    """D_CD cannot distinguish an anisotropic tensor from an isotropic one of
    the same trace.  This is a property of the reported scalar, and it is part
    of what the comparison with the shell estimator is about."""
    aniso = np.diag([1.8e-6, 0.6e-6, 0.6e-6])
    iso = np.eye(3) * (np.trace(aniso) / 3)
    v = np.array([0.3e-3, 0.0, 0.0])
    kw = dict(dt=0.5, sigma_x=1.5e-3, sigma_t=2.0, l0=0.0,
              mode=("wrap", "nearest"), t_slice=slice(16, 32))
    a = _centre(fourier_modes(aniso, v), **kw)
    b = _centre(fourier_modes(iso, v), **kw)
    assert np.isclose(a.D_CD, b.D_CD, rtol=0.03)          # identical scalar
    assert not np.allclose(a.D, b.D, rtol=0.2)            # different tensor


# ------------------------------------------- 2. the source's analytic fields
@pytest.mark.parametrize("v_y_mm_s", [0.25, 0.5, 1.0])
def test_translating_blob_recovers_the_speed_and_its_direction(v_y_mm_s):
    v = v_y_mm_s * 1e-3
    f = _centre(translating_blob(v), l0=0.0)
    assert abs(f.v[1] / v - 1) < 0.15
    assert abs(f.v[0]) < 0.15 * abs(f.v[1]) and abs(f.v[2]) < 0.15 * abs(f.v[1])
    assert abs(f.D_CD) < 0.25e-6                          # D consistent with 0


def test_velocity_is_monotone_in_the_true_speed():
    got = [_centre(translating_blob(v), l0=0.0).v[1]
           for v in (0.25e-3, 0.5e-3, 0.75e-3, 1.0e-3)]
    assert all(b > a for a, b in zip(got, got[1:]))


@pytest.mark.parametrize("D_mm2_s", [0.6, 1.0, 1.8])
def test_diffusing_source_recovers_the_dispersion(D_mm2_s):
    D = D_mm2_s * 1e-6
    f = _centre(diffusing_source(D), l0=0.0)
    assert abs(f.D_CD / D - 1) < 0.10
    off = np.array([f.D[0, 1], f.D[0, 2], f.D[1, 2]])
    assert np.all(np.abs(off) < 0.25 * f.D_CD)            # nearly isotropic
    assert f.v_CD < 0.2e-3                                # v consistent with 0


def test_dispersion_is_monotone_in_the_true_dispersion():
    got = [_centre(diffusing_source(D), l0=0.0).D_CD
           for D in (0.6e-6, 1.0e-6, 1.4e-6, 1.8e-6)]
    assert all(b > a for a, b in zip(got, got[1:]))


# ------------------------------------------------ 3. properties of the estimator
def test_the_estimate_is_scale_invariant_on_a_homogeneous_solution():
    """If the medium really obeys the equation with constant coefficients, the
    derivative scale cannot matter: convolution commutes with a linear
    constant-coefficient operator, so the smoothed field satisfies the same
    equation with the same coefficients.  The estimator must show this."""
    C = diffusing_source(1.0e-6)
    vals = [_centre(C, sigma_x=sx, l0=0.0).D_CD for sx in (0.75e-3, 1.5e-3, 3.0e-3)]
    assert max(vals) / min(vals) < 1.02
    assert all(abs(x / 1.0e-6 - 1) < 0.10 for x in vals)


def test_the_estimate_depends_on_scale_when_the_medium_is_heterogeneous():
    """The converse, and the reason the derivative scale is an analysis choice
    rather than a detail: where the medium does not obey the equation with one
    set of coefficients, the answer depends on how far the derivative kernel
    reaches.  Here two sources of different dispersion sit either side of the
    fitted point, and a wider kernel mixes more of both."""
    n = 61
    # the peak of a diffusing source scales as D^(-3/2), so the amplitudes are
    # matched, otherwise the slower source simply dominates by brightness
    lo = diffusing_source(0.4e-6, C0=5e8 * 0.4 ** 1.5, n=n)
    hi = diffusing_source(2.0e-6, C0=5e8 * 2.0 ** 1.5, n=n)
    shift = 6                                   # voxels, i.e. 4.5 mm each way
    C = np.roll(lo, -shift, axis=0) + np.roll(hi, shift, axis=0)
    vals = [_centre(C, sigma_x=sx, l0=0.0).D_CD for sx in (0.75e-3, 1.5e-3, 3.0e-3)]
    assert max(vals) / min(vals) > 1.15
    assert all(np.isfinite(x) for x in vals)
    # and it is the wider kernel that sees more of the faster source
    assert vals[2] > vals[0]


def test_ridge_shrinks_towards_zero_as_l0_grows():
    """The penalty is scaled to the data, so l0 acts whatever the amplitude of
    the concentration is."""
    C = diffusing_source(1.0e-6)
    weak = _centre(C, l0=1e-6).D_CD
    mid = _centre(C, l0=1e-1).D_CD
    strong = _centre(C, l0=1e3).D_CD
    assert abs(strong) < abs(mid) < abs(weak)
    assert abs(strong) < 0.1 * abs(weak)


def test_the_penalty_does_not_depend_on_the_amplitude_of_the_data():
    """Doubling the concentration must not change the estimate: it is the same
    tissue.  Without scaling the penalty to the data this would fail, because
    l0 would be weaker against a larger signal."""
    C = diffusing_source(1.0e-6)
    assert np.isclose(_centre(C, l0=0.1).D_CD, _centre(1000 * C, l0=0.1).D_CD, rtol=1e-9)


def test_fitting_across_the_ends_of_the_record_biases_the_estimate():
    """The temporal convolution reaches about four sigma_t beyond each frame it
    differentiates, so the first and last frames of a finite recording are
    computed partly from data that does not exist.  Including them biases the
    dispersion low by tens of percent, while the interior of the same record
    recovers it.  For a clinical acquisition with sigma_t = 4 s that is some
    16 s at each end of the recording."""
    D = 1.0e-6
    C = diffusing_source(D)
    interior = _centre(C, l0=0.0)
    whole = _centre(C, l0=0.0, t_slice=slice(None))
    assert abs(interior.D_CD / D - 1) < 0.10          # the interior is right
    assert whole.D_CD < 0.8 * D                       # the whole record is not
    assert whole.residual > 10 * interior.residual    # and the residual says so


def test_an_undersampled_recording_is_visibly_wrong_not_silently_wrong():
    """Sampling is an acquisition setting, and below Nyquist the estimate stops
    meaning anything.  The same periodic field is observed over the same 80 s,
    once at 1 Hz and once at the 0.25 Hz of a clinical 3D acquisition.  At
    0.25 Hz the faster spatial modes complete more than half a cycle between
    frames, and the recovered velocity no longer tracks the truth."""
    D, v = 0.3e-6, np.array([0.0, 2.25e-3, 0.0])
    kw = dict(sigma_x=1.5e-3, l0=0.0, mode=("wrap", "nearest"))
    fine = _centre(fourier_modes(D, v, nt=80, dt=1.0), dt=1.0, sigma_t=2.0,
                   t_slice=slice(30, 50), **kw)
    coarse = _centre(fourier_modes(D, v, nt=20, dt=4.0), dt=4.0, sigma_t=4.0,
                     t_slice=slice(7, 13), **kw)
    assert abs(fine.v[1] / v[1] - 1) < 0.05           # sampled: recovers it
    assert abs(coarse.v[1] / v[1] - 1) > 0.3          # aliased: does not


# ------------------------------------------------------------ bookkeeping
def test_kernel_is_spherical_and_grows_with_s():
    n3, n7, n11 = (len(cdi3d.kernel_offsets(s)) for s in (3, 7, 11))
    assert n3 < n7 < n11
    for s in (3, 7, 11):
        off = cdi3d.kernel_offsets(s)
        assert np.all(np.sum(off ** 2, axis=1) <= (s / 2.0) ** 2 + 1e-9)
        assert (off == 0).all(axis=1).sum() == 1          # contains its centre


def test_edge_voxels_are_absent_rather_than_wrong():
    C = diffusing_source(1.0e-6)
    n = C.shape[0]
    out = cdi3d.fit_volume(C, DX, 0.5, 1.5e-3, 2.0, s=7, centres=[(0, 0, 0), (n // 2,) * 3])
    assert (0, 0, 0) not in out and (n // 2,) * 3 in out


def test_fit_reports_the_size_of_the_problem_it_solved():
    C = diffusing_source(1.0e-6)
    f = _centre(C, l0=0.0)
    nt_used = len(range(*slice(C.shape[3] // 3, 2 * C.shape[3] // 3).indices(C.shape[3])))
    assert f.n_rows == len(cdi3d.kernel_offsets(7)) * nt_used
    assert 0.0 <= f.residual < 1.0 and np.isfinite(f.condition)
