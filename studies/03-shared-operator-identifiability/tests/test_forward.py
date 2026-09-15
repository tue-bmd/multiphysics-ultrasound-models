"""Limiting cases and analytic identities of the three forward models."""
import numpy as np
import pytest

from operatorid import Acquisition, Inversion
from operatorid.constants import (GAMMA_NOMINAL, TRACK_KERNEL_M,
                               TRANSMIT_HZ, psf_widths)
from operatorid.models import bmode as BM, ceus as CE, swe as SW
from operatorid.params import TRUTH

A = Acquisition()
P = dict(TRUTH)


# ------------------------------------------------------------------ B-mode --

def test_zero_point_spread_returns_the_ideal_profile():
    x = A.bmode.positions()
    assert np.allclose(BM.profile(x, P, 0.0), BM.ideal(x, P), rtol=0, atol=0)


def test_the_profile_depends_on_the_widths_only_in_quadrature():
    """The exact degeneracy the whole study is built on, asserted directly:
    any pair with the same sqrt(sig_e^2 + w^2) gives the identical profile."""
    x = A.bmode.positions()
    tot = np.hypot(P["sig_e"], 0.35e-3)
    ref = BM.profile(x, P, 0.35e-3)
    for frac in (0.2, 0.5, 0.8, 0.95):
        q = dict(P, sig_e=frac * tot)
        w = np.sqrt(tot ** 2 - q["sig_e"] ** 2)
        assert np.allclose(BM.profile(x, q, w), ref, rtol=1e-12, atol=1e-14)


def test_the_profile_integrates_to_the_lesion_area():
    """Convolution conserves the integral, so blurring cannot change the area
    of the lesion above background."""
    x = np.linspace(-40e-3, 70e-3, 20001)
    area = [np.trapezoid(BM.profile(x, P, w) - P["b0"], x)
            for w in (0.0, 0.35e-3, 2.0e-3)]
    assert np.allclose(area, 2 * P["R"] * P["db"], rtol=1e-6)


def test_a_wide_point_spread_flattens_the_lesion():
    x = A.bmode.positions()
    contrast = [BM.profile(x, P, w).max() - BM.profile(x, P, w).min()
                for w in (0.2e-3, 2.0e-3, 8.0e-3)]
    assert contrast[0] > contrast[1] > contrast[2]


# -------------------------------------------------------------------- CEUS --

def test_blur_adds_to_the_variance_and_conserves_the_mass():
    x = np.linspace(-60e-3, 120e-3, 40001)
    t = np.array([8.0])
    dt = t[0] - P["t0"]
    for w in (0.0, 0.875e-3, 3.0e-3):
        c = CE.frames(x, t, P, w)[0]
        m0 = np.trapezoid(c, x)
        mean = np.trapezoid(x * c, x) / m0
        var = np.trapezoid((x - mean) ** 2 * c, x) / m0
        assert m0 == pytest.approx(P["a_C"], rel=1e-6)
        assert mean == pytest.approx(P["v"] * dt, rel=1e-6)
        assert var == pytest.approx(CE.observed_variance(P["D"], dt, w), rel=1e-6)


def test_the_variance_is_linear_in_time_with_the_blur_as_intercept():
    """The separation the study relies on: slope 2D, intercept w^2."""
    t = np.linspace(4.0, 20.0, 9)
    w = 0.875e-3
    var = CE.observed_variance(P["D"], t - P["t0"], w)
    slope, intercept = np.polyfit(t - P["t0"], var, 1)
    assert slope == pytest.approx(2 * P["D"], rel=1e-10)
    assert intercept == pytest.approx(w ** 2, rel=1e-10)


def test_frames_before_arrival_are_empty():
    x = A.ceus.positions()
    f = CE.frames(x, np.array([0.5 * P["t0"], 2 * P["t0"]]), P, 0.875e-3)
    assert np.all(f[0] == 0.0)
    assert f[1].max() > 0.0


# --------------------------------------------------------------------- SWE --

def test_zero_point_spread_returns_the_ideal_field():
    x, t = A.swe.positions(), A.swe.times()
    a = SW.displacement(x, t, P, 0.0, A.swe.f0, A.swe.bandwidth)
    b = SW.ideal(x, t, P, A.swe.f0, A.swe.bandwidth)
    assert np.allclose(a, b, rtol=0, atol=0)


def test_the_blur_is_a_low_pass_that_does_not_depend_on_distance():
    """Attenuation grows with propagation distance, the point spread does not.
    That difference is what makes them separable."""
    om = 2 * np.pi * np.array([100.0, 200.0, 400.0])
    f = SW.blur_factor(om, P["c_s"], 0.63e-3)
    assert np.all(np.diff(f) < 0)              # higher frequencies suppressed
    assert np.all(f <= 1.0)
    assert SW.blur_factor(0.0, P["c_s"], 0.63e-3) == pytest.approx(1.0)


def test_the_pulse_arrives_at_the_time_the_wave_speed_implies():
    x, t = A.swe.positions(), A.swe.times()
    u = SW.displacement(x, t, P, 0.0, A.swe.f0, A.swe.bandwidth)
    peak = t[np.argmax(np.abs(u), axis=1)]
    slope = np.polyfit(x, peak, 1)[0]
    assert 1.0 / slope == pytest.approx(P["c_s"], rel=0.02)


def test_attenuation_decays_the_amplitude_with_distance():
    x, t = A.swe.positions(), A.swe.times()
    u = SW.displacement(x, t, P, 0.0, A.swe.f0, A.swe.bandwidth)
    amp = np.abs(u).max(axis=1)
    assert np.all(np.diff(amp) < 0)


# --------------------------------------------------------------- the ratios -

def test_the_width_map_is_the_documented_expression():
    """`psf_widths` is not a free choice: B-mode is the reference, contrast
    scales with the wavelength, and the shear window adds a fixed tracking
    kernel in quadrature."""
    f, k = TRANSMIT_HZ, TRACK_KERNEL_M
    for w0 in (0.12e-3, 0.35e-3, 1.0e-3):
        W = psf_widths(w0)
        assert W["bmode"] == pytest.approx(w0, rel=1e-14)
        assert W["ceus"] == pytest.approx(f["bmode"] / f["ceus"] * w0, rel=1e-14)
        assert W["swe"] == pytest.approx(
            np.hypot(f["bmode"] / f["swe"] * w0, k), rel=1e-14)


def test_the_shear_width_is_not_proportional_to_the_aperture():
    """The tracking kernel is a fixed length, so the shear ratio is not a
    constant.  A linear 1.8 is right only at the nominal aperture, and the
    catalogue prior spans a factor of eight."""
    r = {w0: float(psf_widths(w0)["swe"] / w0)
         for w0 in (0.12e-3, 0.35e-3, 1.0e-3)}
    assert r[0.35e-3] == pytest.approx(1.807, abs=0.002)
    assert r[0.12e-3] > 3.0            # a narrow aperture: the kernel dominates
    assert r[1.0e-3] < 1.5             # a wide one: diffraction dominates
    assert r[0.12e-3] > 2 * r[1.0e-3]


def test_the_corrections_scale_the_windows_they_belong_to():
    W0 = psf_widths(0.35e-3)
    Wc = psf_widths(0.35e-3, {"ceus": 1.15})
    assert Wc["ceus"] == pytest.approx(1.15 * W0["ceus"], rel=1e-14)
    assert Wc["bmode"] == pytest.approx(W0["bmode"], rel=1e-14)
    assert Wc["swe"] == pytest.approx(W0["swe"], rel=1e-14)


def test_a_common_probe_does_not_mean_a_common_point_spread():
    """One aperture component, three different widths."""
    inv = Inversion(free=("w0", "sig_e"), level="shared", windows=("bmode",))
    p = inv.truth_physical()
    w = {m: inv.width(p, m) for m in ("bmode", "swe", "ceus")}
    assert w["bmode"] < w["swe"] < w["ceus"]
    for m, v in psf_widths(p["w0"]).items():
        assert w[m] == pytest.approx(float(v), rel=1e-12)
