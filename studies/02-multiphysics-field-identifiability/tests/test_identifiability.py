"""Structural rank, null directions and the coupling mechanism.

These tests assert the identifiability statements that the results rest on, so
that a change to the forward model that would invalidate one of them fails here
first.
"""
import numpy as np
import pytest

from fieldid import Acquisition, Inversion
from fieldid.coupling import CorrelatedPrior
from fieldid.infer import diagnostics as DG
from fieldid.infer import sampling as SA
from fieldid.params import variant


def _relax_acq():
    a = Acquisition()
    a.relax.enabled = True
    return a


# ------------------------------------------------- the phi times A degeneracy

def test_ceus_has_one_null_direction_and_it_is_phi_times_A():
    inv = Inversion(free=("phi", "D", "vmag", "vth", "vaz", "A", "t0"),
                    windows=("ceus",))
    sr = DG.structural_rank(inv)
    assert sr["rank"] == sr["n_par"] - 1
    v = sr["null_space_log"][:, 0]
    i, j = inv.space.index("phi"), inv.space.index("A")
    assert abs(abs(v[i]) - 1 / np.sqrt(2)) < 0.02
    assert abs(abs(v[j]) - 1 / np.sqrt(2)) < 0.02
    assert v[i] * v[j] < 0                      # opposite signs: a ratio is free
    others = [k for k in range(len(v)) if k not in (i, j)]
    assert np.abs(v[others]).max() < 0.02


def test_calibrating_the_delivered_amount_restores_phi():
    loose = Inversion(free=("phi", "D", "vmag", "vth", "vaz", "A", "t0"),
                      windows=("ceus",))
    tight = Inversion(free=("phi", "D", "vmag", "vth", "vaz", "A", "t0"),
                      windows=("ceus",),
                      overrides=dict(A=variant("A", lo=1 / 1.1, hi=1.1,
                                               role="calibrated")))
    rng = np.random.default_rng(0)
    y = tight.simulate(rng)
    a = SA.laplace_summary(tight, SA.laplace(tight, y), ["phi"], n=4000)
    b = SA.laplace_summary(loose, SA.laplace(loose, loose.simulate(
        np.random.default_rng(0))), ["phi"], n=4000)
    assert a["phi"]["contraction"] < 0.2
    assert b["phi"]["contraction"] > 0.7
    assert b["phi"]["contraction"] > 5 * a["phi"]["contraction"]


def test_normalized_curves_carry_no_information_about_phi_or_A():
    acq = Acquisition()
    acq.ceus.normalized = True
    inv = Inversion(free=("phi", "D", "vmag", "vth", "vaz", "A", "t0"),
                    windows=("ceus",), acq=acq)
    j = DG.jacobian(inv)
    scale = np.abs(j).max()
    for n in ("phi", "A"):
        col = j[:, inv.space.index(n)]
        assert np.abs(col).max() < 1e-12 * scale


# ------------------------------------------------------ the shear-wave window

def test_shear_window_sees_essentially_nothing_of_phi_or_k():
    inv = Inversion(free=("mu", "eta_s", "phi", "k", "F0", "Tp"),
                    windows=("swe",))
    sp = DG.spectrum(inv)
    lam = dict(zip(inv.free, np.diag(DG.fisher(inv))))
    assert lam["phi"] < 1e-5             # no measurable information
    assert lam["k"] < 0.05
    assert lam["mu"] > 1e4               # and a great deal about the modulus
    assert sp["contraction"][0] > 0.999


# --------------------------------------------------------- relaxation window

def test_relaxation_null_direction_is_the_analytic_one():
    """With the applied stress unknown, the relaxation observation determines
    P L / M, M S_v and k / S_v, so exactly one direction is free and in
    logarithms it is (M, S_v, k, P) proportional to (+1, -1, -1, +1)."""
    inv = Inversion(free=("M", "S_v", "k", "P"), windows=("relax",),
                    acq=_relax_acq(),
                    overrides=dict(P=variant("P", lo=100.0, hi=2500.0,
                                             role="nuisance")))
    sr = DG.structural_rank(inv)
    assert sr["rank"] == 3
    v = sr["null_space_log"][:, 0]
    v = v / v[inv.space.index("M")]
    expect = {"M": 1.0, "S_v": -1.0, "k": -1.0, "P": 1.0}
    for n, e in expect.items():
        assert v[inv.space.index(n)] == pytest.approx(e, abs=0.02)


def test_knowing_the_storage_breaks_the_relaxation_degeneracy():
    """Fixing S_v leaves the relaxation observation with three unknowns and
    three combinations, so the rank becomes full."""
    inv = Inversion(free=("M", "k", "P"), windows=("relax",), acq=_relax_acq(),
                    overrides=dict(P=variant("P", lo=100.0, hi=2500.0,
                                             role="nuisance")))
    sr = DG.structural_rank(inv)
    assert sr["rank"] == 3


# ------------------------------------------------------------- the mechanism

def _joint(level, windows, overrides=None):
    free = ["phi", "k", "mu", "eta_s", "F0", "Tp",
            "D", "vmag", "vth", "vaz", "A", "t0"]
    if "relax" in windows:
        free += ["M", "P"]
    free += ["S_v"] if level == "independent" else ["C_v"]
    acq = _relax_acq()
    return Inversion(free=tuple(free), level=level, windows=windows, acq=acq,
                     overrides=overrides or {})


def test_under_the_current_protocol_storage_is_exactly_unidentified():
    inv = _joint("independent", ("swe", "ceus"))
    j = DG.jacobian(inv)
    assert np.abs(j[:, inv.space.index("S_v")]).max() == 0.0
    assert np.abs(j).max() > 0.0


def test_coupling_alone_does_not_help_without_a_mechanical_relaxation():
    """With only the shear window and the contrast window, imposing
    S_v = phi C_v cannot transfer anything, because no observation depends on
    S_v at all."""
    ov = dict(A=variant("A", lo=1 / 1.1, hi=1.1, role="calibrated"))
    inv = _joint("constitutive", ("swe", "ceus"), ov)
    j = DG.jacobian(inv)
    assert np.abs(j[:, inv.space.index("C_v")]).max() == 0.0
    assert np.abs(j).max() > 0.0


def _k_contraction(level, ov, seed=3):
    inv = _joint(level, ("swe", "ceus", "relax"), ov)
    y = inv.simulate(np.random.default_rng(seed))
    s = SA.laplace_summary(inv, SA.laplace(inv, y), ["k", "S_v"], n=6000, seed=5)
    return s["k"]["contraction"], s["S_v"]["contraction"]


def test_the_coupling_transfers_little_while_the_compliance_is_unknown():
    """With the relaxation observation present but C_v spanning two decades,
    imposing S_v = phi C_v barely narrows the permeability: the prior on the
    product is dominated by C_v, not by phi, so pinning phi buys little."""
    ov = dict(A=variant("A", lo=1 / 1.1, hi=1.1, role="calibrated"),
              P=variant("P", lo=100.0, hi=2500.0, role="nuisance"))
    ind, _ = _k_contraction("independent", ov)
    con, _ = _k_contraction("constitutive", ov)
    assert 0.8 < con / ind < 1.05


def test_with_a_calibrated_compliance_the_coupling_contracts_the_permeability():
    """Once C_v is independently calibrated, the same relation carries the
    contrast-derived phi into the mechanical problem and the permeability
    narrows substantially."""
    ov = dict(A=variant("A", lo=1 / 1.1, hi=1.1, role="calibrated"),
              P=variant("P", lo=100.0, hi=2500.0, role="nuisance"))
    ind, _ = _k_contraction("independent", ov)
    ovc = dict(ov)
    ovc["C_v"] = variant("C_v", lo=1.0e-3 / 1.5, hi=1.0e-3 * 1.5,
                         role="calibrated")
    con, sv = _k_contraction("constitutive", ovc)
    assert con < 0.5 * ind
    assert sv < 0.35


def test_network_prior_preserves_scatter():
    c = np.array([[1.0, 0.6, 0.3, 0.2],
                  [0.6, 1.0, 0.2, 0.1],
                  [0.3, 0.2, 1.0, 0.4],
                  [0.2, 0.1, 0.4, 1.0]])
    cp = CorrelatedPrior(("phi", "k", "D", "vmag"), c, "test", 40)
    for n in ("k", "D", "vmag"):
        s = cp.conditional_sd(n, "phi")
        assert 0.0 < s < 1.0          # conditioning on phi narrows but never fixes
    assert cp.conditional_sd("k", "phi") == pytest.approx(np.sqrt(1 - 0.36), rel=1e-12)


# ------------------------------------------ a near-invariant direction of the
#                                             joint model
def test_a_four_parameter_direction_is_exactly_invariant_for_two_windows():
    """k -> a k, C_v -> a C_v, M -> M / a, P -> P / a leaves the relaxation
    prediction exactly unchanged and the contrast prediction untouched.

    The relaxation sees only P L / M, M S_v and k / S_v, and with S_v = phi C_v
    at fixed phi this transformation preserves all three.  The invariance is
    **exact for those two windows and not for the joint model**: the shear
    window does see the direction, through the Biot inertial coupling in the
    effective density.  What it sees is small rather than nothing, so this is a
    near-invariant direction of the joint model and an exact invariance of two
    of its three windows.  The size is reported as a change in chi-squared over
    the whole observation vector and is asserted here rather than converted into
    any other unit."""
    acq = Acquisition()
    acq.relax.enabled = True
    inv = Inversion(free=("mu",), windows=("swe", "ceus", "relax"), acq=acq)
    p0 = inv.truth_physical()
    sl, sig = inv.window_slices(), inv.sigma()

    def scaled(a):
        q = dict(p0)
        q["k"] *= a
        q["C_v"] *= a
        q["S_v"] = q["phi"] * q["C_v"]
        q["M"] /= a
        q["P"] /= a
        return q

    for a, swe_max in ((2.0, 0.01), (10.0, 0.30)):
        q = scaled(a)
        d = {}
        for w in ("relax", "ceus", "swe"):
            r = (inv._window(w, q) - inv._window(w, p0)) / sig[sl[w]]
            d[w] = float((r ** 2).sum())
        assert d["relax"] < 1e-12, (a, d["relax"])
        assert d["ceus"] == 0.0
        assert d["swe"] < swe_max, (a, d["swe"])


def test_the_profile_bounds_do_not_depend_on_how_tight_a_prior_is():
    """Physical bounds come from the catalogue range, not from the prior, so
    calibrating a quantity cannot shrink the box a likelihood-only profile is
    optimized in."""
    from fieldid.infer.diagnostics import physical_bounds
    free = ("phi", "C_v", "D", "vmag", "vth", "vaz", "A", "t0")
    loose = Inversion(free=free, windows=("ceus",))
    tight = Inversion(free=free, windows=("ceus",),
                      overrides=dict(C_v=variant("C_v", lo=1e-3 / 1.5, hi=1e-3 * 1.5,
                                                 role="calibrated")))
    i = loose.space.index("C_v")
    lo_a, hi_a = physical_bounds(loose)
    lo_b, hi_b = physical_bounds(tight)
    # same physical width, expressed in each configuration's own units
    wa = (hi_a[i] - lo_a[i]) * loose.space.prior_sd()[i]
    wb = (hi_b[i] - lo_b[i]) * tight.space.prior_sd()[i]
    assert wa == pytest.approx(wb, rel=1e-12)
    # and in standardized units the calibrated box is far wider, as it must be
    assert (hi_b[i] - lo_b[i]) > 5 * (hi_a[i] - lo_a[i])


def test_two_configurations_sharing_a_likelihood_share_an_imaging_profile():
    """E3 and E5 differ only in their prior, so their imaging-only profiles must
    coincide.  They did not, because a single warm-started sweep is path
    dependent; the inner problem is now solved from two starts."""
    from fieldid.infer.diagnostics import profile_likelihood
    from fieldid.coupling import CorrelatedPrior
    import numpy as np
    c = np.array([[1.0, 0.87, -0.33, -0.53], [0.87, 1.0, -0.30, -0.41],
                  [-0.33, -0.30, 1.0, 0.10], [-0.53, -0.41, 0.10, 1.0]])
    cp = CorrelatedPrior(("phi", "k", "D", "vmag"), c, "test", 30)
    free = ("phi", "k", "D", "vmag", "vth", "vaz", "A", "t0", "C_v")
    a = Inversion(free=free, level="constitutive", windows=("ceus",))
    b = Inversion(free=free, level="network", windows=("ceus",), corr=cp)
    y = a.simulate(np.random.default_rng(0))
    pa = profile_likelihood(a, y, "k", n=5, span=2.0, level="imaging")
    pb = profile_likelihood(b, y, "k", n=5, span=2.0, level="imaging")
    assert np.allclose(pa["delta"], pb["delta"], atol=1e-6)


# ------------------------------------ matched prior information is not a
#                                      measurement
def test_a_matched_storage_prior_is_not_labelled_as_a_calibration():
    """The matched control carries the marginal the coupled arm induces, so that
    the two differ only in the coupling.  Nothing about it is measured, and its
    role says so."""
    from fieldid.experiments import matched_Sv, CV_CAL
    from fieldid.infer.diagnostics import AUXILIARY_ROLES
    e = matched_Sv(CV_CAL["C_v"])["S_v"]
    assert e.role == "matched"
    assert e.role != "calibrated"
    # it still enters the middle profile level, because a comparison needs the
    # same information counted at the same level in both arms
    assert e.role in AUXILIARY_ROLES


def test_the_middle_profile_level_holds_both_auxiliary_roles():
    from fieldid.infer.diagnostics import prior_residual_at, x_to_z
    from fieldid.experiments import matched_Sv, CV_CAL
    ov = dict(A=variant("A", lo=1 / 1.1, hi=1.1, role="calibrated"))
    ov.update(matched_Sv(CV_CAL["C_v"]))
    inv = Inversion(free=("phi", "k", "S_v", "D", "vmag", "vth", "vaz", "A", "t0"),
                    windows=("ceus",), overrides=ov)
    z = x_to_z(inv, inv.space.x0(inv.truth))
    assert prior_residual_at(inv, z, "imaging").size == 0
    assert prior_residual_at(inv, z, "imaging_plus_auxiliary").size == 2
    assert prior_residual_at(inv, z, "posterior").size == len(inv.free)


def test_the_flat_imaging_profile_does_not_need_an_inadmissible_porosity():
    """The imaging profile of k is flat, and the minimizer does reach the
    ceiling of the vascular volume fraction on its way.  The claim is only
    worth something if the flatness survives a ceiling no tissue could exceed,
    so that is computed rather than argued."""
    from fieldid.infer.diagnostics import profile_domain_sensitivity
    ov = dict(A=variant("A", lo=1 / 1.1, hi=1.1, role="calibrated"))
    inv = Inversion(free=("phi", "k", "mu", "eta_s", "F0", "Tp", "D", "vmag",
                          "vth", "vaz", "A", "t0"),
                    windows=("swe", "ceus"), overrides=ov)
    y = inv.simulate(np.random.default_rng(0))
    s = profile_domain_sensitivity(inv, y, "k", ceiling={"phi": 0.30},
                                   n=3, span=3.0, level="imaging")
    assert s["delta_max"] < 1.0
    assert s["max_abs_difference"] < 0.05
