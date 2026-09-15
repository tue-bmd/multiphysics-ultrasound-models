"""Comparison fairness, and the machinery the comparison depends on.

A comparison between arms is only about the thing under test if everything else
is identical.  These assert that it is.
"""
import numpy as np
import pytest

from operatorid import Acquisition, Inversion
from operatorid.constants import GAMMA_NOMINAL, psf_widths
from operatorid import experiments as EX
from operatorid.infer import diagnostics as DG
from operatorid.infer import sampling as SA
from operatorid.params import CATALOGUE, PSF_NAMES, TRUTH, psf_variants, variant

MAIN = {c.name: c for c in EX.matrix()}["main"]


# ------------------------------------------------------------- matched priors

def test_the_independent_widths_carry_the_marginal_sharing_would_induce():
    """Sharing introduces a relationship between the widths.  It must not also
    introduce a different marginal, or the comparison would credit the prior.

    The map is monotone, so the 5th and 95th percentiles of each width are the
    images of those of the aperture component, exactly."""
    e0 = CATALOGUE["w0"]
    lo, hi = psf_widths(e0.lo), psf_widths(e0.hi)
    for m in PSF_NAMES:
        e = psf_variants()[PSF_NAMES[m]]
        assert e.lo == pytest.approx(float(lo[m]), rel=1e-12)
        assert e.hi == pytest.approx(float(hi[m]), rel=1e-12)
    # for the two linear windows the induced marginal is lognormal with the
    # same log width; for the shear window the quadrature makes it narrower,
    # which is stated in `psf_variants`
    for m in ("bmode", "ceus"):
        assert psf_variants()[PSF_NAMES[m]].sd == pytest.approx(e0.sd, rel=1e-12)
    assert psf_variants()[PSF_NAMES["swe"]].sd < e0.sd


def test_every_arm_of_every_case_analyzes_exactly_the_same_data():
    """The dataset is made once per case and handed to every arm, bit for bit,
    together with its noise scale.

    This is not a formality.  The contrast noise is a fraction of the contrast
    peak and the peak depends on the operator, so an arm that generated its own
    data would see a different realization *and* a different noise level
    whenever its assumption about the operator was wrong, and the comparison
    would no longer be between analyses of one experiment."""
    for case in EX.matrix():
        y, sigma = EX.generate(case, seed=0)
        for a in EX.ARMS:
            inv = case.build(a, sigma=sigma)
            assert np.array_equal(inv.sigma(), sigma), (case.name, a)
            assert inv.windows == tuple(w for w in
                                        ("bmode", "swe", "ceus")
                                        if w in case.windows), (case.name, a)
        # and the misspecified case really is misspecified
        if case.name == "wrong_ratio":
            y_ok, s_ok = EX.generate(
                [c for c in EX.matrix() if c.name == "main"][0], seed=0)
            assert not np.allclose(y, y_ok), case.name
            assert not np.allclose(sigma, s_ok), \
                "the noise scale must follow the generating operator"


def test_only_the_oracle_knows_the_generating_corrections():
    case = {c.name: c for c in EX.matrix()}["wrong_ratio"]
    assert case.build("oracle").gamma["ceus"] == pytest.approx(1.15)
    for a in ("independent", "shared", "shared_calibrated"):
        assert case.build(a).gamma == GAMMA_NOMINAL, a
    # and the oracle therefore uses the true width, which is what makes it one
    orc = case.build("oracle")
    assert orc.width(orc.truth_physical(), "ceus") == pytest.approx(
        float(psf_widths(TRUTH["w0"], {"ceus": 1.15})["ceus"]), rel=1e-12)


def test_the_arms_agree_on_everything_but_the_operator():
    """Same windows, same noise, same tissue and nuisance priors."""
    y, sigma = EX.generate(MAIN, seed=0)
    arms = {a: MAIN.build(a, sigma=sigma) for a in EX.ARMS}
    for a in EX.ARMS:
        for m in ("bmode", "swe", "ceus"):
            assert arms[a].width(arms[a].truth_physical(), m) == pytest.approx(
                float(psf_widths(TRUTH["w0"])[m]), rel=1e-12), (a, m)
    tissue = [n for n in arms["shared"].free
              if n not in ("w0",) and not n.startswith(("w_", "gamma_"))]
    for n in tissue:
        for a in EX.ARMS:
            if n not in arms[a].free:
                continue
            e1, e2 = arms[a].space.entry(n), arms["shared"].space.entry(n)
            assert (e1.lo, e1.hi, e1.kind) == (e2.lo, e2.hi, e2.kind), (a, n)


def test_the_oracle_is_the_generating_width_and_is_not_estimated():
    inv = MAIN.build("oracle")
    assert "w0" not in inv.free
    assert not any(n.startswith("w_") for n in inv.free)
    p = inv.truth_physical()
    for m, v in psf_widths(TRUTH["w0"]).items():
        assert inv.width(p, m) == pytest.approx(float(v), rel=1e-12)


def test_no_tissue_parameter_is_shared_between_windows():
    """This study isolates operator sharing: the tissue parameters of the three
    windows are disjoint, in every arm, so a difference cannot come from them."""
    from operatorid.model import WINDOW_FREE
    seen = {}
    for w, names in WINDOW_FREE.items():
        for n in names:
            assert n not in seen, (n, w, seen.get(n))
            seen[n] = w


def test_the_calibrated_arm_estimates_the_corrections_it_does_not_assume():
    inv = MAIN.build("shared_calibrated")
    assert "gamma_ceus" in inv.free and "gamma_swe" in inv.free
    assert "w0" in inv.free
    # and they act on the window each belongs to, and only that one
    p = dict(inv.truth_physical()); p["gamma_ceus"] = 1.20
    assert inv.width(p, "ceus") == pytest.approx(
        1.20 * float(psf_widths(p["w0"])["ceus"]), rel=1e-12)
    assert inv.width(p, "bmode") == pytest.approx(p["w0"], rel=1e-12)


# ----------------------------------------------------------- optimizer bounds

def test_expanded_bounds_stay_inside_the_physical_domain():
    inv = MAIN.build("shared")
    lo, hi = DG.physical_bounds(inv)
    m, sd = DG.standardize(inv)
    val = {n: (inv.space.entry(n).to_value(m[i] + sd[i] * lo[i]),
               inv.space.entry(n).to_value(m[i] + sd[i] * hi[i]))
           for i, n in enumerate(inv.free)}
    assert val["t0"][0] == pytest.approx(0.0, abs=1e-9)
    assert val["t0"][1] <= 20.0 + 1e-9
    assert val["x_c"][0] >= -1e-9 and val["x_c"][1] <= 30.0e-3 + 1e-9
    assert val["w0"][1] <= 15.0e-3 + 1e-9
    # a quantity with no stated domain keeps the full two-decade widening
    assert val["c_s"][1] == pytest.approx(5.0 * 100, rel=1e-6)


def test_the_bounds_do_not_depend_on_how_tight_a_prior_is():
    a = Inversion(free=("sig_e", "R", "x_c", "b0", "db", "w_bmode"),
                  level="independent", windows=("bmode",),
                  overrides=psf_variants())
    ov = dict(psf_variants())
    ov["w_bmode"] = variant("w_bmode", lo=0.30e-3, hi=0.40e-3,
                            catalogue={"w_bmode": ov["w_bmode"]})
    b = Inversion(free=a.free, level="independent", windows=("bmode",),
                  overrides=ov)
    i = a.space.index("w_bmode")
    la, ha = DG.physical_bounds(a)
    lb, hb = DG.physical_bounds(b)
    wa = (ha[i] - la[i]) * a.space.prior_sd()[i]
    wb = (hb[i] - lb[i]) * b.space.prior_sd()[i]
    assert wa == pytest.approx(wb, rel=1e-12)      # same physical width
    assert (hb[i] - lb[i]) > 5 * (ha[i] - la[i])   # far wider in prior units


# ------------------------------------------------------------------ sampling

def test_a_chain_advanced_in_pieces_is_the_same_chain():
    inv = Inversion(free=("c_s", "alpha0", "a_S"), level="oracle",
                    windows=("swe",))
    y = inv.simulate(np.random.default_rng(2))
    z0 = DG.x_to_z(inv, inv.space.x0(inv.truth))
    cov = np.eye(3) * 0.01
    whole = SA.AMChain(inv, y, z0, cov, seed=7).advance(40, thin=2)
    ch = SA.AMChain(inv, y, z0, cov, seed=7)
    parts = np.vstack([ch.advance(13, thin=2), ch.advance(27, thin=2)])
    assert np.array_equal(whole, parts)
    assert ch.n == 40


def test_the_sampler_recovers_a_gaussian_posterior_width():
    inv = Inversion(free=("c_s", "alpha0", "a_S"), level="oracle",
                    windows=("swe",))
    y = inv.simulate(np.random.default_rng(2))
    z_map, _ = SA.find_map(inv, y)
    lap = SA.laplace(inv, y, z_map)
    mc = SA.preconditioned_metropolis(inv, y, n_steps=4000, n_chains=2, seed=3,
                                      thin=2, lap=lap, z_map=z_map)
    for a, b in zip(mc["samples"].std(axis=0, ddof=1),
                    np.sqrt(np.diag(lap["cov"]))):
        assert a == pytest.approx(b, rel=0.2)


# --------------------------------------------------------------- checkpoints

def test_a_checkpoint_refuses_an_incompatible_configuration(tmp_path):
    from operatorid.provenance import (IncompatibleCheckpoint, load_checkpoint,
                                    save_checkpoint)
    p = str(tmp_path / "c.pkl")
    save_checkpoint(p, {"n": 1}, {"case": "main", "seed": 0})
    assert load_checkpoint(p, {"case": "main", "seed": 0}) == {"n": 1}
    with pytest.raises(IncompatibleCheckpoint):
        load_checkpoint(p, {"case": "main", "seed": 1})


def test_a_checkpoint_refuses_a_changed_package(tmp_path, monkeypatch):
    from operatorid import provenance as PR
    p = str(tmp_path / "c.pkl")
    PR.save_checkpoint(p, {"n": 1}, {"case": "main"})
    monkeypatch.setattr(PR, "code_hash", lambda: "0" * 64)
    with pytest.raises(PR.IncompatibleCheckpoint):
        PR.load_checkpoint(p, {"case": "main"})
    # and allows it only when told to, explicitly
    assert PR.load_checkpoint(p, {"case": "main"},
                              allow_code_change=True) == {"n": 1}


def test_the_configuration_fingerprint_moves_with_the_configuration():
    from operatorid.provenance import config_hash
    a = config_hash({"x": 1, "y": [1, 2]})
    assert a == config_hash({"y": [1, 2], "x": 1})     # order does not matter
    assert a != config_hash({"x": 1, "y": [1, 3]})
