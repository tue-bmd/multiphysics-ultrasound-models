"""Parameters, priors, noise, determinism, inference machinery, generators."""
import numpy as np
import pytest

from fieldid import Acquisition, Inversion
from fieldid.compat import trapezoid
from fieldid.coupling import CorrelatedPrior, apply, extra_free
from fieldid.infer import diagnostics as DG
from fieldid.infer import sampling as SA
from fieldid.models import transport as TP
from fieldid.models.richer import Richer
from fieldid.params import CATALOGUE, Space, TRUTH, Z90, induced_Sv, variant


# ----------------------------------------------------------------- parameters

def test_transforms_round_trip():
    for n, e in CATALOGUE.items():
        assert e.to_value(e.to_x(e.nominal)) == pytest.approx(e.nominal, rel=1e-14)


def test_prior_interval_matches_the_stated_percentiles():
    for n, e in CATALOGUE.items():
        if e.kind != "normal":
            continue
        assert e.to_value(e.mean - Z90 * e.sd) == pytest.approx(e.lo, rel=1e-10)
        assert e.to_value(e.mean + Z90 * e.sd) == pytest.approx(e.hi, rel=1e-10)


def test_storage_marginal_is_the_product_of_its_factors():
    m, s = induced_Sv()
    e = CATALOGUE["S_v"]
    assert e.mean == pytest.approx(m, rel=1e-12)
    assert e.sd == pytest.approx(s, rel=1e-12)
    assert TRUTH["S_v"] == pytest.approx(TRUTH["phi"] * TRUTH["C_v"], rel=1e-14)


def test_prior_override_changes_only_the_named_entry():
    sp = Space(["k", "P"], {"P": variant("P", lo=100.0, hi=2500.0, role="nuisance")})
    assert sp.entry("P").role == "nuisance"
    assert sp.entry("P").sd > CATALOGUE["P"].sd
    assert sp.entry("k").sd == CATALOGUE["k"].sd


def test_coupling_levels_declare_their_own_storage_parameter():
    assert extra_free("independent") == ["S_v"]
    assert extra_free("constitutive") == ["C_v"]
    assert extra_free("network") == ["C_v"]
    p = apply("constitutive", dict(phi=0.03, C_v=2e-3))
    assert p["S_v"] == pytest.approx(6e-5, rel=1e-14)
    q = apply("independent", dict(phi=0.03, S_v=1.0))
    assert q["S_v"] == 1.0


# ---------------------------------------------------------------- inversion --

def test_physical_dictionary_is_complete_and_uses_fixed_constants():
    inv = Inversion(free=("mu",), windows=("swe",))
    p = inv.truth_physical()
    for n in ("mu", "eta_s", "phi", "k", "rho", "rho_f", "eta_b", "alpha"):
        assert n in p


def test_simulation_is_deterministic_in_the_seed():
    inv = Inversion(free=("mu", "eta_s"), windows=("swe", "ceus"))
    a = inv.simulate(np.random.default_rng(7))
    b = inv.simulate(np.random.default_rng(7))
    assert np.array_equal(a, b)
    c = inv.simulate(np.random.default_rng(8))
    assert not np.array_equal(a, c)


def test_residuals_are_standard_normal_at_the_truth():
    inv = Inversion(free=("mu", "eta_s"), windows=("swe", "ceus"))
    rng = np.random.default_rng(11)
    x0 = inv.space.x0(inv.truth)
    r = (inv.simulate(rng) - inv.predict(x0)) / inv.sigma()
    assert abs(r.mean()) < 0.05
    assert abs(r.std() - 1.0) < 0.05


def test_window_slices_partition_the_observation_vector():
    inv = Inversion(free=("mu",), windows=("swe", "ceus"))
    sl = inv.window_slices()
    n = inv.predict(inv.space.x0(inv.truth)).size
    assert sl["swe"].start == 0
    assert sl["ceus"].stop == n
    assert sl["swe"].stop == sl["ceus"].start


def test_normalized_noise_propagation_matches_a_monte_carlo_estimate():
    acq = Acquisition()
    acq.ceus.normalized = True
    inv = Inversion(free=("D",), windows=("ceus",), acq=acq)
    p = inv.truth_physical()
    t = acq.ceus.times()
    b = TP.occupancy(acq.ceus.positions(), t, p)
    s_abs = acq.ceus.sigma_rel * b.max()
    rng = np.random.default_rng(0)
    draws = np.array([TP.normalize(b + rng.normal(0, s_abs, b.shape), t)
                      for _ in range(300)])
    emp = draws.std(axis=0).ravel()
    stated = inv.sigma()
    ok = emp > 0.05 * emp.max()             # where the curve carries signal
    ratio = np.median(stated[ok] / emp[ok])
    assert 0.7 < ratio < 1.4


# --------------------------------------------------------------- diagnostics -

def test_jacobian_converges_at_second_order():
    inv = Inversion(free=("mu", "eta_s", "phi", "k"), windows=("swe",))
    e = DG.jacobian_convergence(inv)
    assert e[0] > e[1] > e[2] > 0.0
    assert e[0] / e[1] > 3.0                 # halving the step quarters the error


def test_standardized_coordinates_round_trip():
    inv = Inversion(free=("mu", "k", "t0"), windows=("swe", "ceus"))
    x = inv.space.x0(inv.truth)
    assert np.allclose(DG.z_to_x(inv, DG.x_to_z(inv, x)), x)


def test_prior_precision_is_the_identity_for_independent_normal_priors():
    inv = Inversion(free=("mu", "k"), windows=("swe",))
    assert np.allclose(DG.prior_precision(inv), np.eye(2))


def test_prior_precision_uses_the_correlation_when_one_is_supplied():
    c = np.array([[1.0, 0.5, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    cp = CorrelatedPrior(("phi", "k", "D", "vmag"), c, "test", 30)
    inv = Inversion(free=("phi", "k", "D", "vmag", "C_v"), level="network",
                    windows=("ceus",), corr=cp)
    lam = DG.prior_precision(inv)
    assert lam[0, 1] == pytest.approx(np.linalg.inv(c)[0, 1], rel=1e-12)
    assert lam[4, 4] == pytest.approx(1.0, rel=1e-12)


def test_laplace_agrees_with_the_sampler():
    acq = Acquisition()
    acq.swe.n_x = 6
    acq.swe.duration = 8.0e-3
    inv = Inversion(free=("mu", "eta_s", "F0"), windows=("swe",), acq=acq)
    y = inv.simulate(np.random.default_rng(2))
    lap = SA.laplace(inv, y)
    mc = SA.adaptive_metropolis(inv, y, n_steps=12000, n_chains=3, seed=1,
                                cov0=lap["cov"])
    assert np.all(mc["rhat"] < 1.1)
    sd_l = np.sqrt(np.diag(lap["cov"]))
    sd_m = mc["samples"].std(axis=0)
    assert np.allclose(sd_l, sd_m, rtol=0.4)
    assert np.allclose(lap["mean"], mc["samples"].mean(axis=0),
                       atol=0.4 * sd_l.max())


def test_contraction_is_one_when_a_parameter_enters_no_window():
    inv = Inversion(free=("mu", "S_v"), windows=("swe",))
    y = inv.simulate(np.random.default_rng(4))
    s = SA.laplace_summary(inv, SA.laplace(inv, y), ["S_v"], n=8000)
    assert s["S_v"]["contraction"] == pytest.approx(1.0, abs=0.05)


def test_profile_likelihood_is_minimal_near_the_truth():
    inv = Inversion(free=("mu", "eta_s"), windows=("swe",))
    y = inv.simulate(np.random.default_rng(6))
    pr = DG.profile_likelihood(inv, y, "mu", n=9, span=1.5)
    assert abs(pr["grid"][int(np.argmin(pr["nll"]))]) < 0.5


# ----------------------------------------------------------------- generators

def test_richer_generator_reduces_to_the_base_model_in_its_null_limit():
    acq = Acquisition()
    acq.relax.enabled = True
    inv = Inversion(free=("mu",), windows=("swe", "ceus", "relax"), acq=acq)
    p = inv.truth_physical()
    g = Richer(push_width=0.0, slow_fraction=0.0, k_spread=0.0)
    for w in ("swe", "ceus", "relax"):
        a = inv._window(w, p, gen=None)
        b = inv._window(w, p, gen=g)
        assert np.allclose(a, b, rtol=1e-10, atol=1e-18 * max(np.abs(a).max(), 1))


def test_richer_generator_conserves_tracer_amount():
    acq = Acquisition()
    inv = Inversion(free=("mu",), windows=("ceus",), acq=acq)
    p = inv.truth_physical()
    g = Richer()
    t = acq.ceus.times()
    d = 0.9e-3
    ax = (np.arange(72) - 35.5) * d
    grid = np.meshgrid(ax, ax, ax, indexing="ij")
    v = TP.drift_vector(p["vmag"], p["vth"], p["vaz"])
    r = np.column_stack([a.ravel() for a in grid]) + v * 5.0

    class _Cfg:
        def positions(self_): return r
        def times(self_): return np.array([p["t0"] + 5.0])
    b = g.ceus(_Cfg(), p)
    assert TP.total_mass(b, (d, d, d))[0] == pytest.approx(p["phi"] * p["A"],
                                                           rel=5e-3)


def test_richer_generator_actually_differs_from_the_base_model():
    acq = Acquisition()
    acq.relax.enabled = True
    inv = Inversion(free=("mu",), windows=("swe", "ceus", "relax"), acq=acq)
    p = inv.truth_physical()
    g = Richer()
    for w in ("swe", "ceus", "relax"):
        a = inv._window(w, p, gen=None)
        b = inv._window(w, p, gen=g)
        rel = np.abs(a - b).max() / np.abs(a).max()
        assert rel > 1e-3, w


def test_closed_form_summary_matches_the_sampled_one():
    from fieldid.infer import summaries as SU
    inv = Inversion(free=("phi", "D", "vmag", "vth", "vaz", "A", "t0"),
                    windows=("ceus",))
    y = inv.simulate(np.random.default_rng(5))
    lap = SA.laplace(inv, y)
    names = ["phi", "D", "vmag", "A"]
    exact = SU.summarize_gaussian(inv, lap, names)
    drawn = SA.laplace_summary(inv, lap, names, n=60000, seed=9)
    for n in names:
        assert exact[n]["contraction"] == pytest.approx(
            drawn[n]["contraction"], rel=0.05)
        assert exact[n]["median"] == pytest.approx(drawn[n]["median"], rel=0.02)


def test_closed_form_summary_handles_a_derived_quantity():
    from fieldid.infer import summaries as SU
    inv = Inversion(free=("phi", "C_v", "D", "vmag", "vth", "vaz", "A", "t0"),
                    level="constitutive", windows=("ceus",))
    y = inv.simulate(np.random.default_rng(5))
    lap = SA.laplace(inv, y)
    s = SU.summarize_gaussian(inv, lap, ["S_v"])
    assert s["S_v"]["truth"] == pytest.approx(TRUTH["phi"] * TRUTH["C_v"], rel=1e-12)
    assert 0.0 < s["S_v"]["contraction"] <= 1.0


# --------------------------------------- physical domains and profile records

def test_expanded_bounds_stay_inside_the_physical_domain():
    """Widening the catalogue range by two decades would admit a vascular
    volume fraction of ten.  The widening is clipped to the domain, so the
    optimizer can reach the edge of what the model can mean and no further."""
    free = ("phi", "k", "D", "vmag", "vth", "vaz", "A", "t0")
    inv = Inversion(free=free, windows=("ceus",))
    lo, hi = DG.physical_bounds(inv)
    m, sd = DG.standardize(inv)
    val = {n: (inv.space.entry(n).to_value(m[i] + sd[i] * lo[i]),
               inv.space.entry(n).to_value(m[i] + sd[i] * hi[i]))
           for i, n in enumerate(free)}
    assert val["phi"][1] == pytest.approx(1.0, rel=1e-12)
    assert val["vth"] == pytest.approx((0.0, np.pi), abs=1e-12)
    assert val["vaz"] == pytest.approx((-np.pi, np.pi), abs=1e-12)
    assert val["t0"][0] == pytest.approx(0.0, abs=1e-12)
    # a quantity with no hard upper limit keeps the full two-decade widening
    assert val["k"][1] == pytest.approx(1e-9, rel=1e-9)


def test_the_profile_records_what_the_optimizer_did():
    """A flat profile is only a statement about the likelihood if the minimizer
    stayed inside the admissible region and converged; both are recorded."""
    inv = Inversion(free=("phi", "D", "vmag", "vth", "vaz", "A", "t0"),
                    windows=("ceus",))
    y = inv.simulate(np.random.default_rng(4))
    pr = DG.profile_likelihood(inv, y, "phi", n=5, span=1.5, level="imaging")
    lo, hi = DG.physical_bounds(inv)
    z = np.asarray(pr["z_hat"])
    assert z.shape == (5, len(inv.free))
    assert np.all(z >= lo - 1e-9) and np.all(z <= hi + 1e-9)
    assert np.asarray(pr["at_bound"]).shape == z.shape
    # the profiled coordinate is set, not fitted, so it is never a bound hit
    assert not np.asarray(pr["at_bound"])[:, inv.space.index("phi")].any()
    audit = DG.profile_audit(inv, pr)
    assert audit["n_not_converged"] == 0
    assert audit["per_parameter"]["phi"]["outside_domain"] == 0


def test_a_chain_advanced_in_pieces_is_the_same_chain():
    """The checkpointed sampler exists because the useful chain length is longer
    than one process may live.  Splitting a chain must not change it."""
    inv = Inversion(free=("mu", "eta_s", "F0"), windows=("swe",))
    y = inv.simulate(np.random.default_rng(2))
    z0 = DG.x_to_z(inv, inv.space.x0(inv.truth))
    cov = np.eye(3) * 0.01
    whole = SA.AMChain(inv, y, z0, cov, seed=7).advance(40, thin=2)
    ch = SA.AMChain(inv, y, z0, cov, seed=7)
    parts = np.vstack([ch.advance(13, thin=2), ch.advance(27, thin=2)])
    assert np.array_equal(whole, parts)
    assert ch.n == 40


def test_the_sampler_recovers_a_gaussian_posterior_width():
    """The preconditioned sampler is validated against the case where the
    Laplace approximation is exact by construction: a well-determined,
    near-linear block."""
    inv = Inversion(free=("mu", "eta_s", "F0"), windows=("swe",))
    y = inv.simulate(np.random.default_rng(2))
    z_map, _ = SA.find_map(inv, y)
    lap = SA.laplace(inv, y, z_map)
    mc = SA.preconditioned_metropolis(inv, y, n_steps=4000, n_chains=2, seed=3,
                                      thin=2, lap=lap, z_map=z_map)
    sd_mc = mc["samples"].std(axis=0, ddof=1)
    sd_la = np.sqrt(np.diag(lap["cov"]))
    for a, b in zip(sd_mc, sd_la):
        assert a == pytest.approx(b, rel=0.15)
    d = SA.directional_diagnostics(inv, mc["chains"], z_map, n_dir=3)
    assert len(d["rhat"]) == 3 and len(d["ess"]) == 3
