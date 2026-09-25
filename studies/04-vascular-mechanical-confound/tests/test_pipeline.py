from dataclasses import dataclass

import numpy as np

from vmconf import ceus as CE
from vmconf import inference as INF
from vmconf import likelihood as LK
from vmconf import mech, provenance


def test_vascular_response_matches_direct_sum():
    rng = np.random.default_rng(0)
    tau = np.exp(rng.normal(np.log(8e-4), 0.5, 400))
    w = rng.random(400); w /= w.sum()
    om = mech.band()
    r = mech.VascularResponse(w, tau, 2000.0)
    for mu, s in ((2000.0, 1.0), (1400.0, 0.8), (3000.0, 0.9)):
        direct = mech.vascular_modulus(w, tau * (2000.0 / mu) * s ** -4, om, 500.0)
        assert np.allclose(r(om, mu, s, 500.0), direct, rtol=2e-3)


def test_grid_marginals_normalise_and_recover_a_delta():
    g = INF.Grid3(np.linspace(1000, 3000, 21), np.linspace(0.5, 2.0, 21),
                  np.linspace(0.6, 1.0, 21))
    lp = np.full(g.shape, -1e9); lp[10, 5, 15] = 0.0
    m = g.marginal(lp, 0)
    assert abs(m.sum() - 1.0) < 1e-12
    assert abs(g.summary(lp, 0)["median"] - g.mu[10]) < 1e-6
    assert abs(g.summary(lp, 2)["median"] - g.s[15]) < 1e-6


def test_noise_is_actually_applied():
    rng = np.random.default_rng(1)
    y = np.ones(50)
    z = LK.add_relative_noise(y, 0.03, rng)
    assert not np.allclose(y, z)
    assert abs(np.std(z) - 0.03) < 0.015


def test_loglik_peaks_at_truth():
    rng = np.random.default_rng(2)
    truth = np.linspace(1.0, 2.0, 24)
    obs = LK.add_relative_noise(truth, 1e-4, rng)
    assert LK.gaussian_loglik(obs, truth, 0.03) > LK.gaussian_loglik(obs, 1.3 * truth, 0.03)


def test_correlated_loglik_is_looser_than_independent():
    rng = np.random.default_rng(3)
    truth = np.linspace(1.0, 2.0, 24)
    obs = LK.add_relative_noise(truth, 0.03, rng)
    wrong = 1.05 * truth
    d_ind = LK.gaussian_loglik(obs, truth, 0.03) - LK.gaussian_loglik(obs, wrong, 0.03)
    d_cor = LK.correlated_loglik(obs, truth, 0.03, 0.7) - LK.correlated_loglik(obs, wrong, 0.03, 0.7)
    assert d_cor < d_ind          # correlation reduces the evidence against a smooth error


def test_ceus_interpolant_holds_at_edges_and_flags():
    tab = dict(rows=[dict(s=0.6, v=1e-3, auc=1e-11), dict(s=0.8, v=2e-3, auc=2e-11),
                     dict(s=1.0, v=3e-3, auc=3e-11)])
    it = CE.Interpolant(tab)
    A = it(0.8)
    assert abs(float(A) - 2e-11) < 1e-20
    assert not it.in_range(0.4) and it.in_range(0.7)
    assert abs(float(it(0.4)) - 1e-11) < 1e-22   # held, not extrapolated


def test_ceus_interpolant_rejects_nonpositive_amounts():
    tab = dict(rows=[dict(s=0.6, auc=1e-11), dict(s=1.0, auc=0.0)])
    try:
        CE.Interpolant(tab)
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive CEUS amount was accepted")


def test_provenance_serializes_dataclass_parameters_without_local_paths(monkeypatch):
    @dataclass
    class Params:
        seed: int = 3
        centre: tuple = (0.0, -0.005, 0.0)

    class Net:
        n = 2
        params = Params()

    monkeypatch.setattr(provenance, "code_hashes", lambda: {"vmconf_sha256": "x"})
    monkeypatch.setattr(
        provenance,
        "network_fingerprints",
        lambda net: {"topology": "t", "path": "p"},
    )
    monkeypatch.setattr(
        provenance.sys,
        "argv",
        ["/private/tmp/study/run.py", "--out", "/private/tmp/output.json"],
    )
    record = provenance.record(
        Net(), purpose="test", output="/private/tmp/output.json"
    )
    assert record["network"]["params"] == {
        "seed": 3,
        "centre": [0.0, -0.005, 0.0],
    }
    assert "porovasc_path" not in record
    assert record["argv"] == ["run.py", "--out", "output.json"]
    assert record["settings"]["output"] == "output.json"
    assert provenance._jsonable(float("nan")) is None


def test_joint_free_does_not_couple_the_modalities():
    """The uncoupled control must not let CEUS sharpen the mechanical s."""
    g = INF.Grid3(np.linspace(1800, 2200, 9), np.linspace(0.8, 1.2, 9),
                  np.linspace(0.7, 1.0, 21))
    tab = dict(rows=[dict(s=s, v=1e-3 * s, auc=1e-11 * s ** 2) for s in np.linspace(0.7, 1.0, 7)])
    it = CE.Interpolant(tab)
    rng = np.random.default_rng(4)
    w = rng.random(50); w /= w.sum()
    tau = np.exp(rng.normal(np.log(8e-4), 0.4, 50))
    resp = mech.VascularResponse(w, tau, 2000.0)
    om = mech.band()
    swe = INF.swe_predict(resp, om, 2000.0, 1.0, 0.8, 300.0)
    data = dict(swe=swe, ceus=np.atleast_1d(np.asarray(it(0.8), float)))
    sp = INF.SwePredictor(g, resp, om, 300.0)
    a = INF.log_posterior(g, "joint_free", data, sp, 0.03, 0.10, ceus_interp=it)
    b = INF.log_posterior(g, "swe", data, sp, 0.03, 0.10, ceus_interp=it)
    wa = g.summary(a, 2)["width"]; wb = g.summary(b, 2)["width"]
    assert abs(wa - wb) < 1e-6 * max(wb, 1e-12)
