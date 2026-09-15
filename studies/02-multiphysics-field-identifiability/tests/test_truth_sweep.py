import pytest

from fieldid import experiments as EX
from fieldid.params import TRUTH
from fieldid import truth_sweep as TS


def test_explicit_scenarios_share_a_complete_consistent_truth():
    spec = {"derive_storage": True, "scenarios": [
        {"name": "a", "values": {"phi": 0.02, "C_v": 2e-3}},
        {"name": "b", "values": {"phi": 0.05, "k": 1e-12}},
    ]}
    scenarios = TS.expand_scenarios(spec)
    assert scenarios[0]["truth"]["S_v"] == pytest.approx(4e-5)
    assert scenarios[1]["truth"]["S_v"] == pytest.approx(
        0.05 * TRUTH["C_v"])
    assert set(scenarios[0]["truth"]) == set(TRUTH)


def test_cartesian_grid_expands_every_combination():
    spec = {"grid": {"phi": [0.01, 0.03], "D": [3e-7, 1e-6, 3e-6]}}
    scenarios = TS.expand_scenarios(spec)
    assert len(scenarios) == 6
    assert len({(s["truth"]["phi"], s["truth"]["D"]) for s in scenarios}) == 6


def test_invalid_truth_is_rejected():
    with pytest.raises(ValueError, match="cannot exceed one"):
        TS.expand_scenarios({"scenarios": [
            {"name": "bad", "values": {"phi": 1.2}}]})
    with pytest.raises(ValueError, match="positive parameter"):
        TS.expand_scenarios({"scenarios": [
            {"name": "bad", "values": {"k": 0.0}}]})


def test_calibrations_are_recentered_without_changing_their_width():
    cfg = {c.name: c for c in EX.matrix()}["E8_relax_constitutive_Cvcal"]
    truth = dict(TRUTH, C_v=2e-3)
    shifted = TS.recentered_config(cfg, truth)
    old, new = cfg.overrides["C_v"], shifted.overrides["C_v"]
    assert new.nominal == pytest.approx(2e-3)
    assert new.hi / new.lo == pytest.approx(old.hi / old.lo)
    assert cfg.overrides["C_v"].nominal == pytest.approx(1e-3)


def test_plan_has_a_case_limit_that_is_visible_before_running():
    spec = {"grid": {"phi": [0.01, 0.02], "k": [1e-13, 1e-12]},
            "configs": ["E2_ceus_Acal", "E3_joint_independent"],
            "n_replicates": 3}
    p = TS.plan(spec)
    assert p["n_scenarios"] == 4
    assert p["n_fits"] == 24


def test_small_sweep_writes_reproducible_outputs(tmp_path):
    spec = {
        "derive_storage": True,
        "n_replicates": 1,
        "configs": ["E2_ceus_Acal"],
        "report": ["phi", "D"],
        "scenarios": [
            {"name": "reference", "values": {}},
            {"name": "high_phi", "values": {"phi": 0.05}},
        ],
    }
    out = tmp_path / "sweep"
    result = TS.run(spec, str(out), seed=4)
    assert result["manifest"]["n_fits"] == 2
    assert len(result["summary"]) == 4
    first = (out / "truth_sweep.json").read_bytes()
    TS.run(spec, str(out), seed=4)
    assert (out / "truth_sweep.json").read_bytes() == first
    assert (out / "truth_sweep_long.csv").exists()
