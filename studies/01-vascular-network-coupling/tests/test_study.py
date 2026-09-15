"""Fast tests of the study layer: provenance, sampling design, pairing.

None of these builds a gland.
"""
import json
import os

import numpy as np
import pytest
from types import SimpleNamespace

from porovasc import study as S
from porovasc.config import BASE, apply_globals
from porovasc.physics import flow as F, drainage as DR


# ------------------------------------------------------------- provenance
def test_run_refuses_any_nonempty_directory(tmp_path):
    out = str(tmp_path / "r")
    r = S.Run(out, "test", dict(a=1), [1], 1)
    r.write(dict(arm="a", seed=1, rve=0, condition="a", x=1.0))
    assert r.finish()
    assert os.path.exists(os.path.join(out, "COMPLETE"))
    with pytest.raises(SystemExit):
        S.Run(out, "test", dict(a=1), [1], 1)
    out2 = str(tmp_path / "r2"); os.makedirs(out2)
    open(os.path.join(out2, "anything.txt"), "w").close()      # any content counts
    with pytest.raises(SystemExit):
        S.Run(out2, "test", dict(a=1), [1], 1)
    assert json.load(open(os.path.join(out, "manifest.json")))["schema"] == S.SCHEMA >= 4


def test_run_rejects_duplicate_keys_and_marks_failure(tmp_path):
    out = str(tmp_path / "r")
    r = S.Run(out, "test", {}, [1], 2)
    r.write(dict(arm="a", seed=1, rve=0, condition="a"))
    with pytest.raises(SystemExit):
        r.write(dict(arm="a", seed=1, rve=0, condition="a"))
    assert not r.finish()                       # one record short
    assert os.path.exists(os.path.join(out, "FAILED"))


def test_records_hold_null_not_nan(tmp_path):
    out = str(tmp_path / "r")
    r = S.Run(out, "test", {}, [1], 1)
    r.write(dict(arm="a", seed=1, rve=0, condition="a", v=np.nan, w=np.inf, u=[1.0, np.nan]))
    r.finish()
    line = open(os.path.join(out, "records.jsonl")).readline()
    assert "NaN" not in line and "Infinity" not in line
    rec = json.loads(line)
    assert rec["v"] is None and rec["w"] is None and rec["u"] == [1.0, None]


# ---------------------------------------------------------- sampling design
def test_sampling_volumes_do_not_overlap_and_fit_in_the_gland():
    rng = np.random.default_rng(0)
    half = 3e-3
    c = S.rve_centres(6, rng, half)
    from porovasc.geometry.network import GLAND_SEMI
    for i, a in enumerate(c):
        assert np.all(np.sum((S._corners(a, half) / GLAND_SEMI) ** 2, axis=1) < 1.0)
        for b in c[i + 1:]:
            assert not S.cubes_overlap(a, b, half)


def test_sphere_containment_predicate():
    from porovasc.geometry.network import GLAND_SEMI
    a = GLAND_SEMI[0]
    assert S.sphere_inside_gland((0, 0, 0), 6e-3)
    assert not S.sphere_inside_gland((a - 5e-3, 0, 0), 6e-3)        # pokes out of the tip
    assert not S.sphere_inside_gland((a - 6e-3, 0, 0), 6e-3)        # touches the capsule
    assert S.sphere_inside_gland((a - 6.5e-3, 0, 0), 6e-3)
    assert not S.support_inside_gland((0, GLAND_SEMI[1] - 3.1e-3, 0), 3e-3, 0.0)  # a cube corner outside
    assert S.support_inside_gland((0, GLAND_SEMI[1] - 4e-3, 0), 3e-3, 0.0)


def test_large_supports_are_kept_inside_the_gland():
    """Sweep arms with a 6 mm compression sphere, and 6 mm lesion spheres, must
    be placed so that the whole sphere lies inside the gland."""
    for seed in range(30):
        rng = np.random.default_rng(seed)
        for c in S.rve_centres(6, rng, 3e-3, R_support=6e-3):
            assert S.sphere_inside_gland(c, 6e-3)
        c = S.rve_centres(2, np.random.default_rng(seed), 3e-3, R_support=6e-3,
                          min_sep=S.lesion_separation(6e-3, 3e-3, 2e-3, 0.375e-3))
        assert S.sphere_inside_gland(c[0], 6e-3)


def test_lesion_separation_is_enforced():
    half, R_max, vox = 3e-3, 2e-3, 0.375e-3
    sep = S.lesion_separation(6e-3, half, R_max, vox)
    assert sep > 6e-3 + half                    # cube corner, kernel and voxel all counted
    for seed in range(25):
        c = S.rve_centres(2, np.random.default_rng(seed), half, min_sep=sep)
        assert np.linalg.norm(np.asarray(c[0]) - np.asarray(c[1])) >= sep


def test_segment_box_distance_is_exact():
    p0 = np.array([[-5e-3, 0, 0], [0, 3e-3, 0]])
    p1 = np.array([[5e-3, 0, 0], [1e-3, 3e-3, 0]])
    d = S.segment_box_distance(p0, p1, [0, 0, 0], 1e-3)
    assert d[0] == 0.0                          # passes straight through
    assert np.isclose(d[1], 2e-3, rtol=1e-9)    # parallel, 2 mm away
    # random oblique segments against a dense sample: never above it, and
    # equal to it within the sampling resolution
    rng = np.random.default_rng(1); h = 1e-3
    for _ in range(100):
        a = rng.uniform(-4e-3, 4e-3, 3); b = rng.uniform(-4e-3, 4e-3, 3)
        ex = S.segment_box_distance(a, b, [0, 0, 0], h)[0]
        s_ = np.linspace(0, 1, 20001)[:, None]; q = a + s_ * (b - a)
        dense = np.sqrt((np.maximum(np.abs(q) - h, 0) ** 2).sum(1)).min()
        assert ex <= dense + 1e-12 and abs(ex - dense) < 1e-5 * h + 1e-9


def test_input_voxels_leave_room_for_the_kernel():
    rng = np.random.default_rng(0)
    half, vox, R = 3e-3, 0.375e-3, 2e-3
    for v in S.input_voxels(rng, (0, 0, 0), half, 20, vox, R):
        assert np.all(np.abs(np.asarray(v)) <= half - R - vox + 1e-12)


# ------------------------------------------------------------------ pairing
def _fake_net(parent, r, Lpath, p0, p1, twin=None, r_dau=None):
    n = len(r)
    return SimpleNamespace(n=n, parent=np.array(parent), kind=np.zeros(n, int),
                           twin=np.full(n, -1) if twin is None else np.array(twin),
                           term=np.zeros(n, bool), r=np.array(r), Lpath=np.array(Lpath),
                           p0=np.array(p0, float), p1=np.array(p1, float),
                           r_dau=np.full((n, 2), np.nan) if r_dau is None else np.array(r_dau, float))


def test_fingerprints_separate_topology_from_path_length():
    a = _fake_net([-1, 0], [1e-4, 8e-5], [1e-3, 1e-3], [[0, 0, 0], [1e-3, 0, 0]],
                  [[1e-3, 0, 0], [2e-3, 0, 0]])
    b = _fake_net([-1, 0], [1e-4, 8e-5], [1.3e-3, 1e-3], [[0, 0, 0], [1e-3, 0, 0]],
                  [[1e-3, 0, 0], [2e-3, 0, 0]])          # tortuosity only
    c = _fake_net([-1, 0], [1e-4, 5e-5], [1e-3, 1e-3], [[0, 0, 0], [1e-3, 0, 0]],
                  [[1e-3, 0, 0], [2e-3, 0, 0]])          # different branching
    d = _fake_net([-1, 0], [1e-4, 8e-5], [1e-3, 1e-3], [[0, 0, 0], [1e-3, 0, 0]],
                  [[1e-3, 0, 0], [2e-3, 0, 0]], r_dau=[[8e-5, 6e-5], [np.nan, np.nan]])
    ta, pa = S.fingerprints(a); tb, pb = S.fingerprints(b); tc, pc = S.fingerprints(c)
    td, _ = S.fingerprints(d)
    assert ta == tb and pa != pb                 # same topology, different paths
    assert ta != tc                              # different branching
    assert ta != td                              # different omitted daughters, hence bed


def test_pressures_reach_the_drainage_module():
    cfg = dict(BASE); cfg["P_ART_mmHg"] = 60.0
    apply_globals(cfg)
    try:
        assert np.isclose(DR._flow.P_ART / F.MMHG, 60.0)
    finally:
        cfg["P_ART_mmHg"] = 70.0; apply_globals(cfg)


def test_drainage_equilibrium_equals_declared_total():
    """The relaxation must approach the declared equilibrium volume, which is
    the check that the loading fraction is applied exactly once."""
    from porovasc.geometry import network as N
    net = N.build(N.Params(seed=3, d_term_gland=120e-6, rve_half=2e-3,
                           rve_centres=((0.0, 0.0, 0.0),)))
    fl = F.solve(net)
    dr = DR.solve(net, fl, (0.0, 0.0, 0.0), 2e-3, dP_ext=100.0, H=5e3, t_max=100.0)
    assert np.isclose(dr.f[-1], 1.0, atol=1e-6)
    # and the amplitude identity on the clipped support
    strain = dr.dP_ext / 5e3
    assert np.isclose(-dr.dV_inf / strain / dr.V_tissue, dr.phi, rtol=1e-10)


def test_travel_bound_accumulates_slowest_streamlines():
    """A ten-segment path at rho_max 0.9 whose mean-flow times fit the record
    but whose slowest streamlines do not must be flagged."""
    from porovasc.physics import transport as T
    old = T.RHO_MAX; T.RHO_MAX = 0.9
    try:
        # mean-flow route 12 s: with the bolus and the bed it fits the 81.92 s
        # record; the slowest streamline at rho 0.9 runs at 0.38 of the mean
        # speed, and the same route then does not fit
        n = 10; L = np.full(n, 2e-3); v = np.full(n, 1.667e-3); d = np.full(n, 100e-6)
        net = SimpleNamespace(n=n, Lpath=L, d=d, kind=np.zeros(n, int), gen=np.arange(n),
                              parent=np.r_[-1, np.arange(n - 1)])
        fl = SimpleNamespace(v=v, t_arr=np.cumsum(L / v), t_seg=L / v)
        mean_route = T.travel_time_bound(net, fl, poiseuille=False)
        slow_route = T.travel_time_bound(net, fl, poiseuille=True)
        assert mean_route < T.T_END < slow_route
        assert np.isclose(T.bed_quantile(0.999), 3.266, atol=2e-3)
    finally:
        T.RHO_MAX = old


# ------------------------------------------------------------------ analysis
def _rec(arm, seed, rve, role, tau, ests, kernel="plug"):
    vox = [dict(input=i, est=[dict(kernel=kernel, R_mm=1.0, **e)]) for i, e in enumerate(ests)]
    return dict(arm=arm, seed=seed, rve=rve, condition=arm, role=role, same_topology=True,
                same_path=arm == "baseline", bed_constant="held_from_baseline", wrapped=False,
                clearance_mm=5.0 if role == "reference" else None,
                compression=dict(tau_rc=tau, phi_region=0.02, k_rc_index=1e-12, c_k_rc=80.0,
                                 k_bundle_straight=1e-11, k_bundle_tort=8e-12,
                                 support=dict(T2=1.6, d_perm=1e-4)),
                permeability=dict(k_network_mean=1e-13, k_network_face=[1e-13, 2e-13, 0.0], n_blocked=1,
                                  c_k_network=1e3, k_bundle_straight=1e-11, k_bundle_tort=8e-12,
                                  support=dict(phi=0.01, d_perm=1e-4, T2=1.5)),
                voxels=vox)


def _ok(v, D, acc=(0, 1, 2), **kw):
    e = dict(status="ok", v=v, D=D, at_bound=False, bed_share=0.5, area=1.0, accepted=list(acc))
    e.update(kw); return e


def _manifest(out, seeds, arms):
    json.dump(dict(kind="test", schema=S.SCHEMA, seeds=seeds,
                   effective=dict(radii_mm=[1.0], poiseuille=[False], n_inputs=4,
                                  arms=[dict(name=a) for a in arms])), open(out / "manifest.json", "w"))


def _write(out, recs):
    out.mkdir(exist_ok=True)
    with open(out / "records.jsonl", "w") as fh:
        for r in recs: fh.write(json.dumps(r) + "\n")
    (out / "COMPLETE").write_text("x")


def test_paired_effects_are_reduced_gland_by_gland():
    """Halving v in two glands and doubling it in the third must give gland
    effects [-0.301, -0.301, +0.301] whatever the number of voxels per gland;
    invalid pairs are excluded and counted; a changed causality set is split."""
    from porovasc import analyse_paired as A
    recs = []
    for seed, f in ((1, 0.5), (2, 0.5), (3, 2.0)):
        for rve in (0, 1):
            n = 4 if seed == 1 else 2
            recs += [_rec("baseline", seed, rve, "sample", 0.01, [_ok(1e-3, 1e-6)] * n),
                     _rec("arm", seed, rve, "sample", 0.02, [_ok(f * 1e-3, 1e-6)] * n)]
    recs[1]["voxels"][0]["est"][0]["at_bound"] = True
    recs[3]["voxels"][0]["est"][0]["accepted"] = [0, 1]
    s = A.summarise_arm(recs, "arm", ["plug"], [1.0])
    k = s["kinetics"]["plug_R1.00_new_front20"]["all"]
    v = k["effects"]["v"]
    assert v["n_glands"] == 3 and v["boot95"] is not None
    assert np.allclose(sorted(v["values"]), [np.log10(0.5), np.log10(0.5), np.log10(2.0)], atol=1e-9)
    assert k["transitions"]["ok>ok+at_bound"] == 1 and k["transitions"]["ok>ok"] == 15
    assert k["different_accepted"]["n"] == 1 and k["same_accepted"]["n"] == 14
    assert k["effects"]["bed_share"]["kind"] == "diff" and k["effects"]["area"]["n_glands"] == 3
    assert np.isclose(s["volume"]["tau_rc"]["all"]["median"], np.log10(2.0))
    assert s["volume"]["n_blocked"]["all"]["kind"] == "diff" and s["volume"]["T2_cube"]["all"]["n_glands"] == 3
    z = s["volume"]["k_network_z"]["all"]                                # zero permeability: no log ratio,
    assert z["n_glands"] == 0 and z["pairs"] == 6 and z["absent"] == 6   # and the absence is counted
    assert z["transitions"] == {"zero>zero": 6}
    assert k["same_accepted"]["v"]["kind"] == "log"


def test_analysis_keys_cover_the_estimator_grid():
    """Every (kernel, radius, transfer function, rule) in the manifest gets its
    own summary; entries without tf/rule fields are read as new/front20."""
    from porovasc import analyse_paired as A
    recs = [_rec("baseline", 1, 0, "sample", 0.01, [_ok(1.0, 1.0)]),
            _rec("arm", 1, 0, "sample", 0.01, [_ok(2.0, 1.0)])]
    for r in recs:
        for e in r["voxels"][0]["est"]:
            e["tf"] = "new"; e["rule"] = "front20"
        r["voxels"][0]["est"].append(dict(e, tf="old", rule="moment", v=e["v"] * 3))
    s = A.summarise_arm(recs, "arm", ["plug"], [1.0], ["new", "old"], ["front20", "moment"])
    keys = set(s["kinetics"])
    assert keys == {"plug_R1.00_new_front20", "plug_R1.00_new_moment", "plug_R1.00_old_front20", "plug_R1.00_old_moment"}
    assert s["kinetics"]["plug_R1.00_new_front20"]["all"]["valid_pairs"] == 1
    assert s["kinetics"]["plug_R1.00_old_moment"]["all"]["valid_pairs"] == 1
    assert s["kinetics"]["plug_R1.00_new_moment"]["all"]["valid_pairs"] == 0   # not run: missing, not silent
    assert A._est(recs[0], 0, "plug", 1.0) is not None


def test_absent_directional_pairs_are_counted_not_dropped():
    """Six volume pairs, one with a blocked x direction in the arm only: the
    gland effect is reported from the five usable pairs, and the summary says
    that one pair was absent and why."""
    from porovasc import analyse_paired as A
    recs = []
    for rve in range(6):
        b = _rec("baseline", 1, rve, "sample", 0.01, [_ok(1.0, 1.0)])
        a = _rec("arm", 1, rve, "sample", 0.01, [_ok(1.0, 1.0)])
        a["permeability"]["k_network_face"] = [2e-13, 2e-13, 0.0]
        recs += [b, a]
    recs[-1]["permeability"]["k_network_face"][0] = 0.0                # positive > zero in one pair
    x = A.summarise_arm(recs, "arm", ["plug"], [1.0])["volume"]["k_network_x"]["all"]
    assert x["pairs"] == 6 and x["usable"] == 5 and x["absent"] == 1
    assert x["transitions"] == {"positive>positive": 5, "positive>zero": 1}
    assert np.isclose(x["median"], np.log10(2.0))


def test_baseline_summary_takes_volume_medians_before_gland_medians():
    """Two volumes with voxel medians 1 and 100 in one gland give a gland value
    of 50.5, not the pooled-voxel median."""
    from porovasc import analyse_paired as A
    recs = [_rec("baseline", 1, 0, "sample", 0.01, [_ok(1.0, 1.0)] * 9),          # nine voxels at 1
            _rec("baseline", 1, 1, "sample", 0.01, [_ok(100.0, 1.0)] * 1)]        # one voxel at 100
    b = A.summarise_baseline(recs, ["plug"], [1.0])
    assert np.isclose(b["kinetics"]["plug_R1.00_new_front20"]["v_mm_s"]["per_seed"][1], 50.5 * 1e3)
    assert np.isclose(b["kinetics"]["plug_R1.00_new_front20"]["bed_share"]["per_seed"][1], 0.5)
    assert b["volume"]["k_network_z"]["absent"] == 2 and b["volume"]["k_network_z"]["usable"] == 0


def test_accepted_set_split_is_reduced_gland_by_gland():
    """Gland 1 has three volumes at -1, gland 2 one volume at +1: the gland-level
    median of the same-accepted effect is 0, not the pooled -1."""
    from porovasc import analyse_paired as A
    recs = []
    for rve in range(3):
        recs += [_rec("baseline", 1, rve, "sample", 0.01, [_ok(1.0, 1.0)]),
                 _rec("arm", 1, rve, "sample", 0.01, [_ok(0.1, 1.0)])]
    recs += [_rec("baseline", 2, 0, "sample", 0.01, [_ok(1.0, 1.0)]),
             _rec("arm", 2, 0, "sample", 0.01, [_ok(10.0, 1.0)])]
    s = A.summarise_arm(recs, "arm", ["plug"], [1.0])
    sa = s["kinetics"]["plug_R1.00_new_front20"]["all"]["same_accepted"]
    assert sa["n"] == 4 and sa["v"]["n_glands"] == 2 and np.isclose(sa["v"]["median"], 0.0)


def test_transitions_keep_wrapped_and_bound_together():
    from porovasc import analyse_paired as A
    recs = [_rec("baseline", 1, 0, "sample", 0.01, [_ok(1.0, 1.0)]),
            _rec("arm", 1, 0, "sample", 0.01, [_ok(1.0, 1.0, status="wrapped", at_bound=True)])]
    k = A.summarise_arm(recs, "arm", ["plug"], [1.0])["kinetics"]["plug_R1.00_new_front20"]["all"]
    assert k["transitions"] == {"ok>wrapped+at_bound": 1} and k["valid_pairs"] == 0


def test_load_refuses_old_schema_duplicates_and_double_files(tmp_path):
    from porovasc import analyse_paired as A
    recs = [_rec("baseline", 1, 0, "sample", 0.01, [_ok(1.0, 1.0)])]
    out = tmp_path / "ok"; _write(out, recs); _manifest(out, [1], [])
    A.load(str(out))                                                    # accepted
    old = tmp_path / "old"; _write(old, recs); _manifest(old, [1], [])
    m = json.load(open(old / "manifest.json")); m["schema"] = 3; json.dump(m, open(old / "manifest.json", "w"))
    with pytest.raises(SystemExit):
        A.load(str(old))
    dup = tmp_path / "dup"; _write(dup, recs + recs); _manifest(dup, [1], [])
    with pytest.raises(SystemExit):
        A.load(str(dup))
    both = tmp_path / "both"; _write(both, recs); _manifest(both, [1], [])
    (both / "records.jsonl.gz").write_bytes(b"")
    with pytest.raises(SystemExit):
        A.load(str(both))


def test_analysis_end_to_end_writes_summary_with_provenance(tmp_path):
    from porovasc import analyse_paired as A
    recs = []
    for seed in (1, 2, 3):
        recs += [_rec("baseline", seed, 0, "sample", 0.01, [_ok(1e-3, 1e-6)] * 2),
                 _rec("arm", seed, 0, "sample", 0.02, [_ok(0.5e-3, 1e-6)] * 2)]
    out = tmp_path / "run"; _write(out, recs); _manifest(out, [1, 2, 3], ["arm"])
    A.main([str(out)])
    summ = json.load(open(out / "analysis" / "summary.json"))
    assert summ["arms"][0]["kinetics"]["plug_R1.00_new_front20"]["all"]["effects"]["v"]["n_glands"] == 3
    assert summ["analysis"]["partial"] is False and len(summ["analysis"]["sha256"]) == 64
    assert summ["analysis"]["boot_seed"] == A.BOOT_SEED and summ["analysis"]["n_boot"] == A.N_BOOT
