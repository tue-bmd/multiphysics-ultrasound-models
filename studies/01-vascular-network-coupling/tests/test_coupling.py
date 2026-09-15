"""The shared-dependence machinery, the flow coherence, and the corrected
matched comparison between the two estimators."""
import os
import types

import numpy as np
import pytest

from porovasc import analyse_coupling as AC
from porovasc import analyse_estimators as AE
from porovasc import cross_settings as CS
from porovasc.homogenise import darcy as DA
from porovasc.run_coupling import curve_descriptors, spectrum_width


# ------------------------------------------------------------ flow coherence
def _fake(dirs, Q, L=1e-3, r=20e-6, centre=(0.0, 0.0, 0.0)):
    """A bundle of straight segments through the origin, as the coherence
    function needs them."""
    dirs = np.asarray(dirs, float)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    p0 = np.zeros_like(dirs) + np.asarray(centre, float)
    p1 = p0 + dirs * L
    net = types.SimpleNamespace(p0=p0, p1=p1,
                                Lchord=np.full(len(dirs), L),
                                Lpath=np.full(len(dirs), L),
                                r=np.full(len(dirs), r))
    fl = types.SimpleNamespace(Q=np.asarray(Q, float))
    return net, fl


def test_coherence_is_one_when_every_vessel_carries_flow_the_same_way():
    n = 12
    net, fl = _fake([[1.0, 0.0, 0.0]] * n, [1e-9] * n)
    out = DA.coherence(net, fl, (0.0, 0.0, 0.0), 5e-3)
    assert out["chi"] == pytest.approx(1.0, rel=1e-12)


def test_coherence_falls_to_zero_when_flows_oppose():
    n = 12
    dirs = [[1.0, 0.0, 0.0]] * n
    Q = [1e-9] * (n // 2) + [-1e-9] * (n // 2)   # half the bundle runs backwards
    net, fl = _fake(dirs, Q)
    out = DA.coherence(net, fl, (0.0, 0.0, 0.0), 5e-3)
    assert out["chi"] == pytest.approx(0.0, abs=1e-12)
    # throughput is unchanged, which is the whole point of reporting both
    assert out["vessel_speed"] > 0


def test_coherence_separates_the_two_speeds():
    n = 12
    Q = [1e-9] * (n // 2) + [-1e-9] * (n // 2)
    net, fl = _fake([[1.0, 0.0, 0.0]] * n, Q)
    out = DA.coherence(net, fl, (0.0, 0.0, 0.0), 5e-3)
    assert out["darcy_speed"] == pytest.approx(0.0, abs=1e-18)
    assert out["vessel_speed"] == pytest.approx(1e-9 / (np.pi * 20e-6 ** 2), rel=1e-12)


def test_coherence_declines_to_answer_on_too_few_segments():
    net, fl = _fake([[1.0, 0.0, 0.0]] * 3, [1e-9] * 3)
    assert DA.coherence(net, fl, (0.0, 0.0, 0.0), 5e-3)["chi"] is None


# ------------------------------------------------------- curve descriptors
def test_curve_descriptors_recover_a_known_gaussian():
    t = np.arange(0, 60, 0.02)
    mu, sd = 12.0, 2.5
    y = np.exp(-0.5 * ((t - mu) / sd) ** 2)
    d = curve_descriptors(t, y)
    assert d["mtt"] == pytest.approx(mu, rel=1e-4)
    assert d["width"] == pytest.approx(sd, rel=1e-3)
    assert d["t_peak"] == pytest.approx(mu, abs=0.02)
    assert d["area"] == pytest.approx(sd * np.sqrt(2 * np.pi), rel=1e-4)


def test_curve_descriptors_return_nothing_for_an_empty_curve():
    t = np.arange(0, 10, 0.1)
    assert curve_descriptors(t, np.zeros_like(t))["mtt"] is None


def test_spectrum_width_is_zero_for_a_single_time_constant():
    tau = np.logspace(-2, 2, 40)
    w = np.zeros_like(tau); w[17] = 1.0
    assert spectrum_width(tau, w) == pytest.approx(0.0, abs=1e-12)


def test_spectrum_width_grows_with_the_spread_of_the_weights():
    tau = np.logspace(-2, 2, 81)
    narrow = np.exp(-0.5 * ((np.log10(tau) - 0) / 0.2) ** 2)
    wide = np.exp(-0.5 * ((np.log10(tau) - 0) / 0.8) ** 2)
    assert spectrum_width(tau, narrow) < spectrum_width(tau, wide)
    assert spectrum_width(tau, wide) == pytest.approx(0.8, rel=0.1)


# ------------------------------------------------- the coupling effect table
def _crec(arm, seed, rve, phi, tau, v):
    return dict(arm=arm, condition=arm, seed=seed, rve=rve,
                network=dict(phi_cube=phi, k_mean=1e-12, k_x=1e-12, k_z=1e-12,
                             T2=1.5, d_perm=2e-5, chi_chi=0.05,
                             chi_vessel_speed=1e-3, n_blocked=0),
                drainage=dict(relax_per_strain=phi, tau_rc=tau, spectrum_width=0.4,
                              phi_region=phi),
                transport=dict(curves=[dict(t_peak=10.0, mtt=12.0, width=3.0,
                                            bed_share=0.5)] * 3,
                               est=[dict(v=v, D=1e-6)] * 3))


def test_effects_are_paired_and_reduced_over_glands():
    recs = []
    for s in (100, 101, 102):
        recs.append(_crec("baseline", s, 0, 0.03, 1.0, 1e-3))
        recs.append(_crec("H_3k", s, 0, 0.03, 5.0 / 3.0, 1e-3))
    eff = AC.arm_effects(recs, "H_3k")
    tau = eff["drainage"]["tau_rc"]
    assert tau["n_glands"] == 3
    assert tau["median"] == pytest.approx(np.log10(5.0 / 3.0), rel=1e-9)
    assert AC.verdict(tau) == "moved"
    # nothing else moved, which is what makes the modulus arm a control
    assert AC.verdict(eff["network"]["phi_cube"]) == "undetermined"
    assert AC.verdict(eff["transport"]["v"]) == "undetermined"


def test_a_quantity_absent_from_one_arm_is_counted_not_ignored():
    recs = []
    for s in (100, 101, 102):
        recs.append(_crec("baseline", s, 0, 0.03, 1.0, 1e-3))
        a = _crec("x", s, 0, 0.03, 1.0, 1e-3)
        a["transport"]["est"] = [dict(v=None, D=None)] * 3
        recs.append(a)
    eff = AC.arm_effects(recs, "x")
    assert eff["transport"]["v"]["n_volumes_with_effect"] == 0
    assert eff["transport"]["v"]["n_volumes_absent"] == 3
    assert eff["transport"]["v"]["n_glands"] == 0
    assert eff["transport"]["v"]["accounting"]["lost"] == 9


def test_per_voxel_quantities_are_reduced_within_the_volume_first():
    r = _crec("baseline", 100, 0, 0.03, 1.0, 1e-3)
    r["transport"]["est"] = [dict(v=1e-3, D=1e-6), dict(v=3e-3, D=1e-6),
                             dict(v=2e-3, D=1e-6)]
    assert AC.value(r, ("transport", "est", "v")) == pytest.approx(2e-3)


# ----------------------------------------- the matched estimator comparison
def _erec(seed, rve, shell, grid):
    """One record with one evaluation point carrying the given estimates."""
    est = []
    for fdt, R, v, D, bad in shell:
        est.append(dict(estimator="shell", psf_fwhm_mm=[1.1, 1.1, 1.1],
                        frame_dt_s=fdt, R_mm=R, v=v, D=D, at_bound=bad))
    for fdt, st, sx, v, D, bad in grid:
        est.append(dict(estimator="grid", psf_fwhm_mm=[1.1, 1.1, 1.1],
                        frame_dt_s=fdt, sigma_t_s=st, sigma_x_mm=sx,
                        v=v, D=D, negative_D=bad))
    return dict(seed=seed, rve=rve, points=[dict(est=est)])


PSF = (1.1, 1.1, 1.1)


def test_matched_uses_only_points_where_both_estimators_returned_a_value():
    recs = [
        _erec(100, 0, [(1.0, 1.0, 3e-3, 6e-7, False)], [(1.0, 4.0, 1.5, 1e-3, 1e-7, False)]),
        _erec(101, 0, [(1.0, 1.0, 9e-3, 6e-7, False)], [(1.0, 4.0, 1.5, 1e-3, 1e-7, True)]),
        _erec(102, 0, [(1.0, 1.0, 3e-3, 6e-7, False)], [(1.0, 4.0, 1.5, 1e-3, 1e-7, False)]),
    ]
    m = AE.matched(recs, PSF, 1.0, 1.0, 4.0, 1.5)
    assert m["n_points"] == 3
    assert m["n_shell_valid"] == 3 and m["n_grid_valid"] == 2
    assert m["n_jointly_valid"] == 2
    # the joint median excludes the gland whose grid fit failed
    assert m["hier"]["s_v"]["median"] == pytest.approx(3e-3)
    # while each estimator's own median includes it, which is the confound
    assert m["hier"]["so_v"]["median"] == pytest.approx(3e-3)
    assert m["ratio_v"] == pytest.approx(3.0)


def test_matched_reports_failure_rates_for_both_estimators():
    recs = [_erec(100, 0, [(1.0, 1.0, 3e-3, 6e-7, True)],
                  [(1.0, 4.0, 1.5, 1e-3, 1e-7, False)])]
    m = AE.matched(recs, PSF, 1.0, 1.0, 4.0, 1.5)
    assert m["shell_failure"] == pytest.approx(1.0)
    assert m["grid_failure"] == pytest.approx(0.0)
    assert m["n_jointly_valid"] == 0


def test_matched_refuses_a_setting_that_does_not_identify_one_estimate():
    r = _erec(100, 0, [(1.0, 1.0, 3e-3, 6e-7, False), (1.0, 1.0, 4e-3, 7e-7, False)],
              [(1.0, 4.0, 1.5, 1e-3, 1e-7, False)])
    with pytest.raises(SystemExit):
        AE.matched([r], PSF, 1.0, 1.0, 4.0, 1.5)


def test_common_frames_are_the_intersection_of_the_two_estimators():
    recs = [_erec(100, 0,
                  [(0.25, 1.0, 3e-3, 6e-7, False), (1.0, 1.0, 3e-3, 6e-7, False)],
                  [(1.0, 4.0, 1.5, 1e-3, 1e-7, False), (2.0, 4.0, 1.5, 1e-3, 1e-7, False)])]
    assert AE.common_frames(recs) == [1.0]


def test_matched_will_not_compare_across_frame_intervals():
    """The reported difference must not mix a 0.25 s shell estimate with a grid
    estimate made at 1 s; asking for a frame the grid never ran at fails."""
    recs = [_erec(100, 0,
                  [(0.25, 1.0, 3e-3, 6e-7, False), (1.0, 1.0, 3e-3, 6e-7, False)],
                  [(1.0, 4.0, 1.5, 1e-3, 1e-7, False)])]
    with pytest.raises(SystemExit):
        AE.matched_table(recs, PSF, frame_dt=0.25)        # the grid never ran there
    assert AE.matched_table(recs, PSF)                    # the shared one works


def test_matched_table_says_so_when_the_two_estimators_share_no_frame(capsys):
    recs = [_erec(100, 0, [(0.25, 1.0, 3e-3, 6e-7, False)],
                  [(1.0, 4.0, 1.5, 1e-3, 1e-7, False)])]
    assert AE.matched_table(recs, PSF) == []
    assert "share no frame interval" in capsys.readouterr().out


# ------------------------------ generation against analysis in cross_settings
def test_generation_and_analysis_are_reported_apart():
    rows = [("plug", 1.0, "new", "front20", 1.2, 0.50, 5, "ok"),
            ("plug", 1.5, "new", "front20", 1.6, 0.90, 5, "ok"),
            ("plug", 1.0, "old", "front20", 0.9, 0.30, 5, "ok"),
            ("pois", 1.0, "new", "front20", 2.4, 1.00, 5, "ok"),
            ("pois", 1.5, "new", "front20", 3.0, 1.80, 5, "ok"),
            ("pois", 1.0, "old", "front20", 1.8, 0.60, 5, "ok")]
    out = CS.split_generation_and_analysis(rows)
    # within one kernel the analysis alone spans a factor of three on D
    assert out["per_kernel"]["plug"]["D"][2] == pytest.approx(3.0, rel=1e-9)
    assert out["per_kernel"]["pois"]["D"][2] == pytest.approx(3.0, rel=1e-9)
    # and the kernel effect is a clean factor of two, held at fixed analysis
    assert out["generation"]["D"][0] == pytest.approx(0.5, rel=1e-9)
    assert out["generation"]["D"][1] == pytest.approx(0.5, rel=1e-9)


# --------------------------------- paired voxel effects and validity handling
def _trec(arm, seed, rve, est, curves=None):
    """One record whose transport block carries the given per-voxel entries."""
    curves = curves or [dict(area=1.0, t_peak=10.0, mtt=12.0, width=3.0,
                             bed_share=0.5) for _ in est]
    return dict(arm=arm, condition=arm, seed=seed, rve=rve,
                network=dict(phi_cube=0.03, k_mean=1e-12, k_x=1e-12, k_z=1e-12,
                             T2=1.5, d_perm=2e-5, chi_chi=0.05,
                             chi_vessel_speed=1e-3, n_blocked=0),
                drainage=dict(relax_per_strain=0.03, tau_rc=1.0,
                              spectrum_width=0.4, phi_region=0.03),
                transport=dict(status="ok", curves=curves, est=est))


def test_an_estimate_at_an_optimizer_bound_is_not_an_estimate():
    b = _trec("baseline", 100, 0, [dict(v=1e-3, D=1e-6, at_bound=False),
                                   dict(v=2e-3, D=2e-6, at_bound=False)])
    a = _trec("x", 100, 0, [dict(v=1e-4, D=1e-6, at_bound=True),
                            dict(v=4e-3, D=4e-6, at_bound=False)])
    e, acc = AC.paired_effect(b, a, ("transport", "est", "v"), "log")
    # only the second voxel is valid in both, so the effect is its effect alone
    assert e == pytest.approx(np.log10(2.0), rel=1e-12)
    assert acc == dict(n=2, both=1, lost=1, gained=0, neither=0)


def test_voxels_are_paired_by_index_not_pooled():
    """Taking a median in each condition first would compare voxel 0 of one with
    voxel 1 of the other when one of them drops out."""
    b = _trec("baseline", 100, 0, [dict(v=1e-3, D=1e-6, at_bound=False),
                                   dict(v=1e-2, D=1e-6, at_bound=False)])
    a = _trec("x", 100, 0, [dict(v=2e-3, D=1e-6, at_bound=False),
                            dict(v=None, D=None, at_bound=False)])
    paired, acc = AC.paired_effect(b, a, ("transport", "est", "v"), "log")
    assert paired == pytest.approx(np.log10(2.0), rel=1e-12)
    assert acc["both"] == 1 and acc["lost"] == 1
    # the pooled alternative would have compared 2e-3 against a median of 5.5e-3
    pooled = np.log10(AC.value(a, ("transport", "est", "v"))
                      / AC.value(b, ("transport", "est", "v")))
    assert abs(pooled - paired) > 0.3


def test_a_wrapped_transport_block_yields_no_transport_effect():
    b = _trec("baseline", 100, 0, [dict(v=1e-3, D=1e-6, at_bound=False)])
    a = _trec("x", 100, 0, [dict(v=2e-3, D=1e-6, at_bound=False)])
    a["transport"]["status"] = "wrapped"
    e, acc = AC.paired_effect(b, a, ("transport", "est", "v"), "log")
    assert e is None and acc["n"] == 0


def test_the_curve_area_is_in_the_reported_table():
    assert "area" in AC.FAMILIES["transport"]


def test_flow_dependent_descriptors_are_excluded_from_the_geometry_invariants():
    assert "chi" not in AC.GEOMETRY_INVARIANTS
    assert "vessel_speed" not in AC.GEOMETRY_INVARIANTS
    assert "phi_cube" in AC.GEOMETRY_INVARIANTS and "k_mean" in AC.GEOMETRY_INVARIANTS


# ------------------------------------------- the arms actually do something
@pytest.mark.slow
def test_the_rejected_arms_really_do_nothing_under_this_study():
    """rho_0.5 and P_END_30 were in an earlier version of COUPLING_ARMS.  Under
    the plug kernel and a bed conductance held from the baseline, neither is
    read at all, so both leave geometry, flow and every contrast curve
    bit-identical.  This is why they were removed, and it is asserted rather
    than remembered."""
    import numpy as np
    from porovasc import study as S
    from porovasc.config import BASE, SENS, COUPLING_REJECTED, apply_globals
    from porovasc.physics import flow as F, transport as TR

    cfg0 = dict(BASE); cfg0["d_term_gland"] = 80e-6
    apply_globals(cfg0)
    centres = S.rve_centres(1, np.random.default_rng(100), cfg0["rve_half"],
                            R_support=cfg0["R_comp"])
    net = S.build_arm(S.Arm("baseline", cfg0), 100, centres)
    fl0 = F.solve(net, R_lat=cfg0["R_LAT"]); G0 = float(fl0.G_bed0)
    rin = np.random.default_rng(100 + 1_000_000)
    inputs = S.input_voxels(rin, centres[0], cfg0["rve_half"], 2,
                            cfg0["vox_half_mm"] * 1e-3, 1e-3)
    ks = [k for c in inputs
          for k in TR.shell_kernels(c, 1e-3, cfg0["vox_half_mm"] * 1e-3, n_dir=8)]
    ref = np.array([TR.propagate(net, fl0, ks, poiseuille=False).tic[i]
                    for i in range(len(ks))])
    for name in COUPLING_REJECTED:
        arm = S.Arm(name, dict(cfg0, **SENS[name]))
        apply_globals(arm.cfg)
        n2 = S.build_arm(arm, 100, centres)
        fl = S.solve_arm(n2, arm, G0)
        assert n2.n == net.n and np.array_equal(n2.r, net.r), name
        assert np.array_equal(fl.Q, fl0.Q), name
        cur = np.array([TR.propagate(n2, fl, ks, poiseuille=False).tic[i]
                        for i in range(len(ks))])
        assert np.array_equal(cur, ref), name
        apply_globals(cfg0)


@pytest.mark.slow
def test_the_control_arms_hold_on_a_real_run(tmp_path):
    """End to end: a small real study, and the control invariants asserted on
    its records rather than described in prose."""
    from porovasc.run_coupling import run
    out = str(tmp_path / "coupling")
    assert run(out, ["H_3k", "P_ART_60"], n_real=2, n_rve=1, seed0=100,
               n_inputs=3, small=True)
    man, recs = AC.load(out)
    effs = [(a, loc, AC.arm_effects(recs, a))
            for a, loc in (("H_3k", "mechanics"), ("P_ART_60", "flow"))]
    assert AC.control_checks(effs)
    # a control that moves nothing at all is not a control for anything: the
    # flow arm has to reach the transport
    flow = {a: e for a, _, e in effs}["P_ART_60"]
    assert abs(flow["transport"]["mtt"]["median"]) > 0


# ------------------------------------- the wrap stop must actually stop
@pytest.mark.slow
def test_a_wrapping_arm_stops_the_run_and_writes_nothing(tmp_path):
    """Regression: the stop path used to build a manifest entry keyed by a
    tuple, which json cannot write.  That TypeError was caught by the same
    `except Exception` that handles build failures, so the run carried on and
    finished carrying invalid transport.  The stop now sits outside that
    handler and the keys are strings."""
    import json
    from porovasc.run_coupling import run
    out = str(tmp_path / "wrap")
    with pytest.raises(SystemExit) as e:
        run(out, ["H_3k"], n_real=1, n_rve=1, seed0=100, n_inputs=2,
            small=True, t_end=0.5)          # nothing can traverse in 0.5 s
    assert "would wrap" in str(e.value)
    assert "--t-end" in str(e.value)
    recs = os.path.join(out, "records.jsonl")
    assert not os.path.exists(recs) or os.path.getsize(recs) == 0
    man = json.load(open(os.path.join(out, "manifest.json")))
    assert man["travel_time_bounds"]                      # serialized
    assert all(isinstance(k, str) for k in man["travel_time_bounds"])
    assert not os.path.exists(os.path.join(out, "COMPLETE"))


@pytest.mark.slow
def test_the_probe_uses_the_study_configuration(tmp_path):
    """Regression: the probe hardcoded one sampling volume while the study
    defaults to two.  Sampling volumes are refined regions of the tree, so a
    bound measured on a one-volume gland does not bound a two-volume one."""
    import numpy as np
    from porovasc import study as S
    from porovasc.config import BASE, apply_globals
    from porovasc.physics import flow as F, transport as TR
    from porovasc.run_coupling import probe_t_end, EST

    cfg = dict(BASE); cfg["d_term_gland"] = 80e-6
    apply_globals(cfg)
    sizes = []
    for n_rve in (1, 2):
        centres = S.rve_centres(n_rve, np.random.default_rng(100), cfg["rve_half"],
                                R_support=cfg["R_comp"])
        sizes.append(S.build_arm(S.Arm("baseline", cfg), 100, centres).n)
    assert sizes[0] != sizes[1], "the number of volumes must change the gland"

    # the probe, at the study's configuration, must report the bound of the
    # gland the study will actually build
    centres = S.rve_centres(2, np.random.default_rng(100), cfg["rve_half"],
                            R_support=cfg["R_comp"])
    net = S.build_arm(S.Arm("baseline", cfg), 100, centres)
    fl = F.solve(net, R_lat=cfg["R_LAT"])
    direct = float(TR.travel_time_bound(net, fl,
                                        poiseuille=(EST["kernel"] == "pois")))
    need = probe_t_end([], n_real=1, n_rve=2, seed0=100, small=True)
    assert need >= direct
    assert need / 2 < direct or need == pytest.approx(BASE["T_END_s"])
