"""Paired studies: sensitivity arms and lesions, each against its own baseline.

    python -m porovasc.run_paired sweep  --out runs/sweep_alpha06 --arm alpha_0.6 --n-real 5
    python -m porovasc.run_paired lesion --out runs/lesion_tort16 --tort 1.6 --n-real 10
    python -m porovasc.run_paired base   --out runs/baseline --n-real 5 --n-rve 6

Every arm is built from the same seed, the same sampling-volume centres and the
same input voxels as its baseline, and uses the bed constant calibrated on that
baseline.  Each record carries the quantities of one sampling volume in one
condition, together with the identifiers needed to pair it: arm, seed, rve,
condition.

For the lesion studies the first sampling volume holds the lesion and the second
is a spatially matched reference region, placed far enough away that no
lesion-modified segment can reach its acquisition support.  It is not an
unaffected counterfactual: when the intervention changes the branching draws the
graph diverges globally, and pressures and flows redistribute through the whole
network.  Each record stores the realized clearance, the topology fingerprint of
its build and whether that fingerprint equals the baseline's.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import traceback

import numpy as np

from .geometry import network as N
from .physics import flow as F, drainage as DR, transport as TR
from .homogenise import darcy as DA
from .config import BASE, SENS, apply_globals
from . import study as S

# arms whose intervention is the calibration target itself: holding the bed
# constant fixed would make them inert, so they recalibrate by design
CALIBRATION_ARMS = ("P_END_30", "P_END_40")

# arms that must leave the geometry, the network permeability and the contrast
# kinetics untouched: their records are compared with the baseline record of
# the same volume and the run fails if anything but the compression differs
NULL_ARMS = {"H_3k", "H_10k", "Rcomp_1.5", "Rcomp_6"}
NULL_KEYS = ("permeability", "voxels", "fingerprint", "n_seg")

def measure(net, fl, cfg, centre, inputs, poiseuille=(False, True), radii=(1.0,), n_dir=48,
            tfs=("new",), rules=("front20",), wrapped=False):
    """Everything measured on one sampling volume: the supports, the compression
    relaxation, the network permeability, and the two-kernel estimates for the
    given input voxels under every (kernel, radius, transfer function,
    causality rule), with the accepted shell directions kept.  The curves are
    propagated once per kernel and radius; the transfer function and the rule
    only change the fit."""
    h = cfg["rve_half"]; vox = cfg["vox_half_mm"] * 1e-3
    rec = dict(centre_mm=list(np.asarray(centre) * 1e3))
    # compression, on the sphere support
    dr = DR.solve(net, fl, centre, cfg["R_comp"], dP_ext=100.0, H=cfg["H"])
    R_comp = cfg["R_comp"]
    sph = DA.support_stats(net, N.sphere_fraction(net.p0, net.p1, np.asarray(centre), R_comp),
                           4 / 3 * np.pi * R_comp ** 3)
    # the relaxation and the descriptors now share one support: the compression
    # loads the clipped part of each vessel, and dr.phi is that same clipped phi
    k_rc = float(F.ETA * dr.phi * R_comp ** 2 / (dr.tau_rc * cfg["H"]))
    rec["compression"] = dict(tau_rc=float(dr.tau_rc), phi_region=float(dr.phi), k_rc_index=k_rc,
                              support=sph, support_mismatch=float(abs(dr.phi - sph["phi"]) / max(sph["phi"], 1e-30)),
                              c_k_rc=float(dr.phi * sph["d_perm"] ** 2 / k_rc) if k_rc > 0 else None,
                              **DA.bundle(dr.phi, sph["d_perm"], sph["T2"]))
    # network permeability, on the cube support
    da = DA.measure(net, fl, centre, h)
    cube = DA.support_stats(net, N.cube_fraction(net.p0, net.p1, np.asarray(centre), h), (2 * h) ** 3)
    rec["permeability"] = dict(k_network_face=[float(x) for x in da.k_face],
                               k_network_mean=float(da.k_mean), n_blocked=int(da.n_blocked),
                               c_k_network=da.c_k, support=cube,
                               **DA.bundle(cube["phi"], cube["d_perm"], cube["T2"]))
    # contrast kinetics
    vox_recs = [dict(input=i, centre_mm=list(np.asarray(c) * 1e3), est=[]) for i, c in enumerate(inputs)]
    for pois in poiseuille:
        for R_mm in radii:
            R = R_mm * 1e-3
            allk, index = [], []
            for c in inputs:
                ks = TR.shell_kernels(c, R, vox, n_dir=n_dir)
                index.append((len(allk), len(ks))); allk += ks
            tr = TR.propagate(net, fl, allk, poiseuille=pois)
            for i, (i0, nk) in enumerate(index):
                tin = tr.tic[i0]
                kern = "pois" if pois else "plug"
                if tin.sum() <= 0:
                    for tf in tfs:
                        for rule in rules:
                            vox_recs[i]["est"].append(dict(kernel=kern, R_mm=R_mm, tf=tf, rule=rule,
                                                           status="no_signal"))
                    continue
                outs = [tr.tic[k] for k in range(i0 + 1, i0 + nk)]
                area = float(tin.sum() * TR.DT)
                bed_share = float(tr.tic_bed[i0].sum() / max(tin.sum(), 1e-30))
                tail = float(TR.tail_fraction(tin))
                for tf in tfs:
                    for rule in rules:
                        v, D, n, r2, info = TR.identify_shell(tr.t, tin, outs, R, tf=tf, rule=rule, detail=True)
                        status = "ok" if np.isfinite(v) else ("few_pairs" if n < 3 else "no_fit")
                        if wrapped:
                            status = "wrapped"          # record is shorter than the travel time
                        at_bound = bool(np.isfinite(v) and (v <= 1.01e-4 or v >= 0.099 or D <= 1.01e-9 or D >= 0.99e-4))
                        vox_recs[i]["est"].append(dict(
                            kernel=kern, R_mm=R_mm, tf=tf, rule=rule, status=status,
                            v=float(v) if np.isfinite(v) else None, D=float(D) if np.isfinite(D) else None,
                            n_pairs=int(n), r2=float(r2) if np.isfinite(r2) else None, at_bound=at_bound,
                            accepted=info["accepted"],
                            arrival_margin_s=[None if m is None else round(m, 4) for m in info["margin"]],
                            area=area, bed_share=bed_share, tail=tail))
    rec["voxels"] = vox_recs
    return rec


def _same(a, b):
    """Structural equality with exact float comparison, for the null check."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


def run(kind, out, cfg0, arms, n_real, n_rve, seed0, n_inputs, small=False):
    seeds = [seed0 + i for i in range(n_real)]
    expected = len(seeds) * n_rve * (1 + len(arms))
    radii = (1.0,) if small else tuple(cfg0["shell_radii_mm"])
    n_dir = 8 if small else int(cfg0["n_dir"])
    kernels = (False,) if small else tuple(cfg0["poiseuille"])
    tfs = tuple(cfg0["tfs"]); rules = tuple(cfg0["causal_rules"])
    # a lesion study must place its volumes so that the lesion sphere cannot
    # reach the acquisition support of the control volume
    les = next((a.lesion for a in arms if a.lesion), None)
    min_sep = S.lesion_separation(les["radius"], cfg0["rve_half"], max(radii) * 1e-3,
                                  cfg0["vox_half_mm"] * 1e-3) if les else 0.0
    # every support that will be attached to a volume must lie inside the gland:
    # the largest compression sphere of any condition, and the lesion sphere
    R_support = max([cfg0["R_comp"]] + [a.cfg["R_comp"] for a in arms] + ([les["radius"]] if les else []))
    # push the configuration once before recording it, so that the manifest
    # holds the record length this run actually used, not the module default
    apply_globals(cfg0)
    effective = dict(radii_mm=list(radii), n_dir=n_dir, poiseuille=list(kernels), tfs=list(tfs),
                     causal_rules=list(rules), dt_s=TR.DT, t_end_s=TR.T_END,
                     n_inputs=n_inputs, n_rve=n_rve, min_centre_separation_mm=min_sep * 1e3,
                     support_radius_mm=R_support * 1e3,
                     arms=[dict(name=a.name, lesion=a.lesion, recalibrate=a.recalibrate,
                                changed={k: v for k, v in a.cfg.items() if cfg0.get(k) != v})
                           for a in arms])
    run_ = S.Run(out, kind, cfg0, seeds, expected, extra=dict(effective=effective))
    gbed = {}; fingerprints = {}; separations = {}; null_checks = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        centres = S.rve_centres(n_rve, rng, cfg0["rve_half"], min_sep=min_sep, R_support=R_support)
        assert all(S.support_inside_gland(c, cfg0["rve_half"], R_support) for c in centres)
        if n_rve > 1:
            separations[str(seed)] = float(min(np.linalg.norm(np.asarray(a) - np.asarray(b)) * 1e3
                                               for i, a in enumerate(centres) for b in centres[i + 1:]))
        rin = np.random.default_rng(seed + 1_000_000)
        inputs = {q: S.input_voxels(rin, c, cfg0["rve_half"], n_inputs, cfg0["vox_half_mm"] * 1e-3,
                                    max(radii) * 1e-3) for q, c in enumerate(centres)}
        base = S.Arm("baseline", cfg0)
        # a lesion is centred on the first sampling volume of this seed
        conditions = [base]
        for arm in arms:
            conditions.append(S.Arm(arm.name, arm.cfg,
                                    lesion=None if arm.lesion is None else dict(arm.lesion, centre=centres[0]),
                                    recalibrate=arm.recalibrate))
        G0 = None; fp0 = None; base_recs = {}
        for arm in conditions:
            apply_globals(arm.cfg)
            t0 = time.time()
            try:
                net = S.build_arm(arm, seed, centres)
                if G0 is None:
                    fl = F.solve(net, R_lat=arm.cfg["R_LAT"]); G0 = float(fl.G_bed0); gbed[str(seed)] = G0
                    bed_constant = "baseline_fit"
                else:
                    fl = S.solve_arm(net, arm, G0)
                    bed_constant = "target_arm_fit" if arm.recalibrate else "held_from_baseline"
                fp = S.fingerprints(net)
                if fp0 is None:
                    fp0 = fp; fingerprints[str(seed)] = dict(topology=fp[0], path=fp[1])
                bound = TR.travel_time_bound(net, fl, poiseuille=any(kernels))
                wrapped = bool(bound > TR.T_END)
            except Exception:
                for q in range(n_rve):
                    run_.write(dict(arm=arm.name, seed=seed, rve=q, condition=arm.name,
                                    error=traceback.format_exc()))
                continue
            print("  [%s seed %d] %d segments, G_bed0 %.3e (%.0f s)"
                  % (arm.name, seed, net.n, fl.G_bed0, time.time() - t0), flush=True)
            for q, c in enumerate(centres):
                t1 = time.time()
                try:
                    role = "lesion" if (arm.lesion and q == 0) else ("reference" if arm.lesion else "sample")
                    clearance = S.support_clearance(net, c, arm.cfg["rve_half"],
                                                    arm.cfg["vox_half_mm"] * 1e-3, max(radii) * 1e-3)
                    if role == "reference" and not clearance > 0:
                        raise RuntimeError("a lesion-modified segment reaches the reference support "
                                           "(clearance %.3f mm)" % (clearance * 1e3))
                    rec = measure(net, fl, arm.cfg, c, inputs[q], poiseuille=kernels,
                                  radii=radii, n_dir=n_dir, tfs=tfs, rules=rules, wrapped=wrapped)
                    rec.update(arm=arm.name, seed=seed, rve=q, condition=arm.name, role=role,
                               g_bed0=float(fl.G_bed0), bed_constant=bed_constant,
                               fingerprint=dict(topology=fp[0], path=fp[1]),
                               same_topology=bool(fp[0] == fp0[0]), same_path=bool(fp[1] == fp0[1]),
                               travel_time_bound_s=float(bound), wrapped=wrapped,
                               clearance_mm=float(clearance * 1e3) if np.isfinite(clearance) else None,
                               n_seg=int(net.n))
                    if arm.name == "baseline":
                        base_recs[q] = rec
                    elif arm.name in NULL_ARMS:
                        same = all(_same(rec.get(k), base_recs[q].get(k)) for k in NULL_KEYS)
                        rec["null_check"] = "pass" if same else "fail"
                        null_checks.setdefault(arm.name, []).append(rec["null_check"])
                        if not same:
                            raise RuntimeError("null arm %s changed geometry or transport" % arm.name)
                except Exception:
                    rec = dict(arm=arm.name, seed=seed, rve=q, condition=arm.name,
                               error=traceback.format_exc())
                run_.write(rec)
                print("     rve %d (%s) done (%.0f s)%s" % (q, arm.name, time.time() - t1,
                                                            "  ERROR" if "error" in rec else ""), flush=True)
    run_.note(g_bed0=gbed, baseline_fingerprint=fingerprints, min_separation_mm=separations,
              null_checks=null_checks)
    ok = run_.finish()
    print(("COMPLETE" if ok else "FAILED") + ": %d records in %s" % (run_.n_written, out))
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["base", "sweep", "lesion"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", help="name of a sensitivity arm (see config.SENS)")
    ap.add_argument("--tort", type=float); ap.add_argument("--alpha", type=float)
    ap.add_argument("--gamma", type=float)
    ap.add_argument("--radius", type=float, default=6e-3)
    ap.add_argument("--n-real", type=int, default=5)
    ap.add_argument("--n-rve", type=int, default=6)
    ap.add_argument("--n-inputs", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--t-end", type=float,
                    help="length of the transform record in s (default %g); an arm whose "
                         "travel-time bound exceeds it wraps and its curves are invalid" % TR.T_END)
    a = ap.parse_args()

    cfg0 = dict(BASE)
    if a.t_end:
        cfg0["T_END_s"] = a.t_end
    if a.small:                                 # smoke configuration, about a minute
        cfg0["d_term_gland"] = 80e-6
    if a.kind == "base":
        arms = []
        n_rve = a.n_rve
    elif a.kind == "sweep":
        if a.arm not in SENS:
            raise SystemExit("unknown arm %r; choose from %s" % (a.arm, ", ".join(sorted(SENS))))
        cfg = dict(cfg0); cfg.update(SENS[a.arm])
        arms = [S.Arm(a.arm, cfg, recalibrate=a.arm in CALIBRATION_ARMS)]
        n_rve = a.n_rve
    else:
        over = {k: v for k, v in (("tort_mean", a.tort), ("alpha", a.alpha), ("gamma_med", a.gamma))
                if v is not None}
        if not over:
            raise SystemExit("give one of --tort, --alpha, --gamma")
        name = "lesion_" + "_".join("%s%g" % (k.split("_")[0], v) for k, v in over.items())
        arms = [S.Arm(name, dict(cfg0), lesion=dict(radius=a.radius, **over))]   # centre set per seed
        n_rve = 2                      # one lesion volume and one control
    run(a.kind, a.out, cfg0, arms, a.n_real, n_rve, a.seed0, a.n_inputs, small=a.small)


if __name__ == "__main__":
    main()
