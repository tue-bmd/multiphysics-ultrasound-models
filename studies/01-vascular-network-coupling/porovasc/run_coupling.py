"""Do drainage and intravascular transport respond to the same changes in
vascular organization?

    python -m porovasc.run_coupling --out runs/coupling --n-real 5 --n-rve 2 --t-end auto
    python -m porovasc.run_coupling --out runs/coupling_small --n-real 2 --n-rve 1 --small

One paired study, one set of controlled perturbations, three families of
observable measured on the same sampling volume:

  drainage    the response of the compressed region: the equilibrium volume
              expelled per unit applied strain, the e-folding time of the
              residual, and the width of the relaxation spectrum;

  transport   what a contrast bolus does in the same volume: curve descriptors
              that no estimator is involved in (time to peak, mean transit time,
              the width of the curve, the area, the share carried by the lumped
              bed), and one convection-dispersion estimate;

  network     descriptors computed directly on the network and on nothing else:
              vascular volume fraction, directional face-to-face permeability,
              permeability weighted diameter, mean squared tortuosity, and the
              flow coherence chi.

The arms are chosen so that they do not all act at the same place.  Four act on
the vascular geometry.  Two act only on the matrix modulus, which enters the
compliance law and nothing else, so the transport family should not move.  Two
act only on the driving pressure, which changes the flow without touching the
geometry, so the geometric descriptors should not move.  The experiment
therefore reports which dependencies appear and which do not; it is built so
that a claim that phi and k alone explain both responses can fail.

The physics is unchanged from the rest of the package.  This runner records
quantities the existing solvers already compute and that the paired runner does
not store, and it holds the bed conductance calibrated on the baseline exactly
as `run_paired` does, so that an arm is compared with its own baseline gland.
"""
from __future__ import annotations
import argparse
import time
import traceback

import numpy as np

from . import study as S
from .config import BASE, SENS, COUPLING_ARMS, COUPLING_LOCUS, apply_globals
from .geometry import network as N
from .homogenise import darcy as DA
from .physics import drainage as DR
from .physics import flow as F
from .physics import transport as TR

#: the one estimator setting this runner uses.  The dependence of an estimate on
#: the analysis setting is a separate result, measured by `cross_settings`; here
#: the setting is fixed so that a difference between arms is a difference in the
#: tissue.
EST = dict(kernel="plug", R_mm=1.0, tf="new", rule="front20")


def curve_descriptors(t, y):
    """Estimator-free descriptors of one contrast curve.

    These exist so that the transport family does not depend on any inverse
    formulation: a change in them is a change in what arrived, not a change in
    what a fit made of it."""
    y = np.asarray(y, float)
    dt = float(t[1] - t[0])
    area = float(y.sum() * dt)
    if area <= 0 or not np.isfinite(area):
        return dict(area=None, t_peak=None, mtt=None, width=None)
    u = y / area
    mtt = float((t * u).sum() * dt)
    var = float(((t - mtt) ** 2 * u).sum() * dt)
    return dict(area=area, t_peak=float(t[int(np.argmax(y))]), mtt=mtt,
                width=float(np.sqrt(var)) if var > 0 else None)


def spectrum_width(tau_grid, spectrum):
    """Weighted standard deviation of log10(tau) under the relaxation spectrum.

    A single exponential would give zero.  The width says how far the drainage
    is from one time constant, which is a property of the network and not of the
    modulus."""
    w = np.asarray(spectrum, float)
    tg = np.asarray(tau_grid, float)
    m = np.isfinite(w) & (w > 0) & (tg > 0)
    if m.sum() == 0:
        return None
    if m.sum() == 1:
        return 0.0          # all the weight on one time constant
    w = w[m] / w[m].sum()
    lg = np.log10(tg[m])
    mu = float((w * lg).sum())
    return float(np.sqrt((w * (lg - mu) ** 2).sum()))


def volume_record(net, fl, cfg, centre, inputs, small=False, wrapped=False):
    """The three families on one sampling volume.

    When the arm's travel-time bound exceeds the transform record the contrast
    curves wrap, and every transport number derived from them is meaningless.
    In that case the transport block is not computed at all: recording a number
    that is known to be invalid, even beside a status flag, invites it to be
    read."""
    R_comp = cfg["R_comp"]
    h = cfg["rve_half"]      # the cube support run_paired uses, so that the
                             # descriptors of the two studies are comparable
    rec = dict(centre_mm=[float(x) * 1e3 for x in centre])

    # ---- network descriptors, computed on the network alone -----------------
    da = DA.measure(net, fl, centre, h)
    cube = DA.support_stats(net, N.cube_fraction(net.p0, net.p1, np.asarray(centre), h),
                            (2 * h) ** 3)
    coh = DA.coherence(net, fl, centre, R_comp)
    rec["network"] = dict(phi_cube=float(cube["phi"]), d_perm=float(cube["d_perm"]),
                          T2=float(cube["T2"]), n_seg_cube=int(cube.get("n_seg", 0) or 0),
                          k_mean=float(da.k_mean),
                          k_x=float(da.k_face[0]), k_y=float(da.k_face[1]),
                          k_z=float(da.k_face[2]), n_blocked=int(da.n_blocked),
                          cube_half_mm=h * 1e3, **{"chi_%s" % k: v for k, v in coh.items()})

    # ---- drainage ------------------------------------------------------------
    dr = DR.solve(net, fl, centre, R_comp, dP_ext=100.0, H=cfg["H"])
    strain = 100.0 / cfg["H"]
    rec["drainage"] = dict(
        tau_rc=float(dr.tau_rc) if np.isfinite(dr.tau_rc) else None,
        phi_region=float(dr.phi),
        dV_inf=float(dr.dV_inf), V_blood=float(dr.V_blood), V_tissue=float(dr.V_tissue),
        # the equilibrium amplitude, made dimensionless by the applied strain.
        # Under the compliance law C_i = V_i / H this equals the vascular volume
        # fraction of the compressed region; recording both lets the identity be
        # checked rather than assumed.
        relax_per_strain=float(-dr.dV_inf / dr.V_tissue / strain)
        if (dr.V_tissue > 0 and strain > 0) else None,
        spectrum_width=spectrum_width(dr.tau_grid, dr.spectrum),
        dP_ext=100.0, H=float(cfg["H"]), R_comp_mm=R_comp * 1e3)

    # ---- transport -----------------------------------------------------------
    if wrapped:
        rec["transport"] = dict(setting=dict(EST), status="wrapped",
                                curves=[], est=[])
        return rec
    R = EST["R_mm"] * 1e-3
    vox = cfg["vox_half_mm"] * 1e-3
    n_dir = 8 if small else int(cfg["n_dir"])
    allk, index = [], []
    for c in inputs:
        ks = TR.shell_kernels(c, R, vox, n_dir=n_dir)
        index.append((len(allk), len(ks)))
        allk += ks
    tr = TR.propagate(net, fl, allk, poiseuille=(EST["kernel"] == "pois"))
    curves, ests = [], []
    for (i0, nk) in index:
        tin = tr.tic[i0]
        if tin.sum() <= 0:
            curves.append(dict(area=None, t_peak=None, mtt=None, width=None,
                               bed_share=None))
            ests.append(dict(status="no_signal", v=None, D=None))
            continue
        d = curve_descriptors(tr.t, tin)
        d["bed_share"] = float(tr.tic_bed[i0].sum() / max(tin.sum(), 1e-30))
        curves.append(d)
        outs = [tr.tic[k] for k in range(i0 + 1, i0 + nk)]
        v, D, n_pairs, r2 = TR.identify_shell(tr.t, tin, outs, R,
                                              tf=EST["tf"], rule=EST["rule"])
        ok = bool(np.isfinite(v) and np.isfinite(D))
        at_bound = bool(ok and (v <= 1.01e-4 or v >= 0.099 or D <= 1.01e-9 or D >= 0.99e-4))
        ests.append(dict(status="ok" if ok else "no_fit",
                         v=float(v) if ok else None, D=float(D) if ok else None,
                         n_pairs=int(n_pairs), r2=float(r2) if np.isfinite(r2) else None,
                         at_bound=at_bound))
    rec["transport"] = dict(setting=dict(EST), n_dir=n_dir, status="ok",
                            curves=curves, est=ests)
    return rec


def run(out, arm_names, n_real, n_rve, seed0, n_inputs, small=False, t_end=None):
    cfg0 = dict(BASE)
    if t_end:
        cfg0["T_END_s"] = t_end
    if small:                             # smoke configuration, about a minute
        cfg0["d_term_gland"] = 80e-6
    arms = [S.Arm(name, dict(cfg0, **SENS[name])) for name in arm_names]
    seeds = [seed0 + i for i in range(n_real)]
    expected = len(seeds) * n_rve * (1 + len(arms))
    apply_globals(cfg0)
    effective = dict(n_rve=n_rve, n_inputs=n_inputs, small=bool(small),
                     dt_s=TR.DT, t_end_s=TR.T_END, estimator=dict(EST),
                     support_radius_mm=cfg0["R_comp"] * 1e3,
                     arms=[dict(name=a.name, locus=COUPLING_LOCUS.get(a.name),
                                purpose=COUPLING_ARMS.get(a.name),
                                changed={k: v for k, v in a.cfg.items() if cfg0.get(k) != v})
                           for a in arms])
    run_ = S.Run(out, "coupling", cfg0, seeds, expected, extra=dict(effective=effective))
    gbed, fps, bounds = {}, {}, {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        centres = S.rve_centres(n_rve, rng, cfg0["rve_half"], R_support=cfg0["R_comp"])
        rin = np.random.default_rng(seed + 1_000_000)
        inputs = {q: S.input_voxels(rin, c, cfg0["rve_half"], n_inputs,
                                    cfg0["vox_half_mm"] * 1e-3, EST["R_mm"] * 1e-3)
                  for q, c in enumerate(centres)}
        G0 = None
        for arm in [S.Arm("baseline", cfg0)] + arms:
            apply_globals(arm.cfg)
            t0 = time.time()
            try:
                net = S.build_arm(arm, seed, centres)
                if G0 is None:
                    fl = F.solve(net, R_lat=arm.cfg["R_LAT"])
                    G0 = float(fl.G_bed0)
                    gbed[str(seed)] = G0
                    fps[str(seed)] = dict(zip(("topology", "path"), S.fingerprints(net)))
                    bed = "baseline_fit"
                else:
                    fl = S.solve_arm(net, arm, G0)
                    bed = "held_from_baseline"
                # an arm that lengthens the paths can take longer to traverse
                # than the transform record, in which case its curves wrap and
                # every transport number for it is invalid
                bound = float(TR.travel_time_bound(
                    net, fl, poiseuille=(EST["kernel"] == "pois")))
            except Exception:
                for q in range(n_rve):
                    run_.write(dict(arm=arm.name, seed=seed, rve=q, condition=arm.name,
                                    error=traceback.format_exc()))
                continue

            # The wrap check is deliberately *outside* the try above.  Inside it,
            # any fault in the stop path itself - a manifest that will not
            # serialize, say - would be caught as an ordinary build failure, the
            # loop would continue, and the study would finish carrying invalid
            # transport.  Keys are strings for the same reason: json cannot write
            # a tuple key, and that failure has no business reaching the caller
            # as a build error.
            bounds["%s|%d" % (arm.name, seed)] = bound
            if bound > TR.T_END:
                need = TR.T_END
                while need < bound:
                    need *= 2
                run_.note(travel_time_bounds=bounds)
                run_.finish()
                raise SystemExit(
                    "arm %r on seed %d has a travel-time bound of %.1f s, longer "
                    "than the %.1f s transform record: its curves would wrap.\n"
                    "Rerun the whole study with --t-end %.2f, so that every arm is "
                    "compared on one record length.\n"
                    "`--probe-t-end` reports the length needed without running the "
                    "study." % (arm.name, seed, bound, TR.T_END, need))

            print("  [%s seed %d] %d segments, G_bed0 %.3e (%.0f s)"
                  % (arm.name, seed, net.n, fl.G_bed0, time.time() - t0), flush=True)
            for q, c in enumerate(centres):
                t1 = time.time()
                try:
                    rec = volume_record(net, fl, arm.cfg, c, inputs[q], small=small)
                    rec.update(arm=arm.name, seed=seed, rve=q, condition=arm.name,
                               locus=COUPLING_LOCUS.get(arm.name, "baseline"),
                               n_seg=int(net.n), bed_constant=bed,
                               travel_time_bound_s=bound, status="ok")
                    run_.write(rec)
                except Exception:
                    run_.write(dict(arm=arm.name, seed=seed, rve=q, condition=arm.name,
                                    error=traceback.format_exc()))
                print("     rve %d done (%.0f s)" % (q, time.time() - t1), flush=True)
    run_.note(g_bed0=gbed, baseline_fingerprint=fps, travel_time_bounds=bounds)
    ok = run_.finish()
    print(("COMPLETE" if ok else "FAILED") + ": %d records in %s" % (run_.n_written, out))
    return ok


def probe_t_end(arm_names, n_real, n_rve, seed0, small=False):
    """Longest travel time over the study, without running it.

    One record length has to serve every arm, because comparing an arm analysed
    on a longer record with one analysed on a shorter record would confound the
    intervention with the acquisition.

    The probe reproduces the study's configuration exactly: the same seeds, the
    same number of sampling volumes, and the same centre draw.  That matters
    because the sampling volumes are refined regions of the tree, so the number
    and placement of them changes the network, and a bound measured on a
    one-volume gland does not bound a two-volume one.  It costs one build and
    one flow solve per arm per seed, and nothing else."""
    cfg0 = dict(BASE)
    if small:
        cfg0["d_term_gland"] = 80e-6
    seeds = [seed0 + i for i in range(n_real)]
    arms = [S.Arm(n, dict(cfg0, **SENS[n])) for n in arm_names]
    n_build = len(seeds) * (1 + len(arms))
    print("probing %d gland build(s): %d seed(s) x %d condition(s), %d volume(s) each"
          % (n_build, len(seeds), 1 + len(arms), n_rve), flush=True)
    worst = 0.0
    per_arm = {}
    for seed in seeds:
        apply_globals(cfg0)
        centres = S.rve_centres(n_rve, np.random.default_rng(seed),
                                cfg0["rve_half"], R_support=cfg0["R_comp"])
        G0 = None
        for arm in [S.Arm("baseline", cfg0)] + arms:
            apply_globals(arm.cfg)
            net = S.build_arm(arm, seed, centres)
            if G0 is None:
                fl = F.solve(net, R_lat=arm.cfg["R_LAT"])
                G0 = float(fl.G_bed0)
            else:
                fl = S.solve_arm(net, arm, G0)
            b = float(TR.travel_time_bound(net, fl,
                                           poiseuille=(EST["kernel"] == "pois")))
            per_arm[arm.name] = max(per_arm.get(arm.name, 0.0), b)
            worst = max(worst, b)
            print("  seed %d  %-16s %8.1f s   (%d segments)"
                  % (seed, arm.name, b, net.n), flush=True)
    print("\nlongest per arm, over %d seed(s):" % len(seeds))
    for name, b in sorted(per_arm.items(), key=lambda kv: -kv[1]):
        print("  %-16s %8.1f s" % (name, b))
    need = float(BASE["T_END_s"])
    while need < worst:
        need *= 2
    print("\nlongest overall %.1f s; run the study with --t-end %.2f" % (worst, need))
    return need


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="output directory; not needed with --probe-t-end")
    ap.add_argument("--arms", default=",".join(COUPLING_ARMS),
                    help="comma separated arm names from porovasc.config.SENS")
    ap.add_argument("--n-real", type=int, default=5, help="glands (seeds)")
    ap.add_argument("--n-rve", type=int, default=2, help="sampling volumes per gland")
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--n-inputs", type=int, default=12)
    ap.add_argument("--t-end", default=None,
                    help="length of the transform record in seconds, or 'auto' to "
                         "probe the study's own configuration first and use the "
                         "length it needs")
    ap.add_argument("--probe-t-end", action="store_true",
                    help="report the record length every arm needs, over the same "
                         "seeds and sampling volumes the study will use, and stop "
                         "without running it")
    ap.add_argument("--small", action="store_true",
                    help="smoke configuration: a coarser tree and 8 shell directions "
                         "instead of the full set.  Not for reported results.")
    a = ap.parse_args(argv)
    names = [x for x in a.arms.split(",") if x]
    bad = [n for n in names if n not in SENS]
    if bad:
        raise SystemExit("unknown arm(s): %s" % bad)
    if a.probe_t_end:
        probe_t_end(names, a.n_real, a.n_rve, a.seed0, small=a.small)
        raise SystemExit(0)
    if not a.out:
        raise SystemExit("--out is required unless --probe-t-end is given")
    if a.t_end is None:
        t_end = None
    elif str(a.t_end).lower() == "auto":
        # Probe first, then run with what the probe asked for.  This is the same
        # computation `--probe-t-end` does, over the same seeds and volumes, and
        # it removes the one step where a number has to be carried by hand from
        # one command to the next.
        print("=" * 70)
        print("PREFLIGHT: how long a transform record does this study need?")
        print("=" * 70, flush=True)
        t_end = probe_t_end(names, a.n_real, a.n_rve, a.seed0, small=a.small)
        print("\n" + "=" * 70)
        print("STUDY: running with --t-end %.2f" % t_end)
        print("=" * 70, flush=True)
    else:
        t_end = float(a.t_end)
    ok = run(a.out, names, a.n_real, a.n_rve, a.seed0, a.n_inputs,
             small=a.small, t_end=t_end)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
