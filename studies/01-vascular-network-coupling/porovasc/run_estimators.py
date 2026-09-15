"""Two estimators, one tissue, one acquisition.

    python -m porovasc.run_estimators --out runs/estimators --n-real 3 --n-rve 2

The question is narrow and the design follows from it: when two published ways
of estimating convective dispersion are applied to the same tissue through the
same acquisition, do they report the same thing?  Everything that could differ
other than the estimator is therefore held fixed.  Both are given curves
sampled from one propagated grid, at one voxel pitch, one frame rate and one
point spread function, at the same positions.  What is swept is, separately,
the acquisition (frame rate, point spread function) and the analysis settings
of each estimator (shell radius for one; derivative scale for the other).

    shell estimator, physics.transport.identify_shell
        one input voxel and a spherical shell of output voxels at radius R;
        a scalar (v, D) through a one-dimensional transfer function along the
        chord; no derivatives, no regularisation.

    grid estimator, physics.cdi3d.fit_volume
        a solid spherical kernel; the equation fitted directly for a tensor
        and a velocity; Gaussian derivatives at scale sigma_x; ridge.

Three properties of the setup are deliberate and are not defects.

The gland is uniformly resolved (no refinement inside the sampling volumes).
The grid estimator's derivatives reach about four sigma_x beyond the kernel,
so with a refined sampling volume every fit would straddle a step in vessel
density at the boundary of the refinement, and would be measuring that step.

The sampling volumes are placed so that the whole propagated block lies inside
the gland, which needs a containment radius of `half + margin`, some 11 mm,
rather than the 2 mm of a shell acquisition.  A consequence worth reporting is
that sigma_x = 3 mm, the largest the source used, needs 17 mm and does not fit
inside a prostate-sized gland at all.

The record is the long one (163.84 s), because the grid estimator's temporal
smoothing costs about four sigma_t at each end of the recording, and at a
clinical frame rate a shorter record leaves too few usable frames.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import traceback

import numpy as np

from .geometry import network as N
from .physics import flow as F, transport as TR, cdi3d
from .homogenise import darcy as DA
from .config import BASE, apply_globals
from . import grid as G
from . import study as S

# Acquisition settings, swept.  The frame rate spans what separates the two
# published studies: 2D contrast imaging runs at video rate, 3D at around or
# below 1 Hz.  0.25 s is the fastest the propagated block can afford in memory;
# 1 s is what a current 3D acquisition achieves; 4 s is the rate of the 2019
# acquisition the grid estimator was published on.
FRAME_DT = (0.25, 0.5, 1.0, 2.0, 4.0)      # s, for the shell estimator
# The grid estimator gains nothing from frames much finer than its own temporal
# smoothing (at fixed sigma_t the frame rate changes the fit by under a
# percent), and the derivative fields of a finely sampled block cost hundreds
# of megabytes, so it is evaluated from 1 s down.
FRAME_DT_GRID = (1.0, 2.0, 4.0)            # s
# Point spread function, as (x, y, z) full widths at half maximum.  None, the
# isotropic 1.1 mm the grid estimator was published with (its system's lateral
# resolution at 25 mm), and a realistic anisotropic one for a mechanically
# swept endocavity probe at 4 MHz: axial a few tenths of a millimetre, lateral
# of order 1 mm, elevational two to three times the lateral.  The anisotropic
# case is the one that can put an instrumental anisotropy into a fitted tensor.
PSF_FWHM = ((0.0, 0.0, 0.0), (1.1e-3, 1.1e-3, 1.1e-3), (0.8e-3, 1.1e-3, 2.5e-3))

# analysis settings of each estimator, swept
SIGMA_X = (0.75e-3, 1.5e-3)                # m; 3 mm does not fit inside the gland
SIGMA_T = (2.0, 4.0, 8.0)                  # s; swept independently of the frame rate, so that
                                           # temporal smoothing is not confounded with sampling
KERNEL_S = 7                               # voxels
L0 = 0.1
SHELL_R_MM = (0.75, 1.0, 1.5, 2.0)
N_DIR = 48
TF, RULE = "new", "front20"

DX = 0.75e-3                               # voxel pitch of the acquisition
HALF = 3.0e-3                              # half-width of the sampling volume
EVAL_STEP = 2                              # evaluation lattice, in voxels
T_END = 163.84                             # s


def eval_indices(n, m, step=EVAL_STEP):
    """Grid indices at which both estimators are evaluated: a lattice inside
    the sampling volume, leaving the margin free for the derivative support."""
    lo, hi = m, n - m - 1
    ax = list(range(lo, hi + 1, step))
    return [(i, j, k) for i in ax for j in ax for k in ax]


def ground_truth(net, fl, centre, radii=(1.0e-3, 2.625e-3)):
    """What the flow actually is in the support of each estimator.

    There is a true velocity and there is no true dispersion coefficient.  The
    velocity is well defined: microbubbles follow the blood, so the speed a
    tracer samples is the flow-weighted mean speed of the vessels in the
    support, which the flow solution gives exactly.  A dispersion coefficient
    is not well defined, because the tissue does not obey a convection-
    dispersion equation in the first place: the contrast field is the
    superposition of the curves of thousands of discrete vessels, and no
    (v, D) generates it.  So v can be compared with a truth and D cannot, and
    that asymmetry is the point rather than a gap in the bookkeeping.

    The speeds are reported on two supports, because the two estimators do not
    share one: 1 mm is a typical shell radius, 2.625 mm is the radius of the
    seven-voxel kernel of the grid estimator.

    Weighting is by flux, not by volume: a bubble samples a streamline in
    proportion to the flow it carries, so the flow-weighted mean is what a
    tracer reports, and the volume-weighted mean is not.

    The pressure gradient is fitted, by flow-weighted least squares, to the
    nodal pressures of the segments in the support.  It is the quantity that
    has been missing from every comparison so far: the constitutive relation
    for an intrinsic velocity is

        v = k |grad p| / (eta phi),

    so without grad p an estimate of v cannot be referred to k at all, and a
    correlation between the two says as much about where in the tree the
    volume sits as about its permeability.  With it, `v_darcy` below is what
    that relation predicts, from quantities all measured on the network, and
    comparing it with `v_flow` tests the relation itself before any estimator
    is blamed for not reporting it.  The permeability used is the bundle form
    on the same support, phi d_perm^2 / (32 <T^2>), so that every term refers
    to the same piece of tissue."""
    speed = np.abs(fl.Q) / (np.pi * net.r ** 2)
    centre = np.asarray(centre, float)
    out = {}
    for R in radii:
        w = N.sphere_fraction(net.p0, net.p1, centre, R)
        m = w > 0
        q = np.abs(fl.Q) * w
        vol = np.pi * net.r ** 2 * net.Lpath * w
        tot = q.sum()
        key = "R%.2fmm" % (R * 1e3)
        st = DA.support_stats(net, w, 4 / 3 * np.pi * R ** 3)
        rec = dict(phi=st["phi"], d_perm=st["d_perm"], T2=st["T2"], n_seg=int(m.sum()),
                   v_flow_mm_s=None, v_vol_mm_s=None, v_flow_cv=None,
                   grad_p_Pa_per_mm=None, v_darcy_mm_s=None)
        if tot > 0:
            vbar = float((speed * q).sum() / tot)
            rec["v_flow_mm_s"] = vbar * 1e3
            rec["v_vol_mm_s"] = float((speed * vol).sum() / vol.sum() * 1e3)
            rec["v_flow_cv"] = float(np.sqrt(((speed - vbar) ** 2 * q).sum() / tot) / vbar)
            # The pressure field of a perfused bed is not smooth in space: an
            # artery at 70 mmHg and its paired vein at 8 mmHg can sit a tenth of
            # a millimetre apart, so a gradient fitted over both is dominated by
            # which of the two a node belongs to, and comes out as the whole
            # arteriovenous drop over the width of the support.  The separation
            # is therefore reported directly, and the gradient is fitted on the
            # arterial side alone, where a pressure field along the tree does
            # exist.  This is why a Darcy velocity is not the right reference
            # for what a contrast estimator sees at this scale.
            pn = fl.p[:net.n]
            for side, sel in (("art", m & net.art), ("vein", m & net.vein)):
                ps = pn[sel] / F.MMHG
                if ps.size:
                    rec["p_%s_mmHg" % side] = [float(ps.min()), float(ps.max())]
                    rec["n_%s" % side] = int(sel.sum())
            sel = m & net.art
            if sel.sum() >= 4:
                X = net.p1[sel] - centre
                A = np.c_[X, np.ones(len(X))]
                wt = np.abs(fl.Q)[sel]
                if wt.sum() > 0:
                    W = wt / wt.sum()
                    try:
                        g = np.linalg.solve((A * W[:, None]).T @ A, (A * W[:, None]).T @ pn[sel])[:3]
                        rec["grad_p_art_Pa_per_mm"] = float(np.linalg.norm(g)) * 1e-3
                    except np.linalg.LinAlgError:
                        pass
            # The quantities a contrast curve does encode.  For a perfused bed
            # the central volume theorem relates them: the mean transit time is
            # the blood volume divided by the flow through it.  These are
            # defined whatever the spatial arrangement of the vessels, which a
            # Darcy velocity is not.
            Vol = 4 / 3 * np.pi * R ** 3
            # The lumped bed hangs off a terminal node, so its flow is a point
            # quantity: it counts when that node lies inside the support, not in
            # proportion to how much of the parent segment does.
            inside = np.linalg.norm(net.p1[fl.bed_a] - centre, axis=1) <= R
            fin = float(np.abs(fl.Q_bed)[inside].sum())
            rec["perfusion_ml_min_100g"] = (fin * 6e7 / (Vol * 1e6 * 1.05) * 100) if Vol > 0 else None
            rec["V_blood_mm3"] = float(vol.sum() * 1e9)
            rec["mtt_true_s"] = float(vol.sum() / fin) if fin > 0 else None
            rec["n_bed"] = int(inside.sum())
        out[key] = rec
    return out


def shell_estimate(g, centre, R, tf=TF, rule=RULE, n_dir=N_DIR):
    """The shell estimator, given curves interpolated from the same grid the
    other estimator sees."""
    pos = centre + R * TR.fibonacci_sphere(n_dir)
    tin = G.sample_at(g, [centre])[0]
    if tin.sum() <= 0:
        return dict(status="no_signal")
    outs = list(G.sample_at(g, pos))
    v, D, n, r2, info = TR.identify_shell(g.t, tin, outs, R, tf=tf, rule=rule, detail=True)
    ok = np.isfinite(v)
    at_bound = bool(ok and (v <= 1.01e-4 or v >= 0.099 or D <= 1.01e-9 or D >= 0.99e-4))
    return dict(status="ok" if ok else ("few_pairs" if n < 3 else "no_fit"),
                v=float(v) if ok else None, D=float(D) if ok else None,
                n_pairs=int(n), r2=float(r2) if np.isfinite(r2) else None,
                at_bound=at_bound, n_accepted=len(info["accepted"]))


def measure_volume(net, fl, centre, poiseuille=False, progress=False):
    """Every estimate at one sampling volume, over the acquisition settings and
    the analysis settings of both estimators."""
    sigma_max = max(SIGMA_X)
    n, origin, m = G.grid_for(centre, HALF, DX, sigma_max, KERNEL_S)
    base = G.propagate_grid(net, fl, centre, HALF, dx=DX, sigma_x=sigma_max, s=KERNEL_S,
                            frame_dt=min(FRAME_DT), poiseuille=poiseuille, chunk=3000,
                            progress=progress)
    idxs = eval_indices(n, m)
    rows = {ix: dict(index=list(ix), centre_mm=list(base.position_of(ix) * 1e3),
                     truth=ground_truth(net, fl, base.position_of(ix)), est=[])
            for ix in idxs}
    for psf in PSF_FWHM:
        gp = G.apply_psf(base, psf)
        for fdt in FRAME_DT:
            gf = G.rebin(gp, fdt)
            nt = len(gf.t)
            # grid estimator: the derivative fields are built once per setting, and
            # sigma_t is swept separately from the frame rate so that temporal
            # smoothing and sampling are not confounded
            for sigma_t in (SIGMA_T if fdt in FRAME_DT_GRID else ()):
                if sigma_t < 2 * fdt:
                    continue                        # unresolved: fewer than two samples per sigma
                drop = int(np.ceil(4 * sigma_t / fdt))
                if nt - 2 * drop < 4:
                    continue                        # too few frames left after the record ends
                sl = slice(drop, nt - drop)
                acq = dict(psf_fwhm_mm=[x * 1e3 for x in psf], frame_dt_s=fdt,
                           sigma_t_s=sigma_t, n_frames=nt, n_frames_used=nt - 2 * drop)
                for sx in SIGMA_X:
                    fits = cdi3d.fit_volume(gf.C, DX, fdt, sx, sigma_t, s=KERNEL_S, l0=L0,
                                            centres=idxs, mode=("reflect", "nearest"), t_slice=sl)
                    for ix in idxs:
                        f = fits.get(ix)
                        rows[ix]["est"].append(dict(
                            estimator="grid", **acq, sigma_x_mm=sx * 1e3, s=KERNEL_S, l0=L0,
                            D=None if f is None else float(f.D_CD),
                            v=None if f is None else float(f.v_CD),
                            # the eigenvalues of the fitted tensor, smallest first:
                            # the trace alone cannot show whether an anisotropy in
                            # the estimate came from the tissue or from the beam
                            D_eig=None if f is None else [float(x) for x in np.sort(np.linalg.eigvalsh(f.D))],
                            residual=None if f is None else f.residual,
                            # a fit of an unconstrained tensor can return a negative
                            # trace; that is not a small dispersion, it is a fit that
                            # found no dispersive process, and it is counted separately
                            negative_D=None if f is None else bool(f.D_CD <= 0),
                            status="no_signal" if f is None else "ok"))
            # shell estimator: the same grid, the same points, on the same interior
            drop = max(1, int(round(2.0 / fdt)))
            sl = slice(drop, nt - drop)
            acq = dict(psf_fwhm_mm=[x * 1e3 for x in psf], frame_dt_s=fdt,
                       n_frames=nt, n_frames_used=nt - 2 * drop)
            gs = G.Grid(C=gf.C[:, :, :, sl], t=gf.t[sl], origin=gf.origin, dx=gf.dx,
                        frame_dt=gf.frame_dt, vox_half=gf.vox_half)
            for ix in idxs:
                c = gs.position_of(ix)
                for R_mm in SHELL_R_MM:
                    rows[ix]["est"].append(dict(estimator="shell", **acq, R_mm=R_mm,
                                                tf=TF, rule=RULE,
                                                **shell_estimate(gs, c, R_mm * 1e-3)))
    return [rows[ix] for ix in idxs], dict(block=n, margin=m, n_eval=len(idxs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-real", type=int, default=3)
    ap.add_argument("--n-rve", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--poiseuille", action="store_true")
    ap.add_argument("--small", action="store_true",
                    help="a coarse gland, for a smoke test only")
    ap.add_argument("--d-term-um", type=float,
                    help="terminal diameter of the explicit tree in micrometres "
                         "(default %g); the tree is uniformly resolved, so this sets "
                         "how finely the contrast field is structured"
                         % (BASE["d_term_gland"] * 1e6))
    a = ap.parse_args()

    cfg = dict(BASE)
    cfg["T_END_s"] = T_END
    if a.small:
        cfg["d_term_gland"] = 80e-6
    if a.d_term_um:
        cfg["d_term_gland"] = a.d_term_um * 1e-6
    cfg["d_term_rve"] = cfg["d_term_gland"]          # uniform resolution, no refinement
    apply_globals(cfg)

    sigma_max = max(SIGMA_X)
    R_support = HALF + G.margin_voxels(sigma_max, DX, KERNEL_S) * DX
    seeds = [a.seed0 + i for i in range(a.n_real)]
    effective = dict(frame_dt_s=list(FRAME_DT), frame_dt_grid_s=list(FRAME_DT_GRID),
                     psf_fwhm_mm=[[x * 1e3 for x in p] for p in PSF_FWHM],
                     sigma_x_mm=[s * 1e3 for s in SIGMA_X], sigma_t_s=list(SIGMA_T),
                     kernel_s=KERNEL_S, l0=L0,
                     shell_radii_mm=list(SHELL_R_MM), n_dir=N_DIR, tf=TF, rule=RULE,
                     dx_mm=DX * 1e3, half_mm=HALF * 1e3, eval_step_voxels=EVAL_STEP,
                     t_end_s=TR.T_END, dt_s=TR.DT, poiseuille=bool(a.poiseuille),
                     support_radius_mm=R_support * 1e3, uniform_d_term_um=cfg["d_term_gland"] * 1e6)
    run_ = S.Run(a.out, "estimators", cfg, seeds, len(seeds) * a.n_rve,
                 extra=dict(effective=effective))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        centres = S.rve_centres(a.n_rve, rng, HALF, R_support=R_support)
        t0 = time.time()
        apply_globals(cfg)
        net = N.build(N.Params(seed=seed, d_term_gland=cfg["d_term_gland"],
                               d_term_rve=cfg["d_term_rve"]))
        fl = F.solve(net, R_lat=cfg["R_LAT"])
        print("  [seed %d] %d segments (%.0f s)" % (seed, net.n, time.time() - t0), flush=True)
        for q, c in enumerate(centres):
            t1 = time.time()
            try:
                rows, info = measure_volume(net, fl, np.asarray(c), poiseuille=a.poiseuille)
                run_.write(dict(seed=seed, rve=q, centre_mm=list(np.asarray(c) * 1e3),
                                n_seg=int(net.n), **info, points=rows))
            except Exception:
                run_.write(dict(seed=seed, rve=q, error=traceback.format_exc()))
            print("     rve %d done (%.0f s)" % (q, time.time() - t1), flush=True)
    ok = run_.finish()
    print(("COMPLETE" if ok else "FAILED") + ": %d records in %s" % (run_.n_written, a.out))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
