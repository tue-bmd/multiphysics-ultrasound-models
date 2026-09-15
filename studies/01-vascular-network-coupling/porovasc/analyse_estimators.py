"""Summary of a two-estimator run.

    python -m porovasc.analyse_estimators runs/estimators
    python -m porovasc.analyse_estimators runs/estimators --psf 0.0

The reduction is the same three steps used everywhere in this package, because
the experimental unit is the gland and not the voxel: the median over the
evaluation points of a sampling volume, then over the volumes of a gland, then
the list over glands with its median and range.

What is reported, per estimator and per setting:

  yield        how many evaluation points returned a usable value at all, and
               why the rest did not.  This is not bookkeeping: the shell
               estimator's yield collapses as the frame interval grows, and a
               median taken over the survivors is a median over a selected
               minority, not over the tissue.

  D, v         the estimates, and the dispersivity D / v.  The dispersivity is
               the more interpretable of the three: if dispersion in a vascular
               bed is D = alpha_L v with alpha_L a geometric length, then D
               carries no information that v does not already carry, and
               alpha_L is the quantity that reflects the structure.

  matched      the two estimators at one common frame interval, on the
               evaluation points at which both returned a usable value, over
               every grid setting rather than a selected one.  Comparing each
               estimator on its own surviving subset, or across different frame
               intervals, confounds the estimator with the acquisition and with
               which points survived; both confounds are reported so that their
               size is visible.

  quality      for the grid estimator the residual of its own least squares
               (1.0 means the convection-dispersion model explains nothing of
               the data) and the fraction of points at which the fitted tensor
               has a negative trace, which is not a small dispersion but a fit
               that found no dispersive process.  For the shell estimator the
               number of shell directions the causality rule accepted, and the
               fraction of fits that ended at an optimiser bound.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os

import numpy as np

from .analyse_paired import hierarchy, _med
from .study import SCHEMA
from . import validate as VAL


def load(folder, partial=False):
    """Read a run, after validating the archive (see `porovasc.validate`)."""
    VAL.require(folder, partial=partial)
    man = json.load(open(os.path.join(folder, "manifest.json")))
    path, err = VAL._records_path(folder)
    if err:
        raise SystemExit(err)
    op = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    recs = [json.loads(l) for l in op if l.strip()]
    errors = [r for r in recs if "error" in r]
    return man, [r for r in recs if "error" not in r], errors


def _rows(recs, keep, value):
    """Hierarchy rows: the median of `value` over the evaluation points of each
    sampling volume, for the estimates selected by `keep`."""
    out = []
    for r in recs:
        vals = []
        for p in r["points"]:
            for e in p["est"]:
                if keep(e):
                    x = value(e)
                    if x is not None and np.isfinite(x):
                        vals.append(x)
        out.append(dict(seed=r["seed"], rve=r["rve"], val=_med(vals)))
    return out


def _counts(recs, keep):
    """Totals over every evaluation point, for the yield line."""
    n = ok = neg = bound = 0
    res, acc = [], []
    for r in recs:
        for p in r["points"]:
            for e in p["est"]:
                if not keep(e):
                    continue
                n += 1
                if e.get("D") is not None:
                    ok += 1
                if e.get("negative_D"):
                    neg += 1
                if e.get("at_bound"):
                    bound += 1
                if e.get("residual") is not None:
                    res.append(e["residual"])
                if e.get("n_accepted") is not None:
                    acc.append(e["n_accepted"])
    return dict(n=n, ok=ok, negative=neg, at_bound=bound,
                residual=_med(res), accepted=_med(acc))


def _same_psf(a, b):
    """The record stores the point spread function as (x, y, z); a run made
    before it could be anisotropic stores one number."""
    a = [a, a, a] if np.isscalar(a) else list(a)
    b = [b, b, b] if np.isscalar(b) else list(b)
    return np.allclose(a, b)


def _psf_label(p):
    p = [p, p, p] if np.isscalar(p) else list(p)
    return ("%.1f mm" % p[0]) if len(set(p)) == 1 else ("%.1f/%.1f/%.1f mm" % tuple(p))


def _fmt(h, scale=1.0, fmt="%8.3g"):
    if not h["n_glands"]:
        return "%8s" % "-"
    return fmt % (h["median"] * scale)


def _range(h, scale=1.0):
    if not h["n_glands"]:
        return ""
    return "[%.3g, %.3g]" % (h["min"] * scale, h["max"] * scale)


def shell_table(recs, psf):
    print("\nSHELL ESTIMATOR   (psf %s)" % _psf_label(psf))
    print("  the frame interval is an acquisition setting; the estimator has no temporal")
    print("  scale of its own, so it inherits whatever the acquisition delivers")
    print("  %-7s %-6s %7s %7s %9s %9s %9s %9s" %
          ("frame/s", "R/mm", "yield", "accept", "v mm/s", "D mm2/s", "D/v mm", "at bound"))
    frames = sorted({e["frame_dt_s"] for r in recs for p in r["points"]
                     for e in p["est"] if e["estimator"] == "shell"})
    radii = sorted({e["R_mm"] for r in recs for p in r["points"]
                    for e in p["est"] if e["estimator"] == "shell"})
    for fdt in frames:
        for R in radii:
            k = lambda e: (e["estimator"] == "shell" and _same_psf(e["psf_fwhm_mm"], psf)
                           and e["frame_dt_s"] == fdt and e["R_mm"] == R)
            c = _counts(recs, k)
            if not c["n"]:
                continue
            usable = lambda e: k(e) and e.get("D") is not None and not e.get("at_bound")
            hv = hierarchy(_rows(recs, usable, lambda e: e["v"]), boot=0)
            hD = hierarchy(_rows(recs, usable, lambda e: e["D"]), boot=0)
            ha = hierarchy(_rows(recs, usable, lambda e: e["D"] / e["v"] if e["v"] else None), boot=0)
            print("  %-7.2f %-6.2f %6.0f%% %7.1f %s %s %s %8.0f%%"
                  % (fdt, R, 100.0 * c["ok"] / c["n"], c["accepted"] or 0,
                     _fmt(hv, 1e3), _fmt(hD, 1e6), _fmt(ha, 1e3),
                     100.0 * c["at_bound"] / max(c["ok"], 1)))


def grid_table(recs, psf):
    print("\nGRID ESTIMATOR   (psf %s)" % _psf_label(psf))
    print("  sigma_t is the estimator's own temporal scale, swept separately from the frame")
    print("  interval; residual 1.0 means the model explains nothing of the data")
    print("  %-7s %-8s %-8s %8s %8s %9s %9s %9s" %
          ("frame/s", "sigma_t/s", "sigma_x/mm", "residual", "neg. D", "v mm/s", "D mm2/s", "D/v mm"))
    keys = sorted({(e["frame_dt_s"], e["sigma_t_s"], e["sigma_x_mm"])
                   for r in recs for p in r["points"]
                   for e in p["est"] if e["estimator"] == "grid"})
    for fdt, st, sx in keys:
        k = lambda e: (e["estimator"] == "grid" and _same_psf(e["psf_fwhm_mm"], psf)
                       and e["frame_dt_s"] == fdt and e["sigma_t_s"] == st
                       and e["sigma_x_mm"] == sx)
        c = _counts(recs, k)
        if not c["n"]:
            continue
        # a negative trace is not a small dispersion; the medians are over the
        # points where the fit found a dispersive process at all
        usable = lambda e: k(e) and e.get("D") is not None and not e.get("negative_D")
        hv = hierarchy(_rows(recs, usable, lambda e: e["v"]), boot=0)
        hD = hierarchy(_rows(recs, usable, lambda e: e["D"]), boot=0)
        ha = hierarchy(_rows(recs, usable, lambda e: e["D"] / e["v"] if e["v"] else None), boot=0)
        print("  %-7.2f %-8.1f %-8.2f %8.3f %7.0f%% %s %s %s"
              % (fdt, st, sx, c["residual"] or float("nan"),
                 100.0 * c["negative"] / max(c["n"], 1),
                 _fmt(hv, 1e3), _fmt(hD, 1e6), _fmt(ha, 1e3)))


def psf_effect(recs, frame_dt=None):
    """The reported dispersion under each named acquisition blur.

    Both estimators are read at one frame interval they share.  Reading the
    shell estimator at its fastest interval and the grid estimator at whatever
    intervals it ran would make this table disagree with the matched comparison
    below it, and with the figure, for a reason that has nothing to do with the
    blur."""
    frames = common_frames(recs)
    fdt = (frames[0] if frames else None) if frame_dt is None else frame_dt
    psfs = sorted({tuple(e["psf_fwhm_mm"]) if not np.isscalar(e["psf_fwhm_mm"])
                   else (e["psf_fwhm_mm"],) * 3
                   for r in recs for p in r["points"] for e in p["est"]})
    sts = sorted({e["sigma_t_s"] for r in recs for p in r["points"]
                  for e in p["est"] if e["estimator"] == "grid"})
    if not sts:
        return
    print("\nPOINT SPREAD FUNCTION, the one acquisition setting both estimators share")
    print("  each blur is a named condition, never the mean of its three widths;")
    print("  both estimators at frame interval %s" % ("?" if fdt is None else "%.2f s" % fdt))
    print("  %-16s %26s %26s" % ("psf", "shell, R 1 mm", "grid, sigma_t %.0f s, sigma_x 1.5 mm" % sts[-1]))
    for psf in psfs:
        ks = lambda e: (e["estimator"] == "shell" and _same_psf(e["psf_fwhm_mm"], psf)
                        and (fdt is None or e["frame_dt_s"] == fdt) and e["R_mm"] == 1.0
                        and e.get("D") is not None and not e.get("at_bound"))
        kg = lambda e: (e["estimator"] == "grid" and _same_psf(e["psf_fwhm_mm"], psf)
                        and (fdt is None or e["frame_dt_s"] == fdt)
                        and e["sigma_t_s"] == sts[-1] and e["sigma_x_mm"] == 1.5
                        and e.get("D") is not None and not e.get("negative_D"))
        hs = hierarchy(_rows(recs, ks, lambda e: e["D"]), boot=0)
        hg = hierarchy(_rows(recs, kg, lambda e: e["D"]), boot=0)
        print("  %-16s %20s mm2/s %20s mm2/s"
              % (_psf_label(psf), _fmt(hs, 1e6), _fmt(hg, 1e6)))


SHELL_VALID = lambda e: e.get("D") is not None and not e.get("at_bound")
GRID_VALID = lambda e: e.get("D") is not None and not e.get("negative_D")


def _pick(point, want):
    """The single estimate at a point matching every key in `want`, or None."""
    hits = [e for e in point["est"]
            if all((_same_psf(e["psf_fwhm_mm"], v) if k == "psf_fwhm_mm"
                    else e.get(k) == v) for k, v in want.items())]
    if len(hits) > 1:
        raise SystemExit("%d estimates match %s at one point; the setting keys "
                         "do not identify an estimate uniquely" % (len(hits), want))
    return hits[0] if hits else None


def common_frames(recs):
    """Frame intervals at which both estimators were actually run."""
    f = {}
    for est in ("shell", "grid"):
        f[est] = {e["frame_dt_s"] for r in recs for p in r["points"]
                  for e in p["est"] if e["estimator"] == est}
    return sorted(f["shell"] & f["grid"])


def matched(recs, psf, frame_dt, shell_R, sigma_t, sigma_x):
    """Both estimators at one frame interval, on the evaluation points at which
    both returned a usable value.

    The comparison the experiment exists to make is between estimator
    formulations, so everything else has to be held equal: the same tissue, the
    same propagated curves, the same point spread function, the same frame
    interval, and the same evaluation points.  Comparing each estimator on its
    own surviving subset confounds the estimate with which points survived, and
    comparing across frame intervals confounds the estimator with the
    acquisition.  Both are reported anyway, as `shell_own` and `grid_own`, so
    that the size of that confound is visible rather than assumed away."""
    want_s = dict(estimator="shell", psf_fwhm_mm=psf, frame_dt_s=frame_dt, R_mm=shell_R)
    want_g = dict(estimator="grid", psf_fwhm_mm=psf, frame_dt_s=frame_dt,
                  sigma_t_s=sigma_t, sigma_x_mm=sigma_x)
    n_pts = n_s = n_g = n_both = 0
    rows = {k: [] for k in ("s_v", "s_D", "g_v", "g_D", "so_v", "so_D", "go_v", "go_D")}
    for r in recs:
        for p in r["points"]:
            es, eg = _pick(p, want_s), _pick(p, want_g)
            if es is None and eg is None:
                continue
            n_pts += 1
            ok_s = es is not None and SHELL_VALID(es)
            ok_g = eg is not None and GRID_VALID(eg)
            n_s += ok_s
            n_g += ok_g
            key = dict(seed=r["seed"], rve=r["rve"])
            if ok_s:
                rows["so_v"].append(dict(val=es["v"], **key))
                rows["so_D"].append(dict(val=es["D"], **key))
            if ok_g:
                rows["go_v"].append(dict(val=eg["v"], **key))
                rows["go_D"].append(dict(val=eg["D"], **key))
            if ok_s and ok_g:
                n_both += 1
                rows["s_v"].append(dict(val=es["v"], **key))
                rows["s_D"].append(dict(val=es["D"], **key))
                rows["g_v"].append(dict(val=eg["v"], **key))
                rows["g_D"].append(dict(val=eg["D"], **key))
    h = {k: hierarchy(v, boot=0) for k, v in rows.items()}
    out = dict(frame_dt_s=frame_dt, shell_R_mm=shell_R, sigma_t_s=sigma_t,
               sigma_x_mm=sigma_x, psf=list(psf),
               n_points=n_pts, n_shell_valid=n_s, n_grid_valid=n_g,
               n_jointly_valid=n_both,
               shell_failure=1.0 - n_s / n_pts if n_pts else float("nan"),
               grid_failure=1.0 - n_g / n_pts if n_pts else float("nan"),
               joint_fraction=n_both / n_pts if n_pts else float("nan"),
               hier=h)
    for q in ("v", "D"):
        a, b = h["s_" + q]["median"], h["g_" + q]["median"]
        out["ratio_" + q] = (a / b) if (a is not None and b) else float("nan")
        ao, bo = h["so_" + q]["median"], h["go_" + q]["median"]
        out["ratio_own_" + q] = (ao / bo) if (ao is not None and bo) else float("nan")
    return out


def matched_table(recs, psf, frame_dt=None, shell_R=1.0):
    """The whole grid of settings at one frame interval, with no selection.

    There is no "setting most favourable to each estimator" here: a favourable
    setting can only be named against a criterion, and no criterion is available
    because the tissue has no true dispersion coefficient to be close to.  Every
    grid setting is therefore reported, and the spread of the ratio across them
    is part of the answer."""
    frames = common_frames(recs)
    if not frames:
        print("\nMATCHED COMPARISON: the two estimators share no frame interval; "
              "no matched comparison is possible")
        return []
    fdt = frames[0] if frame_dt is None else frame_dt
    if fdt not in frames:
        raise SystemExit("frame interval %g s is not common to both estimators; "
                         "available: %s" % (fdt, frames))
    keys = sorted({(e["sigma_t_s"], e["sigma_x_mm"]) for r in recs for p in r["points"]
                   for e in p["est"]
                   if e["estimator"] == "grid" and e["frame_dt_s"] == fdt
                   and _same_psf(e["psf_fwhm_mm"], psf)})
    print("\nMATCHED COMPARISON   psf %s, frame interval %.2f s, shell R %.2f mm"
          % (_psf_label(psf), fdt, shell_R))
    print("  both estimators on the same curves at the same frame interval, and the")
    print("  medians taken over the evaluation points at which both returned a value")
    print("  %-8s %-8s %7s %7s %7s | %8s %8s %6s | %8s %8s %6s"
          % ("sigma_t", "sigma_x", "shell", "grid", "joint",
             "v shell", "v grid", "ratio", "D shell", "D grid", "ratio"))
    print("  %-8s %-8s %7s %7s %7s | %8s %8s %6s | %8s %8s %6s"
          % ("s", "mm", "fail", "fail", "points", "mm/s", "mm/s", "", "mm2/s", "mm2/s", ""))
    out = []
    for st, sx in keys:
        m = matched(recs, psf, fdt, shell_R, st, sx)
        out.append(m)
        if not m["n_points"]:
            continue
        h = m["hier"]
        print("  %-8.1f %-8.2f %6.0f%% %6.0f%% %7d | %8s %8s %6.1f | %8s %8s %6.1f"
              % (st, sx, 100 * m["shell_failure"], 100 * m["grid_failure"],
                 m["n_jointly_valid"],
                 _fmt(h["s_v"], 1e3), _fmt(h["g_v"], 1e3), m["ratio_v"],
                 _fmt(h["s_D"], 1e6), _fmt(h["g_D"], 1e6), m["ratio_D"]))
    good = [m for m in out if np.isfinite(m.get("ratio_D", float("nan")))]
    if good:
        rv = [m["ratio_v"] for m in good]
        rD = [m["ratio_D"] for m in good]
        print("  over the %d grid settings at this frame interval: "
              "v ratio %.1f to %.1f, D ratio %.1f to %.1f"
              % (len(good), min(rv), max(rv), min(rD), max(rD)))
        worst = max(good, key=lambda m: abs(np.log(m["ratio_D"])))
        print("  on each estimator's own valid points instead of the joint set, the D "
              "ratio at the widest setting is %.1f rather than %.1f"
              % (worst["ratio_own_D"], worst["ratio_D"]))
    return out


def frame_sweep(recs, psf, shell_R=1.0):
    """The matched ratio at every frame interval both estimators share, so that
    the acquisition effect and the estimator effect can be told apart."""
    frames = common_frames(recs)
    if len(frames) < 2:
        return []
    print("\nTHE SAME COMPARISON AT EVERY SHARED FRAME INTERVAL")
    print("  %-8s %-8s %-8s %7s %7s %7s %8s %8s"
          % ("frame/s", "sigma_t", "sigma_x", "shell", "grid", "joint", "v ratio", "D ratio"))
    out = []
    for fdt in frames:
        keys = sorted({(e["sigma_t_s"], e["sigma_x_mm"]) for r in recs for p in r["points"]
                       for e in p["est"]
                       if e["estimator"] == "grid" and e["frame_dt_s"] == fdt
                       and _same_psf(e["psf_fwhm_mm"], psf)})
        for st, sx in keys:
            m = matched(recs, psf, fdt, shell_R, st, sx)
            if not m["n_jointly_valid"]:
                continue
            out.append(m)
            print("  %-8.2f %-8.1f %-8.2f %6.0f%% %6.0f%% %7d %8.1f %8.1f"
                  % (fdt, st, sx, 100 * m["shell_failure"], 100 * m["grid_failure"],
                     m["n_jointly_valid"], m["ratio_v"], m["ratio_D"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--psf", default="1.1",
                    help="point spread function for the tables, one number or three "
                         "comma separated, in mm (e.g. 0 or 1.1 or 0.8,1.1,2.5)")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--frame", type=float, default=None,
                    help="frame interval for the matched comparison, in seconds; "
                         "default the smallest interval both estimators share")
    ap.add_argument("--shell-R", type=float, default=1.0,
                    help="shell radius for the matched comparison, in mm")
    ap.add_argument("--sweep", action="store_true",
                    help="also repeat the matched comparison at every shared frame interval")
    a = ap.parse_args()
    psf = [float(x) for x in str(a.psf).split(",")]
    a.psf = psf * 3 if len(psf) == 1 else psf
    man, recs, errors = load(a.folder, partial=a.partial)
    eff = man.get("effective", {})
    npts = sum(len(r["points"]) for r in recs)
    print("=" * 92)
    print("%s   %d volumes over %d glands, %d evaluation points, tree %g um, record %.2f s"
          % (a.folder, len(recs), len({r["seed"] for r in recs}), npts,
             eff.get("uniform_d_term_um", float("nan")), eff.get("t_end_s", float("nan"))))
    if errors:
        print("  %d records carry errors and were excluded" % len(errors))
    shell_table(recs, a.psf)
    grid_table(recs, a.psf)
    psf_effect(recs, a.frame)
    matched_table(recs, a.psf, a.frame, a.shell_R)
    if a.sweep:
        frame_sweep(recs, a.psf, a.shell_R)


if __name__ == "__main__":
    main()
