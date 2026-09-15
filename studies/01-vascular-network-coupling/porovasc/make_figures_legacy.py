"""Figures for RESULTS.md.

    python docs/make_figures.py --baseline runs/baseline \
                               --lesion   runs/lesion_tort16 \
                               --estimators runs/estimators

Each figure corresponds to one section of RESULTS.md and is generated from the
run that section quotes.  Titles state what is plotted, not what it means; the
reading belongs in the text.

Two rules hold throughout, so that a figure and the table beside it cannot
disagree:

  aggregation   every summary of an estimate uses the same three-step reduction
                as the tables - median over the evaluation points of a sampling
                volume, then over the volumes of a gland, then over glands.
                Pooling evaluation points across glands would weight a gland by
                how many of its points happened to yield a fit.  Counts, such as
                the yield, are counts over points and are labelled as such.

  conditions    the point spread function is a named condition, never a number
                obtained by averaging its three widths.  An anisotropic blur of
                0.8 by 1.1 by 2.5 mm is not a 1.5 mm blur.

The acquisition condition used for the tables is passed with --psf and printed
on the figure, so that the figure and RESULTS.md quote the same condition.

The compression figure builds a gland (about a minute) because the relaxation
is not stored in any run record; pass --skip-gland to omit it.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


from porovasc import validate as VAL
from porovasc import cross_settings as CS
from porovasc.analyse_estimators import _rows, _psf_label, _same_psf
from porovasc.analyse_paired import hierarchy

PARTIAL = False          # set from the command line; figures from an archive
                         # that does not validate are marked on the figure


def _check(folder):
    rep = VAL.check(folder)
    if not rep["ok"]:
        VAL.report(rep)
        if not PARTIAL:
            raise SystemExit("%s does not validate; pass --partial to draw from it "
                             "anyway, which stamps the figure as provisional" % folder)
    return rep


def _summary(folder):
    """The summary, through the same checked loader `cross_settings` uses.

    Validating the archive is not enough for a figure drawn from a summary: the
    summary is a derived file and can describe an earlier state of the archive.
    `cross_settings.load` checks that too, and a figure must not be drawn from a
    summary that a table would refuse."""
    return CS.load(folder, partial=PARTIAL)


def _records(folder):
    _check(folder)
    plain = os.path.join(folder, "records.jsonl")
    path = plain if os.path.exists(plain) else plain + ".gz"
    op = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    return [json.loads(l) for l in op if l.strip() and "error" not in l]


def _stamp(fig, folder):
    """A figure drawn from an archive that does not validate says so."""
    if PARTIAL:
        fig.text(0.99, 0.01, "provisional: %s did not validate" % os.path.basename(folder),
                 ha="right", va="bottom", fontsize=6, color="C3")


def _hmed(recs, keep, value, scale=1.0):
    """The gland-level median of `value`, by the same reduction as the tables."""
    h = hierarchy(_rows(recs, keep, value), boot=0)
    return (h["median"] * scale) if h["n_glands"] else np.nan


def _parse(key):
    kern, R, tf, rule = key.split("_")
    return kern, float(R[1:]), tf, rule


# ----------------------------------------------------------------- figure 1
def fig_support(folder, path):
    """Reported v and D against the shell radius, on a fixed tissue."""
    s = _summary(folder)
    kin = s["baseline"]["kinetics"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    styles = {("plug", "front20"): "o-", ("plug", "front10"): "s--", ("plug", "moment"): "^:",
              ("pois", "front20"): "o-", ("pois", "front10"): "s--", ("pois", "moment"): "^:"}
    for kern in ("plug", "pois"):
        for rule in ("front10", "front20", "moment"):
            R, v, D = [], [], []
            for key in kin:
                k, r, tf, ru = _parse(key)
                if k != kern or ru != rule or tf != "new":
                    continue
                hv, hD = kin[key]["v_mm_s"], kin[key]["D_mm2_s"]
                if hv["n_glands"] and hD["n_glands"]:
                    R.append(r); v.append(hv["median"]); D.append(hD["median"])
            if not R:
                continue
            o = np.argsort(R); R = np.array(R)[o]
            c = "C0" if kern == "plug" else "C1"
            ax[0].plot(R, np.array(v)[o], styles[(kern, rule)], color=c, ms=4,
                       label="%s, %s" % (kern, rule))
            ax[1].plot(R, np.array(D)[o], styles[(kern, rule)], color=c, ms=4)
    ax[0].set_xlabel("shell radius (mm)"); ax[0].set_ylabel("reported v (mm/s)")
    ax[1].set_xlabel("shell radius (mm)"); ax[1].set_ylabel("reported D (mm$^2$/s)")
    ax[0].legend(fontsize=7, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Shell estimator on one fixed tissue, corrected transfer function, "
                 "five glands", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


# ----------------------------------------------------------------- figure 2
def fig_estimators(folder, path, psf=(1.1, 1.1, 1.1)):
    """Yield against frame interval, grid residual against its temporal scale,
    and the reported dispersion under each named acquisition blur.

    Every dispersion summary here uses the hierarchical reduction of the tables.
    The blur panel treats each point spread function as a named condition."""
    recs = _records(folder)
    E = [e for r in recs for p_ in r["points"] for e in p_["est"]]

    def psf_key(e):
        p_ = e["psf_fwhm_mm"]
        return tuple(p_) if not np.isscalar(p_) else (p_, p_, p_)

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))

    # (a) shell yield against frame interval.  A yield is a count over
    # evaluation points and is reported as one, which is why it is not
    # aggregated hierarchically.
    frames = sorted({e["frame_dt_s"] for e in E if e["estimator"] == "shell"})
    for R in sorted({e["R_mm"] for e in E if e["estimator"] == "shell"}):
        y = []
        for f in frames:
            sel = [e for e in E if e["estimator"] == "shell" and psf_key(e) == tuple(psf)
                   and e["frame_dt_s"] == f and e["R_mm"] == R]
            y.append(100.0 * sum(e.get("D") is not None for e in sel) / max(len(sel), 1))
        ax[0].plot(frames, y, "o-", ms=4, label="R = %.2f mm" % R)
    ax[0].set_xscale("log"); ax[0].set_xlabel("frame interval (s)")
    ax[0].set_ylabel("evaluation points with a fit (%)")
    ax[0].set_title("shell estimator, yield over points", fontsize=9)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    # (b) grid residual against sigma_t, hierarchically aggregated
    for f in sorted({e["frame_dt_s"] for e in E if e["estimator"] == "grid"}):
        sts, res = [], []
        for st in sorted({e["sigma_t_s"] for e in E if e["estimator"] == "grid"}):
            keep = (lambda st_=st, f_=f: (lambda e: (
                e["estimator"] == "grid" and _same_psf(e["psf_fwhm_mm"], psf)
                and e["frame_dt_s"] == f_ and e["sigma_t_s"] == st_
                and e["sigma_x_mm"] == 1.5 and e.get("residual") is not None)))()
            m = _hmed(recs, keep, lambda e: e["residual"])
            if np.isfinite(m):
                sts.append(st); res.append(m)
        if sts:
            ax[1].plot(sts, res, "o-", ms=4, label="frame %.0f s" % f)
    ax[1].set_xlabel("$\\sigma_t$ (s)"); ax[1].set_ylabel("least-squares residual")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("grid estimator, gland-level median", fontsize=9)
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)

    # (c) the point spread function as a named condition, at one frame interval
    psfs = sorted({psf_key(e) for e in E})
    shell_frames = sorted({e["frame_dt_s"] for e in E if e["estimator"] == "shell"})
    grid_frames = sorted({e["frame_dt_s"] for e in E if e["estimator"] == "grid"})
    common = sorted(set(shell_frames) & set(grid_frames))
    fdt = common[0] if common else None
    stmax = max(e["sigma_t_s"] for e in E if e["estimator"] == "grid")
    sh, gr = [], []
    for p_ in psfs:
        ks = (lambda p0=p_: (lambda e: (
            e["estimator"] == "shell" and psf_key(e) == p0 and e["R_mm"] == 1.0
            and (fdt is None or e["frame_dt_s"] == fdt)
            and e.get("D") is not None and not e.get("at_bound"))))()
        kg = (lambda p0=p_: (lambda e: (
            e["estimator"] == "grid" and psf_key(e) == p0 and e["sigma_x_mm"] == 1.5
            and e["sigma_t_s"] == stmax
            and (fdt is None or e["frame_dt_s"] == fdt)
            and e.get("D") is not None and not e.get("negative_D"))))()
        sh.append(_hmed(recs, ks, lambda e: e["D"], 1e6))
        gr.append(_hmed(recs, kg, lambda e: e["D"], 1e6))
    x = np.arange(len(psfs))
    ax[2].plot(x, sh, "o-", ms=5, label="shell, R = 1 mm")
    ax[2].plot(x, gr, "s-", ms=5, label="grid, $\\sigma_x$ = 1.5 mm")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([_psf_label(p_) for p_ in psfs], fontsize=7, rotation=20)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("point spread function, named condition")
    ax[2].set_ylabel("reported D (mm$^2$/s)")
    ax[2].set_title("acquisition blur%s" % ("" if fdt is None else ", frame %.2f s" % fdt),
                    fontsize=9)
    ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)

    fig.suptitle("Two estimators on the same curves; gland-level medians, "
                 "acquisition condition psf %s" % _psf_label(psf), fontsize=9)
    _stamp(fig, folder)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


# ----------------------------------------------------------------- figure 3
def fig_lesion(folder, path):
    """The effect of the lesion on v and D under every analysis setting, for
    the lesion volume and for the remote reference volume."""
    s = _summary(folder)
    arm = s["arms"][0]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for j, (name, lab) in enumerate((("v", "v"), ("D", "D"))):
        for i, (role, colour) in enumerate((("lesion", "C3"), ("reference", "C0"))):
            vals, lo, hi = [], [], []
            for key in sorted(arm["kinetics"]):
                e = arm["kinetics"][key].get(role)
                if not e:
                    continue
                h = e["effects"][name]
                if h["n_glands"]:
                    vals.append(h["median"]); lo.append(h["min"]); hi.append(h["max"])
            x = np.arange(len(vals)) + 0.0
            ax[j].errorbar(x, vals, yerr=[np.array(vals) - lo, np.array(hi) - np.array(vals)],
                           fmt="o", ms=3, lw=0.8, color=colour, alpha=0.8, label=role)
        ax[j].axhline(0, color="k", lw=0.8)
        ax[j].set_xlabel("analysis setting")
        ax[j].set_title("effect on %s" % lab, fontsize=9)
        ax[j].grid(alpha=0.3)
    ax[0].set_ylabel("log$_{10}$(lesion gland / baseline gland)")
    ax[0].legend(fontsize=8)
    fig.suptitle("Tortuosity raised to 1.6 inside a sphere; every analysis setting, "
                 "bars are the range over glands", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


# ----------------------------------------------------------------- figure 4
def fig_compression(path, seed=100, centre=(-5e-3, -8e-3, 0.0)):
    """The drained fraction over time, its relaxation spectrum, and the
    identity between relaxation amplitude and vascular volume fraction."""
    from porovasc.geometry import network as N
    from porovasc.physics import flow as F, drainage as D
    from porovasc.config import BASE, apply_globals
    apply_globals(dict(BASE))
    net = N.build(N.Params(seed=seed, rve_centres=(centre,)))
    fl = F.solve(net)
    c = np.asarray(centre)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for R in (1.5e-3, 3.0e-3, 6.0e-3):
        d = D.solve(net, fl, c, R, dP_ext=100.0, H=5e3)
        ax[0].semilogx(d.t[1:], d.f[1:], label="R = %.1f mm" % (R * 1e3))
        if abs(R - 3e-3) < 1e-9:
            # the spectrum is a non-negative least-squares fit on a grid of time
            # constants, so it is sparse by construction; drawn as lines it would
            # read as an oscillation rather than as a set of weights
            ax[1].vlines(d.tau_grid, 0, d.spectrum, color="C0", lw=1.4)
            ax[1].set_xscale("log")
            ax[1].axvline(d.tau_rc, color="C3", lw=1, ls="--",
                          label="$\\tau_{rc}$ = %.0f ms" % (d.tau_rc * 1e3))
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("drained fraction")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("relaxation time (s)"); ax[1].set_ylabel("spectral weight")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    # the identity, over several compression radii and moduli
    # the identity holds for every compressed radius and every modulus, and the
    # points for different moduli coincide because the ratio does not contain H
    pts = {}
    for H, mark in ((3e3, "o"), (5e3, "s"), (10e3, "^")):
        xs, ys = [], []
        for R in (1.5e-3, 2.0e-3, 3.0e-3, 4.0e-3, 6.0e-3):
            d = D.solve(net, fl, c, R, dP_ext=100.0, H=H)
            xs.append(d.phi); ys.append(-d.dV_inf / d.V_tissue / (100.0 / H))
        pts[H] = (xs, ys, mark)
    lim = [0, max(max(v[1]) for v in pts.values()) * 1.15]
    ax[2].plot(lim, lim, "k-", lw=0.8, zorder=0)
    for H, (xs, ys, mark) in pts.items():
        ax[2].plot(xs, ys, mark, ms=7, mfc="none", label="H = %g kPa" % (H / 1e3))
    ax[2].legend(fontsize=7)
    ax[2].set_xlim(lim); ax[2].set_ylim(lim)
    ax[2].set_xlabel("vascular volume fraction of the sphere")
    ax[2].set_ylabel("relaxation / applied strain")
    ax[2].grid(alpha=0.3)
    fig.suptitle("Compression relaxation of the network, seed %d" % seed, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def main(argv=None):
    global PARTIAL
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline"); ap.add_argument("--lesion"); ap.add_argument("--estimators")
    ap.add_argument("--psf", default="1.1",
                    help="acquisition condition for the estimator figure: one number "
                         "or three comma separated, in mm FWHM.  Must match the "
                         "condition RESULTS.md quotes.")
    ap.add_argument("--skip-gland", action="store_true")
    ap.add_argument("--partial", action="store_true",
                    help="draw from an archive that does not validate, stamping the "
                         "figure as provisional")
    a = ap.parse_args(argv)
    PARTIAL = a.partial
    psf = [float(x) for x in str(a.psf).split(",")]
    psf = tuple(psf * 3 if len(psf) == 1 else psf)
    os.makedirs(OUT, exist_ok=True)
    if a.baseline:
        print(fig_support(a.baseline, os.path.join(OUT, "support_dependence.png")))
    if a.estimators:
        print(fig_estimators(a.estimators, os.path.join(OUT, "two_estimators.png"), psf))
    if a.lesion:
        print(fig_lesion(a.lesion, os.path.join(OUT, "lesion_tortuosity.png")))
    if not a.skip_gland:
        print(fig_compression(os.path.join(OUT, "compression.png")))


if __name__ == "__main__":
    main()
