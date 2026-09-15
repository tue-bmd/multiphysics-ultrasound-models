"""The three figures of RESULTS.md.

    python -m porovasc.figures --out figures

Each figure is drawn from one named source, which is printed on the figure
itself so that a figure separated from this repository still says where it came
from.  Nothing here computes a result; the figures read what the runs wrote.

  fig1_network.png     one built gland, seed 100                (live)
  fig2_coupling.png    runs/coupling, 90 records                (archive)
  fig3_estimator.png   runs/estimators2, analysis settings      (archive)
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PLAIN = dict(fontsize=8, color="0.35")


def _stamp(fig, text):
    fig.text(0.01, 0.01, text, ha="left", va="bottom", **PLAIN)


# ------------------------------------------------------------------ figure 1 -

def fig_network(out, seed=100):
    """One gland: the arterial tree coloured by pressure, the veins behind it.

    Vessels thinner than `d_min` are dropped from the drawing only.  The whole
    tree is about 54 000 segments and drawing all of them produces a solid
    block in which nothing is visible; the count that is drawn is printed.
    """
    from .geometry import network as N
    from .geometry.network import GLAND_SEMI
    from .physics import flow as F
    from .config import BASE, apply_globals

    apply_globals(dict(BASE))
    net = N.build(N.Params(seed=seed))
    fl = F.solve(net)

    p0, p1 = net.p0 * 1e3, net.p1 * 1e3          # mm
    d = net.d * 1e6                              # um
    art = net.kind == 0
    p_mmHg = fl.p[np.arange(net.n)] / (133.322)

    fig = plt.figure(figsize=(11, 4.6))

    # -- left: the tree ----------------------------------------------------
    # The feeders start outside the organ.  They are kept in the model and left
    # out of the drawing, because on a shared axis they set the limits and the
    # gland shrinks to a blob.
    semi = np.asarray(GLAND_SEMI) * 1e3                       # mm
    inside = (np.abs(p0) <= semi * 1.02).all(1) & (np.abs(p1) <= semi * 1.02).all(1)

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    d_min = 40.0
    sel = art & (d >= d_min) & inside
    lc = Line3DCollection(np.stack([p0[sel], p1[sel]], axis=1),
                          linewidths=np.clip(d[sel] / 150.0, 0.25, 2.2),
                          cmap="autumn_r", array=p_mmHg[sel])
    ax.add_collection3d(lc)

    vsel = (~art) & (d >= d_min * 1.8) & inside
    ax.add_collection3d(Line3DCollection(
        np.stack([p0[vsel], p1[vsel]], axis=1),
        linewidths=0.5, colors="#3b6ea5", alpha=0.35))

    ax.set_xlim(-semi[0], semi[0])
    ax.set_ylim(-semi[1], semi[1])
    ax.set_zlim(-semi[2], semi[2])
    ax.set_box_aspect(tuple(semi / semi.max()))
    ax.view_init(elev=16, azim=-62)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("y (mm)", fontsize=8)
    ax.set_zlabel("z (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(lc, ax=ax, shrink=0.55, pad=0.11)
    cb.set_label("arterial pressure (mmHg)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("one gland, 22 mL, %d of %d segments drawn\n"
                 "arteries by pressure, veins in blue, $d \\geq$ %.0f $\\mu$m"
                 % (sel.sum() + vsel.sum(), net.n, d_min), fontsize=9)

    # -- right: velocity against diameter, with the literature relation -----
    ax2 = fig.add_subplot(1, 2, 2)
    v = np.abs(fl.v) * 1e3
    ok = art & (d > 0) & (v > 0)
    ax2.scatter(d[ok], v[ok], s=1.0, alpha=0.08, color="#c0392b", linewidths=0,
                rasterized=True)
    dd = np.logspace(np.log10(d[ok].min()), np.log10(d[ok].max()), 200)
    ax2.plot(dd, 4 * 108 / np.pi * dd ** 1.101 / 1e3, "k-", lw=1.6,
             label="$Q = 108\\,d^{3.101}$, Skinner (1979)")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("segment diameter ($\\mu$m)", fontsize=9)
    ax2.set_ylabel("blood speed (mm/s)", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=8, loc="upper left", frameon=False)
    r = v[ok] / (4 * 108 / np.pi * d[ok] ** 1.101 / 1e3)
    ax2.set_title("simulated / literature: median %.2f, 5-95%% [%.2f, %.2f]"
                  % (np.median(r), *np.percentile(r, [5, 95])), fontsize=9)

    _stamp(fig, "live check, seed %d; %.1f mL gland" % (seed, 22.0))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(out, "fig1_network.png")
    fig.savefig(path, dpi=170); plt.close(fig)
    return path


# ------------------------------------------------------------------ figure 2 -

# (family, quantity as named in analyse_coupling.FAMILIES, label), bottom to top
COUPLING_ROWS = [
    ("transport", "mtt",              "mean transit time"),
    ("transport", "t_peak",           "time to peak"),
    ("transport", "D",                "estimated dispersion"),
    ("transport", "v",                "estimated drift"),
    ("drainage",  "tau_rc",           "relaxation time"),
    ("drainage",  "relax_per_strain", "amplitude per strain"),
    ("network",   "vessel_speed",     "vessel speed"),
    ("network",   "k_mean",           "permeability"),
    ("network",   "T2",               "mean squared tortuosity"),
    ("network",   "phi_cube",         "vascular volume fraction"),
]
FAMILY_COLOR = {"network": "#2b6cb0", "drainage": "#c05621", "transport": "#2f7d32"}
ARMS = [("tort_1.45", "tortuosity\n(geometry)"),
        ("H_3k",      "matrix modulus\n(control)"),
        ("P_ART_60",  "inlet pressure\n(control)")]


def fig_coupling(out, archive):
    """Effects come from `analyse_coupling.arm_effects`, so the figure and the
    table in RESULTS.md are the same reduction of the same records."""
    from . import analyse_coupling as AC

    _man, recs = AC.load(archive)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)

    for ax, (arm, title) in zip(axes, ARMS):
        eff = AC.arm_effects(recs, arm)
        for i, (fam, name, _lab) in enumerate(COUPLING_ROWS):
            h = eff[fam][name]
            if h["median"] is None:
                continue
            ax.plot([h["min"], h["max"]], [i, i], "-", lw=2.4,
                    color=FAMILY_COLOR[fam], alpha=0.35, solid_capstyle="round")
            ax.plot(h["median"], i, "o", ms=5.5, color=FAMILY_COLOR[fam], zorder=3)
        ax.axvline(0, color="0.5", lw=0.8, zorder=0)
        ax.set_xlim(-0.32, 0.32)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("effect (log$_{10}$ ratio)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", color="0.92", lw=0.6, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_yticks(np.arange(len(COUPLING_ROWS)))
    axes[0].set_yticklabels([lab for _f, _n, lab in COUPLING_ROWS], fontsize=8.5)
    for i, (fam, _n, _l) in enumerate(COUPLING_ROWS):
        axes[0].get_yticklabels()[i].set_color(FAMILY_COLOR[fam])

    handles = [plt.Line2D([], [], color=c, marker="o", ls="-", lw=2.4, ms=5.5,
                          label=f) for f, c in FAMILY_COLOR.items()]
    axes[-1].legend(handles=handles, fontsize=8, loc="lower right", frameon=False,
                    title="observable family", title_fontsize=8)

    _stamp(fig, "archive: %s, 90 records, 5 glands x 2 volumes; point = median "
                "over glands, bar = range over glands" % os.path.basename(archive))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = os.path.join(out, "fig2_coupling.png")
    fig.savefig(path, dpi=170); plt.close(fig)
    return path


# ------------------------------------------------------------------ figure 3 -

def fig_estimator(out, extract):
    """Reported dispersion against the analysis settings, on fixed curves.

    One tissue, one acquisition, one set of contrast curves.  Only the settings
    of the estimator change.  A settings-independent estimator would put every
    point on one horizontal line.
    """
    import csv as _csv
    rows = list(_csv.DictReader(open(os.path.join(extract,
                                                  "dispersion_by_setting.csv"))))
    man = json.load(open(os.path.join(extract, "manifest.json")))

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    # Only the per-setting medians are drawn.  The spread over sampling volumes
    # within a setting is a different quantity and is in the CSV; showing it
    # here would hide the spread between settings, which is the point.
    dts = sorted({float(r["frame_dt_s"]) for r in rows})
    cmap = plt.get_cmap("viridis")
    col = {dt: cmap(i / max(1, len(dts) - 1) * 0.85) for i, dt in enumerate(dts)}
    mark = {"shell": "o", "grid": "s"}

    for est, mk in mark.items():
        for dt in dts:
            sel = [r for r in rows
                   if r["estimator"] == est and float(r["frame_dt_s"]) == dt]
            if not sel:
                continue
            x = np.array([float(r["support_mm"]) for r in sel])
            y = np.array([float(r["D_median_mm2_s"]) for r in sel])
            ax.scatter(x * (1 + 0.012 * (np.arange(len(x)) % 4 - 1.5)), y,
                       s=34, marker=mk, facecolor=col[dt], edgecolor="0.25",
                       linewidths=0.4, alpha=0.9, zorder=3)

    lo_, hi_ = man["D_median_spread"]
    ax.axhspan(lo_, hi_, color="0.88", zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("estimator spatial support: shell radius, or Gaussian "
                  "$\\sigma_x$ for the grid estimator (mm)", fontsize=9)
    ax.set_ylabel("reported dispersion $D$ (mm$^2$/s)", fontsize=9)

    h = [plt.Line2D([], [], ls="", marker=m, color="0.4", ms=6.5, label=n)
         for n, m in mark.items()]
    h += [plt.Line2D([], [], ls="", marker="o", color=col[dt], ms=6.5,
                     label="%g s" % dt) for dt in dts]
    ax.legend(handles=h, fontsize=8, frameon=False, loc="lower right", ncol=2,
              title="estimator / frame interval", title_fontsize=8)
    ax.set_title("one tissue, one set of contrast curves, %d analysis settings:\n"
                 "reported $D$ spans %.3g to %.3g mm$^2$/s, a factor of %.0f"
                 % (man["settings"], lo_, hi_, hi_ / lo_), fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    _stamp(fig, "derived from archive %s; point = median over sampling volumes; "
                "bound-pinned and negative fits excluded"
           % man["source_archive"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = os.path.join(out, "fig3_estimator.png")
    fig.savefig(path, dpi=170); plt.close(fig)
    return path


# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--coupling", default="runs/coupling")
    ap.add_argument("--estimators", default="runs/estimator_settings")
    ap.add_argument("--only", choices=("network", "coupling", "estimator"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    made = []
    if a.only in (None, "network"):
        made.append(fig_network(a.out))
    if a.only in (None, "coupling") and os.path.isdir(a.coupling):
        made.append(fig_coupling(a.out, a.coupling))
    if a.only in (None, "estimator") and os.path.isdir(a.estimators):
        made.append(fig_estimator(a.out, a.estimators))
    for p in made:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
