"""Overview figure of Study 04: tissue coupling in the reduced model.

Panel functions (letters as in RESULTS.md: Figure 1 A-C, Figure 3 A-B)
    panel_a  Fig. 1A  computational model: the Study-01 network (seed 2) in the
                      gland, the 6-mm sampling volume and its largest vessels
    panel_b  Fig. 1B  shear-wave response of three (mu, eta, s) states inside
                      the 3% observation noise, and
             Fig. 1C  the CEUS AUC of the same states (10% noise band)
    panel_c  Fig. 3A  (mu, s) posterior averaged over the 40 noise realizations:
                      SWE alone (grey) and coupled through the AUC-s relation (purple)
    panel_d  Fig. 3B  displaced coupling relation: coverage of the 90% interval
                      for mu and bias in mu against the offset (mean and range)

Data: results/overview_figure/fig_study04_data.npz (written by make_data.py)
and results/overview_figure/net_seed<k>.npz (written by save_net.py).
Rendering needs numpy, scipy and matplotlib only.

Run directly for the original one-row layout (four panels, 17 x 5.6 cm),
written to figures/overview_1x4.pdf/.png.  The two figures used in RESULTS.md
are produced by fig_study04_split.py.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # the study root
DATA = ROOT / "results" / "overview_figure"
FIGS = ROOT / "figures"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONTS = ["Arial Narrow", "Helvetica Neue", "Liberation Sans", "DejaVu Sans"]
PURPLE = "#7B1FA2"; GREY = "0.45"; LIGHT = "0.80"
C_TRUE, C_A, C_B = "black", "#1565C0", "#E65100"      # generating state, two ridge states
FS = 7.0            # base font size (pt)
FS_PANEL = 9.0

D = np.load(DATA / "fig_study04_data.npz")
f, curves, states = D["f"], D["curves"], D["states"]
s_tab, auc = D["s_tab"], D["auc"]
mu_g, s_g = D["mu_grid"] / 1e3, D["s_grid"]
off, cov, bias, width = D["off"], D["cov"], D["bias"], D["width"]
w_mu = D["w_mu"]
s_true, mu_true = float(D["s_true"]), float(D["mu_true"])


def hpd_levels(p, levels=(0.5, 0.9)):
    """Density thresholds enclosing the given probability mass."""
    q = np.sort(p.ravel())[::-1]
    c = np.cumsum(q)
    return [q[np.searchsorted(c, lv * c[-1])] for lv in levels]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONTS, "font.size": FS,
        "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
        "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": FONTS[0], "mathtext.it": FONTS[0] + ":italic",
        "mathtext.bf": FONTS[0] + ":bold", "mathtext.default": "regular",
    })
    from matplotlib.font_manager import findfont, FontProperties
    if "narrow" not in findfont(FontProperties(family=FONTS[0])).lower():
        print("  ! Arial Narrow not found; rendering with a fallback font")
        plt.rcParams.update({"mathtext.fontset": "dejavusans"})


def panel_label(ax, s, x=-0.02, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold",
            ha="right", va="bottom")


ART, VEIN, MATRIX = "#C62828", "#1E5AA8", "#F3D9DD"


def tube(ax, a, b, rad, fc, ec, lw=0.4, dashes=(), zorder=3):
    """Draw a straight vessel segment a->b (mm) of radius rad (mm) as a filled
    band; with `dashes` draw only the two dashed boundary lines at +-rad."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = b - a; n = np.array([-t[1], t[0]]) / max(np.hypot(*t), 1e-9) * rad
    if dashes:
        for sgn in (1, -1):
            ax.plot([a[0] + sgn * n[0], b[0] + sgn * n[0]], [a[1] + sgn * n[1], b[1] + sgn * n[1]],
                    color=ec, lw=lw, dashes=dashes, zorder=zorder)
    else:
        ax.fill([a[0] + n[0], b[0] + n[0], b[0] - n[0], a[0] - n[0]],
                [a[1] + n[1], b[1] + n[1], b[1] - n[1], a[1] - n[1]], fc=fc, ec=ec, lw=lw, zorder=zorder)


def panel_a(ax):
    """The computational model: the Study-01 prostate network (seed 1, x-y
    projection, arteries red, veins blue) in the gland matrix, and one 6-mm
    sampling volume with its largest vessels drawn as tubes."""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle, Ellipse, ConnectionPatch
    net = np.load(DATA / ("net_seed%d.npz" % int(D["seed"])))
    p0, p1, r, frac, kind = net["p0"] * 1e3, net["p1"] * 1e3, net["r"] * 1e6, net["frac"], net["kind"]
    c = net["centre"] * 1e3; h = float(net["half"]) * 1e3; semi = net["gland_semi"] * 1e3
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # --- whole gland: pink matrix, arteries red, veins blue (segments >= 40 um)
    top = ax.inset_axes([0.18, 0.30, 0.82, 0.62]); top.set_axis_off(); top.set_aspect("equal")
    top.add_patch(Ellipse((0, 0), 2 * semi[0], 2 * semi[1], fc=MATRIX, ec="#D9A7B0", lw=0.6, alpha=0.9, zorder=1))
    for k, col in ((1, VEIN), (0, ART)):
        m = (r >= 40.0) & (kind == k)
        top.add_collection(LineCollection(np.stack([p0[m][:, :2], p1[m][:, :2]], axis=1), colors=col,
                                          linewidths=0.15 + 0.7 * (r[m] / 400.0), alpha=0.85, zorder=2))
    top.add_patch(Rectangle((c[0] - h, c[1] - h), 2 * h, 2 * h, fill=False, ec="black", lw=0.8, zorder=4))
    top.set_xlim(-21, 21); top.set_ylim(-18.5, 16)
    top.plot([10, 20], [-17.5, -17.5], color="0.3", lw=0.8); top.text(15, -16.8, "10 mm", ha="center", va="bottom", fontsize=FS - 1.5, color="0.3")
    ax.text(0.58, 0.96, "computational model", ha="center", va="bottom", fontsize=FS)
    # --- one sampling volume: the largest vessels as round-capped tubes (radii x1.5)
    bot = ax.inset_axes([0.0, 0.0, 0.56, 0.56]); bot.set_aspect("equal")
    bot.set_xlim(c[0] - h, c[0] + h); bot.set_ylim(c[1] - h, c[1] + h)
    bot.set_xticks([]); bot.set_yticks([]); bot.set_facecolor(MATRIX)
    for sp in bot.spines.values(): sp.set_linewidth(0.8)
    fig = ax.figure; fig.canvas.draw()
    bb = bot.get_window_extent(); pt_per_mm = (bb.width / fig.dpi * 72) / (2 * h)   # points on paper per mm of tissue
    X = 1.0
    inside = frac > 0
    # vessels >= 60 um at true radius (arteries drawn over veins); the next tier (40-60 um)
    # faint, so that a vessel tapering below the threshold does not look cut
    tier = inside & (r >= 40.0) & (r < 60.0)
    bot.add_collection(LineCollection(np.stack([p0[tier][:, :2], p1[tier][:, :2]], axis=1),
                                      colors=[VEIN if k else ART for k in kind[tier]],
                                      linewidths=np.clip(2 * X * r[tier] * 1e-3 * pt_per_mm, 0.4, None), alpha=0.45,
                                      capstyle="round", zorder=2))
    big = np.where(inside & (r >= 60.0))[0]
    big = big[np.lexsort((r[big], -kind[big]))]          # veins first, arteries drawn on top
    bot.add_collection(LineCollection(np.stack([p0[big][:, :2], p1[big][:, :2]], axis=1),
                                      colors=[VEIN if k else ART for k in kind[big]],
                                      linewidths=2 * X * r[big] * 1e-3 * pt_per_mm, capstyle="round", joinstyle="round", zorder=3))
    # constriction icon: a vessel cross-section with its radius scaled by s
    from matplotlib.patches import Circle
    cx, cy, R = c[0] + 0.68 * h, c[1] - 0.64 * h, 0.19 * h
    bot.add_patch(Rectangle((cx - 1.55 * R, cy - 1.55 * R), 3.1 * R, 3.1 * R, fc="white", ec="none", alpha=0.92, zorder=4))
    bot.add_patch(Circle((cx, cy), R, fc="#F7B6BE", ec="#8E1B1B", lw=0.9, zorder=5))          # lumen and wall
    for sc in (0.75, 0.5, 1.3):                                                                   # the same wall at other s
        bot.add_patch(Circle((cx, cy), sc * R, fc="none", ec="#8E1B1B", lw=0.6, ls=(0, (2.0, 1.4)), zorder=6))
    bot.text(cx, cy + 1.5 * R, "radius $\\times s$", ha="center", va="bottom", fontsize=FS - 1, zorder=6,
             bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
    bot.text(0.04, 0.05, "matrix $\\mu$, $\\eta$", transform=bot.transAxes, ha="left", va="bottom", fontsize=FS - 1, zorder=6,
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    for (xa, ya), (xb, yb) in (((c[0] - h, c[1] + h), (c[0] - h, c[1] + h)), ((c[0] + h, c[1] - h), (c[0] + h, c[1] - h))):
        ax.add_artist(ConnectionPatch(xyA=(xa, ya), coordsA=top.transData, xyB=(xb, yb), coordsB=bot.transData,
                                      color="0.45", lw=0.5, ls=":", zorder=1))
    bot.text(0.5, -0.04, "6 mm", transform=bot.transAxes, ha="center", va="top", fontsize=FS - 1)


def panel_b(axb, axc):
    y0 = curves[0, :24]
    axb.fill_between(f, 0.97 * y0, 1.03 * y0, color=LIGHT, lw=0)
    axb.text(198, 0.97 * y0[-1] - 0.015, "±3% noise", fontsize=FS - 1.5, color="0.35", ha="right", va="top")
    labels = ["$s$ 0.80, $\\mu$ 2.00 kPa, $\\eta$ 1.00 Pa·s",
              "$s$ 0.73, $\\mu$ %.2f kPa, $\\eta$ %.2f Pa·s" % (states[1, 1] / 1e3, states[1, 2]),
              "$s$ 0.83, $\\mu$ %.2f kPa, $\\eta$ %.2f Pa·s" % (states[2, 1] / 1e3, states[2, 2])]
    for k, (col, ls) in enumerate(((C_TRUE, "-"), (C_A, "--"), (C_B, ":"))):
        axb.plot(f, curves[k, :24], color=col, lw=1.0, ls=ls, label=labels[k])
    axb.set_xlabel("frequency (Hz)", labelpad=1); axb.set_ylabel("phase velocity (m/s)", labelpad=1)
    axb.set_xlim(50, 200); axb.set_xticks([50, 100, 150, 200]); axb.set_ylim(1.1, 1.85); axb.set_yticks([1.4, 1.6, 1.8])
    axb.legend(loc="lower right", frameon=False, handlelength=1.0, handletextpad=0.35, borderaxespad=0.0, labelspacing=0.15, fontsize=FS - 1.4)
    axb.set_title("shear-wave response", pad=2)
    # CEUS AUC of the same states
    a0 = np.interp(s_true, s_tab, auc)
    axc.fill_between(s_tab, 0.9 * auc / a0, 1.1 * auc / a0, color=LIGHT, lw=0, label="±10% noise")
    axc.plot(s_tab, auc / a0, color=GREY, lw=0.9)
    for k, col in enumerate((C_TRUE, C_A, C_B)):
        sk = states[k, 0]
        axc.plot(sk, np.interp(sk, s_tab, auc) / a0, "o", ms=4, color=col, mec="white", mew=0.5, zorder=4)
    axc.set_xlabel("vessel radius scale $s$", labelpad=1); axc.set_ylabel("CEUS AUC (rel.)", labelpad=1)
    axc.set_xlim(0.65, 1.05); axc.set_xticks([0.7, 0.8, 0.9, 1.0]); axc.set_ylim(0.5, 2.1); axc.set_yticks([0.5, 1.0, 1.5, 2.0])
    axc.set_title("contrast-transport response", pad=2)


def panel_c(ax):
    from scipy.ndimage import zoom, gaussian_filter
    Z = 4
    mu_f = np.linspace(mu_g[0], mu_g[-1], Z * len(mu_g)); s_f = np.linspace(s_g[0], s_g[-1], Z * len(s_g))
    MU, S = np.meshgrid(mu_f, s_f, indexing="ij")
    for key, col, lw in (("swe_mean", GREY, 0.9), ("cpl_mean", PURPLE, 1.1)):
        p = np.clip(zoom(gaussian_filter(D[key], 0.8), Z, order=3), 0, None); p /= p.sum()
        l90, l50 = hpd_levels(p, (0.9, 0.5))
        ax.contourf(MU, S, p, levels=[l90, p.max()], colors=[col], alpha=0.12)
        ax.contour(MU, S, p, levels=[l90], colors=[col], linewidths=lw, linestyles="dotted")
        ax.contour(MU, S, p, levels=[l50], colors=[col], linewidths=lw)
    ax.plot(mu_true / 1e3, s_true, "+", color="black", ms=7, mew=1.2, zorder=5)
    ax.set_xlabel("matrix elasticity $\\mu$ (kPa)", labelpad=1); ax.set_ylabel("vessel radius scale $s$", labelpad=1)
    ax.set_xlim(1.75, 2.25); ax.set_ylim(0.62, 0.90)
    ax.plot([], [], color=GREY, lw=0.9, label="SWE alone")
    ax.plot([], [], color=PURPLE, lw=1.1, label="SWE + CEUS, coupled")
    ax.plot([], [], color="0.3", lw=0.9, ls="dotted", label="90% (dotted), 50% (solid)")
    ax.legend(loc="lower right", frameon=False, handlelength=1.4, borderaxespad=0.05, labelspacing=0.15, fontsize=FS - 1.3)
    ax.set_title("joint posterior", pad=2)


def panel_d(ax):
    m, lo, hi = cov.mean(0), cov.min(0), cov.max(0)
    ax.fill_between(off, lo, hi, color=PURPLE, alpha=0.15, lw=0)
    ax.plot(off, m, "o-", color=PURPLE, lw=1.1, ms=3.5, label="coverage (90% interval, $\\mu$)")
    ax.axhline(0.9, color="0.3", lw=0.6, ls="dashed")
    ax.text(0.2, 0.91, "nominal 90%", ha="right", va="bottom", fontsize=FS - 1, color="0.3")
    ax.set_xlabel("offset of the assumed AUC–$s$ relation", labelpad=1)
    ax.set_ylabel("coverage", labelpad=1, color=PURPLE)
    ax.set_xlim(-0.005, 0.205); ax.set_ylim(0.35, 1.07); ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2]); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax2 = ax.twinx()
    bm = bias.mean(0)
    ax2.fill_between(off, bias.min(0), bias.max(0), color="0.5", alpha=0.15, lw=0)
    ax2.plot(off, bm, "s--", color="0.25", lw=0.9, ms=3, label="bias in $\\mu$ (%)")
    ax2.set_ylabel("bias in $\\mu$ (%)", labelpad=1, color="0.25"); ax2.set_ylim(-6.5, 1.2); ax2.set_yticks([-6, -4, -2, 0])
    ax2.tick_params(axis="y", labelsize=FS - 0.5)
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(-0.02, -0.02), frameon=False, handlelength=1.6, borderaxespad=0.1, labelspacing=0.2, fontsize=FS - 1)
    ax.set_title("displaced relation: narrow but wrong", pad=2)


def build():
    style()
    fig = plt.figure(figsize=(17 / 2.54, 5.6 / 2.54))
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1.08, 1.10, 0.95, 0.97],
                           left=0.02, right=0.985, top=0.88, bottom=0.20, wspace=0.50)
    axA = plt.subplot(gs[0]); panel_a(axA); panel_label(axA, "A", x=0.02, y=1.0)
    axB = plt.subplot(gs[1]); axC = plt.subplot(gs[2]); panel_b(axB, axC)
    panel_label(axB, "B", x=-0.36, y=1.04); panel_label(axC, "C", x=-0.36, y=1.04)
    axD = plt.subplot(gs[3]); panel_c(axD); panel_label(axD, "D", x=-0.36, y=1.04)
    for ax in (axB, axC, axD):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / ("overview_1x4.%s" % ext), dpi=300)
    print("written figures/overview_1x4.pdf and .png")


if __name__ == "__main__":
    build()
