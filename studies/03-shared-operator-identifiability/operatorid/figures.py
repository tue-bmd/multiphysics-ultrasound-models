"""The three figures.  Titles name what is plotted; they do not state
conclusions."""
from __future__ import annotations
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .acquisition import Acquisition
from .constants import psf_widths
from .experiments import Case
from .infer import sampling as SA
from .infer import summaries as SU
from .models import bmode as BM, ceus as CE, swe as SW
from .params import TRUTH

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _load(results, name):
    p = os.path.join(results, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


# --------------------------------------------------------------- figure 1 ---

def fig_fields(out):
    """What the point spread changes in each observation."""
    a, p = Acquisition(), dict(TRUTH)
    # Predefined sharp-margin setting, used only to make the convolution
    # visible at figure scale.
    p["sig_e"] = 0.30e-3
    w0 = p["w0"]
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 2.5), layout="constrained")

    x = a.bmode.positions()
    ax[0].plot(x * 1e3, BM.ideal(x, p), lw=1.2, color="0.5", label="ideal")
    ax[0].plot(x * 1e3, BM.profile(x, p, float(psf_widths(w0)["bmode"])), lw=1.2,
               color="C0", label="observed")
    ax[0].set_xlabel("position (mm)"); ax[0].set_ylabel("echogenicity (arb.)")
    ax[0].set_title("B-mode")
    ax[0].legend(frameon=False)
    ax[0].set_xlim(5, 19)

    xs, ts = a.swe.positions(), a.swe.times()
    i = len(xs) // 2
    u0 = SW.ideal(xs, ts, p, a.swe.f0, a.swe.bandwidth)[i]
    u1 = SW.displacement(xs, ts, p, float(psf_widths(w0)["swe"]), a.swe.f0,
                         a.swe.bandwidth)[i]
    ax[1].plot(ts * 1e3, u0 * 1e6, lw=1.2, color="0.5", label="ideal")
    ax[1].plot(ts * 1e3, u1 * 1e6, lw=1.2, color="C1", label="observed")
    ax[1].set_xlabel("time (ms)"); ax[1].set_ylabel(r"displacement ($\mu$m)")
    ax[1].set_title("SWE")
    ax[1].legend(frameon=False)
    ax[1].set_xlim(0, 14)

    xc, tc = a.ceus.positions(), a.ceus.times()
    b0 = CE.ideal(xc, tc, p)
    b1 = CE.frames(xc, tc, p, float(psf_widths(w0)["ceus"]))
    c_ref = float(np.max(b0[0]))
    b0, b1 = b0 / c_ref, b1 / c_ref
    time_lines = ((0, "-"), (len(tc) // 2, "--"), (len(tc) - 1, ":"))
    for j, sty in time_lines:
        ax[2].plot(xc * 1e3, b0[j], sty, lw=1.0, color="0.5")
        ax[2].plot(xc * 1e3, b1[j], sty, lw=1.2, color="C2")
    color_handles = [
        plt.Line2D([], [], color="0.5", lw=1.0, label="ideal"),
        plt.Line2D([], [], color="C2", lw=1.2, label="observed"),
    ]
    time_handles = [
        plt.Line2D([], [], color="0.2", ls=sty, lw=1.0,
                   label=f"{tc[j]:.0f} s") for j, sty in time_lines
    ]
    leg1 = ax[2].legend(handles=color_handles, frameon=False, loc="upper right")
    ax[2].add_artist(leg1)
    ax[2].legend(handles=time_handles, frameon=False, loc="center right",
                 title="time", title_fontsize=7)
    ax[2].set_xlabel("position (mm)")
    ax[2].set_ylabel(r"normalized concentration, $c/c_{\max}$")
    ax[2].set_title("CEUS")
    fig.savefig(os.path.join(out, "fig1_fields.png"))
    plt.close(fig)


# --------------------------------------------------------------- figure 2 ---

def _kde_contour(ax, x, y, color, label, levels=(0.5, 0.9), n=140):
    from scipy.stats import gaussian_kde
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), n),
                         np.linspace(y.min(), y.max(), n))
    dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ds = np.sort(kde(xy))[::-1]
    cum = np.cumsum(ds) / ds.sum()
    cuts = sorted(float(ds[np.searchsorted(cum, m)]) for m in levels)
    ax.contour(gx, gy, dens, levels=cuts, colors=color, linewidths=(0.8, 1.3))
    ax.plot([], [], color=color, lw=1.2, label=label)


def fig_posterior(results, out):
    """The margin width against the point spread that confounds it."""
    have = []
    for arm, color, label in (("independent", "C0", "independent operators"),
                              ("shared", "C3", "joint operator")):
        p = os.path.join(results, "mcmc_%s_samples.npz" % arm)
        if os.path.exists(p):
            have.append((np.load(p), color, label))
    if not have:
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.2), layout="constrained")
    xs, ys = [], []
    for d, color, label in have:
        x = np.asarray(d["log_sig_e"], float)
        y = np.asarray(d["log_w_bmode"], float)
        _kde_contour(ax, x, y, color, label)
        xs.append(x); ys.append(y)
    d0 = have[0][0]
    tx, ty = float(d0["truth_log_sig_e"]), float(d0["truth_log_w_bmode"])

    # the curve the B-mode likelihood alone cannot leave: the quadrature sum
    # through the truth.  The independent posterior is that curve.
    tot = np.hypot(np.exp(tx), np.exp(ty))
    s = np.linspace(0.02 * tot, 0.999 * tot, 300)
    ax.plot(np.log(s), np.log(np.sqrt(tot ** 2 - s ** 2)), ls="--", lw=0.8,
            color="0.55", label=r"$\sigma_e^2+w_B^2$ constant")

    # Perfect-model reference: the width is held, so its posterior is a
    # vertical slice.
    orc = _load(results, "main__oracle.json")
    if orc:
        s_ = orc["summary"]["sig_e"]
        ax.plot([np.log(s_["lo"]), np.log(s_["hi"])], [ty, ty], lw=2.6,
                color="C2", solid_capstyle="butt",
                label="perfect model, 90 percent")
    ax.plot(tx, ty, "k+", ms=9, mew=1.4, label="synthetic truth")

    allx = np.concatenate(xs); ally = np.concatenate(ys)
    x0, x1 = np.percentile(allx, [1.0, 99.8])
    y0, y1 = np.percentile(ally, [1.0, 99.8])
    ax.set_xlim(x0 - 0.08 * (x1 - x0), x1 + 0.08 * (x1 - x0))
    ax.set_ylim(min(y0, ty) - 0.15, max(y1, ty) + 0.15)
    ax.set_xlabel(r"$\log \sigma_e$   (margin width)")
    ax.set_ylabel(r"$\log w_B$   (B-mode point spread)")
    ax.set_title("Intrinsic width and B-mode blur", fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.savefig(os.path.join(out, "fig2_posterior.png"))
    plt.close(fig)


# --------------------------------------------------------------- figure 3 ---

ARMS = ("independent", "shared", "shared_calibrated", "oracle")
ARM_COLOR = {
    "independent": "C0",
    "shared": "C3",
    "shared_calibrated": "C1",
    "oracle": "C2",
}
ARM_LABEL = {
    "independent": "independent",
    "shared": "joint operator",
    "shared_calibrated": "joint operator, calibrated",
    "oracle": "perfect model",
}
CONDITIONS = (
    (Case("correct_without_swe", windows=("bmode", "ceus")),
     "B-mode + CEUS\ncorrect"),
    (Case("correct_with_swe"),
     "B-mode + SWE + CEUS\ncorrect"),
    (Case("wrong_without_swe", windows=("bmode", "ceus"),
          gen_gamma={"ceus": 1.15}),
     "B-mode + CEUS\n$w_C$ +15%"),
    (Case("wrong_with_swe", gen_gamma={"ceus": 1.15}),
     "B-mode + SWE + CEUS\n$w_C$ +15%"),
)
OBSERVATION_INDEX = {"bmode": 1, "swe": 2, "ceus": 3}


def _paired_data(case, repetition):
    """Use identical standardized noise for each shared observation."""
    generator = case.generator()
    y0 = generator.predict(generator.space.x0(generator.truth))
    sigma = generator.sigma()
    noise = np.empty_like(y0)
    for observation, slc in generator.window_slices().items():
        rng = np.random.default_rng(
            np.random.SeedSequence([20260915, repetition,
                                    OBSERVATION_INDEX[observation]]))
        noise[slc] = rng.normal(size=slc.stop - slc.start)
    return y0 + sigma * noise, sigma


def _repeated_inference(n_rep):
    """Posterior widths and biases under paired noise realizations."""
    records = {}
    for condition_index, (case, _) in enumerate(CONDITIONS):
        for arm in ARMS:
            widths, biases = [], []
            for repetition in range(n_rep):
                y, sigma = _paired_data(case, repetition)
                inversion = case.build(arm, sigma=sigma)
                mode, _ = SA.find_map(inversion, y, restarts=1)
                laplace = SA.laplace(inversion, y, mode)
                result = SU.summarize_gaussian(
                    inversion, laplace, ("sig_e",))["sig_e"]
                widths.append(result["width_post"])
                biases.append(100.0 * np.expm1(result["bias_log"]))
            records[(condition_index, arm)] = {
                "width": np.asarray(widths),
                "bias": np.asarray(biases),
            }
    return records


def _boxplot(axis, values, positions, colors):
    result = axis.boxplot(
        values, positions=positions, widths=0.18, patch_artist=True,
        showfliers=True, showmeans=True,
        medianprops={"linewidth": 1.3},
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markersize": 3.2},
        flierprops={"marker": "o", "markerfacecolor": "none",
                    "markersize": 2.2, "markeredgewidth": 0.7},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for i, (box, color) in enumerate(zip(result["boxes"], colors)):
        box.set_facecolor(color)
        box.set_alpha(0.30)
        box.set_edgecolor(color)
        box.set_linewidth(1.3)
        for line in result["whiskers"][2 * i:2 * i + 2]:
            line.set_color(color)
        for line in result["caps"][2 * i:2 * i + 2]:
            line.set_color(color)
        result["medians"][i].set_color(color)
        result["means"][i].set_markeredgecolor(color)
        if i < len(result["fliers"]):
            result["fliers"][i].set_markeredgecolor(color)


def fig_calibration(out, n_rep=40):
    """Uncertainty and bias across paired noise realizations."""
    records = _repeated_inference(n_rep)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.0),
                             layout="constrained")
    centers = np.arange(len(CONDITIONS), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(ARMS))
    positions, colors = [], []
    width_values, bias_values = [], []
    for condition_index in range(len(CONDITIONS)):
        for offset, arm in zip(offsets, ARMS):
            positions.append(centers[condition_index] + offset)
            colors.append(ARM_COLOR[arm])
            width_values.append(records[(condition_index, arm)]["width"])
            bias_values.append(records[(condition_index, arm)]["bias"])

    _boxplot(axes[0], width_values, positions, colors)
    _boxplot(axes[1], bias_values, positions, colors)

    labels = [label for _, label in CONDITIONS]
    for axis in axes:
        axis.set_xticks(centers)
        axis.set_xticklabels(labels, fontsize=6.0)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel(r"90% width of $\log\sigma_e$")
    axes[0].set_title("Uncertainty in lesion margin width")
    axes[1].axhline(0.0, color="0.25", linewidth=0.8)
    axes[1].set_ylabel(r"bias in $\sigma_e$ (%)")
    axes[1].set_title("Bias in lesion margin width")

    handles = [Patch(facecolor=ARM_COLOR[arm], label=ARM_LABEL[arm])
               for arm in ARMS]
    axes[0].legend(handles=handles, frameon=False, ncol=2,
                   loc="upper right", fontsize=6)
    fig.savefig(os.path.join(out, "fig3_uncertainty_bias.png"))
    plt.close(fig)


def make_all(results, out):
    os.makedirs(out, exist_ok=True)
    fig_fields(out)
    fig_posterior(results, out)
    fig_calibration(out, n_rep=40)
    print("figures written to", out)
