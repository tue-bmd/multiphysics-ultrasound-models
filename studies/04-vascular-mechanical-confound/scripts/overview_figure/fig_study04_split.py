"""The two overview figures of RESULTS.md, from the panel functions of fig_study04.py.

    figures/fig6_model_responses.pdf/.png             A computational model | B shear-wave response | C contrast-transport response
    figures/fig7_posterior_misspecification.pdf/.png  A joint posterior      | B displaced relation (coverage and bias vs offset)

Reads results/overview_figure/fig_study04_data.npz and net_seed<k>.npz; needs
numpy, scipy and matplotlib only.

    python scripts/overview_figure/fig_study04_split.py [--dpi 400] [--out figures]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

import fig_study04 as F


def build_ac(out, dpi):
    F.style()
    fig = plt.figure(figsize=(12.9 / 2.54, 5.6 / 2.54))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.08, 1.10, 0.95],
                           left=0.025, right=0.985, top=0.88, bottom=0.20, wspace=0.50)
    axA = plt.subplot(gs[0]); F.panel_a(axA); F.panel_label(axA, "A", x=0.02, y=1.0)
    axB = plt.subplot(gs[1]); axC = plt.subplot(gs[2]); F.panel_b(axB, axC)
    F.panel_label(axB, "B", x=-0.36, y=1.04); F.panel_label(axC, "C", x=-0.36, y=1.04)
    for ax in (axB, axC):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "fig6_model_responses.%s" % ext), dpi=dpi)
    plt.close(fig)


def build_de(out, dpi):
    F.style()
    fig = plt.figure(figsize=(11.6 / 2.54, 5.6 / 2.54))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0],
                           left=0.125, right=0.905, top=0.90, bottom=0.20, wspace=0.55)
    axD = plt.subplot(gs[0]); F.panel_c(axD); F.panel_label(axD, "A", x=-0.27, y=1.04)
    axD.spines["top"].set_visible(False); axD.spines["right"].set_visible(False)
    axE = plt.subplot(gs[1]); F.panel_d(axE); F.panel_label(axE, "B", x=-0.26, y=1.04)
    # the twin axes are already labelled and colour-coded; the legend only crowds the panel
    axE.get_legend().remove()
    # no panel titles: the caption explains the two panels
    axD.set_title(""); axE.set_title("")
    axE.set_xlabel("offset of AUC\u2013$s$ relationship", labelpad=1)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "fig7_posterior_misspecification.%s" % ext), dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--out", default=str(F.FIGS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    build_ac(a.out, a.dpi)
    build_de(a.out, a.dpi)
    print("written fig6_model_responses.{pdf,png} and fig7_posterior_misspecification.{pdf,png} in", a.out)
