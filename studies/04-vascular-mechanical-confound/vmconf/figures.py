"""Figures for the stored CEUS, inference, and sensitivity results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path):
    with open(path) as f:
        return json.load(f)


def _seed_files(results, stem):
    files = sorted(Path(results).glob(f"{stem}_seed*.json"))
    if not files:
        raise FileNotFoundError(f"no {stem}_seed*.json files in {results}")
    return files


def ceus_response(results, out):
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    for path in _seed_files(results, "ceus_d30"):
        blob = _load(path)
        rows = sorted(blob["table"]["rows"], key=lambda row: row["s"])
        seed = blob["provenance"]["settings"]["seed"]
        ax.plot(
            [row["s"] for row in rows],
            [row["auc"] for row in rows],
            marker="o",
            markersize=3,
            label=f"seed {seed}",
        )
    ax.set(xlabel="radius scale, s", ylabel="CEUS AUC [a.u. s]")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(frameon=False)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def interval_comparison(results, out):
    blobs = [_load(path) for path in _seed_files(results, "inference_d30")]
    quantities = [("mu", "matrix elasticity, μ"), ("eta", "matrix viscosity, η")]
    models = [("swe", "SWE"), ("joint_shared", "coupled")]
    colors = ["#5b8db8", "#d07a45"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), constrained_layout=True)
    for ax, (quantity, title) in zip(axes, quantities):
        if all("posterior_by_repetition" in blob for blob in blobs):
            distributions = []
            seed_means = []
            for model, _ in models:
                per_seed = [
                    np.array([
                        100 * row[quantity]["width"] / blob["truth"][quantity]
                        for row in blob["posterior_by_repetition"][model]
                    ])
                    for blob in blobs
                ]
                distributions.append(np.concatenate(per_seed))
                seed_means.append([values.mean() for values in per_seed])
            boxes = ax.boxplot(
                distributions,
                positions=np.arange(len(models)),
                widths=0.58,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black"},
            )
            for box, color in zip(boxes["boxes"], colors):
                box.set(facecolor=color, alpha=0.55)
            offsets = np.linspace(-0.07, 0.07, len(blobs))
            for seed_idx, offset in enumerate(offsets):
                values = [seed_means[model_idx][seed_idx]
                          for model_idx in range(len(models))]
                ax.plot(np.arange(len(models)) + offset, values, color="#555555",
                        linewidth=0.7, alpha=0.7)
                ax.scatter(np.arange(len(models)) + offset, values, color=colors,
                           edgecolor="black", linewidth=0.35, s=18, zorder=3)
        else:
            # Backward-compatible view for result files produced before
            # per-repetition posterior summaries were retained.
            for x, ((model, _), color) in enumerate(zip(models, colors)):
                values = np.array([
                    blob["summary"][model][quantity]["width_pct"] for blob in blobs
                ])
                offsets = np.linspace(-0.07, 0.07, len(values))
                ax.bar(x, values.mean(), width=0.58, color=color, alpha=0.65)
                ax.scatter(x + offsets, values, color=color, edgecolor="black",
                           linewidth=0.4)
        ax.set_xticks(range(len(models)), [label for _, label in models])
        ax.set(title=title, ylabel="90% interval width [% of generating value]")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def coupling_sensitivity(results, out):
    blobs = [_load(path) for path in _seed_files(results, "inference_d30")]
    offsets = np.array([row["offset"] for row in blobs[0]["offset_sweep"]])
    bias = np.array(
        [[row["mu"]["bias_pct"] for row in blob["offset_sweep"]] for blob in blobs]
    )
    width = np.array(
        [[row["mu"]["width_pct"] for row in blob["offset_sweep"]] for blob in blobs]
    )
    coverage = np.array(
        [[row["mu"]["coverage"] for row in blob["offset_sweep"]] for blob in blobs]
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), constrained_layout=True)
    for ax, values, ylabel in zip(
        axes,
        (bias, width, coverage),
        ("bias in μ [%]", "90% interval width [%]", "coverage"),
    ):
        ax.plot(offsets, values.mean(axis=0), marker="o", color="#6b4c9a")
        ax.fill_between(
            offsets,
            values.min(axis=0),
            values.max(axis=0),
            color="#6b4c9a",
            alpha=0.18,
            linewidth=0,
        )
        ax.set(xlabel="relation offset in s", ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].axhline(0.9, color="black", linestyle="--", linewidth=1)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def make_all(results="results", out="figures"):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ceus_response(results, out_dir / "fig1_ceus_response.png")
    interval_comparison(results, out_dir / "fig2_interval_comparison.png")
    coupling_sensitivity(results, out_dir / "fig3_coupling_sensitivity.png")
