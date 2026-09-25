"""Summarize and plot the completed generating-parameter sweep."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results/true_value_sweep")
parser.add_argument("--out", default="figures")
parser.add_argument("--summary", default="results/true_value_sweep_summary.json")
parser.add_argument("--s-values", type=float, nargs="+", default=[0.70, 0.80, 0.90])
args = parser.parse_args()


def distribution_summary(values):
    values = np.asarray(values, float)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return dict(
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        minimum=float(values.min()),
        maximum=float(values.max()),
        mean=float(values.mean()),
    )


paths = sorted(Path(args.results).glob("inference_d*_seed*_mu*_eta*_s*.json"))
if not paths:
    raise FileNotFoundError(f"no sweep results found in {args.results}")

grouped = defaultdict(list)
selected_paths = []
for path in paths:
    with open(path) as handle:
        blob = json.load(handle)
    truth = blob["truth"]
    if not any(np.isclose(float(truth["s"]), value) for value in args.s_values):
        continue
    selected_paths.append(path)
    key = (float(truth["mu"]), float(truth["eta"]), float(truth["s"]))
    grouped[key].append(blob)

mu_values = sorted({key[0] for key in grouped})
eta_values = sorted({key[1] for key in grouped})
s_values = sorted({key[2] for key in grouped})
if not grouped or len(s_values) != len(args.s_values):
    raise ValueError("requested radius-scale values are not all present")
expected = len(mu_values) * len(eta_values) * len(s_values)
if len(grouped) != expected:
    raise ValueError(f"incomplete parameter grid: found {len(grouped)} of {expected} combinations")
if len({len(blobs) for blobs in grouped.values()}) != 1:
    raise ValueError("parameter combinations have different numbers of networks")
n_networks = len(next(iter(grouped.values())))

combinations = []
for (mu, eta, s), blobs in sorted(grouped.items()):
    row = dict(mu_true=mu, eta_true=eta, s_true=s, metrics={})
    for quantity in ("mu", "eta"):
        reductions = []
        joint_coverage = []
        swe_coverage = []
        joint_bias = []
        for blob in blobs:
            swe = blob["summary"]["swe"][quantity]
            joint = blob["summary"]["joint_shared"][quantity]
            reductions.append(
                100 * (swe["width_pct"] - joint["width_pct"]) / swe["width_pct"]
            )
            joint_coverage.append(joint["coverage"])
            swe_coverage.append(swe["coverage"])
            joint_bias.append(joint["bias_pct"])
        row["metrics"][quantity] = dict(
            interval_width_reduction_pct=float(np.mean(reductions)),
            joint_coverage=float(np.mean(joint_coverage)),
            swe_coverage=float(np.mean(swe_coverage)),
            joint_bias_pct=float(np.mean(joint_bias)),
        )
    combinations.append(row)

overall = {}
for quantity in ("mu", "eta"):
    overall[quantity] = {
        metric: distribution_summary([
            row["metrics"][quantity][metric] for row in combinations
        ])
        for metric in (
            "interval_width_reduction_pct",
            "joint_coverage",
            "swe_coverage",
            "joint_bias_pct",
        )
    }

summary = dict(
    source_files=sorted(path.name for path in selected_paths),
    n_files=sum(len(blobs) for blobs in grouped.values()),
    n_networks=n_networks,
    repetitions_per_file=int(next(iter(grouped.values()))[0]["provenance"]["settings"]["reps"]),
    grid=dict(mu=mu_values, eta=eta_values, s=s_values),
    combinations=combinations,
    overall=overall,
)
summary_path = Path(args.summary)
summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_path, "w") as handle:
    json.dump(summary, handle, indent=1)


def matrix(quantity, metric, s):
    lookup = {
        (row["mu_true"], row["eta_true"], row["s_true"]): row
        for row in combinations
    }
    return np.array([
        [lookup[(mu, eta, s)]["metrics"][quantity][metric] for mu in mu_values]
        for eta in eta_values
    ])


def text_color(cmap, norm, value):
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "black" if luminance > 0.58 else "white"


def reduction_label(value):
    if abs(value) < 0.05:
        value = 0.0
    return f"{value:.1f}"


out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
quantities = (("mu", "matrix elasticity, μ"), ("eta", "matrix viscosity, η"))

# Precision gain across the generating-parameter grid.
fig, axes = plt.subplots(2, len(s_values), figsize=(12.2, 6.0), constrained_layout=True)
all_reductions = np.concatenate([
    matrix(quantity, "interval_width_reduction_pct", s).ravel()
    for quantity, _ in quantities
    for s in s_values
])
norm = TwoSlopeNorm(vmin=min(float(all_reductions.min()), -0.25),
                    vcenter=0.0, vmax=float(all_reductions.max()))
cmap = plt.get_cmap("RdBu")
for row_index, (quantity, row_label) in enumerate(quantities):
    for column_index, s in enumerate(s_values):
        ax = axes[row_index, column_index]
        values = matrix(quantity, "interval_width_reduction_pct", s)
        image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        for eta_index in range(len(eta_values)):
            for mu_index in range(len(mu_values)):
                value = values[eta_index, mu_index]
                ax.text(mu_index, eta_index, reduction_label(value), ha="center", va="center",
                        fontsize=8, color=text_color(cmap, norm, value))
        ax.set_xticks(range(len(mu_values)), [f"{value / 1000:g}" for value in mu_values])
        ax.set_yticks(range(len(eta_values)), [f"{value:g}" for value in eta_values])
        if row_index == 0:
            ax.set_title(f"s = {s:g}")
        if column_index == 0:
            ax.set_ylabel(f"{row_label}\ngenerating η [Pa s]")
        if row_index == len(quantities) - 1:
            ax.set_xlabel("generating μ [kPa]")
colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
colorbar.set_label("interval-width reduction [%]")
fig.suptitle("Precision gain from coupled SWE-CEUS inference", fontsize=13)
fig.savefig(out_dir / "fig4_sweep_precision.png", dpi=180)
plt.close(fig)

# Coverage of the coupled intervals across the same grid.
fig, axes = plt.subplots(2, len(s_values), figsize=(12.2, 6.0), constrained_layout=True)
norm = Normalize(vmin=0.85, vmax=1.0)
cmap = plt.get_cmap("RdYlGn")
for row_index, (quantity, row_label) in enumerate(quantities):
    for column_index, s in enumerate(s_values):
        ax = axes[row_index, column_index]
        values = matrix(quantity, "joint_coverage", s)
        image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        for eta_index in range(len(eta_values)):
            for mu_index in range(len(mu_values)):
                value = values[eta_index, mu_index]
                ax.text(mu_index, eta_index, f"{value:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color(cmap, norm, value))
        ax.set_xticks(range(len(mu_values)), [f"{value / 1000:g}" for value in mu_values])
        ax.set_yticks(range(len(eta_values)), [f"{value:g}" for value in eta_values])
        if row_index == 0:
            ax.set_title(f"s = {s:g}")
        if column_index == 0:
            ax.set_ylabel(f"{row_label}\ngenerating η [Pa s]")
        if row_index == len(quantities) - 1:
            ax.set_xlabel("generating μ [kPa]")
colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
colorbar.set_label("empirical coverage")
colorbar.ax.axhline(norm(0.9), color="black", linewidth=1)
fig.suptitle("Coverage of coupled 90% intervals", fontsize=13)
fig.savefig(out_dir / "fig5_sweep_coverage.png", dpi=180)
plt.close(fig)

print(f"summarized {sum(len(blobs) for blobs in grouped.values())} files "
      f"over {len(combinations)} parameter combinations")
print(f"written {summary_path}")
print(f"written {out_dir / 'fig4_sweep_precision.png'}")
print(f"written {out_dir / 'fig5_sweep_coverage.png'}")
