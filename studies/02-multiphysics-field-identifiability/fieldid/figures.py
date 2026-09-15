"""Figures.  Titles name what is plotted; they do not state conclusions."""
from __future__ import annotations
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .acquisition import Acquisition
from .model import Inversion
from .models import mechanics as ME, transport as TP
from .params import TRUTH
from .compat import trapezoid

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
    acq = Acquisition()
    acq.relax.enabled = True
    inv = Inversion(free=("mu",), windows=("swe",), acq=acq)
    p = inv.truth_physical()
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.1), layout="constrained")

    x, t = acq.swe.positions(), acq.swe.times()
    u = ME.shear_displacement(x, t, p, acq.swe.f_band, acq.swe.geometry)
    m = np.abs(u).max()
    im = ax[0].imshow(u * 1e6, aspect="auto", origin="lower", cmap="RdBu_r",
                      vmin=-m * 1e6, vmax=m * 1e6,
                      extent=[t[0] * 1e3, t[-1] * 1e3, x[0] * 1e3, x[-1] * 1e3])
    ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("lateral offset (mm)")
    ax[0].set_title("shear window: $u_z(x,t)$")
    plt.colorbar(im, ax=ax[0], label=r"displacement ($\mu$m)", pad=0.03)

    tc = acq.ceus.times()
    b = TP.occupancy(acq.ceus.positions(), tc, p)
    for row in b:
        ax[1].plot(tc, row, lw=0.6, color="0.5", alpha=0.8)
    ax[1].plot(tc, b.mean(axis=0), lw=1.4, color="C0", label="voxel mean")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("occupancy (arb.)")
    ax[1].set_title("contrast window: $b=\\phi c$")
    ax[1].legend(frameon=False)

    tr = acq.relax.times()
    ur = ME.relax_displacement(tr, p)
    u0, ui = ME.relax_limits(p)
    ax[2].plot(tr, ur * 1e6, lw=1.2, color="C3")
    ax[2].axhline(u0 * 1e6, ls=":", lw=0.8, color="0.4")
    ax[2].axhline(ui * 1e6, ls=":", lw=0.8, color="0.4")
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel(r"compaction ($\mu$m)")
    ax[2].set_title("relaxation window: $u(t)$")
    fig.savefig(os.path.join(out, "fig1_fields.png"))
    plt.close(fig)


# --------------------------------------------------------------- figure 2 ---

def fig_spectra(results, out):
    want = [("E1_swe", "SWE alone"), ("E2_ceus_Acal", "CEUS alone"),
            ("E3_joint_independent", "joint, independent"),
            ("E8_relax_independent", "joint + relaxation, independent"),
            ("E8_relax_constitutive_Cvcal",
             "joint + relaxation, $S_v=\\phi C_v$, $C_v$ calibrated")]
    rows = [(lbl, _load(results, n + ".json")) for n, lbl in want]
    rows = [(l, r) for l, r in rows if r]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for i, (lbl, r) in enumerate(rows):
        c = np.array(r["contraction"])
        ax.plot(np.arange(1, len(c) + 1), c, "o-", ms=3, lw=1,
                label="%s (%d par.)" % (lbl, len(c)), color="C%d" % i)
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("direction, least informed first")
    ax.set_ylabel(r"$1/\sqrt{1+\lambda}$")
    ax.set_title("posterior width relative to prior, per eigendirection")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(os.path.join(out, "fig2_spectra.png"))
    plt.close(fig)


# --------------------------------------------------------------- figure 3 ---

def fig_contraction(results, out):
    """Two panels, kept apart on purpose: what the tested reduced model gives,
    and what a contingency observation would add.  A single grouped bar chart
    put most of the visual weight on the contingency and made an extension of
    the acquisition look like the finding."""
    groups = [
        ("the tested reduced model",
         [("E3_joint_independent", "independent priors"),
          ("E5_ceus_only_network", "network prior, contrast alone"),
          ("E5_joint_network", "network prior, both windows")]),
        ("contingency: a relaxation observation added",
         [("E8_relax_independent_matched", "independent, storage prior matched"),
          ("E8_relax_constitutive_Cvcal", "$S_v=\\phi C_v$, $C_v$ calibrated"),
          ("E8_relax_constitutive_Cvcal_norm", "as above, normalized contrast")]),
    ]
    quants = ["k", "S_v", "M", "phi", "mu", "eta_s", "D", "vmag"]
    labels = [r"$k$", r"$S_v$", r"$M$", r"$\phi$", r"$\mu$", r"$\eta_s$",
              r"$D$", r"$|v|$"]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 2.8), sharey=True,
                           layout="constrained")
    for a, (title, want) in zip(ax, groups):
        rows = [(lbl, _load(results, n + ".json")) for n, lbl in want]
        rows = [(l, r) for l, r in rows if r]
        if not rows:
            continue
        w = 0.8 / len(rows)
        xs = np.arange(len(quants))
        for i, (lbl, r) in enumerate(rows):
            v = [r["summary"].get(q, {}).get("contraction", np.nan) for q in quants]
            a.bar(xs + i * w - 0.4 + w / 2, v, width=w, label=lbl, color="C%d" % i)
        a.axhline(1.0, color="0.4", lw=0.8, ls="--")
        a.set_xticks(xs); a.set_xticklabels(labels)
        a.set_yscale("log")
        a.set_title(title, fontsize=8)
        a.legend(frameon=False, fontsize=6, loc="lower left")
    ax[0].set_ylabel("posterior width / prior width")
    fig.savefig(os.path.join(out, "fig3_contraction.png"))
    plt.close(fig)


# --------------------------------------------------------------- figure 4 ---

def fig_amplitude(results, out):
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.2), layout="constrained")
    acq = Acquisition()
    inv = Inversion(free=("phi",), windows=("ceus",), acq=acq)
    p = dict(inv.truth_physical())
    t = acq.ceus.times()
    r = acq.ceus.positions()[13:14]
    for f, c in ((0.5, "C0"), (1.0, "C1"), (2.0, "C2")):
        q = dict(p); q["phi"] = p["phi"] * f
        b = TP.occupancy(r, t, q)[0]
        ax[0].plot(t, b, color=c, lw=1.1, label=r"$\phi\times%.1f$" % f)
        ax[1].plot(t, b / trapezoid(b, t), color=c, lw=1.1)
    ax[0].set_title("absolute occupancy"); ax[1].set_title("unit-area curves")
    for a in ax:
        a.set_xlabel("time (s)")
    ax[0].set_ylabel("occupancy (arb.)"); ax[1].set_ylabel("density (1/s)")
    ax[0].legend(frameon=False)
    fig.savefig(os.path.join(out, "fig4_amplitude.png"))
    plt.close(fig)


# --------------------------------------------------------------- figure 5 ---

def fig_ensemble(results, out, csv_path=None):
    csv_path = csv_path or os.path.join(results, "network_ensemble.csv")
    if not os.path.exists(csv_path):
        return
    import csv as _csv
    cols = {}
    with open(csv_path) as fh:
        for row in _csv.DictReader(fh):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    cols = {k: np.array(v) for k, v in cols.items()}
    from .cli import CENSOR
    keep = np.ones(len(cols["phi"]), bool)
    for n, (lo, hi) in CENSOR.items():
        keep &= (cols[n] > lo) & (cols[n] < hi)
    pairs = [("k", "permeability (m$^2$)"), ("D", "dispersion (m$^2$/s)"),
             ("vmag", "drift (m/s)")]
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.2), layout="constrained")
    for a, (q, lbl) in zip(ax, pairs):
        a.loglog(cols["phi"][keep], cols[q][keep], "o", ms=3.5, color="C0",
                 alpha=0.85, label="retained")
        a.loglog(cols["phi"][~keep], cols[q][~keep], "x", ms=4, color="0.6",
                 label="on a fit bound")
        r = np.corrcoef(np.log(cols["phi"][keep]), np.log(cols[q][keep]))[0, 1]
        a.set_xlabel(r"$\phi$"); a.set_ylabel(lbl)
        a.set_title("log correlation %+.2f, retained" % r)
    ax[1].legend(frameon=False, fontsize=6, loc="lower left")
    fig.savefig(os.path.join(out, "fig5_ensemble.png"))
    plt.close(fig)


def fig_profiles(results, out):
    d = _load(results, "profiles.json")
    if not d:
        return
    want = [("E3_joint_independent", "reduced model, independent"),
            ("E8_relax_independent_matched",
             "+ relaxation, independent, matched"),
            ("E8_relax_constitutive_Cvcal",
             "+ relaxation, $S_v=\\phi C_v$, $C_v$ calibrated")]
    levels = [("imaging", "imaging likelihood alone"),
              ("imaging_plus_auxiliary", "+ auxiliary constraints"),
              ("posterior", "+ every prior")]
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 2.6), sharex=True, sharey=True,
                           layout="constrained")
    for a, (lv, ttl) in zip(ax, levels):
        for i, (n, lbl) in enumerate(want):
            if n not in d or "k" not in d[n] or ("delta_" + lv) not in d[n]["k"]:
                continue
            g = np.array(d[n]["k"]["grid"])
            # in the imaging panel the three curves coincide, which is the
            # result; decreasing marker size and width make the overlap visible
            # rather than leaving only the last one drawn
            a.plot(g, d[n]["k"]["delta_" + lv], "o-", ms=6.5 - 2 * i,
                   lw=2.4 - 0.7 * i, color="C%d" % i, label=lbl, alpha=0.9)
        a.axhline(2.71, color="0.5", lw=0.8, ls="--")
        a.set_yscale("symlog", linthresh=1.0)
        a.set_ylim(-0.3, 2e3)
        a.set_title(ttl, fontsize=8)
    ax[0].set_ylabel(r"$\Delta(-2\log)$")
    ax[0].legend(frameon=False, fontsize=6, loc="upper center")
    fig.supxlabel(r"$\log k$, in prior standard deviations from the truth",
                  fontsize=8)
    ax[0].legend(frameon=False, fontsize=6, loc="upper center")
    fig.savefig(os.path.join(out, "fig6_profiles.png"))
    plt.close(fig)


# --------------------------------------------------------------- figure 7 ---

def _kde_contour(ax, x, y, color, label, levels=(0.5, 0.9), n=120):
    """Highest-density contours of a sample, at the stated probability mass."""
    from scipy.stats import gaussian_kde
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    xs = np.linspace(x.min(), x.max(), n)
    ys = np.linspace(y.min(), y.max(), n)
    gx, gy = np.meshgrid(xs, ys)
    dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    # the level enclosing a given mass is the density quantile of the sample
    ds = np.sort(kde(xy))[::-1]
    cum = np.cumsum(ds) / ds.sum()
    cuts = sorted(float(ds[np.searchsorted(cum, m)]) for m in levels)
    ax.contour(gx, gy, dens, levels=cuts, colors=color, linewidths=(0.8, 1.3))
    ax.plot([], [], color=color, lw=1.2, label=label)


def fig_storage_permeability(results, out):
    """log k against log S_v for the matched independent and coupled arms.

    The mechanism is a two-parameter statement and is drawn as one: the
    relaxation window fixes a combination, which is a ridge in this plane, and
    the coupling adds information about the storage, which cuts across it.  In
    the coupled arm the storage is not a free parameter and is derived from the
    samples as ``S_v = phi C_v``, which is the same relation the model imposes.
    """
    want = [("E8_relax_independent_matched", "C0",
             "independent, storage marginal matched"),
            ("E8_relax_constitutive_Cvcal", "C3",
             "$S_v=\\phi C_v$, $C_v$ calibrated")]
    have = []
    for name, color, label in want:
        p = os.path.join(results, "mcmc_%s_samples.npz" % name)
        if os.path.exists(p):
            have.append((np.load(p, allow_pickle=True), color, label))
    if len(have) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0), layout="constrained")
    xs, ys = [], []
    for d, color, label in have:
        x = np.asarray(d["log_k"], float)
        y = np.asarray(d["log_Sv"], float)
        _kde_contour(ax, x, y, color, label)
        xs.append(x); ys.append(y)
    d0 = have[0][0]
    tx, ty = float(d0["truth_log_k"]), float(d0["truth_log_Sv"])
    # limits from the bulk, not the extremes: a few far draws would otherwise
    # set the frame and leave the contours in the middle of empty axes
    ax_ = np.concatenate(xs); ay_ = np.concatenate(ys)
    x0, x1 = np.percentile(ax_, [0.2, 99.8])
    y0, y1 = np.percentile(ay_, [0.2, 99.8])
    # the relaxation window determines k / S_v, so a line of unit slope through
    # the truth is the combination it fixes; the ridges lie along it
    g = np.array([x0, x1])
    ax.plot(g, ty + (g - tx), ls="--", lw=0.7, color="0.55",
            label=r"$k/S_v$ constant")
    ax.plot(tx, ty, "k+", ms=9, mew=1.4, label="synthetic truth")
    ax.set_xlim(x0 - 0.05 * (x1 - x0), x1 + 0.05 * (x1 - x0))
    ax.set_ylim(y0 - 0.05 * (y1 - y0), y1 + 0.05 * (y1 - y0))
    ax.set_xlabel(r"$\log k$")
    ax.set_ylabel(r"$\log S_v$")
    ax.set_title("posterior samples, 50 and 90 percent contours", fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.savefig(os.path.join(out, "fig7_storage_permeability.png"))
    plt.close(fig)


def make_all(results, out):
    os.makedirs(out, exist_ok=True)
    fig_fields(out)
    fig_spectra(results, out)
    fig_contraction(results, out)
    fig_amplitude(results, out)
    fig_ensemble(results, out)
    fig_profiles(results, out)
    fig_storage_permeability(results, out)
    print("figures written to", out)
