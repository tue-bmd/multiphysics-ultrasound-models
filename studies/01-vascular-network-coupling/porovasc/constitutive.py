"""Which structural quantities do the estimated v and D track?

    python -m porovasc.constitutive runs/baseline
    python -m porovasc.constitutive runs/baseline runs/lesion_tort16

Each sampling volume carries both the structure measured directly on the network
(vascular volume fraction, face-to-face network permeability, permeability
weighted diameter, mean squared tortuosity) and the contrast estimates made on
the same volume.  This module compares them across volumes, so that "the
estimator reports a dispersion" can be turned into "the estimator reports
something that varies with these structural quantities and not with those".

That is an association analysis.  It is the only thing this module claims.

Two candidate hypotheses, neither assumed
-----------------------------------------
Both of the following are relations one might want to hold.  They are stated
here as hypotheses to be examined, because the model does not establish either,
and earlier versions of this file asserted both.

**H1, the single-compartment Darcy closure.**  If the bed behaved as one Darcy
continuum, the superficial flux would be q = (k / eta) |grad p|, and a tracer
travelling with the blood would move at the intrinsic velocity

    v = q / phi = k |grad p| / (eta phi).                                   (H1)

If H1 held with a gradient that varied little between sampling volumes, then v
would track k / phi across volumes, and that is the correlation reported below.

H1 is doubtful in this model, for a reason that is structural rather than
numerical.  Within one millimetre-scale volume the arterial and venous branches
carry blood in opposing directions, so the vector-averaged flux and the scalar
throughput are different quantities and only the first enters a Darcy closure.
`docs/homogenisation_diagnostic.py` measures that directly, as the flow
coherence chi = || sum_e Q_e L_e t_e || / sum_e |Q_e| L_e, together with a
single-compartment Darcy speed and the flow-weighted vessel speed on the same
support.  Read that diagnostic before treating a correlation here as support
for H1: a pressure gradient fitted across arteries and veins together is not the
gradient H1 refers to.

Nothing in this package imposes H1.  The transport model propagates the bolus
through the explicit segments; it contains no Darcy closure and no macroscopic
pressure gradient.

**H2, dispersion proportional to velocity.**  If mechanical dispersion in the
bed followed

    D = alpha_L v                                                           (H2)

with alpha_L a length set by the geometry rather than by the flow, then D would
carry no information that v does not already carry, and alpha_L would be the
structural quantity.  H2 is testable here in two ways: alpha_L should vary less
across volumes than D does, and an intervention acting through the velocity
should move v and D together and leave alpha_L alone.  Both are reported.

What the compression relaxation does and does not establish
-----------------------------------------------------------
The relaxation of the compressed region has an equilibrium amplitude
proportional to the vascular volume fraction of that region.  This is not an
independent measurement of phi: it follows from the compliance law the model
imposes, C_i = V_i / (H + K_wall), under which the volume expelled at
equilibrium is proportional to the lumen volume present.  `physics.drainage`
reproduces that proportionality to a relative 1e-4, which is an implementation
check of the law, not evidence that phi is recoverable from a relaxation
measurement.

Whether an ultrasound-observable relaxation identifies phi in tissue depends on
the true compliance law, on the geometry of the compression, and on what the
measurement actually resolves.  None of those is addressed here.  Treat the
proportionality as a conditional model result.

What this module reports
------------------------
Per sampling volume: the estimated v and D (median over the input voxels of that
volume, for one estimator setting), the derived alpha_L = D / v, and the
structural quantities.  Then rank correlations of each estimate against each
structural quantity across volumes, and for an arm the paired effect on v, on D
and on alpha_L separately.

Rank correlations are used because any relation of this kind would be monotone
rather than linear, the scatter is not Gaussian, and there are at most a few
tens of volumes.  A correlation here is consistent with a hypothesis; with this
many volumes and no control of the confounders it does not establish one.
"""
from __future__ import annotations
import argparse
import itertools
import os
import sys

import numpy as np

from .analyse_paired import load, _est, _valid, _med, _get, hierarchy
from .physics.flow import ETA

# structural quantities measured directly on the network, and what each is
STRUCTURE = {
    "phi": (("permeability", "support", "phi"), "vascular volume fraction of the cube"),
    "k_network_mean": (("permeability", "k_network_mean"), "face-to-face Darcy permeability"),
    "d_perm": (("permeability", "support", "d_perm"), "permeability weighted diameter"),
    "T2": (("permeability", "support", "T2"), "mean squared tortuosity"),
    "phi_region": (("compression", "phi_region"), "vascular volume fraction of the sphere"),
    "k_rc_index": (("compression", "k_rc_index"), "RC index of the compression relaxation"),
}

#: how to read each row of the correlation table
TARGET_NOTE = {
    "k_over_phi": "the combination H1 predicts v would follow, if H1 held",
    "phi": "vascular volume fraction, cube support",
    "k_network_mean": "network permeability under an imposed gradient",
    "d_perm": "permeability weighted diameter",
    "T2": "mean squared tortuosity",
    "phi_region": "vascular volume fraction, compression support",
    "k_rc_index": "RC index of the relaxation, itself a function of the assumed compliance",
}


def spearman(a, b):
    """Rank correlation, without a scipy dependency for this one number."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return np.nan, int(ok.sum())
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return (float((ra * rb).sum() / den) if den > 0 else np.nan), int(ok.sum())


def volume_rows(recs, arm, kernel, R_mm, tf="new", rule="front20"):
    """One row per sampling volume of one arm: the estimates reduced over the
    input voxels of that volume, the derived dispersivity, and the structure."""
    rows = []
    for r in recs:
        if r.get("arm") != arm or "error" in r:
            continue
        vs, Ds = [], []
        for i in range(len(r.get("voxels", []))):
            e = _est(r, i, kernel, R_mm, tf, rule)
            if _valid(e):
                vs.append(e["v"]); Ds.append(e["D"])
        v, D = _med(vs), _med(Ds)
        row = dict(seed=r["seed"], rve=r["rve"], role=r.get("role"), n_valid=len(vs),
                   v=v, D=D, alpha_L=(D / v if (v and D and v > 0) else None))
        for name, (path, _) in STRUCTURE.items():
            row[name] = _get(r, path)
        # the combination H1 would predict for v, formed only so that the
        # hypothesis can be examined; nothing in the model imposes it
        phi = row.get("phi"); k = row.get("k_network_mean")
        row["k_over_phi"] = (k / phi) if (k and phi and phi > 0) else None
        row["k_over_eta_phi"] = (k / (ETA * phi)) if (k and phi and phi > 0) else None
        rows.append(row)
    return rows


def report_correlations(rows, label):
    n = sum(r["v"] is not None for r in rows)
    print("\n%s: %d volumes with an estimate" % (label, n))
    print("  rank correlation across volumes, association only; a correlation is")
    print("  consistent with a hypothesis and does not establish one")
    print("    %-18s %8s %8s %8s   %s"
          % ("structural quantity", "vs v", "vs D", "vs D/v", "reading"))
    targets = ["k_over_phi", "phi", "k_network_mean", "d_perm", "T2",
               "phi_region", "k_rc_index"]
    out = {}
    for name in targets:
        line = "    %-18s" % name
        vals = {}
        for est in ("v", "D", "alpha_L"):
            rho, nn = spearman([r.get(name) for r in rows], [r.get(est) for r in rows])
            vals[est] = rho
            line += " %8s" % ("n/a" if not np.isfinite(rho) else "%+.2f" % rho)
        out[name] = vals
        print(line + "   " + TARGET_NOTE.get(name, ""))
    al = [r["alpha_L"] for r in rows if r["alpha_L"]]
    Ds = [r["D"] for r in rows if r["D"]]
    vs = [r["v"] for r in rows if r["v"]]
    if al:
        al = np.array(al) * 1e3
        print("  dispersivity D/v: median %.2f mm, 5-95%% [%.2f, %.2f], spread factor %.1f"
              % (np.median(al), *np.percentile(al, [5, 95]), max(al) / min(al)))
    if al is not None and len(Ds) > 1 and len(vs) > 1:
        # H2 predicts alpha_L varies less across volumes than D does
        def _sd_log(x):
            x = np.asarray([v for v in x if v and v > 0], float)
            return float(np.std(np.log10(x), ddof=1)) if len(x) > 1 else float("nan")
        print("  H2 check: spread in log10 across volumes, D %.2f, v %.2f, D/v %.2f"
              % (_sd_log(Ds), _sd_log(vs), _sd_log(al)))
        print("            H2 would put D/v below both; it does not by itself confirm H2")
    return out


def report_effects(base_rows, arm_rows, label):
    """For a lesion or sweep arm: the effect on v, on D and on the dispersivity
    separately.  Under H2 an intervention acting through the velocity would move
    v and D alike and leave D/v alone; a change in D/v is evidence against H2 or
    evidence that the intervention did not act through the velocity, and this
    design cannot separate those two."""
    base = {(r["seed"], r["rve"]): r for r in base_rows}
    print("\n%s: paired effect, log10(arm / baseline), over volumes" % label)
    for role in sorted({r.get("role") for r in arm_rows}):
        sel = [r for r in arm_rows if r.get("role") == role]
        print("  role %s" % role)
        for est in ("v", "D", "alpha_L"):
            rows = []
            for a in sel:
                b = base.get((a["seed"], a["rve"]))
                if not b:
                    continue
                x, y = a.get(est), b.get(est)
                rows.append(dict(seed=a["seed"], rve=a["rve"],
                                 val=(float(np.log10(x / y)) if (x and y and x > 0 and y > 0) else None)))
            h = hierarchy(rows, boot=0)
            print("    %-8s median %+0.3f  glands %d  range [%+0.3f, %+0.3f]"
                  % (est, h["median"] if h["n_glands"] else float("nan"), h["n_glands"],
                     h["min"] if h["n_glands"] else float("nan"),
                     h["max"] if h["n_glands"] else float("nan")))


def preamble():
    print("  hypotheses under examination, neither imposed by the model:")
    print("    H1  v = k |grad p| / (eta phi), the single-compartment Darcy closure.")
    print("        Doubtful here: arterial and venous branches oppose within one")
    print("        volume, so a gradient fitted across both is not the gradient H1")
    print("        refers to.  See docs/homogenisation_diagnostic.py.")
    print("    H2  D = alpha_L v, dispersion proportional to velocity through a")
    print("        geometric length.")
    print("  the compression relaxation amplitude is proportional to phi by the")
    print("  imposed compliance law C_i = V_i / (H + K_wall); that is a conditional")
    print("  model result, not a measurement of phi.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--kernel", default="plug")
    ap.add_argument("--radius", type=float, default=1.0, help="shell radius in mm")
    ap.add_argument("--tf", default="new")
    ap.add_argument("--rule", default="front20")
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args(argv)
    for folder in a.folders:
        man, recs, errors = load(folder, partial=a.partial)
        arms = sorted({r.get("arm") for r in recs if r.get("arm") != "baseline"})
        print("=" * 78)
        print("%s  (%s, kernel %s, R = %.2f mm, %s/%s)"
              % (folder, man.get("kind"), a.kernel, a.radius, a.tf, a.rule))
        preamble()
        base = volume_rows(recs, "baseline", a.kernel, a.radius, a.tf, a.rule)
        report_correlations(base, "baseline")
        for arm in arms:
            arm_rows = volume_rows(recs, arm, a.kernel, a.radius, a.tf, a.rule)
            report_correlations(arm_rows, arm)
            report_effects(base, arm_rows, arm)


if __name__ == "__main__":
    main()
