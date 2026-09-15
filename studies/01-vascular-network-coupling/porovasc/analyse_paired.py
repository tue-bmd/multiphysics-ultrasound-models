"""Hierarchical analysis of a paired study.

    python -m porovasc.analyse_paired runs/lesion_tort16
    python -m porovasc.analyse_paired runs/baseline

The experimental unit of an intervention is the gland (the seed), not the
voxel: voxels are nested in sampling volumes, which are nested in glands, and
an intervention acts on the whole gland.  Every quantity is therefore reduced in
the same three steps, by one function, and reported at the top level without
pooling across levels:

  1. within a sampling volume: the median over its valid input voxels (contrast
     kinetics only; volume-level quantities are already one number per volume);
  2. within a gland: the median over its sampling volumes (for a lesion study,
     the lesion volume and the reference volume are kept apart);
  3. across glands: every gland-level value is listed, with the median, the
     range and a percentile bootstrap interval over glands.  No p-values are
     computed: with five glands no rank test can reach the usual thresholds,
     and the list of gland effects says more than a threshold.

Effects are log10 ratios of arm over baseline for strictly positive quantities
and differences for shares and counts.  They are formed only from pairs in
which both fits are valid (status ok, not at a bound); the number of such pairs
and every status transition are reported alongside, so that a change in what
could be estimated cannot hide behind a change in the estimate.  The contrast
kinetic effects are also split by whether the causality rule accepted the same
shell directions in both conditions, which is where discrete jumps of the
estimator come from.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os

import numpy as np

from .study import SCHEMA
from . import validate as VAL

# volume-level quantities: name -> (path, kind); kind "log" for a log ratio,
# "diff" for a difference
VOLUME_KEYS = {
    "tau_rc": (("compression", "tau_rc"), "log"),
    "phi_region": (("compression", "phi_region"), "log"),
    "k_rc_index": (("compression", "k_rc_index"), "log"),
    "c_k_rc": (("compression", "c_k_rc"), "log"),
    "k_bundle_straight_sphere": (("compression", "k_bundle_straight"), "log"),
    "k_bundle_tort_sphere": (("compression", "k_bundle_tort"), "log"),
    "T2_sphere": (("compression", "support", "T2"), "log"),
    "d_perm_sphere": (("compression", "support", "d_perm"), "log"),
    "k_network_mean": (("permeability", "k_network_mean"), "log"),
    "k_network_x": (("permeability", "k_network_face", 0), "log"),
    "k_network_y": (("permeability", "k_network_face", 1), "log"),
    "k_network_z": (("permeability", "k_network_face", 2), "log"),
    "n_blocked": (("permeability", "n_blocked"), "diff"),
    "c_k_network": (("permeability", "c_k_network"), "log"),
    "k_bundle_straight_cube": (("permeability", "k_bundle_straight"), "log"),
    "k_bundle_tort_cube": (("permeability", "k_bundle_tort"), "log"),
    "phi_cube": (("permeability", "support", "phi"), "log"),
    "d_perm_cube": (("permeability", "support", "d_perm"), "log"),
    "T2_cube": (("permeability", "support", "T2"), "log"),
}
# voxel-level quantities of a fit: name -> kind
VOXEL_KEYS = {"v": "log", "D": "log", "area": "log", "bed_share": "diff"}
N_BOOT = 2000
BOOT_SEED = 0          # fixed, so that the interval is reproducible; recorded in the summary


# --------------------------------------------------------------------- input
def load(folder, partial=False):
    """Read a run, after checking that the archive still holds what its manifest
    says it holds.  A summary can outlive its records; `porovasc.validate` is
    what notices."""
    VAL.require(folder, partial=partial)
    man = json.load(open(os.path.join(folder, "manifest.json")))
    path, err = VAL._records_path(folder)
    if err:
        raise SystemExit(err)
    op = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    recs = [json.loads(l) for l in op if l.strip()]
    errors = [r for r in recs if "error" in r]
    return man, [r for r in recs if "error" not in r], errors


def analysis_hash():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _get(rec, path):
    x = rec
    for k in path:
        if x is None: return None
        x = x[k] if isinstance(k, int) and isinstance(x, list) and k < len(x) else (x.get(k) if isinstance(x, dict) else None)
    return x


def _effect(a, b, kind):
    if a is None or b is None:
        return None
    if kind == "diff":
        return float(a - b)
    if not (a > 0 and b > 0):
        return None
    return float(np.log10(a / b))


def _sign(x):
    if x is None: return "missing"
    return "zero" if x <= 0 else "positive"


def volume_accounting(pairs, path, kind, role=None):
    """Per volume pair: the effect, plus the counts a reader needs to know how
    many pairs were absent.  A log ratio is absent when either side is zero or
    missing (a blocked permeability direction, for instance), and the table of
    zero/positive transitions says which."""
    rows = []; trans = {}
    for b, a in pairs:
        if role is not None and a.get("role") != role:
            continue
        xa, xb = _get(a, path), _get(b, path)
        t = _sign(xb) + ">" + _sign(xa); trans[t] = trans.get(t, 0) + 1
        rows.append(dict(seed=a["seed"], rve=a["rve"], val=_effect(xa, xb, kind)))
    usable = sum(r["val"] is not None for r in rows)
    return rows, dict(pairs=len(rows), usable=usable, absent=len(rows) - usable, transitions=trans)


# ----------------------------------------------------------------- hierarchy
def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else None


def hierarchy(rows, boot=N_BOOT, seed=BOOT_SEED):
    """The three-step reduction.  `rows` are dicts with seed, rve and val, where
    val is already the volume-level value (step 1 done by the caller).

    Returns the per-seed values (step 2), the list of gland values, their
    median and range, and a percentile bootstrap interval of the median over
    glands (step 3).  Nothing is pooled across levels."""
    per_vol = {}
    for r in rows:
        if r.get("val") is None:
            continue
        per_vol.setdefault((r["seed"], r["rve"]), []).append(r["val"])
    per_seed_vals = {}
    for (s, q), v in per_vol.items():
        per_seed_vals.setdefault(s, []).append(_med(v))
    per_seed = {s: _med(v) for s, v in per_seed_vals.items()}
    vals = [v for v in per_seed.values() if v is not None]
    out = dict(per_seed=per_seed, n_glands=len(vals), values=vals,
               median=_med(vals), min=min(vals) if vals else None, max=max(vals) if vals else None,
               boot95=None)
    if len(vals) >= 2 and boot:
        rng = np.random.default_rng(seed); arr = np.asarray(vals)
        meds = np.median(rng.choice(arr, size=(boot, len(arr)), replace=True), axis=1)
        out["boot95"] = [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]
    return out


# -------------------------------------------------------------------- pairing
def pair(recs, arm):
    """(baseline record, arm record) for every (seed, rve) present in both."""
    base = {(r["seed"], r["rve"]): r for r in recs if r["arm"] == "baseline"}
    return [(base[(r["seed"], r["rve"])], r) for r in recs
            if r["arm"] == arm and (r["seed"], r["rve"]) in base]


def _est(rec, i, kernel, R_mm, tf="new", rule="front20"):
    for e in rec["voxels"][i]["est"]:
        if e["kernel"] == kernel and e["R_mm"] == R_mm and e.get("tf", "new") == tf \
                and e.get("rule", "front20") == rule:
            return e
    return None


def _grid(kernels, radii, tfs, rules):
    """Every estimator setting, with the key under which it is reported."""
    for kern in kernels:
        for R in radii:
            for tf in tfs:
                for rule in rules:
                    yield "%s_R%.2f_%s_%s" % (kern, R, tf, rule), (kern, R, tf, rule)


def _label(e):
    """Status of a fit for the transition table; a bound is recorded on top of
    the status, so that a wrapped fit at a bound stays visible as both."""
    if e is None:
        return "missing"
    s = e.get("status", "missing")
    return s + "+at_bound" if e.get("at_bound") else s


def _valid(e):
    return e is not None and e.get("status") == "ok" and not e.get("at_bound", False)


def voxel_rows(pairs, kernel, R_mm, tf="new", rule="front20"):
    """Step 1 for the contrast kinetics: per (seed, rve) the median paired
    effect over the jointly valid inputs, for every voxel quantity, overall and
    split by whether the accepted shell directions were identical; plus the
    status transitions of all inputs."""
    rows = []
    for b, a in pairs:
        n_in = min(len(b["voxels"]), len(a["voxels"]))
        eff = {k: [] for k in VOXEL_KEYS}; same = {k: [] for k in ("v", "D")}; diff = {k: [] for k in ("v", "D")}
        trans = {}
        for i in range(n_in):
            eb, ea = _est(b, i, kernel, R_mm, tf, rule), _est(a, i, kernel, R_mm, tf, rule)
            t = _label(eb) + ">" + _label(ea); trans[t] = trans.get(t, 0) + 1
            if not (_valid(eb) and _valid(ea)):
                continue
            for k, kind in VOXEL_KEYS.items():
                eff[k].append(_effect(ea.get(k), eb.get(k), kind))
            bucket = same if eb.get("accepted") == ea.get("accepted") else diff
            for k in ("v", "D"):
                bucket[k].append(_effect(ea.get(k), eb.get(k), "log"))
        rows.append(dict(seed=a["seed"], rve=a["rve"], role=a.get("role"), n_inputs=n_in,
                         n_valid=len(eff["v"]), n_same=len(same["v"]), n_different=len(diff["v"]),
                         transitions=trans,
                         val={k: _med(v) for k, v in eff.items()},
                         same={k: _med(v) for k, v in same.items()},
                         different={k: _med(v) for k, v in diff.items()}))
    return rows


def _merge(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items(): out[k] = out.get(k, 0) + v
    return out


def summarise_arm(recs, arm, kernels, radii, tfs=("new",), rules=("front20",)):
    pairs = pair(recs, arm)
    roles = sorted({a.get("role") for _, a in pairs})
    groups = roles if len(roles) > 1 else [None]
    s = dict(arm=arm, n_pairs=len(pairs), roles=roles,
             same_topology=all(a.get("same_topology") for _, a in pairs),
             same_path=all(a.get("same_path") for _, a in pairs),
             bed_constant=sorted({str(a.get("bed_constant")) for _, a in pairs}),
             wrapped=any(a.get("wrapped") for _, a in pairs),
             clearance_mm=hierarchy([dict(seed=a["seed"], rve=a["rve"], val=a.get("clearance_mm"))
                                     for _, a in pairs if a.get("role") == "reference"], boot=0),
             volume={}, kinetics={})
    for name, (path, kind) in VOLUME_KEYS.items():
        s["volume"][name] = {}
        for ro in groups:
            rows, acc = volume_accounting(pairs, path, kind, ro)
            s["volume"][name][ro or "all"] = dict(kind=kind, **acc, **hierarchy(rows))
    for key, (kern, R, tf, rule) in _grid(kernels, radii, tfs, rules):
            rows = voxel_rows(pairs, kern, R, tf, rule)
            s["kinetics"][key] = {}
            for ro in groups:
                sel = [r for r in rows if ro is None or r["role"] == ro]
                s["kinetics"][key][ro or "all"] = dict(
                    inputs=int(sum(r["n_inputs"] for r in sel)),
                    valid_pairs=int(sum(r["n_valid"] for r in sel)),
                    transitions=_merge([r["transitions"] for r in sel]),
                    effects={k: dict(kind=kind, **hierarchy([dict(seed=r["seed"], rve=r["rve"], val=r["val"][k]) for r in sel]))
                             for k, kind in VOXEL_KEYS.items()},
                    same_accepted=dict(n=int(sum(r["n_same"] for r in sel)),
                                       **{k: dict(kind="log", **hierarchy([dict(seed=r["seed"], rve=r["rve"], val=r["same"][k]) for r in sel]))
                                          for k in ("v", "D")}),
                    different_accepted=dict(n=int(sum(r["n_different"] for r in sel)),
                                            **{k: dict(kind="log", **hierarchy([dict(seed=r["seed"], rve=r["rve"], val=r["different"][k]) for r in sel]))
                                               for k in ("v", "D")}))
    return s


def summarise_baseline(recs, kernels, radii, tfs=("new",), rules=("front20",)):
    """Descriptive summary of the baseline, with the same three steps: median
    over valid voxels within a volume, then over volumes within a gland, then
    the list over glands."""
    base = [r for r in recs if r["arm"] == "baseline"]
    out = dict(n_records=len(base), volume={}, kinetics={})
    for name, (path, kind) in VOLUME_KEYS.items():
        vals = [_get(r, path) for r in base]
        rows = [dict(seed=r["seed"], rve=r["rve"], val=(x if (x is not None and (kind == "diff" or x > 0)) else None))
                for r, x in zip(base, vals)]
        usable = sum(r["val"] is not None for r in rows)
        out["volume"][name] = dict(volumes=len(rows), usable=usable, absent=len(rows) - usable,
                                   **hierarchy(rows, boot=0))
    for key, (kern, R, tf, rule) in _grid(kernels, radii, tfs, rules):
            rows = {k: [] for k in VOXEL_KEYS}; status = {}
            for r in base:
                vals = {k: [] for k in VOXEL_KEYS}
                for i in range(len(r["voxels"])):
                    e = _est(r, i, kern, R, tf, rule)
                    lab = _label(e); status[lab] = status.get(lab, 0) + 1
                    if _valid(e):
                        for k in VOXEL_KEYS: vals[k].append(e.get(k))
                for k in vals:
                    rows[k].append(dict(seed=r["seed"], rve=r["rve"], val=_med(vals[k])))
            out["kinetics"][key] = dict(v_mm_s=_scale(hierarchy(rows["v"], boot=0), 1e3),
                                        D_mm2_s=_scale(hierarchy(rows["D"], boot=0), 1e6),
                                        area=hierarchy(rows["area"], boot=0),
                                        bed_share=hierarchy(rows["bed_share"], boot=0), status=status)
    return out


def _scale(h, f):
    h = dict(h)
    for k in ("median", "min", "max"):
        if h.get(k) is not None: h[k] = h[k] * f
    h["values"] = [v * f for v in h["values"]]
    h["per_seed"] = {s: (None if v is None else v * f) for s, v in h["per_seed"].items()}
    return h


# ------------------------------------------------------------------- report
def _fmt(h):
    if h["n_glands"] == 0: return "n=0"
    vals = " ".join("%+.2f" % v for v in h["values"])
    ci = "  boot95 [%+.2f, %+.2f]" % tuple(h["boot95"]) if h.get("boot95") else ""
    return "median %+.2f  range [%+.2f, %+.2f]%s  glands: %s" % (h["median"], h["min"], h["max"], ci, vals)


def print_summary(man, arms, base):
    eff = man.get("effective", {})
    print("=" * 78)
    print("PAIRED STUDY %s  schema %s  code %s" % (man.get("kind"), man.get("schema"), man.get("code_sha256", "")[:12]))
    print("seeds %s; radii %s mm; kernels %s; transfer functions %s; causality rules %s; inputs %d per volume"
          % (man.get("seeds"), eff.get("radii_mm"), eff.get("poiseuille"), eff.get("tfs", ["new"]),
             eff.get("causal_rules", ["front20"]), eff.get("n_inputs", 0)))
    print("\nBASELINE, gland-level medians of volume medians (SI units)")
    for name, h in base["volume"].items():
        absent = "  (%d of %d volumes absent)" % (h["absent"], h["volumes"]) if h["absent"] else ""
        if h["n_glands"]:
            print("  %-26s median %.3e  range [%.3e, %.3e]  n_glands %d%s" % (name, h["median"], h["min"], h["max"], h["n_glands"], absent))
        else:
            print("  %-26s no usable volume%s" % (name, absent))
    for key, k in base["kinetics"].items():
        print("  %-14s v %s mm/s  D %s mm2/s  status %s" % (key,
              "%.2f" % k["v_mm_s"]["median"] if k["v_mm_s"]["n_glands"] else "n/a",
              "%.2f" % k["D_mm2_s"]["median"] if k["D_mm2_s"]["n_glands"] else "n/a", k["status"]))
    for arm in arms:
        print("\nARM %s: %d paired volumes; roles %s; same topology %s, same path %s; bed constant %s; wrapped %s"
              % (arm["arm"], arm["n_pairs"], arm["roles"], arm["same_topology"], arm["same_path"],
                 arm["bed_constant"], arm["wrapped"]))
        if arm["clearance_mm"]["n_glands"]:
            print("  reference clearance, min over glands: %.2f mm" % arm["clearance_mm"]["min"])
        print("  volume quantities, gland level (log10 arm/baseline, or difference where marked):")
        for name, byrole in arm["volume"].items():
            for role, h in byrole.items():
                if h["n_glands"]:
                    absent = "  (%d of %d pairs absent: %s)" % (h["absent"], h["pairs"], h["transitions"]) if h["absent"] else ""
                    print("    %-26s %-9s %-4s %s%s" % (name, role, h["kind"], _fmt(h), absent))
                elif h["pairs"]:
                    print("    %-26s %-9s %-4s no usable pair of %d: %s" % (name, role, h["kind"], h["pairs"], h["transitions"]))
        print("  contrast kinetics, gland level:")
        for key, byrole in arm["kinetics"].items():
            for role, k in byrole.items():
                print("    %-12s %-9s valid pairs %3d / %3d   transitions %s"
                      % (key, role, k["valid_pairs"], k["inputs"], k["transitions"]))
                for nm, h in k["effects"].items():
                    if h["n_glands"]:
                        print("      %-9s %-4s %s" % (nm, h["kind"], _fmt(h)))
                sa, da = k["same_accepted"], k["different_accepted"]
                print("      accepted set unchanged in %d pairs: v %s, D %s" % (sa["n"], _m(sa["v"]), _m(sa["D"])))
                print("      accepted set changed   in %d pairs: v %s, D %s" % (da["n"], _m(da["v"]), _m(da["D"])))


def _m(h):
    return "n/a" if h["n_glands"] == 0 else "%+.2f (%d glands)" % (h["median"], h["n_glands"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args(argv)
    man, recs, errors = load(a.folder, a.partial)
    eff = man.get("effective", {})
    kernels = ["pois" if p else "plug" for p in eff.get("poiseuille", [False, True])]
    radii = eff.get("radii_mm", [0.75, 1.0, 1.5, 2.0])
    tfs = eff.get("tfs", ["new"]); rules = eff.get("causal_rules", ["front20"])
    arms = [x["name"] for x in eff.get("arms", [])]
    base = summarise_baseline(recs, kernels, radii, tfs, rules)
    arm_summaries = [summarise_arm(recs, arm, kernels, radii, tfs, rules) for arm in arms]
    print_summary(man, arm_summaries, base)
    if errors:
        print("\n%d records carry errors and were excluded" % len(errors))
    outdir = os.path.join(a.folder, "analysis"); os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(dict(manifest=dict(kind=man.get("kind"), schema=man.get("schema"),
                                     code_sha256=man.get("code_sha256"), seeds=man.get("seeds"),
                                     # the effective settings travel with the summary: arms run
                                     # with a longer transform record are not directly comparable
                                     # with the rest, and that must be visible here
                                     effective=man.get("effective")),
                       analysis=dict(sha256=analysis_hash(), partial=bool(a.partial),
                                     n_boot=N_BOOT, boot_seed=BOOT_SEED),
                       baseline=base, arms=arm_summaries, n_errors=len(errors)),
                  f, indent=1, allow_nan=False)
    print("\nsummary written to %s" % os.path.join(outdir, "summary.json"))


if __name__ == "__main__":
    main()
