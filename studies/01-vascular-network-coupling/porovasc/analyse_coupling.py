"""Which perturbations move drainage, which move transport, and which move both?

    python -m porovasc.analyse_coupling runs/coupling

Reads a run written by `porovasc.run_coupling` and reports, for each arm, the
paired gland-level effect on every quantity of the three families, by the same
three-step reduction used everywhere in this package: median over the input
voxels of a sampling volume, median over the volumes of a gland, then the list
over glands with its median and range.  Effects are log10 ratios for strictly
positive quantities and differences for shares.

The table is arranged so that the question can be answered by reading down a
column.  An arm that acts on the geometry should move the network descriptors,
and the two responses that depend on the geometry should move with them.  An arm
that acts only on the matrix modulus should move the drainage and leave the
transport alone; an arm that acts only on the driving pressure should move the
transport and leave the geometric descriptors alone.  Those two are the controls
that make the rest interpretable: without them, a table in which everything
moves says only that the arms did something.

A gland-level range that contains zero is not evidence of an effect, and is
marked as such.  With five glands no rank test reaches a useful threshold, so no
p-values are computed and the list of gland effects is reported instead.

What this cannot show: that a shared dependence implies joint identifiability.
Two observables can both depend on the same structure and still fail to
determine it, and nothing here addresses that.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np

from .analyse_paired import hierarchy, _med
from . import validate as VAL

MIN_GLANDS = 3

#: family -> quantity -> (path into the record, kind, label)
FAMILIES = {
    "network": {
        "phi_cube": (("network", "phi_cube"), "log", "vascular volume fraction"),
        "k_mean": (("network", "k_mean"), "log", "permeability, face to face"),
        "k_x": (("network", "k_x"), "log", "permeability along x"),
        "k_z": (("network", "k_z"), "log", "permeability along z"),
        "T2": (("network", "T2"), "log", "mean squared tortuosity"),
        "d_perm": (("network", "d_perm"), "log", "permeability weighted diameter"),
        "chi": (("network", "chi_chi"), "log", "flow coherence"),
        "vessel_speed": (("network", "chi_vessel_speed"), "log", "flow-weighted vessel speed"),
        # a direction with no inlet-to-outlet path has permeability exactly zero,
        # so its log ratio is undefined; the count of such directions is reported
        # as a difference so that a blocked direction is visible rather than
        # dropping out of the table as missing data
        "n_blocked": (("network", "n_blocked"), "diff", "directions with no through path"),
    },
    "drainage": {
        "relax_per_strain": (("drainage", "relax_per_strain"), "log", "amplitude per applied strain"),
        "tau_rc": (("drainage", "tau_rc"), "log", "relaxation time"),
        "spectrum_width": (("drainage", "spectrum_width"), "log", "width of the relaxation spectrum"),
        "phi_region": (("drainage", "phi_region"), "log", "vascular fraction, compression support"),
    },
    "transport": {
        "area": (("transport", "curves", "area"), "log", "curve area, the amplitude"),
        "t_peak": (("transport", "curves", "t_peak"), "log", "time to peak"),
        "mtt": (("transport", "curves", "mtt"), "log", "mean transit time"),
        "width": (("transport", "curves", "width"), "log", "curve width"),
        "bed_share": (("transport", "curves", "bed_share"), "diff", "share carried by the bed"),
        "v": (("transport", "est", "v"), "log", "estimated drift"),
        "D": (("transport", "est", "D"), "log", "estimated dispersion"),
    },
}

#: network descriptors that depend on the geometry alone.  A flow-only arm must
#: leave every one of them untouched; chi and the vessel speed are deliberately
#: not here, because they depend on the flow and a flow-only arm should move
#: them.
GEOMETRY_INVARIANTS = ("phi_cube", "k_mean", "k_x", "k_z", "T2", "d_perm", "n_blocked")

#: an effect this small is zero: these arms change nothing the quantity sees, so
#: the only departure from zero is floating-point noise
CONTROL_TOL = 1e-9


def load(folder, partial=False):
    VAL.require(folder, partial=partial)
    man = json.load(open(os.path.join(folder, "manifest.json")))
    path, err = VAL._records_path(folder)
    if err:
        raise SystemExit(err)
    import gzip
    op = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    recs = [json.loads(l) for l in op if l.strip()]
    good = [r for r in recs if "error" not in r]
    wrapped = [r for r in good if r.get("status") == "wrapped"]
    if wrapped:
        arms = sorted({r["arm"] for r in wrapped})
        raise SystemExit(
            "%d record(s) in arm(s) %s are marked wrapped: their paths take longer "
            "to traverse than the transform record, so every transport number in "
            "them is invalid.  Rerun those arms with a longer --t-end.  Dropping "
            "them instead would select on the mechanism under study."
            % (len(wrapped), ", ".join(arms)))
    return man, good


def _per_voxel(rec, path):
    """The per-voxel list for a transport quantity, with validity.

    An estimate that ended at an optimizer bound is not an estimate: it is the
    optimizer reporting that it left the feasible region, and averaging it in
    would put the bound itself into the median.  Those are marked invalid
    here."""
    block = rec.get("transport", {}) or {}
    if block.get("status") == "wrapped":
        return []
    out = []
    for d in block.get(path[1], []) or []:
        v = d.get(path[2])
        ok = v is not None and np.isfinite(v)
        if ok and path[1] == "est" and d.get("at_bound"):
            ok = False
        out.append((v if ok else None, ok))
    return out


def value(rec, path):
    """One number per sampling volume, for a volume-level quantity.

    Transport quantities are per voxel and are not reduced here: they are paired
    voxel by voxel first (see `paired_effect`), because taking a median in each
    condition before forming the effect can compare different surviving voxels."""
    if path[0] == "transport" and path[1] in ("curves", "est"):
        vals = [v for v, ok in _per_voxel(rec, path) if ok]
        return _med(vals)
    node = rec
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node if (node is not None and np.isfinite(node)) else None


def paired_effect(base, arm, path, kind):
    """Effect of one arm on one quantity of one sampling volume.

    For a volume-level quantity this is the effect of the two numbers.  For a
    per-voxel transport quantity the effect is formed voxel by voxel on the
    voxels valid in *both* conditions, and only then reduced to the volume by a
    median.  The input voxels are drawn from the same seeded generator in both
    conditions, so voxel i is the same voxel; pairing by index is exact.

    Returns (effect, accounting), where the accounting counts the four validity
    transitions so that a change in what could be estimated cannot hide inside a
    change in the estimate."""
    if not (path[0] == "transport" and path[1] in ("curves", "est")):
        a, b = value(arm, path), value(base, path)
        e = effect(a, b, kind)
        return e, dict(n=1, both=int(e is not None), lost=int(b is not None and a is None),
                       gained=int(a is not None and b is None),
                       neither=int(a is None and b is None))
    pb, pa = _per_voxel(base, path), _per_voxel(arm, path)
    n = min(len(pb), len(pa))
    effs, acc = [], dict(n=n, both=0, lost=0, gained=0, neither=0)
    for i in range(n):
        vb, okb = pb[i]
        va, oka = pa[i]
        if okb and oka:
            acc["both"] += 1
            e = effect(va, vb, kind)
            if e is not None:
                effs.append(e)
        elif okb and not oka:
            acc["lost"] += 1
        elif oka and not okb:
            acc["gained"] += 1
        else:
            acc["neither"] += 1
    return _med(effs), acc


def effect(a, b, kind):
    if a is None or b is None:
        return None
    if kind == "diff":
        return float(a - b)
    if a > 0 and b > 0:
        return float(np.log10(a / b))
    return None


def arm_effects(recs, arm):
    """Gland-level effect of one arm on every quantity, from paired volumes."""
    base = {(r["seed"], r["rve"]): r for r in recs if r["arm"] == "baseline"}
    pairs = [(base[(r["seed"], r["rve"])], r) for r in recs
             if r["arm"] == arm and (r["seed"], r["rve"]) in base]
    out = {}
    for fam, quants in FAMILIES.items():
        out[fam] = {}
        for name, (path, kind, _) in quants.items():
            rows, n_pairs = [], 0
            tot = dict(n=0, both=0, lost=0, gained=0, neither=0)
            for b, a in pairs:
                e, acc = paired_effect(b, a, path, kind)
                for k in tot:
                    tot[k] += acc[k]
                if e is not None:
                    n_pairs += 1
                rows.append(dict(seed=a["seed"], rve=a["rve"], val=e))
            h = hierarchy(rows, boot=0)
            h["n_volumes_with_effect"] = n_pairs
            h["n_volumes_absent"] = len(pairs) - n_pairs
            h["accounting"] = tot
            h["kind"] = kind
            out[fam][name] = h
    out["_n_pairs"] = len(pairs)
    return out


def verdict(h):
    """moved / unmoved / undetermined, from the gland-level range alone."""
    if not h["n_glands"] or h["median"] is None:
        return "no data"
    if h["n_glands"] < MIN_GLANDS:
        return "too few glands"
    lo, hi = h["min"], h["max"]
    if lo is None or hi is None:
        return "unknown"
    if lo > 0 or hi < 0:
        return "moved"
    return "undetermined"


def print_arm(arm, locus, eff):
    print("\n%-12s  acts on: %-10s  %d paired volumes"
          % (arm, locus or "?", eff["_n_pairs"]))
    print("   %-9s %-18s %9s %22s %7s %s"
          % ("family", "quantity", "median", "range over glands", "glands", "verdict"))
    for fam in ("network", "drainage", "transport"):
        for name, (_, kind, label) in FAMILIES[fam].items():
            h = eff[fam][name]
            v = verdict(h)
            if not h["n_glands"]:
                print("   %-9s %-18s %9s %22s %7d %s"
                      % (fam, name, "-", "-", 0, v))
                continue
            acc = h.get("accounting", {})
            note = ""
            if acc.get("lost") or acc.get("gained"):
                note = "  (voxels valid in both %d, lost %d, gained %d, neither %d)" % (
                    acc["both"], acc["lost"], acc["gained"], acc["neither"])
            elif h.get("n_volumes_absent"):
                note = "  (%d volume(s) without an effect)" % h["n_volumes_absent"]
            print("   %-9s %-18s %+9.3f %22s %7d %s%s"
                  % (fam, name, h["median"],
                     "[%+.3f, %+.3f]" % (h["min"], h["max"]),
                     h["n_glands"], v, note))


def control_checks(effs, tol=CONTROL_TOL):
    """Assert what the control arms are for, instead of describing it.

    A modulus-only arm must leave the network and the transport exactly where
    they were: it changes the compliance law and nothing the geometry or the
    curves can see.  A flow-only arm must leave the geometric descriptors
    exactly where they were, while being free to move the flow-dependent ones.
    If either fails, the arm is not doing what its name says, and the geometric
    rows of the table cannot be read as shared dependence."""
    print("\n" + "=" * 78)
    print("CONTROL CHECKS")
    print("  mechanics arms must leave network and transport unmoved;")
    print("  flow arms must leave the geometric descriptors unmoved")
    print("  (chi and the vessel speed depend on the flow and are excluded from that)")
    ok_all = True
    for arm, locus, eff in effs:
        if locus == "mechanics":
            want = [("network", n) for n in FAMILIES["network"]] + \
                   [("transport", n) for n in FAMILIES["transport"]]
        elif locus == "flow":
            want = [("network", n) for n in GEOMETRY_INVARIANTS]
        else:
            continue
        worst, where = 0.0, None
        for fam, n in want:
            h = eff[fam].get(n)
            if not h or not h["n_glands"]:
                continue
            m = max(abs(h["min"] or 0.0), abs(h["max"] or 0.0))
            if m > worst:
                worst, where = m, "%s.%s" % (fam, n)
        ok = worst <= tol
        ok_all &= ok
        print("  %-16s %-10s largest |effect| over %d quantities: %.2e  %s%s"
              % (arm, locus, len(want), worst, "PASS" if ok else "FAIL",
                 "" if ok else "  at %s" % where))
    if not ok_all:
        print("\n  a control arm moved something it should not have.  Either the arm")
        print("  is not the intervention its name claims, or a quantity depends on")
        print("  something it was not thought to.  Do not read the geometry arms")
        print("  until this is resolved.")
    return ok_all


def summary(effs):
    """Which families moved, per arm.  This is the result the experiment is for."""
    print("\n" + "=" * 78)
    print("WHICH FAMILIES MOVED")
    print("  a family counts as moved when at least one of its quantities has a")
    print("  gland-level range that does not contain zero")
    print("  %-12s %-10s %10s %10s %10s" % ("arm", "acts on", "network", "drainage", "transport"))
    for arm, locus, eff in effs:
        cells = []
        for fam in ("network", "drainage", "transport"):
            moved = [n for n in FAMILIES[fam] if verdict(eff[fam][n]) == "moved"]
            total = sum(1 for n in FAMILIES[fam] if eff[fam][n]["n_glands"])
            cells.append("%d of %d" % (len(moved), total))
        print("  %-12s %-10s %10s %10s %10s" % (arm, locus or "?", *cells))
    print("\n  read the two control rows first: an arm acting only on the matrix")
    print("  modulus that moves the transport family, or an arm acting only on the")
    print("  driving pressure that moves the geometric descriptors, would mean the")
    print("  arms are not doing what their names say, and the geometric rows could")
    print("  not then be read as shared dependence")


def check_identity(recs):
    """The compliance-law identity, on the records rather than on trust.

    Under C_i = V_i / H the equilibrium amplitude per applied strain equals the
    vascular volume fraction of the compressed region.  This reports the
    agreement so that the identity is visible as a conditional model result
    rather than asserted."""
    rel, phi = [], []
    for r in recs:
        a = value(r, ("drainage", "relax_per_strain"))
        b = value(r, ("drainage", "phi_region"))
        if a and b:
            rel.append(a); phi.append(b)
    if not rel:
        return
    rel, phi = np.array(rel), np.array(phi)
    err = np.abs(rel - phi) / phi
    print("\nCOMPLIANCE-LAW IDENTITY, amplitude per strain against phi of the region")
    print("  %d volumes, relative difference: median %.2e, max %.2e" %
          (len(rel), float(np.median(err)), float(err.max())))
    print("  this is a property of the imposed law C_i = V_i / (H + K_wall), not")
    print("  evidence that a measured relaxation identifies phi in tissue")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args(argv)
    man, recs = load(a.folder, partial=a.partial)
    loci = {d["name"]: d.get("locus") for d in man.get("effective", {}).get("arms", [])}
    arms = [n for n in loci if any(r["arm"] == n for r in recs)]
    print("=" * 78)
    print("%s   %d records, %d glands, arms: %s"
          % (a.folder, len(recs), len({r["seed"] for r in recs}), ", ".join(arms)))
    check_identity(recs)
    effs = []
    for arm in arms:
        eff = arm_effects(recs, arm)
        effs.append((arm, loci.get(arm), eff))
        print_arm(arm, loci.get(arm), eff)
    ok = control_checks(effs)
    summary(effs)
    print("\n  a shared dependence is not joint identifiability: two observables can")
    print("  both depend on the same structure and still fail to determine it.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
