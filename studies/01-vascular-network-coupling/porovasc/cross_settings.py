"""How much does the reported (v, D) depend on the analysis settings alone?

    python -m porovasc.cross_settings runs/baseline
    python -m porovasc.cross_settings runs/baseline runs/lesion_tort16

Reads analysis/summary.json, which `porovasc.analyse_paired` writes, after
validating that the records that summary was computed from are still intact and
unchanged (`porovasc.validate`).  A summary is a derived file and can outlive
its source; nothing here trusts one that no longer matches its archive.

**Two kinds of setting, kept apart.**  The tissue is held fixed within a run,
but not everything swept here acts at the same stage:

  generation   the velocity kernel, plug against Poiseuille, changes how the
               bolus is propagated through each segment.  It produces different
               curves, so a difference between kernels is a property of the
               modelled transport, not of the analysis, and it must not be
               counted as analysis dependence.

  analysis     the shell radius, the transfer function and the causality rule
               are applied to curves that are already fixed.  A spread across
               these is analysis dependence in the strict sense.

The analysis-only spread is therefore reported with the kernel held fixed, and
the kernel effect is reported separately at fixed analysis settings.  The
combined range over everything is also printed, labelled as such, because it is
what a reader who sweeps everything at once would see.

For a baseline run this prints the gland-level median v and D under every
setting, then varies one factor at a time from a reference setting, so that the
spread due to the shell radius can be compared with the spread due to the
transfer function and to the causality rule.

For an arm it prints the effect (log10 of arm over baseline) under every
setting and reports whether all settings agree on the sign of the effect.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

from . import validate as VAL


def load(folder, partial=False):
    """The summary, after checking that the archive it came from is intact and
    that the summary still describes that archive."""
    rep = VAL.require(folder, partial=partial)
    p = os.path.join(folder, "analysis", "summary.json")
    if not os.path.exists(p):
        raise SystemExit("no %s\nrun first:  python -m porovasc.analyse_paired %s" % (p, folder))
    s = json.load(open(p))
    stale = []
    stored, live = s.get("manifest", {}) or {}, rep.get("manifest", {}) or {}
    for key in ("code_sha256", "records_sha256", "records", "finished"):
        a, b = stored.get(key), live.get(key)
        if a is not None and b is not None and a != b:
            stale.append("%s: summary %s, archive %s" % (key, str(a)[:16], str(b)[:16]))
    rpath, _ = VAL._records_path(folder)
    if rpath and os.path.getmtime(p) < os.path.getmtime(rpath):
        stale.append("the summary is older than records.jsonl")
    if stale:
        msg = ("%s does not describe the archive beside it:\n    %s\n  re-run "
               "porovasc.analyse_paired" % (p, "\n    ".join(stale)))
        if not partial:
            raise SystemExit(msg)
        print("warning  " + msg)
    return s


def parse(key):
    """plug_R1.00_new_front20 -> ("plug", 1.0, "new", "front20")"""
    kern, R, tf, rule = key.split("_")
    return kern, float(R[1:]), tf, rule


def _med(h):
    return h.get("median") if h and h.get("n_glands") else None


def baseline_table(s):
    kin = s["baseline"]["kinetics"]
    rows = []
    for key in kin:
        kern, R, tf, rule = parse(key)
        v, D = _med(kin[key]["v_mm_s"]), _med(kin[key]["D_mm2_s"])
        n = kin[key]["v_mm_s"]["n_glands"]
        rows.append((kern, R, tf, rule, v, D, n, kin[key]["status"]))
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    print("\nBASELINE: the same curves, every analysis setting")
    print("  %-5s %5s %-4s %-8s %9s %10s %7s" % ("kern", "R/mm", "tf", "rule", "v mm/s", "D mm2/s", "glands"))
    for kern, R, tf, rule, v, D, n, st in rows:
        print("  %-5s %5.2f %-4s %-8s %9s %10s %7d"
              % (kern, R, tf, rule,
                 "n/a" if v is None else "%.2f" % v,
                 "n/a" if D is None else "%.2f" % D, n))
    vs = [r[4] for r in rows if r[4] is not None]
    Ds = [r[5] for r in rows if r[5] is not None]
    if vs:
        print("  v  generation and analysis combined: %.2f to %.2f mm/s  (factor %.1f)"
              % (min(vs), max(vs), max(vs) / min(vs)))
    if Ds:
        print("  D  generation and analysis combined: %.2f to %.2f mm2/s (factor %.1f)"
              % (min(Ds), max(Ds), max(Ds) / min(Ds)))
    return rows


def _factor(vals):
    vals = [v for v in vals if v is not None and v > 0]
    return (max(vals) / min(vals)) if len(vals) > 1 else None


def split_generation_and_analysis(rows):
    """Analysis-only spread with the transport kernel held fixed, and the kernel
    effect at fixed analysis settings.

    The kernel changes the propagated curves, so its contribution belongs to the
    modelled transport and not to the analysis.  Reporting one range over both
    would attribute to the analysis a spread that the analysis did not
    produce."""
    have = {(k, R, tf, ru): (v, D) for k, R, tf, ru, v, D, n, st in rows
            if v is not None and D is not None}
    if not have:
        return None
    kerns = sorted({k for k, _, _, _ in have})
    print("\n  ANALYSIS ONLY, transport kernel held fixed")
    print("    %-8s %8s %22s %22s" % ("kernel", "settings", "v range (factor)", "D range (factor)"))
    per_kernel = {}
    for k in kerns:
        sel = {a: b for a, b in have.items() if a[0] == k}
        vs = [b[0] for b in sel.values()]
        Ds = [b[1] for b in sel.values()]
        fv, fD = _factor(vs), _factor(Ds)
        per_kernel[k] = dict(n=len(sel), v=(min(vs), max(vs), fv), D=(min(Ds), max(Ds), fD))
        print("    %-8s %8d %22s %22s"
              % (k, len(sel),
                 "-" if fv is None else "%.2f to %.2f  (x%.1f)" % (min(vs), max(vs), fv),
                 "-" if fD is None else "%.2f to %.2f  (x%.1f)" % (min(Ds), max(Ds), fD)))

    print("\n  GENERATION ONLY, analysis setting held fixed: %s against %s"
          % (kerns[0], kerns[-1]) if len(kerns) > 1 else
          "\n  GENERATION ONLY: only one transport kernel is present")
    gen = None
    if len(kerns) > 1:
        settings = sorted({a[1:] for a in have})
        ratios_v, ratios_D = [], []
        for st in settings:
            a, b = have.get((kerns[0],) + st), have.get((kerns[-1],) + st)
            if a and b and b[0] and b[1]:
                ratios_v.append(a[0] / b[0])
                ratios_D.append(a[1] / b[1])
        if ratios_v:
            gen = dict(n=len(ratios_v), v=(min(ratios_v), max(ratios_v)),
                       D=(min(ratios_D), max(ratios_D)))
            print("    over %d matched analysis settings: v ratio %.2f to %.2f, "
                  "D ratio %.2f to %.2f"
                  % (len(ratios_v), min(ratios_v), max(ratios_v),
                     min(ratios_D), max(ratios_D)))
    # The transfer function is not one analysis choice among several: the
    # corrected and the superseded form are a correct analysis and a wrong one.
    # Holding it fixed as well gives the spread among analyses that are all
    # defensible, which is the number to quote.
    tfs = sorted({a[2] for a in have})
    print("\n  ANALYSIS ONLY, kernel and transfer function both held fixed")
    print("    the transfer function is held because the corrected and the superseded")
    print("    form are not two defensible analyses; this is the spread among analyses")
    print("    that all are, over the shell radius and the causality rule")
    print("    %-8s %-6s %8s %22s %22s"
          % ("kernel", "tf", "settings", "v range (factor)", "D range (factor)"))
    per_cell = {}
    for k in kerns:
        for tf in tfs:
            sel = {a: b for a, b in have.items() if a[0] == k and a[2] == tf}
            if len(sel) < 2:
                continue
            vs = [b[0] for b in sel.values()]
            Ds = [b[1] for b in sel.values()]
            fv, fD_ = _factor(vs), _factor(Ds)
            per_cell[(k, tf)] = dict(n=len(sel), v=(min(vs), max(vs), fv),
                                     D=(min(Ds), max(Ds), fD_))
            print("    %-8s %-6s %8d %22s %22s"
                  % (k, tf, len(sel),
                     "%.2f to %.2f  (x%.1f)" % (min(vs), max(vs), fv),
                     "%.2f to %.2f  (x%.1f)" % (min(Ds), max(Ds), fD_)))

    fD = [per_kernel[k]["D"][2] for k in kerns if per_kernel[k]["D"][2]]
    if fD:
        print("\n    largest analysis-only factor on D within one kernel, every "
              "transfer function: x%.1f" % max(fD))
    corrected = [c["D"][2] for key, c in per_cell.items()
                 if key[1] == "new" and c["D"][2]]
    if corrected:
        print("    largest analysis-only factor on D within one kernel, corrected "
              "transfer function only: x%.1f" % max(corrected))
        cv = [c["v"][2] for key, c in per_cell.items() if key[1] == "new" and c["v"][2]]
        print("    the same for v: x%.1f" % max(cv))
    return dict(per_kernel=per_kernel, generation=gen, per_cell=per_cell)


def one_at_a_time(rows):
    """Spread of the median attributable to each factor separately, from a
    reference setting, so that the shell radius can be compared with the rest."""
    have = {(k, R, tf, ru): (v, D) for k, R, tf, ru, v, D, n, st in rows if v is not None}
    if not have:
        return
    kerns = sorted({k for k, _, _, _ in have}); radii = sorted({R for _, R, _, _ in have})
    tfs = sorted({tf for _, _, tf, _ in have}); rules = sorted({ru for _, _, _, ru in have})
    ref = (kerns[0], radii[len(radii) // 2], "new" if "new" in tfs else tfs[0],
           "front20" if "front20" in rules else rules[0])
    if ref not in have:
        ref = sorted(have)[0]
    print("\n  ONE FACTOR AT A TIME, from the reference setting %s R%.2f %s %s\n"
          "    (the kernel row is a generation effect, the other three are analysis)" % ref)
    for label, idx, values in (("kernel", 0, kerns), ("shell radius", 1, radii),
                               ("transfer function", 2, tfs), ("causality rule", 3, rules)):
        vs, Ds, seen = [], [], []
        for val in values:
            key = list(ref); key[idx] = val; key = tuple(key)
            if key in have:
                vs.append(have[key][0]); Ds.append(have[key][1]); seen.append(val)
        if len(vs) > 1:
            print("    %-18s %-28s v x%.2f   D x%.2f"
                  % (label, ",".join(str(x) for x in seen), max(vs) / min(vs), max(Ds) / min(Ds)))


MIN_GLANDS = 3          # below this the gland range says nothing about the sign


def _cell(h):
    """median with the gland-level range, and whether that range clears zero.
    A median of one sign with a range that straddles zero is not evidence of a
    sign, so a reversal between two settings only counts when both ranges
    clear zero in opposite directions.  With fewer than MIN_GLANDS glands the
    range is not evidence either: one gland always "clears" zero trivially."""
    m = _med(h)
    if m is None:
        return None, None, "none"
    lo, hi = h.get("min"), h.get("max")
    if lo is None or hi is None:
        return m, None, "unknown"
    if h.get("n_glands", 0) < MIN_GLANDS:
        return m, (lo, hi), "too few glands"
    sign = "pos" if lo > 0 else ("neg" if hi < 0 else "straddles")
    return m, (lo, hi), sign


def arm_table(s, folder):
    for arm in s.get("arms", []):
        print("\nARM %s  (%s, %d pairs)" % (arm["arm"], os.path.basename(folder.rstrip("/")), arm["n_pairs"]))
        roles = sorted({r for key in arm["kinetics"] for r in arm["kinetics"][key]})
        for role in roles:
            rows = []
            for key in arm["kinetics"]:
                kern, R, tf, rule = parse(key)
                e = arm["kinetics"][key].get(role)
                if not e:
                    continue
                rows.append((kern, R, tf, rule,
                             _cell(e["effects"]["v"]), _cell(e["effects"]["D"]),
                             e["effects"]["v"]["n_glands"]))
            rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
            print("  role %s: effect = log10(arm / baseline), 0 is no change;" % role)
            print("  [lo, hi] is the range over glands, * marks a range that does not contain 0")
            print("   %-5s %5s %-4s %-8s %24s %24s %7s"
                  % ("kern", "R/mm", "tf", "rule", "v  median [gland range]", "D  median [gland range]", "glands"))
            for kern, R, tf, rule, vc, Dc, n in rows:
                def fmt(c):
                    m, rng, sign = c
                    if m is None:
                        return "n/a"
                    star = "*" if sign in ("pos", "neg") else " "
                    return "%+.3f [%+.3f,%+.3f]%s" % (m, rng[0], rng[1], star) if rng else "%+.3f" % m
                print("   %-5s %5.2f %-4s %-8s %24s %24s %7d" % (kern, R, tf, rule, fmt(vc), fmt(Dc), n))
            for name, i in (("v", 4), ("D", 5)):
                cells = [r[i] for r in rows if r[i][0] is not None]
                if not cells:
                    continue
                vals = [c[0] for c in cells]
                pos = sum(1 for c in cells if c[2] == "pos")
                neg = sum(1 for c in cells if c[2] == "neg")
                straddle = sum(1 for c in cells if c[2] == "straddles")
                few = sum(1 for c in cells if c[2] == "too few glands")
                tail = (", %d with fewer than %d glands" % (few, MIN_GLANDS)) if few else ""
                if pos and neg:
                    verdict = ("REVERSAL: %d settings robustly positive, %d robustly negative "
                               "(gland ranges clear 0 in both directions)%s" % (pos, neg, tail))
                elif pos or neg:
                    verdict = ("%d settings robustly %s, %d with a gland range containing 0%s: "
                               "consistent direction, no reversal"
                               % (pos or neg, "positive" if pos else "negative", straddle, tail))
                elif straddle:
                    verdict = ("no setting resolves the sign: %d gland ranges contain 0%s"
                               % (straddle, tail))
                else:
                    verdict = "not assessable: every setting has fewer than %d glands" % MIN_GLANDS
                print("   %s: medians %+.3f to %+.3f over settings; %s" % (name, min(vals), max(vals), verdict))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--partial", action="store_true",
                    help="read an archive that does not validate, marking every "
                         "number derived from it as provisional")
    a = ap.parse_args(argv)
    for folder in a.folders:
        s = load(folder, partial=a.partial)
        eff = s["manifest"].get("effective", {})
        print("=" * 78)
        print("%s   record %.2f s, dt %.3f s, %d inputs per volume"
              % (folder, eff.get("t_end_s", float("nan")), eff.get("dt_s", float("nan")),
                 eff.get("n_inputs", 0)))
        rows = baseline_table(s)
        split_generation_and_analysis(rows)
        one_at_a_time(rows)
        arm_table(s, folder)


if __name__ == "__main__":
    main()
