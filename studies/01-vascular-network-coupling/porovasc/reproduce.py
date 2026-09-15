"""Re-derive one archived record with the code that is running now.

    python -m porovasc.reproduce runs/baseline --seed 100 --rve 0
    python -m porovasc.reproduce runs/baseline --seed 100 --rve 0 --transport 2

A manifest records the sha256 of the sources that wrote it.  When that hash
differs from the current checkout, the provenance of the archive cannot be
verified by comparison, and the source version may no longer exist.  A hash
mismatch does not say the numbers are wrong, and it does not say they are right.

This answers the question the hash cannot: rebuild the same gland from the same
seed with the current code and recompute the quantities the record holds, then
report the difference.  Agreement to floating-point tolerance is direct evidence
that whatever changed between the two source versions does not reach these
numbers.  A disagreement localises what did.

What is compared, in increasing cost:

  geometry     the topology and path fingerprints against the manifest, and the
               sampling-volume centres against the record.  These are exact
               string and float comparisons and cost one gland build.
  flow         the calibrated bed constant against the manifest.
  volume       the compression and permeability blocks of the record.  Cheap
               once the gland exists.
  transport    the contrast estimate for the first `--transport` input voxels,
               at the settings the record used.  This is the expensive part and
               is off by default.

The comparison is a reproduction check, not a test of correctness: it says
whether this code reproduces that archive, not whether either is right.

**Scope.**  One invocation checks selected quantities of one record.  It does
not verify an archive, and it does not verify other archives that happen to
share a source hash.  What it supports is a statement of the form "representative
records reproduce within the stated tolerances under the current code", and
nothing broader.  Checking a baseline record inside a lesion study says nothing
about the lesion arm; pass `--arm` to rebuild the intervention itself, which
reconstructs that arm's configuration and lesion from the manifest and compares
against the record's own fingerprints rather than the baseline's.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os

import numpy as np

from . import study as S
from . import validate as VAL
from .config import apply_globals
from .geometry import network as N
from .homogenise import darcy as DA
from .physics import drainage as DR
from .physics import flow as F
from .physics import transport as TR

TOL = 1e-9              # relative agreement that counts as a reproduction


def _read_record(folder, arm, seed, rve):
    path, err = VAL._records_path(folder)
    if err:
        raise SystemExit(err)
    op = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    with op as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("arm", "baseline") == arm and r.get("seed") == seed
                    and r.get("rve") == rve and "error" not in r):
                return r
    raise SystemExit("no record for arm %s seed %s rve %s" % (arm, seed, rve))


def _cmp(out, label, new, old, tol=TOL):
    """Record one comparison.  Strings compare exactly, numbers relatively.

    A record stores a failed fit as null and the solver returns NaN for the
    same thing, so the two are treated as the same absence: agreeing that no
    value exists is a reproduction, not a difference."""
    if isinstance(new, float) and np.isnan(new):
        new = None
    if isinstance(old, float) and np.isnan(old):
        old = None
    if old is None and new is None:
        out.append((label, "both absent", 0.0, True))
        return
    if old is None and new is not None:
        # The archive does not store this quantity for this record.  Not every
        # archive kind stores every quantity: the coupling runner writes no
        # per-record fingerprints for its intervention arms, so there is nothing
        # to compare against.  Reporting that as a difference would be a false
        # failure; it is an absence, and it is named as one and excluded from
        # the verdict rather than quietly dropped.
        out.append((label, "not stored by this archive: %s vs -" % (new,),
                    float("nan"), None))
        return
    if isinstance(old, str) or isinstance(new, str):
        ok = (old == new)
        out.append((label, "%s vs %s" % (new, old), 0.0 if ok else 1.0, ok))
        return
    if new is None:
        out.append((label, "one absent: %s vs %s" % (new, old), float("inf"), False))
        return
    old = float(old); new = float(new)
    if old == 0.0 and new == 0.0:
        d = 0.0
    elif old == 0.0:
        d = abs(new)
    else:
        d = abs(new - old) / abs(old)
    out.append((label, "%.10g vs %.10g" % (new, old), d, d <= tol))


def reproduce(folder, seed, rve, arm="baseline", n_transport=0, partial=False):
    rep = VAL.require(folder, partial=partial)
    man = rep["manifest"]
    cfg = dict(man["config"])
    eff = man.get("effective", {})
    rec = _read_record(folder, arm, seed, rve)
    out = []

    n_rve = int(eff.get("n_rve", 1))
    min_sep = float(eff.get("min_centre_separation_mm", 0.0)) * 1e-3
    R_support = float(eff.get("support_radius_mm", cfg["R_comp"] * 1e3)) * 1e-3
    apply_globals(cfg)

    rng = np.random.default_rng(seed)
    centres = S.rve_centres(n_rve, rng, cfg["rve_half"], min_sep=min_sep,
                            R_support=R_support)
    c = centres[rve]
    for i, ax in enumerate("xyz"):
        _cmp(out, "centre %s (mm)" % ax, c[i] * 1e3, rec["centre_mm"][i], 1e-12)

    # Rebuild the condition the record belongs to, not the baseline.  An arm
    # changes the configuration and may carry a lesion; reconstructing only the
    # baseline would compare a different object with the record.
    spec = next((d for d in eff.get("arms", []) if d.get("name") == arm), None)
    if arm != "baseline" and spec is None:
        raise SystemExit("the manifest does not describe an arm named %r; it has %s"
                         % (arm, [d.get("name") for d in eff.get("arms", [])]))
    cfg_arm = dict(cfg, **(spec.get("changed") or {})) if spec else cfg
    lesion = spec.get("lesion") if spec else None
    if lesion:
        lesion = dict(lesion, centre=centres[0])      # as run_paired places it
    the_arm = S.Arm(arm, cfg_arm, lesion=lesion,
                    recalibrate=bool(spec.get("recalibrate")) if spec else False)
    apply_globals(cfg_arm)
    net = S.build_arm(the_arm, seed, centres)
    fp = S.fingerprints(net)
    # an arm record carries its own fingerprints; only a baseline record is
    # described by the manifest's baseline fingerprint
    want_fp = rec.get("fingerprint") or (
        (man.get("baseline_fingerprint") or {}).get(str(seed), {})
        if arm == "baseline" else {})
    _cmp(out, "topology fingerprint", fp[0], want_fp.get("topology"))
    _cmp(out, "path fingerprint", fp[1], want_fp.get("path"))
    _cmp(out, "segments", net.n, rec.get("n_seg"))

    if arm == "baseline":
        fl = F.solve(net, R_lat=cfg_arm["R_LAT"])
        _cmp(out, "bed constant G_bed0", fl.G_bed0,
             (man.get("g_bed0") or {}).get(str(seed)))
    else:
        # an arm holds the bed constant calibrated on its own baseline gland,
        # unless its intervention is that calibration
        G0 = (man.get("g_bed0") or {}).get(str(seed))
        if G0 is None:
            raise SystemExit("the manifest carries no baseline bed constant for "
                             "seed %d, so this arm cannot be rebuilt" % seed)
        fl = S.solve_arm(net, the_arm, float(G0))
    cfg = cfg_arm

    # ---- the volume-level blocks ------------------------------------------
    comp = rec.get("compression", {})
    if comp:
        dr = DR.solve(net, fl, c, cfg["R_comp"], dP_ext=100.0, H=cfg["H"])
        _cmp(out, "compression tau_rc", dr.tau_rc, comp.get("tau_rc"), 1e-6)
        _cmp(out, "compression phi_region", dr.phi, comp.get("phi_region"))
    perm = rec.get("permeability", {})
    if perm:
        h = cfg["rve_half"]          # the cube support run_paired uses
        da = DA.measure(net, fl, c, h)
        _cmp(out, "permeability k_mean", da.k_mean, perm.get("k_network_mean"))
        for i, ax in enumerate("xyz"):
            _cmp(out, "permeability k_%s" % ax, da.k_face[i],
                 (perm.get("k_network_face") or [None] * 3)[i])
        sup = perm.get("support", {})
        cube = DA.support_stats(net, N.cube_fraction(net.p0, net.p1, np.asarray(c), h),
                                (2 * h) ** 3)
        for k in ("phi", "d_perm", "T2"):
            _cmp(out, "support %s" % k, cube.get(k), sup.get(k))

    # ---- transport, the expensive part ------------------------------------
    if n_transport and rec.get("voxels"):
        rin = np.random.default_rng(seed + 1_000_000)
        radii = tuple(eff.get("radii_mm", cfg["shell_radii_mm"]))
        inputs = {q: S.input_voxels(rin, cc, cfg["rve_half"], int(eff.get("n_inputs", 24)),
                                    cfg["vox_half_mm"] * 1e-3, max(radii) * 1e-3)
                  for q, cc in enumerate(centres)}
        R = max(radii) * 1e-3 if len(radii) == 1 else 1.0e-3
        vox = cfg["vox_half_mm"] * 1e-3
        n_dir = int(eff.get("n_dir", 48))
        for i in range(min(n_transport, len(rec["voxels"]))):
            ci = inputs[rve][i]
            _cmp(out, "voxel %d centre x (mm)" % i, ci[0] * 1e3,
                 rec["voxels"][i]["centre_mm"][0], 1e-12)
            ks = TR.shell_kernels(ci, R, vox, n_dir=n_dir)
            tr = TR.propagate(net, fl, ks, poiseuille=False)
            outs = [tr.tic[k] for k in range(1, len(ks))]
            v, D, npair, r2 = TR.identify_shell(tr.t, tr.tic[0], outs, R,
                                                tf="new", rule="front20")
            want = next((e for e in rec["voxels"][i]["est"]
                         if e.get("kernel") == "plug" and abs(e.get("R_mm", 0) - R * 1e3) < 1e-9
                         and e.get("tf") == "new" and e.get("rule") == "front20"), None)
            if want is None:
                out.append(("voxel %d estimate" % i, "no matching setting in the record",
                            float("inf"), False))
                continue
            _cmp(out, "voxel %d v" % i, v, want.get("v"), 1e-6)
            _cmp(out, "voxel %d D" % i, D, want.get("D"), 1e-6)
    return man, rec, out


def report(folder, man, out, seed, rve):
    print("=" * 78)
    print("%s   seed %d, volume %d" % (folder, seed, rve))
    print("  archive written by %s" % (man.get("code_sha256") or "?")[:12])
    print("  this checkout is    %s" % S.code_hash()[:12])
    print("  %-28s %-46s %s" % ("quantity", "now vs archive", "rel. diff"))
    worst = 0.0
    for label, text, d, ok in out:
        if ok is None:
            print("  %-28s %-46s %10s   NOT STORED" % (label, text, "-"))
            continue
        worst = max(worst, d if np.isfinite(d) else 1e300)
        print("  %-28s %-46s %10.2e %s" % (label, text, d, "" if ok else "  DIFFERS"))
    bad = [o for o in out if o[3] is False]
    absent = [o for o in out if o[3] is None]
    compared = len(out) - len(absent)
    print()
    if not bad:
        print("  REPRODUCED, this record only: all %d compared quantities agree\n"
              "  within the tolerance stated for each (%g relative for the\n"
              "  deterministic blocks, 1e-6 for a fitted estimate).  This covers the\n"
              "  quantities compared above in this record; it is not a statement\n"
              "  about the archive, nor about other archives that share its source\n"
              "  hash." % (compared, TOL))
    else:
        print("  NOT REPRODUCED: %d of %d compared quantities differ.  The archive\n"
              "  and this code do not compute the same thing; the differing rows say\n"
              "  where." % (len(bad), compared))
    if absent:
        print("\n  %d quantity(ies) are not stored by this archive and could not be\n"
              "  compared: %s.  They are excluded from the verdict above, which\n"
              "  therefore covers less than the full record."
              % (len(absent), ", ".join(o[0] for o in absent)))
    return not bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rve", type=int, default=0)
    ap.add_argument("--arm", default="baseline")
    ap.add_argument("--transport", type=int, default=0,
                    help="also recompute the estimate for this many input voxels")
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args(argv)
    man, rec, out = reproduce(a.folder, a.seed, a.rve, a.arm, a.transport, a.partial)
    ok = report(a.folder, man, out, a.seed, a.rve)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
