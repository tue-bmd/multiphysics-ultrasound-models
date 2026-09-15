"""Extract a (phi, k, D, |v|) ensemble from the vascular-network simulator.

    python scripts/network_ensemble.py --porovasc /path/to/porovascular-tissue-sim \
                                       --seed 100 --centres 4 --out results/ensemble.csv

One row per sampling volume.  On the same support and the same gland:

  phi   vascular volume fraction of the cube of half-width ``--half``;
  k     face-to-face network permeability of that cube, the response to an
        imposed macroscopic gradient;
  D, v  the effective dispersion and drift a convection-dispersion model fitted
        to the propagated contrast curves reports on a spherical shell of radius
        ``--shell`` about the same center.

These are the quantities the reduced model in this package calls phi, k, D and
|v|, measured on the network rather than assumed.  The fractional vascular
compliance C_v is deliberately not extracted: in the network simulator it is
1 / (H + K_wall) by construction, so an ensemble would only return the value it
was given.

The script is an importer.  It does not modify the vascular-network package, and
that package is frozen.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--porovasc", required=True, help="path to porovascular-tissue-sim")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--centres", type=int, default=4)
    ap.add_argument("--half", type=float, default=2.0e-3, help="cube half-width [m]")
    ap.add_argument("--shell", type=float, default=1.0e-3, help="shell radius [m]")
    ap.add_argument("--vox", type=float, default=0.375e-3, help="voxel half-width [m]")
    ap.add_argument("--n-dir", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.porovasc)
    from porovasc.geometry import network as N
    from porovasc.physics import flow as F, transport as TR
    from porovasc.homogenise import darcy as DA
    from porovasc.config import BASE, apply_globals

    apply_globals(dict(BASE))
    net = N.build(N.Params(seed=a.seed))
    fl = F.solve(net)

    rng = np.random.default_rng(a.seed)
    centres = []
    while len(centres) < a.centres:
        c = rng.normal(0, 4e-3, 3)
        if N.in_gland(c):
            centres.append(c)

    allk, index = [], []
    for c in centres:
        ks = TR.shell_kernels(c, a.shell, a.vox, n_dir=a.n_dir)
        index.append((len(allk), len(ks)))
        allk += ks
    tr = TR.propagate(net, fl, allk, poiseuille=False)

    rows = []
    for (c, (i0, nk)) in zip(centres, index):
        d = DA.measure(net, fl, c, a.half)
        tin = tr.tic[i0]
        if tin.sum() <= 0 or not np.isfinite(d.k_mean) or d.k_mean <= 0:
            continue
        outs = [tr.tic[j] for j in range(i0 + 1, i0 + nk)]
        v, D, n_pairs, r2 = TR.identify_shell(tr.t, tin, outs, a.shell,
                                              tf="new", rule="front20")
        if not (np.isfinite(v) and np.isfinite(D) and v > 0 and D > 0):
            continue
        rows.append(dict(seed=a.seed,
                         cx=float(c[0]), cy=float(c[1]), cz=float(c[2]),
                         phi=float(d.phi), k=float(d.k_mean),
                         D=float(D), vmag=float(v),
                         n_pairs=int(n_pairs), r2=float(r2),
                         n_seg=int(net.n)))

    fields = ["seed", "cx", "cy", "cz", "phi", "k", "D", "vmag",
              "n_pairs", "r2", "n_seg"]
    new = not os.path.exists(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print("seed %d: %d usable of %d volumes -> %s"
          % (a.seed, len(rows), a.centres, a.out))


if __name__ == "__main__":
    main()
