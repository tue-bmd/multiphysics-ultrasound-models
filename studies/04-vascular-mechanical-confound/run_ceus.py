"""Stage A (expensive, cached): tabulate the CEUS observables against s.

    python run_ceus.py --dterm 40e-6 --seed 1 --out results/ceus_seed1.json

One flow solve and one shell propagation per value of s.  Re-run only when the
network or the acquisition changes; `run_inference.py` reads the table.
"""
import argparse
import time

import numpy as np
from porovasc.geometry import network as N

from vmconf import ceus, provenance

ap = argparse.ArgumentParser()
ap.add_argument("--dterm", type=float, default=40e-6)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--s-min", type=float, default=0.60)
ap.add_argument("--s-max", type=float, default=1.05)
ap.add_argument("--n-s", type=int, default=10)
ap.add_argument("--dz", type=float, default=1.5e-3, help="shell radius [m]")
ap.add_argument("--n-dir", type=int, default=48)
ap.add_argument("--n-inputs", type=int, default=8,
                help="input-voxel locations summarized per radius scale")
ap.add_argument("--estimator", action="store_true",
                help="also run the shell contrast-kinetic estimator.  Off by "
                     "default: the observable is the CEUS AUC, which needs "
                     "no estimator and is ~40x cheaper to compute.")
ap.add_argument("--out", default="results/ceus.json")
a = ap.parse_args()

centre = (0.0, -5e-3, 0.0)
net = N.build(N.Params(n_feeders=4, d_term_gland=a.dterm,
                       d_term_rve=max(a.dterm - 10e-6, 20e-6),
                       rve_centres=(centre,), rve_half=3e-3, seed=a.seed))
print("network: %d segments" % net.n)
s_values = np.linspace(a.s_min, a.s_max, a.n_s)
t0 = time.time()
table = ceus.response_table(net, s_values, centre, dz=a.dz, n_dir=a.n_dir,
                            n_inputs=a.n_inputs, estimator=a.estimator)
print("tabulated %d radius scales in %.1f s" % (len(s_values), time.time() - t0))
print("\n    s          CEUS AUC       IQR across inputs" +
      ("     v [mm/s]  D [mm2/s]  (recorded, NOT used)" if a.estimator else ""))
for r in table["rows"]:
    line = "  %5.3f    %11.4e     %11.4e" % (r["s"], r["auc"], r.get("auc_iqr", 0.0))
    if a.estimator:
        line += "     %8.3f  %9.3f" % (1e3 * r["v"], 1e6 * r["D"])
    print(line)
print("\nObservable: area under the CEUS time-intensity curve (AUC).  No "
      "contrast-kinetic estimator is used.")
print("It is the calibrated-amplitude case: informative about the radius scale "
      "only when the input")
print("concentration and the acoustic sensitivity are calibrated.")
prov = provenance.record(net, **vars(a))
provenance.save(a.out, prov, dict(table=table))
print("\nwritten", a.out)
