"""Stage B: posterior inference, the comparison arms, and coverage.

    python run_inference.py --ceus results/ceus.json --reps 40

Reads the tabulated CEUS response, builds the SWE forward model on the same
network, then for each repetition draws noise, evaluates every arm on a dense
(mu, eta, s) grid and records the posterior summaries.  Coverage is reported
with its binomial standard error, because forty repetitions do not pin a 90%
interval to better than about five points.
"""
import argparse
import json
import time

import numpy as np
from porovasc.geometry import network as N

from vmconf import ceus as CE
from vmconf import forward, mech, provenance
from vmconf import inference as INF
from vmconf import likelihood as LK

ap = argparse.ArgumentParser()
ap.add_argument("--ceus", default="results/ceus.json")
ap.add_argument("--reps", type=int, default=40)
ap.add_argument("--mu", type=float, default=2000.0)
ap.add_argument("--eta", type=float, default=1.0)
ap.add_argument("--s-true", type=float, default=0.80)
ap.add_argument("--share", type=float, default=0.20)
ap.add_argument("--sigma-swe", type=float, default=0.03)
ap.add_argument("--sigma-ceus", type=float, default=0.10)
ap.add_argument("--relation-sd", type=float, default=0.02)
ap.add_argument("--relation-offset", type=float, default=0.05)
ap.add_argument("--offset-sweep", type=float, nargs="*",
                default=[0.0, 0.02, 0.05, 0.10, 0.15, 0.20],
                help="displacements used for coupling-relation sensitivity; "
                     "an empty list disables the sweep")
ap.add_argument("--with-alpha", action="store_true")
ap.add_argument("--f-lo", type=float, default=50.0)
ap.add_argument("--f-hi", type=float, default=200.0,
                help="upper band edge; above ~200 Hz the Poiseuille assumption "
                     "fails for a large share of the lumen weight")
ap.add_argument("--n-mu", type=int, default=61)
ap.add_argument("--n-eta", type=int, default=61)
ap.add_argument("--n-s", type=int, default=61)
ap.add_argument("--out", default="results/inference.json")
a = ap.parse_args()

with open(a.ceus) as f:
    blob = json.load(f)
tab = blob["table"]
cfg = blob["provenance"]["settings"]
centre = (0.0, -5e-3, 0.0)
net = N.build(N.Params(n_feeders=4, d_term_gland=cfg["dterm"],
                       d_term_rve=max(cfg["dterm"] - 10e-6, 20e-6),
                       rve_centres=(centre,), rve_half=3e-3, seed=cfg["seed"]))
half = 3e-3
omega = mech.band(a.f_lo, a.f_hi, 24)

vol, phi0 = forward.region_weights(net, centre, half)
w, tau_ref = mech.segment_times(net.r, net.Lpath, vol, a.mu)
resp = mech.VascularResponse(w, tau_ref, a.mu)
dG = mech.amplitude_for_share(w, tau_ref, a.eta, a.share, f_ref=200.0)
inval = mech.poiseuille_validity(net.r[vol > 0], vol[vol > 0], omega)
print("phi %.4f   dG %.1f Pa   Poiseuille-invalid weight: %.0f%% at %.0f Hz, %.0f%% at %.0f Hz"
      % (phi0, dG, 100 * inval[0], a.f_lo, 100 * inval[-1], a.f_hi))

interp = CE.Interpolant(tab)
grid = INF.Grid3(np.linspace(0.5 * a.mu, 1.6 * a.mu, a.n_mu),
                 np.linspace(0.3 * a.eta, 2.2 * a.eta, a.n_eta),
                 np.linspace(interp.s[0], interp.s[-1], a.n_s))
if not bool(interp.in_range(a.s_true)):
    raise ValueError("s_true lies outside the tabulated CEUS range")

swe_true = INF.swe_predict(resp, omega, a.mu, a.eta, a.s_true, dG, with_alpha=a.with_alpha)
swe_pred = INF.SwePredictor(grid, resp, omega, dG, with_alpha=a.with_alpha)
print("SWE grid tensor %s, %.0f MB" % (swe_pred.pred.shape, swe_pred.pred.nbytes / 1e6))
ceus_true = np.atleast_1d(np.asarray(interp(a.s_true), float))

ARMS = ["swe", "ceus", "joint_free", "joint_shared", "joint_wrong", "perfect_model"]
rng = np.random.default_rng(12345)
# Draw every realization up front, so the arms and the offset sweep are
# evaluated on identical data and the comparison is paired
DATA = [dict(swe=LK.add_relative_noise(swe_true, a.sigma_swe, rng),
             ceus=LK.add_relative_noise(ceus_true, a.sigma_ceus, rng))
        for _ in range(a.reps)]
acc = {arm: [] for arm in ARMS}
t0 = time.time()
for rep in range(a.reps):
    data = DATA[rep]
    for arm in ARMS:
        lp = INF.log_posterior(grid, arm, data, swe_pred,
                               a.sigma_swe, a.sigma_ceus, ceus_interp=interp,
                               s_true=a.s_true, relation_sd=a.relation_sd,
                               relation_offset=a.relation_offset)
        acc[arm].append(dict(mu=grid.summary(lp, 0), eta=grid.summary(lp, 1),
                             s=grid.summary(lp, 2)))
    if rep == 0:
        print("first repetition in %.1f s; %d to go" % (time.time() - t0, a.reps - 1))

truth = dict(mu=a.mu, eta=a.eta, s=a.s_true)
print("\n%-20s %-26s %-26s %-26s" % ("arm", "mu", "eta", "s"))
print("%-20s %-26s %-26s %-26s" % ("", "bias%  width%  cover", "bias%  width%  cover",
                                   "bias%  width%  cover"))
summary = {}
for arm in ARMS:
    row = {}
    cells = []
    for p in ("mu", "eta", "s"):
        med = np.array([r[p]["median"] for r in acc[arm]])
        wid = np.array([r[p]["width"] for r in acc[arm]])
        cov = np.mean([(r[p]["lo"] <= truth[p] <= r[p]["hi"]) for r in acc[arm]])
        se = float(np.sqrt(max(cov * (1 - cov), 1e-12) / a.reps))
        row[p] = dict(bias_pct=float(100 * (med.mean() - truth[p]) / truth[p]),
                      width_pct=float(100 * wid.mean() / truth[p]),
                      coverage=float(cov), coverage_se=float(se))
        cells.append("%+6.2f %6.2f %.2f+-%.2f" % (row[p]["bias_pct"], row[p]["width_pct"], cov, se))
    summary[arm] = row
    print("%-20s %-26s %-26s %-26s" % (arm, cells[0], cells[1], cells[2]))

# ---- sensitivity to displacement of the coupling relation ----------------
sweep = []
sweep_by_repetition = []
if a.offset_sweep:
    print("\ncoupling displaced by relation_offset, evaluated on the same data:")
    print("  offset     s: bias%%   width%%   coverage        mu: bias%%  width%%")
    for off in a.offset_sweep:
        rows = []
        for rep in range(a.reps):
            lp = INF.log_posterior(grid, "joint_shared" if off == 0 else "joint_wrong",
                                   DATA[rep], swe_pred, a.sigma_swe, a.sigma_ceus,
                                   ceus_interp=interp, s_true=a.s_true,
                                   relation_sd=a.relation_sd, relation_offset=off)
            rows.append(dict(mu=grid.summary(lp, 0), eta=grid.summary(lp, 1),
                             s=grid.summary(lp, 2)))
        r = {}
        for q in ("s", "mu", "eta"):
            med = np.array([x[q]["median"] for x in rows])
            wid = np.array([x[q]["width"] for x in rows])
            cov = np.mean([(x[q]["lo"] <= truth[q] <= x[q]["hi"]) for x in rows])
            r[q] = dict(bias_pct=float(100 * (med.mean() - truth[q]) / truth[q]),
                        width_pct=float(100 * wid.mean() / truth[q]),
                        coverage=float(cov),
                        coverage_se=float(np.sqrt(max(cov * (1 - cov), 1e-12) / a.reps)))
        print("  %6.3f   %+8.2f %8.2f   %.2f+-%.2f      %+8.2f %8.2f"
              % (off, r["s"]["bias_pct"], r["s"]["width_pct"], r["s"]["coverage"],
                 r["s"]["coverage_se"], r["mu"]["bias_pct"], r["mu"]["width_pct"]))
        sweep.append(dict(offset=float(off), **r))
        sweep_by_repetition.append(dict(offset=float(off), posterior=rows))
    print("  A displaced relation biases the estimate.  The interval does widen in")
    print("  response - the data partly detect the inconsistency - but not enough to")
    print("  keep coverage, so the failure shows in coverage and NOT in interval")
    print("  width alone.  Reporting contraction without coverage would miss it.")

prov = provenance.record(net, omega=omega.tolist(), dG=float(dG),
                         poiseuille_invalid=inval.tolist(), **vars(a))
provenance.save(
    a.out,
    prov,
    dict(
        truth=truth,
        observations=DATA,
        posterior_by_repetition=acc,
        summary=summary,
        offset_sweep=sweep,
        offset_sweep_by_repetition=sweep_by_repetition,
    ),
)
print("\nwritten", a.out)
print("coverage is reported with its binomial standard error; %d repetitions "
      "give about +-%.2f near 0.9" % (a.reps, np.sqrt(0.9 * 0.1 / a.reps)))
