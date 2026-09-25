"""Validity and resolution diagnostics for the reduced mechanical model.

1. Poiseuille-invalid vessels. The response is bounded by comparing all vessel
   contributions with a conservative calculation that removes contributions
   outside the quasi-static validity range at each frequency.

2. The lumped capillary bed.  The explicit tree stops at d_term and everything
   below it is a lumped conductance.  The transport solve uses that bed; the
   mechanical model does not see it at all.  That asymmetry matters, because the
   point of the study is that both observables come from one vasculature.  Its
   size is measured here as a resolution sensitivity: refine the explicit tree
   and see whether the mechanical result converges.
"""
import argparse

import numpy as np
from porovasc.geometry import network as N

from vmconf import forward, mech, provenance

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--mu", type=float, default=2000.0)
ap.add_argument("--eta", type=float, default=1.0)
ap.add_argument("--share", type=float, default=0.20)
ap.add_argument("--dterms", type=float, nargs="+", default=[60e-6, 40e-6, 30e-6])
ap.add_argument("--out", default="results/diagnostics.json")
a = ap.parse_args()

centre, half = (0.0, -5e-3, 0.0), 3e-3
NU = 3.6e-3 / 1050.0
payload = {}

print("=" * 72)
print("1. CALIBER STRATIFICATION AND THE POISEUILLE BRACKET")
print("=" * 72)
d_bracket = min(a.dterms)
net = N.build(N.Params(n_feeders=4, d_term_gland=d_bracket,
                       d_term_rve=max(d_bracket - 10e-6, 20e-6),
                       rve_centres=(centre,), rve_half=half, seed=a.seed))
vol, phi = forward.region_weights(net, centre, half)
m = vol > 0
r, w = net.r[m], vol[m] / vol[m].sum()
print("region: %d segments, d_term = %.0f um, phi = %.4f"
      % (m.sum(), 1e6 * d_bracket, phi))
edges = np.array([0, 15, 25, 40, 75, 150, 1e9]) * 1e-6      # radius bins
print("\n  radius band     share of    share of    max valid")
print("   (um)            segments    lumen vol   freq (Hz)")
strat = []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (r >= lo) & (r < hi)
    if not sel.any(): continue
    fmax = 7 * NU / max(hi, 1e-9) ** 2 / (2 * np.pi)
    print("  %5.0f - %-7s  %8.3f    %8.3f    %9.1f"
          % (1e6 * lo, ("%.0f" % (1e6 * hi)) if hi < 1e8 else "inf",
             sel.mean(), w[sel].sum(), fmax))
    strat.append(dict(lo_um=1e6 * lo, hi_um=None if hi > 1e8 else 1e6 * hi,
                      frac_segments=float(sel.mean()), frac_volume=float(w[sel].sum()),
                      f_max_Hz=float(fmax)))
payload["stratification"] = strat

om = mech.band(50.0, 200.0, 24)
w_all, tau_all = mech.segment_times(net.r, net.Lpath, vol, a.mu)
dG = mech.amplitude_for_share(w_all, tau_all, a.eta, a.share)
print("\n   f(Hz)   invalid    speed, all      speed, valid     bracket")
print("           weight     vessels (m/s)   only (m/s)       width (%)")
brack = []
for i, f in enumerate([50.0, 100.0, 158.1, 200.0]):
    o = np.array([2 * np.pi * f])
    keep = net.r <= np.sqrt(7 * NU / o[0])
    wk, tk = mech.segment_times(net.r, net.Lpath, vol * keep, a.mu)
    retained = float(vol[keep].sum() / vol.sum())
    inv = float(mech.poiseuille_validity(r, w, o)[0])
    gA = mech.kelvin_voigt(o, a.mu, a.eta) + mech.vascular_modulus(w_all, tau_all, o, dG)
    gV = mech.kelvin_voigt(o, a.mu, a.eta) + mech.vascular_modulus(
        wk, tk, o, dG * retained
    )
    cA = mech.shear_wave(gA, o)[0][0]; cV = mech.shear_wave(gV, o)[0][0]
    print("  %6.1f   %7.2f    %12.4f   %12.4f   %9.2f"
          % (f, inv, cA, cV, 100 * abs(cA - cV) / cA))
    brack.append(dict(f=f, invalid_weight=inv, retained_weight=retained,
                      c_all=float(cA), c_valid=float(cV),
                      bracket_pct=float(100 * abs(cA - cV) / cA)))
payload["poiseuille_bracket"] = brack
payload["poiseuille_bracket_d_term_um"] = float(1e6 * d_bracket)

print("\n" + "=" * 72)
print("2. WHERE THE EXPLICIT TREE STOPS: RESOLUTION SENSITIVITY")
print("=" * 72)
print("\n Two amplitude conventions, side by side.  'share' re-derives the vascular")
print(" increment at each resolution to keep a fixed fraction of measured viscosity;")
print(" 'coupling' holds a fixed dimensionless constant k calibrated on the coarsest")
print(" network, so a larger phi gives a larger increment.")
print("\n d_term   n_seg    phi     tau_gm   in-band  | fixed SHARE      | fixed COUPLING")
print("  (um)                              (ms)             | dG(Pa)  c(100Hz) | dG(Pa)  c(100Hz)")
res = []
k_fixed = None
for dt in sorted(a.dterms, reverse=True):
    nn = N.build(N.Params(n_feeders=4, d_term_gland=dt,
                          d_term_rve=max(dt - 10e-6, 20e-6),
                          rve_centres=(centre,), rve_half=half, seed=a.seed))
    v2, p2 = forward.region_weights(nn, centre, half)
    w2, t2 = mech.segment_times(nn.r, nn.Lpath, v2, a.mu)
    d2 = mech.amplitude_for_share(w2, t2, a.eta, a.share)
    if k_fixed is None:                       # calibrate k once, on the coarsest
        k_fixed = d2 / (p2 * a.mu)
    d2f = forward.amplitude_fixed_coupling(k_fixed, p2, a.mu)
    s2 = mech.spectrum_summary(w2, t2, 50.0, 200.0)
    o = np.array([2 * np.pi * 100.0])
    gm = mech.kelvin_voigt(o, a.mu, a.eta)
    c = mech.shear_wave(gm + mech.vascular_modulus(w2, t2, o, d2), o)[0][0]
    cf = mech.shear_wave(gm + mech.vascular_modulus(w2, t2, o, d2f), o)[0][0]
    print(" %5.0f  %7d  %.4f  %7.3f  %7.2f  | %6.1f %8.4f | %6.1f %8.4f"
          % (1e6 * dt, nn.n, p2, 1e3 * s2["tau_gm"], s2["inband"], d2, c, d2f, cf))
    res.append(dict(d_term_um=1e6 * dt, n_seg=int(nn.n), phi=float(p2),
                    tau_gm_ms=float(1e3 * s2["tau_gm"]), inband=float(s2["inband"]),
                    c_100Hz=float(c), dG=float(d2),
                    c_100Hz_fixed_coupling=float(cf), dG_fixed_coupling=float(d2f),
                    k_fixed=float(k_fixed)))
payload["resolution"] = res

if len(res) > 1:
    a0, a1 = res[0], res[-1]
    print("\n  across the tested range: phi %+.1f%%, c(100Hz) %+.2f%% at fixed share, "
          "%+.2f%% at fixed coupling"
          % (100 * (a1["phi"] / a0["phi"] - 1),
             100 * (a1["c_100Hz"] / a0["c_100Hz"] - 1),
             100 * (a1["c_100Hz_fixed_coupling"] / a0["c_100Hz_fixed_coupling"] - 1)))
    print("  If the two differ, resolution-independence under a fixed share is a "
          "property of the")
    print("  normalization and not of the tissue, and must be reported as such.")

print("\nNote: the mechanical model sees ONLY the explicit tree.  The lumped")
print("capillary bed below d_term carries blood volume and is present in the")
print("transport solve, so the two observables are built on different amounts")
print("of vasculature.  If phi and the response keep moving as d_term falls,")
print("the mechanical result has not converged and the asymmetry is material.")
provenance.save(a.out, provenance.record(net, **vars(a)), payload)
print("\nwritten", a.out)
