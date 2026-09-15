"""Does a single-continuum drift-dispersion closure describe this network?

    python docs/homogenisation_diagnostic.py --seed 100

A diagnostic, not part of the package: it changes nothing and adds nothing to
the model.  It asks, at a range of averaging volumes, whether the quantities a
one-compartment continuum needs actually exist.

Three questions, after the scheme proposed for this purpose:

  chi      the flow-direction coherence of an averaging volume,

               chi = || sum_e Q_e L_e t_e || / sum_e |Q_e| L_e,

           the ratio of the vector-averaged flux to the scalar throughput.
           chi near 1 means the blood in the volume moves coherently and a
           single drift velocity is meaningful; chi much less than 1 means
           arterial and venous branches carry blood in opposing directions and
           cancel, so throughput can be large while the mean vector velocity is
           near zero.  The two are different quantities and only the second
           enters a Darcy closure.

  scale    whether chi, the vascular volume fraction and the permeability
           settle as the averaging volume grows.  If they settle and chi
           becomes appreciable, a single continuum is adequate at that scale
           and the network is merely sparse below it.  If chi keeps falling,
           the cancellation is structural rather than a sampling artefact.

  pressure whether the vascular pressure inside one averaging volume is
           single-valued or separates into arterial and venous populations.  A
           single-compartment Darcy law needs a single-valued p.

Permeability here is the response to an imposed macroscopic gradient, measured
by the face-to-face experiment of homogenise.darcy: seal four faces, hold the
other two at fixed pressures, solve, and average the flux.  That is a property
of the network's connectivity, and it is not the same thing as the native flux,
which is what chi measures.
"""
from __future__ import annotations
import argparse

import numpy as np

from porovasc.geometry import network as N
from porovasc.physics import flow as F
from porovasc.homogenise import darcy as DA
from porovasc.config import BASE, apply_globals


def chi_and_structure(net, fl, centre, R):
    """Coherence, volume fraction and throughput of one spherical volume.

    The coherence itself is `homogenise.darcy.coherence`, so that the diagnostic
    and the recorded descriptor cannot drift apart; this adds the per-side
    pressures, which only the diagnostic uses."""
    out = DA.coherence(net, fl, centre, R)
    if out["chi"] is None and out["n_seg"] < 10:
        return None
    w = N.sphere_fraction(net.p0, net.p1, np.asarray(centre), R)
    m = w > 0
    pn = fl.p[:net.n]
    for side, sel in (("art", m & net.art), ("vein", m & net.vein)):
        if sel.sum():
            ps = pn[sel] / F.MMHG
            out["p_%s" % side] = (float(ps.min()), float(np.median(ps)), float(ps.max()))
    return out


def bimodality(net, fl, centre, R):
    """How much of the pressure range inside the volume is empty between the
    arterial and venous populations.  One means fully separated."""
    w = N.sphere_fraction(net.p0, net.p1, np.asarray(centre), R)
    m = w > 0
    pn = fl.p[:net.n] / F.MMHG
    a, v = pn[m & net.art], pn[m & net.vein]
    if a.size < 5 or v.size < 5:
        return None
    gap = a.min() - v.max()
    span = max(a.max(), v.max()) - min(a.min(), v.min())
    return float(gap / span) if span > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--permeability", action="store_true",
                    help="also measure the face-to-face permeability (slow)")
    a = ap.parse_args()
    apply_globals(dict(BASE))
    net = N.build(N.Params(seed=a.seed))
    fl = F.solve(net)
    print("gland seed %d: %d segments" % (a.seed, net.n))
    rng = np.random.default_rng(0)
    centres = []
    while len(centres) < a.samples:
        c = rng.normal(0, 5e-3, 3)
        if N.in_gland(c):
            centres.append(c)

    print("\n%8s %7s %9s %9s %12s %12s %9s" %
          ("R (mm)", "n_seg", "chi", "phi", "Darcy mm/s", "vessel mm/s", "p gap"))
    for R in (0.5e-3, 1.0e-3, 2.0e-3, 3.0e-3, 5.0e-3, 8.0e-3):
        rows = [chi_and_structure(net, fl, c, R) for c in centres]
        rows = [r for r in rows if r]
        if not rows:
            continue
        gaps = [g for g in (bimodality(net, fl, c, R) for c in centres) if g is not None]
        print("%8.2f %7d %9.4f %9.4f %12.4f %12.2f %9s"
              % (R * 1e3, int(np.median([r["n_seg"] for r in rows])),
                 np.median([r["chi"] for r in rows]),
                 np.median([r["phi"] for r in rows]),
                 np.median([r["darcy_speed"] for r in rows]) * 1e3,
                 np.median([r["vessel_speed"] for r in rows]) * 1e3,
                 "%.2f" % np.median(gaps) if gaps else "-"))

    if a.permeability:
        print("\nface-to-face permeability against cube half-width (imposed gradient)")
        print("%8s %7s %13s %9s" % ("h (mm)", "n_seg", "k (m^2)", "phi"))
        for h in (1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3):
            ks, ph = [], []
            for c in centres[:12]:
                try:
                    d = DA.measure(net, fl, c, h)
                except Exception:
                    continue
                st = DA.support_stats(net, N.cube_fraction(net.p0, net.p1, np.asarray(c), h),
                                      (2 * h) ** 3)
                if np.isfinite(d.k_mean) and d.k_mean > 0:
                    ks.append(d.k_mean); ph.append(st["phi"])
            if ks:
                print("%8.2f %7s %13.3e %9.4f"
                      % (h * 1e3, "-", np.median(ks), np.median(ph)))


if __name__ == "__main__":
    main()
