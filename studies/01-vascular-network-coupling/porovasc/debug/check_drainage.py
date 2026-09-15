"""Drainage invariants.

  1. compliance identity: the drained fraction tends to 1, i.e. the volume
     expelled equals sum_region C_i dP_ext.  Hence the fractional tissue
     relaxation is phi * dP_ext / H_eff: the closure's central claim, derived.
  2. scaling: tau_rc ~ 1/H (H doubled -> tau halved) and tau_rc ~ 1/d^2 under
     uniform radius scaling.
  3. tau_rc is an output, expected in the tens-of-ms range.
  4. spectrum: report its width; a single-tau model is an approximation.
"""
from __future__ import annotations
import numpy as np
from ..geometry.network import Network
from ..physics.flow import Flow
from ..physics import drainage as D


def _hdr(t): print("\n" + t); print("-" * len(t))
def _pf(ok, msg): print(("PASS  " if ok else "FAIL  ") + msg); return ok


def check(net: Network, fl: Flow, centre, R_comp, H=5e3, dP=500.0) -> bool:
    ok = True
    d0 = D.solve(net, fl, centre, R_comp, dP_ext=dP, H=H)

    _hdr("1. Compliance identity and the relaxation amplitude")
    print("      compressed region: sphere R = %.1f mm, %d segments, blood volume %.3f mm^3 in %.1f mm^3 tissue"
          % (R_comp * 1e3, d0.in_region.sum(), d0.V_blood * 1e9, d0.V_tissue * 1e9))
    print("      phi_region = %.2f %%" % (100 * d0.phi))
    ok &= _pf(abs(d0.f[-1] - 1) < 0.02, "drained fraction at t=%.1f s = %.4f (expected 1)" % (d0.t[-1], d0.f[-1]))
    eps_applied = dP / H
    relax = -d0.dV_inf / d0.V_tissue
    print("      applied strain dP/H = %.2e ; fractional tissue relaxation = %.2e ; ratio = %.4f  vs phi = %.4f"
          % (eps_applied, relax, relax / eps_applied, d0.phi))
    ok &= _pf(abs(relax / eps_applied - d0.phi) / d0.phi < 1e-6,
              "fractional relaxation / applied strain == phi_region (derived on the network)")

    _hdr("2. Time constant (an OUTPUT) and its spectrum")
    print("      tau_rc (time to f = 1 - 1/e = 0.632) = %.1f ms" % (d0.tau_rc * 1e3))
    w = d0.spectrum; tg = d0.tau_grid
    lt = np.log(tg); mu = (w * lt).sum(); sd = np.sqrt((w * (lt - mu) ** 2).sum())
    print("      spectrum: log-mean tau = %.1f ms, log-sd = %.2f (factor %.1f); 10-90%% [%.1f, %.1f] ms"
          % (np.exp(mu) * 1e3, sd, np.exp(sd),
             np.exp(np.interp([0.1, 0.9], np.cumsum(w), lt))[0] * 1e3,
             np.exp(np.interp([0.1, 0.9], np.cumsum(w), lt))[1] * 1e3))
    for tt in (0.005, 0.01, 0.02, 0.05, 0.1, 0.5):
        print("        f(%.3f s) = %.3f" % (tt, np.interp(tt, d0.t, d0.f)))
    ok &= _pf(1e-3 < d0.tau_rc < 1.0, "tau_rc within 1 ms .. 1 s (vascular branch)")

    _hdr("3. Scaling tests")
    d1 = D.solve(net, fl, centre, R_comp, dP_ext=dP, H=2 * H)
    rH = d1.tau_rc / d0.tau_rc
    ok &= _pf(abs(rH - 0.5) < 0.05, "H doubled -> tau_rc ratio %.3f (expected 0.500)" % rH)
    d2 = D.solve(net, fl, centre, R_comp, dP_ext=dP, H=H, radius_scale=1.2)
    rd = d2.tau_rc / d0.tau_rc
    ok &= _pf(abs(rd - 1 / 1.2 ** 2) < 0.08, "radii x1.2 -> tau_rc ratio %.3f (expected %.3f = 1/1.2^2)" % (rd, 1 / 1.44))
    print("      (radius scaling changes C by r^2 and G by r^4; the bed conductance is held fixed, hence the tolerance)")

    _hdr("4. Sensitivity of tau_rc to the compressed-region size")
    for Rc in (0.5 * R_comp, R_comp, 2 * R_comp):
        dd = D.solve(net, fl, centre, Rc, dP_ext=dP, H=H)
        print("      R_comp = %.1f mm: tau_rc = %.1f ms, phi_region = %.2f %%" % (Rc * 1e3, dd.tau_rc * 1e3, 100 * dd.phi))

    print("\n" + "=" * 60 + "\nDRAINAGE " + ("OK" if ok else "NOT OK"))
    return ok, d0
