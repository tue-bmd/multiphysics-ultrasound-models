"""Flow invariants and physiological checks.

Physiology anchors.  Bracketed numbers index the reference list in this study's
README.  A quantity with no verifiable source carries no comparison.

  perfusion                  15-20 mL/min/100 g, 0.21 mL/g/min normal [7]
  velocity vs diameter       Q = 108 d^3.101, Q in um^3/s and d in um, arterial
                             side 5-900 um [12], fitted to [13].  Spans the
                             whole tree, so the check is segment by segment.
  arteriolar-end pressure    35 mmHg     IMPOSED by calibration, not a check
  venular-end pressure       18.9 +/- 1.6 mmHg, human postcapillary venules,
                             servo-null micropuncture, lip [14].  NOT imposed.
  capillaries per terminal   24-40, hamster tibialis [15]
  feeder -> terminal transit 1141 +/- 262 ms renal cortex, multi-delay ASL [16]
"""
from __future__ import annotations
import numpy as np
from ..geometry.network import Network, GLAND_SEMI
from ..physics.flow import Flow, ETA, MMHG, P_ART, P_VEN, P_ART_END


def _hdr(t): print("\n" + t); print("-" * len(t))
def _pf(ok, msg): print(("PASS  " if ok else "FAIL  ") + msg); return ok


def check(net: Network, fl: Flow) -> bool:
    ok = True
    n = net.n; A = net.kind == 0; V = net.kind == 1
    seg_end = np.arange(n)

    _hdr("1. Kirchhoff: net flow into every interior node is zero")
    inflow = np.zeros(n + 2)
    np.add.at(inflow, seg_end, fl.Q)           # flow arrives at the end node
    np.add.at(inflow, fl.start, -fl.Q)         # and leaves the start node
    np.add.at(inflow, fl.bed_a, -fl.Q_bed)
    np.add.at(inflow, fl.bed_v, fl.Q_bed)
    interior = np.ones(n + 2, bool); interior[[fl.inlet, fl.outlet]] = False
    Qref = np.abs(fl.Q).max()
    res = np.abs(inflow[interior]).max() / Qref
    ok &= _pf(res < 1e-9, "max |sum Q| / max|Q| over %d interior nodes = %.1e" % (interior.sum(), res))
    Qin = -inflow[fl.inlet]; Qout = inflow[fl.outlet]
    ok &= _pf(abs(Qin - Qout) / Qin < 1e-9, "inlet flow = outlet flow: %.3f vs %.3f mL/min" % (Qin * 6e7, Qout * 6e7))

    _hdr("2. Pressure monotone along every path")
    dpA = fl.p[fl.start[A]] - fl.p[seg_end[A]]     # should be > 0 (flow p0 -> p1)
    dpV = fl.p[seg_end[V]] - fl.p[fl.start[V]]     # veins: flow p1 -> p0, so end > start
    ok &= _pf(dpA.min() > -1e-9 * P_ART, "arteries: pressure drops along p0->p1 in all %d segments (min dp = %.2e Pa)" % (A.sum(), dpA.min()))
    ok &= _pf(dpV.min() > -1e-9 * P_ART, "veins: pressure drops along p1->p0 in all %d segments (min dp = %.2e Pa)" % (V.sum(), dpV.min()))
    ok &= _pf((fl.Q[A] > 0).all() and (fl.Q[V] < 0).all(), "flow direction: arteries forward, veins backward, everywhere")

    _hdr("3. Total gland flow (OUTPUT of the pressure-driven solve)")
    Vg = 4 / 3 * np.pi * np.prod(GLAND_SEMI) * 1e6            # mL
    mass_g = Vg * 1.05
    perf = Qin * 6e7 / mass_g * 100
    print("      total flow %.2f mL/min for a %.1f mL gland -> %.1f mL/min/100 g   (lit 15-20)" % (Qin * 6e7, Vg, perf))
    ok &= _pf(5 < perf < 60, "perfusion within a factor ~3 of physiology")

    _hdr("4. Velocities by caliber (arteries), against the Skinner relation [12]")
    d = net.d * 1e6; v = np.abs(fl.v) * 1e3
    # Skinner 1979 [12]: Q = 108 d^3.101 with Q in um^3/s and d in um, fitted to
    # in situ measurements over 5-900 um on the arterial side.  Dividing by the
    # lumen area gives mean velocity, here in mm/s.
    v_lit = lambda dia_um: 4 * 108 / np.pi * dia_um ** 1.101 / 1e3
    for lo, hi in [(400, 700), (150, 400), (60, 150), (30, 60)]:
        m = A & (d >= lo) & (d < hi)
        if m.sum():
            dm = np.median(d[m]); vs = np.median(v[m]); vl = v_lit(dm)
            print("      %3d-%3d um: n=%5d  d median %5.1f um  v median %5.1f mm/s, "
                  "5-95%% [%5.1f, %5.1f]   lit %5.1f, ratio %.2f"
                  % (lo, hi, m.sum(), dm, vs, *np.percentile(v[m], [5, 95]), vl, vs / vl))
    r = v[A] / v_lit(d[A])
    print("      whole arterial tree: n=%d segments, simulated / literature "
          "median %.2f, 5-95%% [%.2f, %.2f]"
          % (A.sum(), np.median(r), *np.percentile(r, [5, 95])))
    ok &= _pf(0.5 < np.median(r) < 2.0,
              "arterial velocities within a factor of two of the relation of [12]")

    _hdr("5. Pressures at the bed")
    term = A & net.term
    pa = fl.p[np.where(term)[0]] / MMHG
    pv = fl.p[fl.bed_v] / MMHG
    print("      arterial terminals: mean %.1f mmHg (target %.0f, imposed by calibration), 5-95%% [%.1f, %.1f]"
          % (pa.mean(), P_ART_END / MMHG, *np.percentile(pa, [5, 95])))
    print("      venular ends      : mean %.1f mmHg, 5-95%% [%.1f, %.1f]      "
          "(lit 18.9 +/- 1.6 [14], NOT imposed)"
          % (pv.mean(), *np.percentile(pv, [5, 95])))
    ok &= _pf(8 < pv.mean() < 25, "venular-end pressure in physiological range")
    print("      NOTE: below the one direct human measurement located [14]; "
          "see RESULTS section 1")
    print("      inlet %.0f mmHg, outlet %.0f mmHg; arterial tree drop %.1f, bed drop %.1f, venous tree drop %.1f mmHg"
          % (P_ART / MMHG, P_VEN / MMHG, P_ART / MMHG - pa.mean(), pa.mean() - pv.mean(), pv.mean() - P_VEN / MMHG))

    _hdr("6. Bed resistance: calibrated value vs capillary geometry (verification of geometry)")
    R_bed_term = 1.0 / (fl.G_bed0)                     # per 30 um-equivalent connection
    d_cap, L_cap = 8e-6, 500e-6
    R_cap = 8 * ETA * L_cap / (np.pi * (d_cap / 2) ** 4)
    N_cap = R_cap / R_bed_term
    print("      R_bed (per %.0f um terminal) = %.2e Pa s/m^3 ; one 8 um x 0.5 mm capillary = %.2e"
          % (net.params.d_term_rve * 1e6, R_bed_term, R_cap))
    print("      -> implied capillaries per terminal arteriole = %.1f   (lit 24-40 [15])" % N_cap)
    ok &= _pf(2 < N_cap < 100, "implied capillary count is within the loose sanity "
                               "bound 2 to 100 (this bound is NOT the literature range)")
    print("      NOTE: DISAGREES with [15] by 3 to 6x.  This is R_cap / R_bed inferred "
          "from the calibrated")
    print("            bed conductance assuming an 8 um x 0.5 mm capillary, so it goes "
          "as d_cap^-4.")
    Qb_frac = np.abs(fl.Q_bed).sum() / Qin
    ok &= _pf(abs(Qb_frac - 1) < 1e-9, "all inlet flow passes through the bed (%.6f)" % Qb_frac)

    _hdr("7. Transit times, feeder inlet -> arterial terminals")
    ta = fl.t_arr[np.where(term)[0]]
    print("      median %.2f s, 5-95%% [%.2f, %.2f] s   (lit 1.14 +/- 0.26 s renal cortex [16])"
          % (np.median(ta), *np.percentile(ta, [5, 95])))
    ok &= _pf(0.2 < np.median(ta) < 15, "transit time within an order of magnitude of the "
                                        "arterial transit time of [16]")
    # flow-weighted transit spread: the raw material of dispersion
    w = fl.Q[np.where(term)[0]]
    tm = (w * ta).sum() / w.sum(); ts = np.sqrt((w * (ta - tm) ** 2).sum() / w.sum())
    print("      flow-weighted mean %.2f s, sd %.2f s, relative dispersion RD = %.2f" % (tm, ts, ts / tm))

    _hdr("8. Flow reaching the RVEs")
    from ..geometry.network import in_cube
    mid = 0.5 * (net.p0 + net.p1)
    for q, c in enumerate(net.params.rve_centres):
        c = np.asarray(c); h = net.params.rve_half
        m = in_cube(mid, c, h)
        Qr = np.abs(fl.Q_bed[in_cube(mid[fl.bed_a], c, h)]).sum()
        Vr = (2 * h) ** 3 * 1e6
        print("      RVE %d: bed flow %.3f mL/min in %.2f mL -> %.1f mL/min/100 g (gland mean %.1f)"
              % (q, Qr * 6e7, Vr, Qr * 6e7 / (Vr * 1.05) * 100, perf))

    print("\n" + "=" * 60 + "\nFLOW " + ("OK" if ok else "NOT OK"))
    return ok
