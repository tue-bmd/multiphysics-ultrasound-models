"""Transport invariants and estimator validation.

  1. TIC area = blood volume (central volume theorem, per kernel):
     for a unit bolus, integral of N_i(t) dt = V_i for every segment.
  2. Estimated v, D from the two-kernel identification against the TRUE
     values from the network's flow-weighted arrival statistics.
  3. distance dependence of the two transfer functions on tissue.
  4. Mechanism split: between-path vs intra-segment (Poiseuille) share.
  5. Venous nuisance: estimates with arterial-only vs full TIC.
"""
from __future__ import annotations
import numpy as np
from ..geometry.network import Network, in_cube, cube_fraction
from ..physics.flow import Flow
from ..physics import transport as TR


def _hdr(t): print("\n" + t); print("-" * len(t))
def _pf(ok, msg): print(("PASS  " if ok else "FAIL  ") + msg); return ok


def flow_axis(net, fl, centre, half):
    """Flow direction in the RVE from the arrival-time gradient (arteries)."""
    mid = 0.5 * (net.p0 + net.p1)
    m = in_cube(mid, centre, half) & (net.kind == 0) & np.isfinite(fl.t_arr)
    X = mid[m] - centre; y = fl.t_arr[m]; wgt = np.abs(fl.Q[m])
    Xw = X * np.sqrt(wgt)[:, None]; yw = (y - np.average(y, weights=wgt)) * np.sqrt(wgt)
    g, *_ = np.linalg.lstsq(np.c_[Xw, np.sqrt(wgt)], yw, rcond=None)
    u = g[:3] / np.linalg.norm(g[:3])
    return u, np.linalg.norm(g[:3])          # unit vector, s per m


def kernel_pair(net, fl, centre, half, dz, khalf):
    u, grad = flow_axis(net, fl, centre, half)
    c1 = np.asarray(centre) - 0.5 * dz * u; c2 = np.asarray(centre) + 0.5 * dz * u
    return [(c1, khalf), (c2, khalf)], u


def truth(tr, dz):
    """Model-free moments of the FULL TICs of the two oracle kernels."""
    s1, s2 = tr.stats[0], tr.stats[1]
    v = dz / (s2["m1"] - s1["m1"])
    D = v ** 3 * (s2["m2"] - s1["m2"]) / (2 * dz)
    return v, D


def series_path(net, start, n_gen):
    """Follow the largest arterial daughter for n_gen generations."""
    ch = net.children(); path = [start]; i = start
    for _ in range(n_gen):
        kids = [c for c in ch[i] if net.kind[c] == 0]
        if not kids: break
        i = max(kids, key=lambda c: net.r[c]); path.append(i)
    return path


def control(net, fl, centre, half, n_gen=4):
    """CONTROL: input and output on ONE arteriolar path, where the series
    transfer-function model is exactly true.  Equivalent of the phantom."""
    _hdr("0. CONTROL: series path (estimator isolated from topology)")
    mid = 0.5 * (net.p0 + net.p1)
    cand = np.where(in_cube(mid, centre, half) & (net.kind == 0) & (net.d > 80e-6) & (net.d < 200e-6))[0]
    cand = sorted(cand, key=lambda i: -len(series_path(net, i, 8)))
    ok = True
    for i in cand[:1]:
        path = series_path(net, i, n_gen)
        k = path[-1]
        dzp = sum(net.Lpath[j] for j in path[1:])          # path length from end of i to end of k
        for pois in (False, True):
            tr = TR.propagate(net, fl, [], poiseuille=pois, extra=path)
            ti, to = tr.N[i], tr.N[k]
            # model-free moments of the two concentrations-present
            def mom(y):
                a = y.sum() * TR.DT; m1 = (tr.t * y).sum() * TR.DT / a
                return m1, ((tr.t - m1) ** 2 * y).sum() * TR.DT / a
            m1i, m2i = mom(ti); m1o, m2o = mom(to)
            vm = dzp / (m1o - m1i); Dm = vm ** 3 * (m2o - m2i) / (2 * dzp)
            vpath = dzp / sum(net.Lpath[j] / abs(fl.v[j]) for j in path[1:])
            print("      path of %d segments, d %.0f -> %.0f um, dz = %.2f mm, path-mean speed %.2f mm/s; %s"
                  % (len(path), net.d[i] * 1e6, net.d[k] * 1e6, dzp * 1e3, vpath * 1e3,
                     "Poiseuille" if pois else "plug flow"))
            print("         moments: v = %.2f mm/s, D = %.4f mm^2/s" % (vm * 1e3, Dm * 1e6))
            for tf in ("new", "old"):
                v, D, r2 = TR.identify(tr.t, ti, to, dzp, tf=tf)
                print("         T_%-3s  : v = %.2f mm/s (%+4.0f%% vs path speed), D = %.4f mm^2/s, alpha_L = %.3f mm, R^2 %.4f"
                      % (tf, v * 1e3, 100 * (v / vpath - 1), D * 1e6, D / v * 1e3, r2))
                if tf == "new" and pois:
                    ok &= _pf(abs(v / vpath - 1) < 0.25, "series control, Poiseuille, T_new: v within 25%% of the path speed")
    return ok


def check(net: Network, fl: Flow, centre, half, dz=3e-3, khalf=0.75e-3) -> bool:
    ok = True
    ok &= control(net, fl, centre, half)
    kernels, u = kernel_pair(net, fl, centre, half, dz, khalf)
    print("flow axis in RVE: %s ; kernels %.1f mm apart, %.1f mm cubes" % (np.round(u, 2), dz * 1e3, 2 * khalf * 1e3))

    _hdr("1. Central volume theorem: TIC area = sum_i Q_i * mean residence_i, per kernel")
    print("      (equals the geometric blood volume for plug flow; under the Poiseuille lift cap the\n"
          "       bubbles under-sample near-wall blood and the visible volume is ~0.66 x geometric)")
    for pois in (False, True):
        tr = TR.propagate(net, fl, kernels, poiseuille=pois)
        mid = 0.5 * (net.p0 + net.p1)
        for k, (c, h) in enumerate(kernels):
            w = cube_fraction(net.p0, net.p1, c, h); m = w > 0
            idx = np.where(m)[0]
            Vgeo = (w * np.pi * net.r ** 2 * net.Lpath)[m].sum()
            Vart_geo = (w * np.pi * net.r ** 2 * net.Lpath)[m & (net.kind == 0)].sum()
            Vvis_art = sum(w[i] * tr.V_vis.get(i, 0.0) for i in idx if net.kind[i] == 0)
            Vvis_all = sum(w[i] * tr.V_vis.get(i, 0.0) for i in idx)
            a_art = tr.tic_art[k].sum() * TR.DT
            a_all = tr.tic[k].sum() * TR.DT
            # bed volume seen by the kernel: each bed node's Q_bed * tau_cap times the
            # fraction of its R_LAT territory inside the cube (same rule as propagate)
            from ..physics.flow import R_LAT as _RLAT
            qn = {}
            for e in range(len(fl.bed_a)): qn[int(fl.bed_a[e])] = qn.get(int(fl.bed_a[e]), 0.0) + abs(fl.Q_bed[e])
            bn = np.array(sorted(qn)); wb = TR._bed_overlap(net.p1[bn], c, h, _RLAT)
            Vbed = sum(wb[j] * qn[int(bn[j])] for j in range(len(bn))) * TR.TAU_CAP
            print("      %-10s kernel %d: %d segments; arterial area %.4f vs visible %.4f mm^3 (ratio %.4f, visible/geometric %.2f); "
                  "full area %.4f vs visible + bed %.4f (ratio %.4f)"
                  % ("Poiseuille" if pois else "plug", k, m.sum(), a_art * 1e9, Vvis_art * 1e9,
                     a_art / Vvis_art, Vvis_art / Vart_geo, a_all * 1e9, (Vvis_all + Vbed) * 1e9,
                     a_all / (Vvis_all + Vbed)))
            ok &= _pf(abs(a_art / Vvis_art - 1) < 0.02 and abs(a_all / (Vvis_all + Vbed) - 1) < 0.02,
                      "%s kernel %d: TIC areas equal Q x mean residence (arterial and full incl. bed)"
                      % ("Poiseuille" if pois else "plug", k))
            if not pois:
                ok &= _pf(abs(Vvis_art / Vart_geo - 1) < 1e-6, "plug kernel %d: visible volume = geometric volume" % k)
            print("         composition of full TIC area: arterial %.0f%%, bed %.0f%%, venous %.0f%%"
                  % (100 * a_art / a_all, 100 * Vbed / a_all, 100 * (a_all - a_art - Vbed) / a_all))
    tr = TR.propagate(net, fl, kernels, poiseuille=True)
    mid = 0.5 * (net.p0 + net.p1)

    _hdr("2. Estimators vs truth (Poiseuille on)")
    vt, Dt = truth(tr, dz)
    s0, s1 = tr.stats[0], tr.stats[1]
    print("      full-TIC first moments: %.2f -> %.2f s (arterial-only arrival means %.2f -> %.2f s)"
          % (s0["m1"], s1["m1"], s0["t_mean"], s1["t_mean"]))
    print("      MOMENT-BASED (model-free) truth on full TICs: v = %.2f mm/s, D = %.3f mm^2/s, alpha_L = %.3f mm"
          % (vt * 1e3, Dt * 1e6, Dt / vt * 1e3))
    res = {}
    for lab, tics in [("full TIC", tr.tic), ("arterial-only TIC", tr.tic_art)]:
        for tf in ("new", "old"):
            v, D, r2 = TR.identify(tr.t, tics[0], tics[1], dz, tf=tf)
            res[(lab, tf)] = (v, D, r2)
            print("      %-18s T_%-3s : v = %6.2f mm/s (%+5.0f%%)  D = %7.3f mm^2/s (%+5.0f%%)  alpha_L = %.3f mm  R^2 = %.3f"
                  % (lab, tf, v * 1e3, 100 * (v / vt - 1), D * 1e6, 100 * (D / Dt - 1), D / v * 1e3, r2))
    v, D, r2 = res[("arterial-only TIC", "new")]
    ok &= _pf(abs(v / vt - 1) < 0.3, "ORACLE (known axis) T_new recovers v within 30%")
    ok &= _pf(0.33 < D / Dt < 3, "ORACLE (known axis) T_new recovers D within a factor 3")

    _hdr("2b. CLINICAL CUDI: voxel input, spherical SHELL of outputs (3D annulus), causal pairs, joint fit")
    vox = 0.375e-3                      # 0.75 mm voxel (4D CEUS)
    for ring in (1.5e-3, 2.0e-3, 3.0e-3):
        ks = TR.ring_kernels(centre, ring, vox, n_dir=48)
        trr = TR.propagate(net, fl, ks, poiseuille=True)
        outs = [trr.tic_art[k] for k in range(1, len(ks))]
        for tf in ("new", "old"):
            vr, Dr, npair, r2r = TR.identify_ring(trr.t, trr.tic_art[0], outs, ring, tf=tf)
            if np.isfinite(vr):
                print("      ring %.1f mm, %2d causal pairs, T_%-3s: v = %6.2f mm/s (%+5.0f%% vs oracle truth)  D = %7.3f mm^2/s (%+5.0f%%)  alpha_L = %.3f mm  R^2 %.2f"
                      % (ring * 1e3, npair, tf, vr * 1e3, 100 * (vr / vt - 1), Dr * 1e6, 100 * (Dr / Dt - 1), Dr / vr * 1e3, r2r))
            else:
                print("      ring %.1f mm: only %d causal pairs, no fit" % (ring * 1e3, npair))
        # full TIC (veins in) for the 2 mm ring
        if abs(ring - 2e-3) < 1e-9:
            outs_full = [trr.tic[k] for k in range(1, len(ks))]
            vr, Dr, npair, r2r = TR.identify_ring(trr.t, trr.tic[0], outs_full, ring, tf="new")
            if np.isfinite(vr):
                print("      ring 2.0 mm, veins IN the TICs, T_new: v = %6.2f mm/s (%+5.0f%%)  D = %7.3f mm^2/s (%+5.0f%%)  alpha_L = %.3f mm  %d pairs"
                      % (vr * 1e3, 100 * (vr / vt - 1), Dr * 1e6, 100 * (Dr / Dt - 1), Dr / vr * 1e3, npair))
    for k in (0, 1):
        kap, mu = TR.mldrw_kappa(tr.t, tr.tic_art[k])
        print("      mLDRW kernel %d: kappa = %.2f 1/s, mu = %.2f s   (v^2/D true = %.2f 1/s)" % (k, kap, mu, vt ** 2 / Dt))

    _hdr("3. Mechanism split: between-path only vs + intra-segment Poiseuille")
    tr0 = TR.propagate(net, fl, kernels, poiseuille=False)
    v0, D0 = truth(tr0, dz)
    v1, D1, _ = TR.identify(tr0.t, tr0.tic_art[0], tr0.tic_art[1], dz, tf="new")
    v2, D2, _ = TR.identify(tr.t, tr.tic_art[0], tr.tic_art[1], dz, tf="new")
    print("      plug flow      : est v %.2f mm/s, D %.3f mm^2/s, alpha_L %.3f mm" % (v1 * 1e3, D1 * 1e6, D1 / v1 * 1e3))
    print("      + Poiseuille   : est v %.2f mm/s, D %.3f mm^2/s, alpha_L %.3f mm" % (v2 * 1e3, D2 * 1e6, D2 / v2 * 1e3))
    print("      -> intra-segment spreading contributes %.0f%% of alpha_L (full remix: lower bound)"
          % (100 * (1 - (D1 / v1) / (D2 / v2))))

    _hdr("4. Distance dependence of the two transfer functions on tissue")
    print("      %6s | %8s %8s %9s | %8s %8s %9s" % ("dz mm", "v_new", "D_new", "aL_new", "v_old", "D_old", "aL_old"))
    for dzz in (1.5e-3, 2.5e-3, 3.5e-3, 4.5e-3):
        kk, _ = kernel_pair(net, fl, centre, half, dzz, khalf)
        trr = TR.propagate(net, fl, kk, poiseuille=True)
        vn, Dn, _ = TR.identify(trr.t, trr.tic_art[0], trr.tic_art[1], dzz, tf="new")
        vo, Do, _ = TR.identify(trr.t, trr.tic_art[0], trr.tic_art[1], dzz, tf="old")
        print("      %6.1f | %8.2f %8.3f %9.3f | %8.2f %8.3f %9.3f"
              % (dzz * 1e3, vn * 1e3, Dn * 1e6, Dn / vn * 1e3, vo * 1e3, Do * 1e6, Do / vo * 1e3))

    print("\n" + "=" * 60 + "\nTRANSPORT " + ("OK" if ok else "NOT OK"))
    return ok, tr
