"""Geometry invariants.  Every check names what it asserts and prints PASS/FAIL
or a measured value next to its literature range.  Run after every change to
geometry/network.py.  If a check fails, the network is not to be used.

Literature anchors (prostate, adult):
  vascular volume fraction        2-10 %      (blood volume, stereology)
  arteriole+venule density >=30um  reported for information only (no cited anchor)
  arteriolar L/d                  5-20
  microvascular tortuosity        1.1-1.5
"""
from __future__ import annotations
import numpy as np
from ..geometry.network import (Network, GLAND_SEMI, D_ROOT, in_gland,
                                in_cube, F_VEIN)


def _hdr(t): print("\n" + t); print("-" * len(t))


def _pf(ok, msg): print(("PASS  " if ok else "FAIL  ") + msg); return ok


def check(net: Network, verbose=True) -> bool:
    ok = True
    prm = net.params
    ch = net.children()
    A = net.art; V = net.vein

    _hdr("1. Murray's law at every arterial bifurcation (imposed exactly)")
    res = []
    for i in np.where(A)[0]:
        kids = [c for c in ch[i] if net.kind[c] == 0]
        if len(kids) == 2:
            res.append(abs(net.r[i] ** 3 - sum(net.r[c] ** 3 for c in kids)) / net.r[i] ** 3)
    res = np.array(res)
    ok &= _pf(res.max() < 1e-9 if len(res) else True,
              "max relative Murray residual over %d bifurcations = %.1e" % (len(res), res.max() if len(res) else 0))
    n1 = sum(1 for i in np.where(A)[0] if len([c for c in ch[i] if net.kind[c] == 0]) == 1)
    print("      single-daughter nodes (other branch below interface or pruned): %d" % n1)

    _hdr("2. No radius floor: terminal calibres sit just above the interface")
    for zone, dt, lab in [(0, prm.d_term_gland, "gland"), (1, prm.d_term_rve, "RVE")]:
        m = A & net.term & (net.zone == zone)
        if m.sum() == 0: print("      %s: no terminals" % lab); continue
        d = net.d[m]; why = net.why[m]
        cal = why == "caliber"
        ok &= _pf(d[cal].min() >= dt * 0.999 if cal.any() else True,
                  "%s terminals by caliber: n=%d, d in [%.0f, %.0f] um, interface %.0f um"
                  % (lab, cal.sum(), d[cal].min() * 1e6 if cal.any() else 0,
                     d[cal].max() * 1e6 if cal.any() else 0, dt * 1e6))
        if (~cal).any():
            print("      %s terminals by pruning (domain/collision): n=%d, d up to %.0f um  <- boundary artifact, reported"
                  % (lab, (~cal).sum(), d[~cal].max() * 1e6))

    _hdr("3. Anatomy: feeders outside the gland, nothing > 500 um inside")
    feed = net.gen == 0
    mid = 0.5 * (net.p0 + net.p1)
    ok &= _pf(not in_gland(mid[feed]).any(), "all %d feeder midpoints outside gland" % feed.sum())
    inside = in_gland(mid) & A
    ok &= _pf(net.d[inside].max() <= prm.d_root * 1.001,
              "max arterial caliber inside gland = %.0f um (limit %.0f)" % (net.d[inside].max() * 1e6, prm.d_root * 1e6))
    print("      gland volume %.1f mL; arterial segments inside: %d" %
          (4 / 3 * np.pi * np.prod(GLAND_SEMI) * 1e6, inside.sum()))
    print("      pruned branches: %s" % net.pruned)

    _hdr("4. Realised vs input distributions (arteries, gen >= 2)")
    m = A & (net.gen >= 2)
    gam = net.Lchord[m] / net.d[m]; T = net.Lpath[m] / net.Lchord[m]
    print("      L/d       : median %.1f, 5-95%% [%.1f, %.1f]   (input median %.0f; lit 5-20)"
          % (np.median(gam), *np.percentile(gam, [5, 95]), prm.gamma_med))
    print("      tortuosity: mean %.2f, 5-95%% [%.2f, %.2f]   (input mean %.2f; lit 1.1-1.5)"
          % (T.mean(), *np.percentile(T, [5, 95]), prm.tort_mean))
    ok &= _pf(0.6 < np.median(gam) / prm.gamma_med < 1.4, "L/d median within 40%% of input")
    # generation count is DERIVED
    for zone, lab in [(0, "gland"), (1, "RVE")]:
        g = net.gen[A & (net.zone == zone)]
        if len(g): print("      %s generations: %d .. %d (derived, not set)" % (lab, g.min(), g.max()))

    _hdr("5. Per-RVE microstructure vs literature")
    rve_ok = True
    for q, c in enumerate(prm.rve_centres):
        c = np.asarray(c); h = prm.rve_half; Vol = (2 * h) ** 3
        m = in_cube(mid, c, h)
        seg_vol = np.pi * net.r ** 2 * net.Lpath
        phiA = seg_vol[m & A].sum() / Vol; phiV = seg_vol[m & V].sum() / Vol
        LV = net.Lpath[m & A].sum() / Vol           # arterial length density
        # section counts on the three mid-planes through the cube centre
        NA = []
        for ax in range(3):
            lo = np.minimum(net.p0[:, ax], net.p1[:, ax]); hi = np.maximum(net.p0[:, ax], net.p1[:, ax])
            cross = (lo < c[ax]) & (hi > c[ax]) & m & A
            NA.append(cross.sum() / (2 * h) ** 2)
        NA = np.array(NA) * 1e-6                       # per mm^2
        # caliber statistics, volume weighted, arteries
        w = seg_vol[m & A]; d = net.d[m & A]
        if w.sum() > 0:
            dm = (w * d).sum() / w.sum(); cv = np.sqrt((w * (d - dm) ** 2).sum() / w.sum()) / dm
            dperm = np.sqrt((w * d ** 2).sum() / w.sum())
        else: dm = cv = dperm = np.nan
        nterm = (m & A & net.term).sum(); nfeed = len(set(net.tree[m & A]))
        print("   RVE %d at %s mm, %.0f mm cube, %d art + %d vein segments, fed by %d tree(s), %d terminals"
              % (q, np.round(c * 1e3, 1), 2 * h * 1e3, (m & A).sum(), (m & V).sum(), nfeed, nterm))
        print("      phi  artery %.2f%%  vein %.2f%%  total %.2f%%          lit 2-10%% (all vessels)"
              % (100 * phiA, 100 * phiV, 100 * (phiA + phiV)))
        print("      L_V  arterial %.1f mm/mm^3                          (no cited anchor; for information)"
              % (LV * 1e-6))
        print("      N_A  arterial on 3 mid-sections %s /mm^2           (no cited anchor; for information)"
              % np.round(NA, 1))
        print("      <d>_vol %.0f um, d_perm %.0f um, CV_d %.2f          (d_perm/<d> = %.2f)"
              % (dm * 1e6, dperm * 1e6, cv, dperm / dm if dm else np.nan))
        if (m & A).sum() < 50 or nterm < 20:
            rve_ok = False
    ok &= _pf(rve_ok, "each RVE has >= 50 arterial segments and >= 20 terminals")

    _hdr("6. No intersections (wall-to-wall clearance, non-adjacent arteries)")
    from scipy.spatial import cKDTree
    ia = np.where(A & (net.gen >= 1))[0]
    pts, rad, own = [], [], []
    for i in ia:
        n = max(2, int(net.Lchord[i] / 1.5e-4) + 1)
        P = net.p0[i][None] + (net.p1[i] - net.p0[i])[None] * np.linspace(0.08, 0.92, n)[:, None]
        pts.append(P); rad.append(np.full(n, net.r[i])); own.append(np.full(n, i))
    pts = np.vstack(pts); rad = np.concatenate(rad); own = np.concatenate(own)
    tr = cKDTree(pts); pairs = tr.query_pairs(1.0e-3, output_type="ndarray")
    a, b = pairs[:, 0], pairs[:, 1]
    same = own[a] == own[b]
    adj = (net.parent[own[a]] == own[b]) | (net.parent[own[b]] == own[a]) | \
          (net.parent[own[a]] == net.parent[own[b]])
    keep = ~same & ~adj
    if keep.any():
        dd = np.linalg.norm(pts[a[keep]] - pts[b[keep]], axis=1) - rad[a[keep]] - rad[b[keep]]
        ok &= _pf(dd.min() > -1e-6, "min clearance between non-adjacent arteries = %.1f um" % (dd.min() * 1e6))
    else:
        _pf(True, "no non-adjacent artery pairs within 1 mm")

    _hdr("7. Connectivity: every segment traces back to a feeder, AND is geometrically continuous")
    has_par = net.parent >= 0
    gap = np.linalg.norm(net.p0[has_par] - net.p1[net.parent[has_par]], axis=1)
    ok &= _pf(gap.max() < 1e-9, "max |child.p0 - parent.p1| over %d segments = %.2e m (arteries and veins)"
              % (has_par.sum(), gap.max()))
    bad = 0
    for i in range(net.n):
        j, k = i, 0
        while net.parent[j] >= 0 and k < 10000: j = net.parent[j]; k += 1
        if net.gen[j] != 0 and net.kind[i] == 0: bad += 1
    ok &= _pf(bad == 0, "%d arterial segments not connected to a feeder" % bad)

    _hdr("8. Veins")
    tw = net.twin[V]
    ok &= _pf(np.allclose(net.r[V], F_VEIN * net.r[tw]) and V.sum() == A.sum() - feed.sum(),
              "%d veins paired to %d non-feeder arteries at %.1fx caliber" % (V.sum(), A.sum() - feed.sum(), F_VEIN))
    # the veins are offset copies of the arteries: they are not collision
    # checked and not confined, so their overlap and their excursions outside
    # the gland are reported rather than asserted
    iv = np.where(V)[0]
    out = np.array([not (in_gland(net.p0[i]) and in_gland(net.p1[i])) for i in iv])
    print("      %d of %d vein segments (%.1f %%) have an endpoint outside the gland (veins are not confined)"
          % (out.sum(), len(iv), 100 * out.mean()))
    pts, rad, own = [], [], []
    for i in np.concatenate([ia, iv]):
        n = max(2, int(net.Lchord[i] / 1.5e-4) + 1)
        P = net.p0[i][None] + (net.p1[i] - net.p0[i])[None] * np.linspace(0.08, 0.92, n)[:, None]
        pts.append(P); rad.append(np.full(n, net.r[i])); own.append(np.full(n, i))
    pts = np.vstack(pts); rad = np.concatenate(rad); own = np.concatenate(own)
    pairs = cKDTree(pts).query_pairs(1.0e-3, output_type="ndarray")
    a, b = pairs[:, 0], pairs[:, 1]
    va, vb = V[own[a]], V[own[b]]
    adj = (net.parent[own[a]] == own[b]) | (net.parent[own[b]] == own[a]) | \
          (net.parent[own[a]] == net.parent[own[b]]) | (net.twin[own[a]] == own[b]) | (net.twin[own[b]] == own[a])
    keep = (own[a] != own[b]) & ~adj & (va | vb)
    if keep.any():
        dd = np.linalg.norm(pts[a[keep]] - pts[b[keep]], axis=1) - rad[a[keep]] - rad[b[keep]]
        segs = set(own[a[keep]][dd < 0]) | set(own[b[keep]][dd < 0])
        print("      %d segments in %d overlapping pairs that involve a vein (veins are not collision checked); "
              "min clearance %.1f um" % (len(segs), int((dd < 0).sum()), dd.min() * 1e6))

    print("\n" + ("=" * 60) + "\nGEOMETRY " + ("OK" if ok else "NOT OK") + "   (%d segments)" % net.n)
    return ok


def figure(net: Network, path="figs/geometry_3d.png", slab=1.0e-3):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prm = net.params; A = net.art; V = net.vein
    fig = plt.figure(figsize=(18, 6))
    # --- panel 1: gland
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    m = A & (net.gen >= 1)
    for i in np.where(m)[0][::max(1, m.sum() // 5000)]:
        ax.plot(*zip(net.p0[i] * 1e3, net.p1[i] * 1e3), color="C%d" % (net.tree[i] % 10),
                lw=max(0.3, net.r[i] * 6e3), alpha=0.7)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 20)
    ax.plot_wireframe(GLAND_SEMI[0] * np.outer(np.cos(u), np.sin(v)) * 1e3,
                      GLAND_SEMI[1] * np.outer(np.sin(u), np.sin(v)) * 1e3,
                      GLAND_SEMI[2] * np.outer(np.ones_like(u), np.cos(v)) * 1e3, color="k", lw=0.2, alpha=0.3)
    for c in prm.rve_centres:
        c = np.asarray(c) * 1e3; h = prm.rve_half * 1e3
        for s in (-1, 1):
            for t in (-1, 1):
                ax.plot([c[0] - h, c[0] + h], [c[1] + s * h] * 2, [c[2] + t * h] * 2, "r", lw=1)
                ax.plot([c[0] + s * h] * 2, [c[1] - h, c[1] + h], [c[2] + t * h] * 2, "r", lw=1)
                ax.plot([c[0] + s * h] * 2, [c[1] + t * h] * 2, [c[2] - h, c[2] + h], "r", lw=1)
    ax.view_init(25, -60); ax.set_title("gland tree (arteries, >= 60 um), RVE cubes in red")
    ax.set_xlabel("x mm"); ax.set_ylabel("y mm"); ax.set_zlabel("z mm")

    # --- panel 2: RVE 0 in 3D, arteries red, veins blue, incl. parents outside (gray)
    c = np.asarray(prm.rve_centres[0]); h = prm.rve_half
    mid = 0.5 * (net.p0 + net.p1)
    inside = in_cube(net.p0, c, h) | in_cube(net.p1, c, h)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    for i in np.where(inside)[0]:
        col = "r" if net.kind[i] == 0 else "b"
        ax.plot(*zip(net.p0[i] * 1e3, net.p1[i] * 1e3), col, lw=max(0.3, net.r[i] * 1.5e4), alpha=0.6)
    ax.view_init(20, -50); ax.set_title("RVE 0 in 3D: arteries red, veins blue\n(segments with an endpoint in the cube)")
    ax.set_xlabel("x mm"); ax.set_ylabel("y mm"); ax.set_zlabel("z mm")

    # --- panel 3: a thin slab through the RVE centre, like a histological section
    ax = fig.add_subplot(1, 3, 3)
    y0 = c[1]
    for i in np.where(inside)[0]:
        a, b = net.p0[i], net.p1[i]
        ya, yb = a[1], b[1]
        lo, hi = min(ya, yb), max(ya, yb)
        if hi < y0 - slab / 2 or lo > y0 + slab / 2: continue
        # clip the chord to the slab
        ta = 0.0 if abs(yb - ya) < 1e-12 else np.clip((y0 - slab / 2 - ya) / (yb - ya), 0, 1)
        tb = 1.0 if abs(yb - ya) < 1e-12 else np.clip((y0 + slab / 2 - ya) / (yb - ya), 0, 1)
        t0, t1 = min(ta, tb), max(ta, tb)
        pa = a + t0 * (b - a); pb = a + t1 * (b - a)
        col = "r" if net.kind[i] == 0 else "b"
        ax.plot([pa[0] * 1e3, pb[0] * 1e3], [pa[2] * 1e3, pb[2] * 1e3], col, lw=max(0.5, net.r[i] * 3e4), alpha=0.8)
    ax.set_aspect("equal"); ax.set_xlim((c[0] - h) * 1e3, (c[0] + h) * 1e3); ax.set_ylim((c[2] - h) * 1e3, (c[2] + h) * 1e3)
    ax.set_title("RVE 0, %.1f mm slab at y = %.1f mm (a 'section')" % (slab * 1e3, y0 * 1e3))
    ax.set_xlabel("x mm"); ax.set_ylabel("z mm")
    fig.tight_layout(); fig.savefig(path, dpi=110); return path
