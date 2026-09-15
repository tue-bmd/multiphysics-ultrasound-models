"""Pressure/flow solve on the network.

State variable: pressure at every node.  Derived per segment: flow Q,
velocity v, transit time.  Pressure-driven: P_art at feeder inlets, P_ven at
venous outlets.  Closed loop artery -> lumped bed -> paired vein.

Node numbering
    artery segment i  : end node = i
    vein segment j    : end node = j            (veins share the index space)
    inlet node        : n_seg      (all feeder start points; one node)
    outlet node       : n_seg + 1  (all vein-root start points; one node)
Segment i runs from node start[i] to node i, where start[i] = parent's end
node, or the inlet/outlet node at the roots.

Bed connections: at every arterial node with fewer than two arterial
children, one connection per missing child to the paired vein's end node,
conductance  G_bed(r) = G_bed0 * (2 r / d_term)^3  with r the radius of the
vessel that would have fed it (Murray-consistent).  G_bed0 is calibrated
once so that the mean arterial-terminal pressure equals P_ART_END at rest.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from dataclasses import dataclass
from ..geometry.network import Network, murray_daughters

ETA = 3.6e-3               # blood viscosity [Pa s]
MMHG = 133.322
P_ART = 70.0 * MMHG        # feeder inlet pressure
P_VEN = 8.0 * MMHG         # venous outlet pressure
P_ART_END = 35.0 * MMHG    # target mean pressure at arterial terminals (rest)


@dataclass
class Flow:
    p: np.ndarray          # node pressure, len n_seg + 2
    Q: np.ndarray          # segment flow, signed along p0->p1
    v: np.ndarray          # segment mean velocity, signed
    t_seg: np.ndarray      # segment transit time |Lpath / v|
    t_arr: np.ndarray      # arrival time at segment END from the inlet (arteries)
    Q_bed: np.ndarray      # flow through each bed connection
    bed_a: np.ndarray      # arterial node of each bed connection
    bed_v: np.ndarray      # venous node of each bed connection
    bed_r: np.ndarray      # feeding radius of each bed connection
    bed_w: np.ndarray      # share of the feeding vessel's bed conductance on this edge
    G_bed0: float
    start: np.ndarray      # start node of each segment
    inlet: int; outlet: int


R_LAT = 0.5e-3   # lateral reach of the capillary bed: an arterial terminal drains
                 # into every venous terminal within R_LAT (share ~ 1/distance).
                 # 0 -> paired vein only (v2.0 behavior).


def _topology(net: Network, R_lat=None):
    R_lat = R_LAT if R_lat is None else R_lat
    n = net.n; inlet, outlet = n, n + 1
    start = np.where(net.parent >= 0, net.parent, -1)
    start[(net.parent < 0) & (net.kind == 0)] = inlet
    start[(net.parent < 0) & (net.kind == 1)] = outlet
    # bed connections
    ch = net.children()
    vein_of = {int(net.twin[j]): j for j in np.where(net.kind == 1)[0]}
    # venous terminal nodes available as bed outlets, with their positions
    vterm = np.array([j for j in np.where(net.kind == 1)[0]
                      if net.term[int(net.twin[j])] or
                      len([c for c in ch[int(net.twin[j])] if net.kind[c] == 0]) < 2])
    vpos = net.p1[vterm]
    from scipy.spatial import cKDTree
    vtree = cKDTree(vpos) if len(vterm) else None
    ba, bv, br, bw = [], [], [], []
    for i in np.where(net.kind == 0)[0]:
        if net.gen[i] == 0: continue                 # feeders never feed the bed
        kids = [c for c in ch[i] if net.kind[c] == 0]
        missing = 2 - len(kids)
        if missing <= 0: continue
        # the radii actually drawn at this node (jittered, and lesion-local where
        # a lesion overrides alpha).  The daughters that were not created are
        # whatever remains after removing the radii of the children that exist:
        # a collision or the gland boundary can remove the LARGER of the pair,
        # so the missing one cannot be assumed to be the smaller draw.
        rd = None if getattr(net, "r_dau", None) is None else net.r_dau[i]
        if rd is not None and np.all(np.isfinite(rd)):
            pool = [float(rd[0]), float(rd[1])]
            for c in kids:
                j = int(np.argmin([abs(net.r[c] - x) for x in pool]))
                pool.pop(j)
            feeds = pool
        else:
            r1, r2 = murray_daughters(net.r[i], net.params.alpha)
            feeds = [r1, r2] if len(kids) == 0 else [min(r1, r2)]
        vnode = vein_of.get(i, None)
        if vnode is None: continue
        # outlets: paired vein always, plus venous terminals within R_lat
        outs = [vnode]; wts = [1.0]
        if R_lat > 0 and vtree is not None:
            for k in vtree.query_ball_point(net.p1[i], R_lat):
                j = int(vterm[k])
                if j == vnode: continue
                dist = np.linalg.norm(vpos[k] - net.p1[i])
                outs.append(j); wts.append(1.0 / max(dist, 50e-6))
            wts[0] = max(wts[0], 1.0 / 50e-6)      # paired vein counts as 'closest'
        wts = np.array(wts) / np.sum(wts)
        for r in feeds:
            for j, wgt in zip(outs, wts):
                ba.append(i); bv.append(j); br.append(r); bw.append(wgt)
    return start, inlet, outlet, np.array(ba), np.array(bv), np.array(br), np.array(bw)


def _solve(net: Network, start, inlet, outlet, ba, bv, br, bw, G_bed0, radius_scale=None):
    n = net.n; N = n + 2
    r = net.r if radius_scale is None else net.r * radius_scale
    G = np.pi * r ** 4 / (8 * ETA * net.Lpath)             # segment conductance
    Gb = G_bed0 * (2 * br / net.params.d_term_rve) ** 3 * bw   # share of each outlet
    rows = np.concatenate([start, np.arange(n), ba, bv])
    cols = np.concatenate([np.arange(n), start, bv, ba])
    vals = np.concatenate([G, G, Gb, Gb])
    A = sp.coo_matrix((-vals, (rows, cols)), shape=(N, N)).tocsr()
    diag = -np.asarray(A.sum(axis=1)).ravel()
    A = A + sp.diags(diag)
    # Dirichlet at inlet/outlet
    b = np.zeros(N)
    for node, val in ((inlet, P_ART), (outlet, P_VEN)):
        b -= A[:, node].toarray().ravel() * val
    keep = np.ones(N, bool); keep[[inlet, outlet]] = False
    idx = np.where(keep)[0]
    p = np.zeros(N); p[inlet] = P_ART; p[outlet] = P_VEN
    p[idx] = spl.spsolve(A[idx][:, idx].tocsc(), b[idx])
    Q = G * (p[start] - p[np.arange(n)])
    Qb = Gb * (p[ba] - p[bv])
    return p, Q, Qb, G


def solve(net: Network, G_bed0=None, radius_scale=None, R_lat=None) -> Flow:
    start, inlet, outlet, ba, bv, br, bw = _topology(net, R_lat)
    term = (net.kind == 0) & net.term
    if G_bed0 is None:
        # calibrate: mean arterial-terminal pressure = P_ART_END
        def f(logG):
            p, *_ = _solve(net, start, inlet, outlet, ba, bv, br, bw, np.exp(logG))
            return p[np.where(term)[0]].mean() - P_ART_END
        lo, hi = np.log(1e-16), np.log(1e-8)
        flo, fhi = f(lo), f(hi)
        assert flo * fhi < 0, "bed calibration bracket failed (%.1f, %.1f mmHg)" % (
            (flo + P_ART_END) / MMHG, (fhi + P_ART_END) / MMHG)
        for _ in range(40):
            mid = 0.5 * (lo + hi); fm = f(mid)
            if fm * flo > 0: lo, flo = mid, fm
            else: hi, fhi = mid, fm
        G_bed0 = float(np.exp(0.5 * (lo + hi)))
    p, Q, Qb, G = _solve(net, start, inlet, outlet, ba, bv, br, bw, G_bed0, radius_scale)
    r = net.r if radius_scale is None else net.r * radius_scale
    v = Q / (np.pi * r ** 2)
    t_seg = np.abs(net.Lpath / np.where(np.abs(v) > 1e-12, v, 1e-12))
    # arrival time at arterial segment ends, accumulated from the inlet
    t_arr = np.full(net.n, np.nan)
    order = np.argsort(net.gen)
    for i in order:
        if net.kind[i] != 0: continue
        t_arr[i] = (0.0 if net.parent[i] < 0 else t_arr[net.parent[i]]) + t_seg[i]
    return Flow(p=p, Q=Q, v=v, t_seg=t_seg, t_arr=t_arr, Q_bed=Qb, bed_a=ba, bed_v=bv,
                bed_r=br, bed_w=bw, G_bed0=G_bed0, start=start, inlet=inlet, outlet=outlet)
