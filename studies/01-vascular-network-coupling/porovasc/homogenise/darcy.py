"""Permeability of the vascular network on a sampling cube, and the bundle-law
descriptors.

FACE-TO-FACE EXPERIMENT.  Every vessel is clipped to the cube of side 2h.  A
piece that leaves through the inlet face x = -h is attached to a reservoir at
P0 + dP, one that leaves through the outlet face x = +h to a reservoir at P0,
and one that leaves through a lateral face is sealed.  Pieces keep the network
node of any endpoint that lies inside the cube, so interior connectivity is
preserved, and their conductance uses the clipped path length.  Bed edges are
kept when both attachment nodes are inside.  With Q the flux through the inlet
reservoir,
        k_network_face = Q eta (2h) / ((2h)^2 dP),
computed along x, y and z.  The scalar is the arithmetic mean of the three,
including directions with no through path, which are exactly zero.

This is the hydraulic permeability of the vascular network on this support.  It
is not the interstitial permeability of tissue.

BUNDLE DESCRIPTORS, all on the same support and all length-weighted by the part
of each vessel inside it:
        k_bundle_straight = phi d_perm^2 / 32
        k_bundle_tort     = phi d_perm^2 / (32 <T^2>)
with d_perm = sqrt(<d^2>) and <T^2> the mean squared tortuosity.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from dataclasses import dataclass
from ..geometry.network import Network, in_cube, cube_fraction, clip_segment_cube, sphere_fraction
from ..physics.flow import Flow, ETA


@dataclass
class Darcy:
    k_face: np.ndarray       # [kx, ky, kz]
    k_mean: float
    phi: float; d_perm: float; d_vol: float; T2: float
    c_k: float               # phi d_perm^2 / k_mean
    c_k_bundle: float        # 32 <T^2>
    n_in: np.ndarray; n_out: np.ndarray   # vessels cut per face
    Q: np.ndarray
    n_blocked: int = 0       # directions with no inlet-to-outlet path


def _cube_system(net, fl, centre, h, axis, dP, P0=0.0):
    """Conductance matrix of the clipped network inside the cube, with Dirichlet
    reservoirs on the two faces normal to `axis`.

    Every segment is clipped to the cube.  An endpoint of the clipped piece is
    either an original network node (when that endpoint of the segment lies
    inside) or a face crossing.  A crossing of the inlet or outlet face joins the
    corresponding reservoir; a crossing of a lateral face is sealed, which for a
    single piece means the piece carries no flux and is dropped, and for a piece
    whose other end is an interior node means that end is a dead end.
    """
    c = np.asarray(centre, float)
    hit, t0, t1 = clip_segment_cube(net.p0, net.p1, c, h)
    idx = np.where(hit)[0]
    nodes = {}
    def nid(k):
        if k not in nodes: nodes[k] = len(nodes)
        return nodes[k]
    RES_IN, RES_OUT = -1, -2
    nid(RES_IN); nid(RES_OUT)
    rows, cols, vals = [], [], []
    n_in = n_out = 0
    tol = 1e-9 * h
    for i in idx:
        a, b = t0[i], t1[i]
        Lp = net.Lpath[i] * (b - a)                     # clipped path length
        if Lp <= 0: continue
        G = np.pi * net.r[i] ** 4 / (8 * ETA * Lp)
        # An end of the clipped piece is an original network node only when the
        # corresponding endpoint of the segment lies strictly inside the cube.
        # Otherwise it is a face crossing, which joins a reservoir when it lies
        # on one of the two faces normal to `axis` (a corner touching such a
        # face included) and is sealed on any other face.  An endpoint that sits
        # exactly on the inlet or outlet face is therefore a reservoir contact,
        # not an interior node.
        ends = []
        for tt, node_of_end, p_end in ((a, int(fl.start[i]), net.p0[i]),
                                       (b, int(i), net.p1[i])):
            q = net.p0[i] + tt * (net.p1[i] - net.p0[i])
            dd = q - c
            on_face = np.abs(dd) >= h - tol
            if not on_face.any():
                ends.append(("n", node_of_end)); continue
            if on_face[axis]:
                ends.append(RES_IN if dd[axis] < 0 else RES_OUT)
            else:
                ends.append("seal")
        ca, cb = ends
        if ca == "seal" or cb == "seal":
            continue                                    # sealed piece carries no flux
        na, nb = nid(ca), nid(cb)
        if ca == RES_IN or cb == RES_IN: n_in += 1
        if ca == RES_OUT or cb == RES_OUT: n_out += 1
        if na == nb: continue
        rows += [na, nb]; cols += [nb, na]; vals += [-G, -G]
    # bed edges: no explicit path to clip, so keep them only when both
    # attachment nodes are inside the cube
    Gb = fl.G_bed0 * (2 * fl.bed_r / net.params.d_term_rve) ** 3 * fl.bed_w
    for e in range(len(fl.bed_a)):
        i, j = int(fl.bed_a[e]), int(fl.bed_v[e])
        if ("n", i) in nodes and ("n", j) in nodes:
            na, nb = nodes[("n", i)], nodes[("n", j)]
            rows += [na, nb]; cols += [nb, na]; vals += [-Gb[e], -Gb[e]]
    N = len(nodes)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    A = A + sp.diags(-np.asarray(A.sum(axis=1)).ravel())
    return A, nodes, n_in, n_out


def face_to_face(net: Network, fl: Flow, centre, h, dP=100.0):
    ks, nin, nout = [], [], []
    for axis in range(3):
        A, nodes, n_in, n_out = _cube_system(net, fl, centre, h, axis, dP)
        ri, ro = nodes[-1], nodes[-2]
        from scipy.sparse.csgraph import connected_components
        ncomp, lab = connected_components(A, directed=False)
        if lab[ri] != lab[ro]:
            ks.append(0.0); nin.append(n_in); nout.append(n_out); continue
        comp = np.where(lab == lab[ri])[0]
        Ac = A[comp][:, comp].tocsr()
        loc = {g: k for k, g in enumerate(comp)}
        ri_c, ro_c = loc[ri], loc[ro]
        p = np.zeros(len(comp)); p[ri_c] = dP; p[ro_c] = 0.0
        keep = np.ones(len(comp), bool); keep[[ri_c, ro_c]] = False
        free = np.where(keep)[0]
        b = -(Ac[free][:, [ri_c, ro_c]] @ p[[ri_c, ro_c]])
        if len(free):
            p[free] = spl.spsolve(Ac[free][:, free].tocsc(), b)
        q = Ac[ri_c] @ p
        Q = float(q.item()) if hasattr(q, "item") else float(q)
        ks.append(Q * ETA * (2 * h) / ((2 * h) ** 2 * dP))
        nin.append(n_in); nout.append(n_out)
    return np.array(ks), np.array(nin), np.array(nout)


def measure(net: Network, fl: Flow, centre, h, dP=100.0) -> Darcy:
    """Face-to-face permeability and the bundle descriptors, all on the same
    cube and all weighted by the length of each vessel inside it."""
    ks, nin, nout = face_to_face(net, fl, centre, h, dP)
    f = cube_fraction(net.p0, net.p1, np.asarray(centre), h)     # in-cube fraction
    m = f > 0
    Vol = (2 * h) ** 3
    w = np.pi * net.r[m] ** 2 * net.Lpath[m] * f[m]              # in-cube lumen volume
    if w.sum() <= 0:
        return Darcy(k_face=ks, k_mean=0.0, n_blocked=int(np.sum(np.asarray(ks) <= 0)), phi=0.0,
                     d_perm=0.0, d_vol=0.0, T2=np.nan, c_k=np.nan, c_k_bundle=np.nan,
                     n_in=nin, n_out=nout, Q=ks * 0)
    phi = w.sum() / Vol
    d = net.d[m]
    d_vol = (w * d).sum() / w.sum(); d_perm = np.sqrt((w * d ** 2).sum() / w.sum())
    T = net.Lpath[m] / net.Lchord[m]; T2 = (w * T ** 2).sum() / w.sum()
    k_mean = float(np.mean(ks))                                  # includes blocked directions
    n_blocked = int(np.sum(np.asarray(ks) <= 0))
    c_k = float(phi * d_perm ** 2 / k_mean) if k_mean > 0 else np.nan
    return Darcy(k_face=ks, k_mean=k_mean, n_blocked=n_blocked, phi=float(phi), d_perm=float(d_perm),
                 d_vol=float(d_vol), T2=float(T2), c_k=c_k, c_k_bundle=float(32 * T2),
                 n_in=nin, n_out=nout, Q=ks * (2 * h) * dP / ETA)


def bundle(phi, d_perm, T2):
    """The two bundle-law descriptors on a given support.

    k_bundle_straight is the parallel-channel law; k_bundle_tort divides it by
    the measured mean squared tortuosity, which is the factor the closure would
    carry for channels of that path length.  Both are descriptors of the same
    support as the phi, d_perm and T2 passed in; they are not interchangeable
    with the solved network permeability."""
    return dict(k_bundle_straight=float(phi * d_perm ** 2 / 32),
                k_bundle_tort=float(phi * d_perm ** 2 / (32 * T2)) if T2 > 0 else float("nan"))


def support_stats(net, frac, Vol):
    """phi, d_vol, d_perm and <T^2> on any support, weighted by the length of
    each vessel inside it.  `frac` is the fraction of each segment inside the
    support (cube_fraction or sphere_fraction)."""
    m = frac > 0
    w = np.pi * net.r[m] ** 2 * net.Lpath[m] * frac[m]
    if w.sum() <= 0:
        return dict(phi=0.0, d_vol=0.0, d_perm=0.0, T2=float("nan"), n_seg=0)
    d = net.d[m]; T = net.Lpath[m] / net.Lchord[m]
    return dict(phi=float(w.sum() / Vol),
                d_vol=float((w * d).sum() / w.sum()),
                d_perm=float(np.sqrt((w * d ** 2).sum() / w.sum())),
                T2=float((w * T ** 2).sum() / w.sum()),
                n_seg=int(m.sum()))


def coherence(net: Network, fl: Flow, centre, R):
    """Flow-direction coherence of a spherical volume, and the two speeds it
    separates.

        chi = || sum_e Q_e L_e t_e || / sum_e |Q_e| L_e

    the vector-averaged flux over the scalar throughput, both length weighted by
    the part of each vessel inside the volume.  chi near one means the blood in
    the volume moves coherently, so a single drift velocity is meaningful; chi
    much less than one means arterial and venous branches carry blood in
    opposing directions and cancel, so the throughput can be large while the
    mean vector velocity is near zero.  Only the second enters a
    single-compartment Darcy closure, which is why the two are reported apart.

    `darcy_speed` is the vector-averaged flux per unit volume; `vessel_speed` is
    the flow-weighted mean speed inside the vessels, which is what a tracer
    samples.  Their ratio is the quantity a one-compartment closure would need
    to be of order one."""
    w = sphere_fraction(net.p0, net.p1, np.asarray(centre, float), R)
    m = w > 0
    out = dict(n_seg=int(m.sum()), chi=None, phi=None,
               darcy_speed=None, vessel_speed=None)
    if m.sum() < 10:
        return out
    u = (net.p1 - net.p0) / net.Lchord[:, None]
    sgn = np.sign(fl.Q)[:, None]
    L = net.Lpath[m] * w[m]
    Q = np.abs(fl.Q)[m]
    vec = ((Q * L)[:, None] * (u[m] * sgn[m])).sum(axis=0)
    gross = float((Q * L).sum())
    V = 4.0 / 3.0 * np.pi * R ** 3
    vol = np.pi * net.r[m] ** 2 * L
    tot = Q.sum()
    out.update(chi=float(np.linalg.norm(vec) / gross) if gross > 0 else None,
               phi=float(vol.sum() / V),
               darcy_speed=float(np.linalg.norm(vec) / V),
               vessel_speed=float(((np.abs(fl.Q) / (np.pi * net.r ** 2))[m] * Q).sum() / tot)
               if tot > 0 else None)
    return out
