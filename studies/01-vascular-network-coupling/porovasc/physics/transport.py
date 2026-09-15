"""Bubble transport on the network and the CEUS estimators.

Frequency-domain propagation.  For arterial segment i with inflow
concentration C_in,i(w) (flux-weighted), the outflow is
    C_out,i(w) = C_in,i(w) h_i(w)
and the bubble NUMBER present in the segment is
    N_i(w) = Q_i C_in,i(w) S_i(w),   S_i(w) = (1 - h_i(w)) / (j w)
(S is the Fourier transform of the residence-time survival function).

h_i is delta(t - L/v) for plug flow, or the flux-weighted Poiseuille
residence-time density  2 t_min^2 / t^3  on [t_min, t_max] with
t_min = L / (2 v) and the bubble-size cap  t_max = t_min / (1 - rho_max^2),
rho_max = 1 - d_bubble / d.  Full remix at every node.

Veins: each vein segment's inflow is the flux-weighted mix of its children's
outflows and the bed inflows at its node; the bed has a gamma impulse
response.  The arterial tree is traversed root-to-leaf, the venous tree
leaf-to-root.

Units: SI.  Concentration is arbitrary (AIF normalized to unit integral),
so N_i is in "bubbles" for one bubble injected.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.optimize import least_squares
from ..geometry.network import Network, in_cube, cube_fraction, clip_segment_cube
from ..physics.flow import Flow

D_BUBBLE = 2.0e-6         # microbubble diameter
RHO_MAX = 0.7             # lift-limited outermost radial position (Segre-Silberberg-like);
                          # caps the Poiseuille residence tail at t_min/(1-RHO_MAX^2) ~ 2 t_min
TAU_CAP, CV_CAP = 1.0, 0.5   # lumped bed residence time and its CV
DT, T_END = 0.02, 81.92      # time grid (4096 samples)
AIF_T0, AIF_MU, AIF_SIG = 5.0, np.log(10.0), 0.55   # lognormal bolus


def time_grid():
    t = np.arange(0, T_END, DT)
    w = 2 * np.pi * np.fft.rfftfreq(len(t), DT)
    return t, w


def aif(t):
    x = np.clip(t - AIF_T0, 1e-9, None)
    a = np.exp(-(np.log(x) - AIF_MU) ** 2 / (2 * AIF_SIG ** 2)) / (x * AIF_SIG * np.sqrt(2 * np.pi))
    a[t <= AIF_T0] = 0
    return a / (a.sum() * DT)


# ---------------------------------------------------------------- kernels
# A segment is a straight chord carrying a path length Lpath.  Flow enters at
# the hydrodynamic inlet, which is p0 for arteries and p1 for veins (the venous
# chords are copies of the arterial ones, so their flow runs backwards along the
# stored direction).  Arc length s below is always measured from that inlet.

_GL = None
def _quad(n=48):
    """Gauss-Legendre nodes and weights on [0, 1]."""
    global _GL
    if _GL is None or len(_GL[0]) != n:
        x, wq = np.polynomial.legendre.leggauss(n)
        _GL = (0.5 * (x + 1), 0.5 * wq)
    return _GL


def rho_cap(d):
    """Outermost radial position a bubble centre can occupy: the lift limit or
    the bubble size, whichever is more restrictive."""
    return float(min(RHO_MAX, max(0.0, 1 - D_BUBBLE / d)))


def _profile(v, d, n=48):
    """Flux-weighted radial distribution of bubble speeds in a segment.
    Returns (u, p) with speeds u(rho) = 2 |v| (1 - rho^2) and weights p summing
    to one, for the capped parabolic profile.  In the degenerate limit rho_c -> 0
    the distribution collapses to the centreline speed 2 |v|.

    48 nodes keep the relative error of a voxel time course below 1e-5 over the
    whole range covered by the studies (rho_max 0.5 to 0.9, transit times up to
    20 s); the error grows with rho_max and with the transit time, because the
    residence density develops a longer tail.
        f(rho) = 4 rho (1 - rho^2) / (2 rho_c^2 - rho_c^4)."""
    rc = rho_cap(d)
    if rc <= 1e-6:                       # centreline only: parabolic limit, not plug
        return np.array([2 * abs(v)]), np.array([1.0])
    x, wq = _quad(n)
    rho = rc * x
    f = 4 * rho * (1 - rho ** 2) / (2 * rc ** 2 - rc ** 4)
    p = f * wq * rc
    p = p / p.sum()
    return 2 * abs(v) * (1 - rho ** 2), p


def _delay_box(w, s0, s1, u):
    """Frequency-domain occupancy of the piece [s0, s1] of a channel of speed u,
    per unit inflow flux: the transit of the piece delayed by its entry time.
        int_{s0/u}^{s1/u} e^{-j w t} dt = (D/u) sinc(w D / (2 pi u)) e^{-j w (s0 + D/2)/u}
    (numerically stable at w = 0, where it is the residence time D/u)."""
    D = s1 - s0
    return (D / u) * np.sinc(w * D / (2 * np.pi * u)) * np.exp(-1j * w * (s0 + 0.5 * D) / u)


def _delta_bins(out, tau, weight):
    """Add a point mass `weight` at time tau, split linearly between the two
    neighbouring samples, which preserves both the mass and the first moment.
    Sample k of the grid represents time k DT."""
    n = len(out)
    x = tau / DT
    i = int(np.floor(x)); f = x - i
    if 0 <= i < n: out[i] += weight * (1 - f) / DT
    if 0 <= i + 1 < n: out[i + 1] += weight * f / DT


def _box_bins(out, t0, t1, height):
    """Add a box of the given height over [t0, t1] to the density `out`.

    The part of the box in each grid cell is deposited at the centroid of that
    overlap with _delta_bins, so both the mass and the first moment of the box
    are preserved exactly.  The second moment is inflated: linear splitting of a
    point mass at phase f between two samples adds f (1 - f) DT^2, which is
    DT^2/6 on average over phases, DT^2/6 for a box covering a full cell and up
    to DT^2/4 for a narrow overlap.  The inflation is therefore phase dependent
    and of order DT^2, and it accumulates along a path of segments."""
    if not (t1 > t0):
        return
    i0 = int(np.floor(t0 / DT)); i1 = int(np.floor(t1 / DT))
    for i in range(i0, i1 + 1):
        a = max(t0, i * DT); b = min(t1, (i + 1) * DT)
        if b > a:
            _delta_bins(out, 0.5 * (a + b), height * (b - a))


def piece_kernel_time(t, s0, s1, v, d, poiseuille):
    """Occupancy impulse response of the piece [s0, s1], per unit inflow flux,
    on the time grid: g(tau) = sum_j p_j 1[s0/u_j <= tau <= s1/u_j].
    Its integral is the residence time of the piece."""
    g = np.zeros(len(t))
    if not poiseuille:
        _box_bins(g, s0 / abs(v), s1 / abs(v), 1.0)
        return g
    u, p = _profile(v, d)
    for uj, pj in zip(u, p):
        _box_bins(g, s0 / uj, s1 / uj, pj)
    return g


def piece_occupancy(w, s0, s1, v, d, poiseuille, t=None):
    """Occupancy spectrum of the piece [s0, s1] (arc length from the inlet) of a
    segment, per unit inflow flux.  Its DC value is the residence time of the
    piece: D/|v| for plug flow, D / (|v| (2 - rho_c^2)) for the capped
    parabolic profile.

    With `t` given the kernel is built on the time grid and transformed, which
    is what the studies use; without it the analytic form is evaluated, which is
    what the analytic tests use."""
    if t is not None:
        return np.fft.rfft(piece_kernel_time(t, s0, s1, v, d, poiseuille)) * DT
    if not poiseuille:
        return _delay_box(w, s0, s1, abs(v))
    u, p = _profile(v, d)
    D = s1 - s0
    U = u[:, None]
    box = (D / U) * np.sinc(w[None, :] * D / (2 * np.pi * U)) * np.exp(-1j * w[None, :] * (s0 + 0.5 * D) / U)
    return (p[:, None] * box).sum(0)


def piece_residence(s0, s1, v, d, poiseuille):
    D = s1 - s0
    if not poiseuille:
        return D / abs(v)
    rc = rho_cap(d)
    if rc <= 1e-6:                       # centreline only: speed 2 |v|
        return D / (2 * abs(v))
    return D / (abs(v) * (2 - rc ** 2))


def _h_plug(w, tau):
    """Returns (H(w), mean residence)."""
    return np.exp(-1j * w * tau), float(tau)


def _h_pois(w, L, v, d, t=None):
    """Transfer function of a whole segment for the capped parabolic profile, on
    the same quadrature as piece_occupancy, so that the pieces of a segment and
    the whole segment stay consistent.  Returns (H(w), mean residence).

    With `t` given the residence density is built on the time grid (point masses
    split linearly between neighbouring samples, which preserves the area and
    the mean) and transformed; this is what the studies use.  Without `t` the
    analytic sum of exponentials is evaluated, which is what the tests use."""
    u, p = _profile(v, d)
    tau = float(piece_residence(0.0, L, v, d, True))
    if t is not None:
        g = np.zeros(len(t))
        for uj, pj in zip(u, p):
            _delta_bins(g, L / uj, pj)
        return np.fft.rfft(g) * DT, tau
    H = (p[:, None] * np.exp(-1j * w[None, :] * L / u[:, None])).sum(0)
    return H, tau


def tail_fraction(y):
    """Fraction of a curve's area in the last 5 % of the record.

    This is a heuristic only: a curve whose delay exceeds the record wraps back
    to early times and can leave a small tail, so a small value does not prove
    that the transform is free of circular wrap.  Use travel_time_bound for
    that."""
    a = y.sum()
    return 0.0 if a <= 0 else float(y[int(0.95 * len(y)):].sum() / a)


def _slowest_segment_times(net, fl, poiseuille):
    """Time a bubble on the slowest streamline needs to cross each segment:
    L/|v| for plug flow, L / (2 |v| (1 - rho_c^2)) for the capped profile."""
    slow = np.asarray(net.Lpath, float) / np.maximum(np.abs(np.asarray(fl.v, float)), 1e-12)
    if poiseuille:
        rc = np.array([rho_cap(d) for d in np.asarray(net.d, float)])
        slow = slow / np.maximum(2 * (1 - rc ** 2), 1e-6)
    return slow


def route_times(net, fl, poiseuille=True):
    """Slowest-streamline travel times accumulated along the tree: from the
    inlet to the end of each arterial segment (root to leaf) and from the start
    of each venous segment to the outlet (leaf to root)."""
    slow = _slowest_segment_times(net, fl, poiseuille)
    t_art = np.zeros(net.n); t_ven = np.zeros(net.n)
    order = np.argsort(net.gen)
    for i in order:
        if net.kind[i] != 0: continue
        p = net.parent[i]
        t_art[i] = slow[i] + (t_art[p] if p >= 0 else 0.0)
    for j in order:                                   # root first for veins too
        if net.kind[j] != 1: continue
        p = net.parent[j]
        t_ven[j] = slow[j] + (t_ven[p] if p >= 0 and net.kind[p] == 1 else 0.0)
    return t_art, t_ven


def venous_return_times(net, fl):
    """Mean-flow time from the end of each venous segment to the outlet."""
    t_out = np.zeros(net.n)
    vein = np.where(net.kind == 1)[0]
    for j in vein[np.argsort(net.gen[vein])]:
        p = net.parent[j]
        t_out[j] = fl.t_seg[j] + (t_out[p] if p >= 0 and net.kind[p] == 1 else 0.0)
    return t_out


def bed_quantile(q=0.999):
    """Quantile of the gamma bed residence, from its distribution."""
    from scipy.stats import gamma
    k = 1 / CV_CAP ** 2
    return float(gamma.ppf(q, k, scale=TAU_CAP / k))


def travel_time_bound(net, fl, poiseuille=True, q=0.999):
    """Latest time at which a voxel curve can still carry signal, compared with
    the record length.

    The bolus lives inside the record; what can push a curve past the end is the
    transport delay.  The bound follows a bubble on the slowest streamline of
    every segment along the longest complete route: the time by which a fraction
    q of the bolus has passed the feeding arteries, the longest arterial route,
    the q quantile of the capillary bed residence, and the longest venous route
    to the outlet.  When the bound exceeds T_END the fixed-length transform wraps
    and the curves are invalid, not truncated."""
    t, _ = time_grid()
    a = aif(t); c = np.cumsum(a) * DT
    t_bolus = float(t[np.searchsorted(c, q * c[-1])])
    if getattr(net, "kind", None) is None or not hasattr(net, "parent"):
        # bare test networks: use what is available
        t_arr = np.asarray(fl.t_arr, float); art = t_arr[np.isfinite(t_arr)]
        slow = _slowest_segment_times(net, fl, poiseuille)
        return float(t_bolus + (art.max() if len(art) else 0.0) + bed_quantile(q) + slow.max())
    t_art, t_ven = route_times(net, fl, poiseuille)
    return float(t_bolus + t_art.max() + bed_quantile(q) + t_ven.max())


def window_ok(y, frac=1e-3):
    """Deprecated name for the tail heuristic; kept so that callers do not
    break.  It is not a proof that the record is free of circular wrap."""
    return tail_fraction(y) < frac


def _survival(w, h, tau):
    """S(w) = (1 - H(w)) / (j w) with the exact DC limit S(0) = mean residence."""
    S = (1 - h) / np.where(np.abs(w) > 0, 1j * w, 1)
    S[0] = tau
    return S


def _h_gamma(w, mean, cv):
    k = 1 / cv ** 2; th = mean / k
    return (1 + 1j * w * th) ** (-k)


_SPH = None
def _sphere_points(n=64):
    """Fixed quasi-uniform points in the unit ball (Fibonacci shells)."""
    global _SPH
    if _SPH is None:
        pts = []
        for r_, m in ((0.35, 8), (0.65, 20), (0.9, 36)):
            i = np.arange(m) + 0.5
            phi = np.arccos(1 - 2 * i / m); th = np.pi * (1 + 5 ** 0.5) * i
            pts.append(r_ * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])
        _SPH = np.vstack(pts)
    return _SPH


def _bed_overlap(P, c, h, R):
    """Fraction of the sphere of radius R about each point of P that lies inside
    the cube (centre c, half-size h).  Only spheres that can overlap are sampled."""
    P = np.atleast_2d(P); out = np.zeros(len(P))
    cand = np.all(np.abs(P - c) < h + R, axis=1)
    if R <= 0:
        out[cand] = in_cube(P[cand], c, h).astype(float); return out
    S = _sphere_points()
    for j in np.where(cand)[0]:
        out[j] = np.mean(in_cube(P[j] + R * S, c, h))
    return out


@dataclass
class Transport:
    t: np.ndarray
    N: dict                # segment index -> number-present time course (only stored segments)
    tic: dict              # kernel id -> TIC (bubbles present)
    tic_art: dict          # kernel id -> arterial-only TIC (diagnostic; NOT what CEUS sees)
    stats: dict            # per kernel: flow-weighted arrival mean/var at kernel centre plane
    poiseuille: bool
    V_vis: dict = None     # segment -> Q * mean residence (= TIC area; < geometric volume under the lift cap)
    tic_bed: dict = None   # kernel id -> bubbles resident in the lumped beds fed from the kernel


def propagate(net: Network, fl: Flow, kernels, poiseuille=True, keep_veins=True, extra=()):
    """Propagate the bolus and return, for every kernel, the number of bubbles
    present in it over time.

    kernels: list of (centre, half) cubes.  extra: segment indices whose whole
    segment occupancy N(t) is also stored (path controls).

    A segment that crosses a kernel contributes only the piece of itself that
    lies inside: the piece is clipped from the chord, its arc length is measured
    from the hydrodynamic inlet of the segment, and its occupancy carries the
    delay of that entry point.  Two kernels on the same long segment therefore
    see the bolus at different times.
    """
    t, w = time_grid()
    A = np.fft.rfft(aif(t)) * DT
    n = net.n
    nk = len(kernels)
    # Clip every segment against every kernel, keeping only the intersections.
    # Arc length is measured from the hydrodynamic inlet: arteries flow p0 -> p1,
    # veins p1 -> p0.  Only the hit lists are stored, so memory scales with the
    # number of intersections rather than with segments times kernels.
    inlet_p1 = net.kind == 1
    K_idx = []; K_s0 = []; K_s1 = []
    seg_kernels = {}
    for k, (c, hh) in enumerate(kernels):
        hit, t0, t1 = clip_segment_cube(net.p0, net.p1, np.asarray(c), hh)
        idx = np.where(hit)[0]
        a = np.where(inlet_p1[idx], 1 - t1[idx], t0[idx]) * net.Lpath[idx]
        b = np.where(inlet_p1[idx], 1 - t0[idx], t1[idx]) * net.Lpath[idx]
        K_idx.append(idx); K_s0.append(a); K_s1.append(b)
        for j, i in enumerate(idx):
            seg_kernels.setdefault(int(i), []).append((k, a[j], b[j]))
    # the lumped bed of a bed-connected node is distributed over a sphere of
    # radius R_LAT at its end node; a kernel receives the fraction it overlaps
    from ..physics.flow import R_LAT as _RLAT
    bed_nodes = np.unique(np.asarray(fl.bed_a, int))
    kbedw = [_bed_overlap(net.p1[bed_nodes], np.asarray(c), hh, _RLAT) for c, hh in kernels]
    kbed = [dict(zip(bed_nodes[wb > 0], wb[wb > 0])) for wb in kbedw]
    bed_kernels = {}                       # bed node -> [(kernel, weight)]
    for k, d_ in enumerate(kbed):
        for i, wb in d_.items():
            bed_kernels.setdefault(int(i), []).append((k, float(wb)))
    need = np.zeros(n, bool)
    for idx in K_idx: need[idx] = True
    for d_ in kbed:
        for i in d_: need[i] = True
    for i in extra: need[i] = True

    nf = len(w)
    Kspec = [np.zeros(nf, complex) for _ in range(nk)]      # all compartments
    Kart = [np.zeros(nf, complex) for _ in range(nk)]       # explicit vessels only, arterial
    Kbed = [np.zeros(nf, complex) for _ in range(nk)]       # lumped bed
    Nstore = {}                                            # whole-segment occupancy (controls)
    Vvis = {}                                              # Q * mean residence per segment

    Hbed = _h_gamma(w, TAU_CAP, CV_CAP)
    Sbed = (1 - Hbed) / np.where(np.abs(w) > 0, 1j * w, 1); Sbed[0] = TAU_CAP
    vin_num = {}; vin_den = {}
    bed_edges = {}
    for e, i in enumerate(fl.bed_a): bed_edges.setdefault(int(i), []).append(e)

    def add_pieces(i, cin, kind):
        """Accumulate the clipped pieces of segment i into their kernels."""
        q = abs(fl.Q[i]); v = fl.v[i]; d = net.d[i]
        for k, s0, s1 in seg_kernels.get(int(i), ()):
            if not (s1 > s0):
                continue
            occ = q * cin * piece_occupancy(w, s0, s1, v, d, poiseuille, t=t)
            Kspec[k] += occ
            if kind == 0:
                Kart[k] += occ

    Cout = {}
    n_child = np.zeros(n, int)
    for i in range(n):
        pi = net.parent[i]
        if pi >= 0: n_child[pi] += 1
    left = n_child.copy()
    order = np.argsort(net.gen, kind="stable")
    for i in order:
        if net.kind[i] != 0: continue
        cin = A if net.parent[i] < 0 else Cout[net.parent[i]]
        L, v, d = net.Lpath[i], fl.v[i], net.d[i]
        h, tau = _h_pois(w, L, v, d, t=t) if poiseuille else _h_plug(w, L / abs(v))
        Cout[i] = cin * h
        if need[i]:
            Vvis[i] = abs(fl.Q[i]) * tau
            add_pieces(i, cin, 0)
        if i in extra:
            Nstore[i] = abs(fl.Q[i]) * cin * piece_occupancy(w, 0.0, L, v, d, poiseuille, t=t)
        pi = net.parent[i]
        if pi >= 0:
            left[pi] -= 1
            if left[pi] == 0: Cout.pop(int(pi), None)
        edges = bed_edges.get(int(i), ())
        if edges:
            qtot = 0.0
            for e in edges:
                j = int(fl.bed_v[e]); qb = abs(fl.Q_bed[e]); qtot += qb
                vin_num[j] = vin_num.get(j, 0) + qb * Cout[i] * Hbed
                vin_den[j] = vin_den.get(j, 0) + qb
            if need[i]:
                nb = qtot * Cout[i] * Sbed         # bubbles resident in the bed fed by i
                for k, wb in bed_kernels.get(int(i), ()):
                    Kbed[k] += wb * nb
                    Kspec[k] += wb * nb
        if left[i] == 0:
            Cout.pop(int(i), None)                 # leaf: spectrum no longer needed
    vch = {}
    for j in np.where(net.kind == 1)[0]:
        pj = net.parent[j]
        if pj >= 0: vch.setdefault(pj, []).append(j)
    Vout = {}
    if keep_veins:
        for j in sorted(np.where(net.kind == 1)[0], key=lambda j: -net.gen[j]):
            num = vin_num.get(j, 0); den = vin_den.get(j, 0)
            for c in vch.get(j, []):
                num = num + Vout[c] * abs(fl.Q[c]); den += abs(fl.Q[c])
                Vout.pop(int(c), None)
            cin = num / den if den > 0 else np.zeros_like(A)
            L, v, d = net.Lpath[j], fl.v[j], net.d[j]
            h, tau = _h_pois(w, L, v, d, t=t) if poiseuille else _h_plug(w, L / abs(v))
            Vout[j] = cin * h
            if need[j]:
                Vvis[j] = abs(fl.Q[j]) * tau
                add_pieces(j, cin, 1)
            if j in extra:
                Nstore[j] = abs(fl.Q[j]) * cin * piece_occupancy(w, 0.0, L, v, d, poiseuille, t=t)
    inv = lambda X: np.fft.irfft(X, n=len(t)) / DT
    tic = {k: inv(Kspec[k]) for k in range(nk)}
    tic_art = {k: inv(Kart[k]) for k in range(nk)}
    tic_bed = {k: inv(Kbed[k]) for k in range(nk)}
    N = {i: inv(Nstore[i]) for i in Nstore}
    # arrival statistics of the pieces actually inside each kernel
    t_in = fl.t_arr - net.Lpath / np.abs(fl.v)          # arrival at the segment inlet
    stats = {}
    for k in range(nk):
        sel = net.kind[K_idx[k]] == 0
        idx = K_idx[k][sel]
        if len(idx):
            frac = (K_s1[k][sel] - K_s0[k][sel]) / net.Lpath[idx]
            q = np.abs(fl.Q[idx]) * frac
            ta = t_in[idx] + K_s0[k][sel] / np.abs(fl.v[idx])   # entry of the clipped piece
        else:
            q = np.zeros(0); ta = np.zeros(0)
        if q.sum() > 0:
            tm = (q * ta).sum() / q.sum()
            stats[k] = dict(t_mean=float(tm), t_var=float((q * (ta - tm) ** 2).sum() / q.sum()),
                            v_mean=float((q * np.abs(fl.v[idx])).sum() / q.sum()))
        else:
            stats[k] = dict(t_mean=np.nan, t_var=np.nan, v_mean=np.nan)
        y = tic[k]; a = y.sum() * DT
        if a > 0:
            m1 = (t * y).sum() * DT / a; m2 = ((t - m1) ** 2 * y).sum() * DT / a
            stats[k].update(m1=float(m1), m2=float(m2), area=float(a))
        else:
            stats[k].update(m1=np.nan, m2=np.nan, area=0.0)
    bound = travel_time_bound(net, fl, poiseuille)
    if bound > T_END:
        import warnings
        warnings.warn("record of %.1f s is shorter than the travel-time bound of %.1f s: "
                      "the transform wraps and the curves are invalid" % (T_END, bound),
                      RuntimeWarning)
    return Transport(t=t, N=N, tic=tic, tic_art=tic_art, stats=stats, poiseuille=poiseuille,
                     V_vis=Vvis, tic_bed=tic_bed)


# ------------------------------------------------------------ estimators
def q_of(w, v, D):
    return np.sqrt(1 + 4j * D * w / v ** 2)


def T_old(w, dz, v, D):
    q = q_of(w, v, D); return np.exp(-v * dz / (2 * D) * (q - 1)) / q


def T_new(w, dz, v, D):
    q = q_of(w, v, D); return np.exp(-v * dz / (2 * D) * (q - 1))


def identify(t, tic_in, tic_out, dz, tf="new", v0=5e-3, D0=1e-6):
    """Two-kernel system identification: fit (v, D) so that
    IFFT(T(w) FFT(tic_in)) matches tic_out.  Curves normalized to unit area."""
    dt = t[1] - t[0]; w = 2 * np.pi * np.fft.rfftfreq(len(t), dt)
    u = tic_in / (tic_in.sum() * dt); y = tic_out / (tic_out.sum() * dt)
    U = np.fft.rfft(u)
    T = T_new if tf == "new" else T_old
    def resid(x):
        v, D = np.exp(x)
        yp = np.fft.irfft(U * T(w, dz, v, D), n=len(t))
        return yp - y
    r = least_squares(resid, np.log([v0, D0]), bounds=(np.log([1e-4, 1e-9]), np.log([0.1, 1e-4])))
    v, D = np.exp(r.x)
    yp = np.fft.irfft(U * T(w, dz, v, D), n=len(t))
    r2 = 1 - np.sum((yp - y) ** 2) / np.sum((y - y.mean()) ** 2)
    return v, D, r2


def mldrw_kappa(t, tic):
    """Single-kernel mLDRW fit: C(t) = A sqrt(k/(2 pi (t-t0))) exp(-k (t-t0-mu)^2 / (2 (t-t0)))."""
    y = tic / tic.max()
    i0 = np.argmax(y > 0.02)
    def model(x):
        A, k, mu, t0 = x
        tt = np.clip(t - t0, 1e-6, None)
        return A * np.sqrt(k / (2 * np.pi * tt)) * np.exp(-k * (tt - mu) ** 2 / (2 * tt))
    x0 = [1.0, 1.0, t[np.argmax(y)] - t[i0], t[i0] - 1]
    r = least_squares(lambda x: model(x) - y, x0,
                      bounds=([0, 1e-3, 0.1, -5], [10, 100, 60, 30]))
    return r.x[1], r.x[2]


# ------------------------------------------------------------ clinical CUDI:
# input voxel + SPHERICAL SHELL of output voxels at radius dz (the 3D analogue
# of the 2D annulus), causal pairs only, ONE (v, D) fitted to all pairs jointly.
# The shell is sampled at n_dir points on a Fibonacci lattice; the clinical
# implementation uses every voxel in the shell.
def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def shell_kernels(centre, dz, vox_half, n_dir=48):
    """Input voxel at centre + n_dir output voxels sampling the spherical shell of radius dz."""
    centre = np.asarray(centre)
    ks = [(centre, vox_half)]
    for u in fibonacci_sphere(n_dir):
        ks.append((centre + dz * u, vox_half))
    return ks


CAUSAL_RULES = ("front20", "front10", "moment")


def _arrival(t, y, rule):
    """Arrival time of a unit-area curve under one causality rule: the time at
    which the wash-in first reaches 20 % (front20) or 10 % (front10) of the
    peak, or the first moment of the curve (moment)."""
    if rule == "moment":
        return float((t * y).sum() / y.sum())
    frac = 0.2 if rule == "front20" else 0.1
    return float(t[np.argmax(y >= frac * y.max())])


def identify_shell(t, tic_in, tic_outs, dz, tf="new", v0=5e-3, D0=1e-6, causal=True, detail=False,
                   rule="front20"):
    """Joint (v, D) over all causal output voxels of a spherical shell.

    causal: keep the outputs whose arrival, under `rule` (front20, front10 or
    moment; see _arrival), is later than the input's.  All curves are
    normalized to unit area and one (v, D) is fitted to all pairs with the
    transfer function `tf` (T_new without the 1/q prefactor, T_old with it)."""
    dt = t[1] - t[0]; w = 2 * np.pi * np.fft.rfftfreq(len(t), dt)
    if rule not in CAUSAL_RULES:
        raise ValueError("unknown causality rule %r" % rule)
    def norm(y):
        s = y.sum() * dt
        return y / s if s > 0 else None
    u = norm(tic_in)
    empty = dict(accepted=[], margin=[], f_in=np.nan)
    if u is None:
        return (np.nan, np.nan, 0, np.nan, empty) if detail else (np.nan, np.nan, 0, np.nan)
    f_in = _arrival(t, u, rule)
    ys = []; acc = []; marg = []
    for j, y in enumerate(tic_outs):
        yn = norm(y)
        if yn is None:
            marg.append(None); continue
        m = float(_arrival(t, yn, rule) - f_in)
        marg.append(m)                                  # every direction, accepted or not
        if causal and m <= 0: continue
        ys.append(yn); acc.append(int(j))
    info = dict(accepted=acc, margin=marg, f_in=float(f_in))
    if len(ys) < 3:
        return (np.nan, np.nan, len(ys), np.nan, info) if detail else (np.nan, np.nan, len(ys), np.nan)
    Y = np.array(ys); U = np.fft.rfft(u)
    T = T_new if tf == "new" else T_old
    def resid(x):
        v, D = np.exp(x)
        yp = np.fft.irfft(U * T(w, dz, v, D), n=len(t))
        return (Y - yp[None, :]).ravel()
    r = least_squares(resid, np.log([v0, D0]), bounds=(np.log([1e-4, 1e-9]), np.log([0.1, 1e-4])))
    v, D = np.exp(r.x)
    yp = np.fft.irfft(U * T(w, dz, v, D), n=len(t))
    r2 = 1 - np.sum((Y - yp) ** 2) / np.sum((Y - Y.mean(1, keepdims=True)) ** 2)
    info["r2"] = float(r2)
    return (v, D, len(ys), r2, info) if detail else (v, D, len(ys), r2)


# backwards-compatible aliases
ring_kernels = shell_kernels
identify_ring = identify_shell


# ------------------------------------------------------------ direction-aware
# identification: first estimate the main transport direction
# from the wash-in front across the shell, then fit only outputs inside a cone
# around it.
def identify_cone(t, tic_in, tic_outs, dirs, dz, tf="new", half_angle_deg=35.0,
                  v0=5e-3, D0=1e-6):
    """dirs: unit vectors from the input voxel to each output voxel.
    Step 1: weighted least squares of front time on position -> gradient g;
            transport direction u = g / |g| (fronts arrive later downstream).
    Step 2: joint (v, D) over causal outputs with angle(dir, u) < half_angle."""
    dt = t[1] - t[0]; w = 2 * np.pi * np.fft.rfftfreq(len(t), dt)
    def norm(y):
        s = y.sum() * dt
        return y / s if s > 0 else None
    def front(y):
        return t[np.argmax(y >= 0.2 * y.max())]
    u_in = norm(tic_in)
    if u_in is None: return np.nan, np.nan, 0, np.nan, None
    f_in = front(u_in)
    ys, X, fr = [], [], []
    for y, d in zip(tic_outs, dirs):
        yn = norm(y)
        if yn is None: continue
        ys.append(yn); X.append(d); fr.append(front(yn) - f_in)
    if len(ys) < 4: return np.nan, np.nan, len(ys), np.nan, None
    X = np.array(X); fr = np.array(fr)
    g, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], fr, rcond=None)
    gv = g[:3]
    if np.linalg.norm(gv) < 1e-12: return np.nan, np.nan, 0, np.nan, None
    u = gv / np.linalg.norm(gv)
    cosang = X @ u
    keep = (cosang > np.cos(np.deg2rad(half_angle_deg))) & (fr > 0)
    if keep.sum() < 3: return np.nan, np.nan, int(keep.sum()), np.nan, u
    Y = np.array(ys)[keep]; U = np.fft.rfft(u_in)
    T = T_new if tf == "new" else T_old
    def resid(x):
        v, D = np.exp(x)
        yp = np.fft.irfft(U * T(w, dz, v, D), n=len(t))
        return (Y - yp[None, :]).ravel()
    r = least_squares(resid, np.log([v0, D0]), bounds=(np.log([1e-4, 1e-9]), np.log([0.1, 1e-4])))
    v, D = np.exp(r.x)
    yp = np.fft.irfft(U * T(w, dz, v, D), n=len(t))
    r2 = 1 - np.sum((Y - yp) ** 2) / np.sum((Y - Y.mean(1, keepdims=True)) ** 2)
    return v, D, int(keep.sum()), r2, u
