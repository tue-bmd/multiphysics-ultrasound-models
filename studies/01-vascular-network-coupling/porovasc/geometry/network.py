"""3D vascular network of the prostate: gland-scale tree + RVE refinement.

Nothing here is calibrated to give
a wanted answer; every constant is either anatomical (cited in the log) or a
distribution whose realized statistics are MEASURED by debug/check_geometry.

Units: SI throughout (metres).  Diameters in comments in micrometres.

Data model: parallel numpy arrays, one entry per segment.
    p0, p1   chord endpoints              (N, 3)
    r        radius                        (N,)
    Lchord   chord length                  (N,)
    Lpath    tortuous path length          (N,)   Lpath = T * Lchord
    parent   parent index or -1            (N,)
    gen      generation (0 = feeder)       (N,)
    tree     feeder id                     (N,)
    kind     0 = artery, 1 = vein          (N,)
    zone     0 = gland, 1 = RVE            (N,)
    rve      RVE id or -1                  (N,)
    term     True if terminal artery       (N,)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import cKDTree

# ------------------------------------------------------------ anatomy
GLAND_SEMI = np.array([20e-3, 15e-3, 17.5e-3])   # transverse, AP, craniocaudal
FEED_D, FEED_L = 0.6e-3, 25e-3                   # penetrating feeders (extracapsular)
D_ROOT = 0.5e-3                                  # largest intraprostatic caliber
F_VEIN = 1.6                                     # vena comitans / artery caliber

# ------------------------------------------------------------ branching
ALPHA = 0.8            # Zamir asymmetry r2/r1
ALPHA_JIT = 0.08       # per-node jitter on alpha (multiplicative sd)
GAMMA_MED = 10.0       # segment chord length / diameter, median
GAMMA_SD = 0.35        # lognormal sd -> ~5-20 range
TORT_MEAN, TORT_SD = 1.25, 0.12   # tortuosity T = path / chord, clipped 1.05-1.6
N_AZ = 8               # candidate azimuths per bifurcation (void bias)
REBUILD_EVERY = 150    # segments added between KD-tree rebuilds (bounds brute force)
GAP = 5e-6             # minimum wall-to-wall clearance between vessels


@dataclass
class Params:
    n_feeders: int = 4
    feed_d: float = FEED_D
    d_root: float = D_ROOT
    d_term_gland: float = 30e-6   # whole gland explicit down to the CEUS visibility floor
    d_term_rve: float = 20e-6     # RVEs refined one step further
    rve_centres: tuple = ((-5e-3, -8e-3, 0.0), (5e-3, -8e-3, 0.0))
    rve_half: float = 3e-3            # 6 mm cube
    seed: int = 1
    alpha: float = ALPHA
    gamma_med: float = GAMMA_MED
    tort_mean: float = TORT_MEAN
    # optional lesion: a sphere in which some parameters are overridden, e.g.
    # dict(centre=(x, y, z), radius=5e-3, tort_mean=1.6, alpha=0.6, gamma_med=6.0).
    # Segments whose START point lies in the sphere are drawn with the lesion
    # values; everything else is unchanged.  Only keys given are overridden.
    lesion: dict | None = None


@dataclass
class Network:
    p0: np.ndarray; p1: np.ndarray; r: np.ndarray
    Lchord: np.ndarray; Lpath: np.ndarray
    parent: np.ndarray; gen: np.ndarray; tree: np.ndarray
    kind: np.ndarray; zone: np.ndarray; rve: np.ndarray; term: np.ndarray
    why: np.ndarray; twin: np.ndarray
    r_dau: np.ndarray | None = None       # realized daughter radii at this node
    params: Params = None
    pruned: dict = field(default_factory=dict)
    lesion: np.ndarray | None = None      # per segment: drawn with lesion parameters

    @property
    def n(self): return len(self.r)
    @property
    def d(self): return 2 * self.r
    @property
    def art(self): return self.kind == 0
    @property
    def vein(self): return self.kind == 1

    def children(self):
        ch = [[] for _ in range(self.n)]
        for i, p in enumerate(self.parent):
            if p >= 0: ch[p].append(i)
        return ch


# ------------------------------------------------------------ helpers
def in_gland(p):
    return np.sum((p / GLAND_SEMI) ** 2, axis=-1) < 1.0


def in_cube(p, c, h):
    return np.all(np.abs(p - c) < h, axis=-1)


def clip_segment_cube(p0, p1, c, h):
    """Clip each chord p0 -> p1 against the axis-aligned cube of centre c and
    half-size h (Liang-Barsky), vectorised.

    Returns (hit, t0, t1) with t in [0, 1] along p0 -> p1; hit is False when the
    chord misses the cube or touches it in a set of zero length.  This is the
    single clipping primitive used for voxel membership, for segment transport
    and for the permeability solve, so that all three use the same geometry.
    """
    p0 = np.atleast_2d(p0); p1 = np.atleast_2d(p1); c = np.asarray(c, float)
    d = p1 - p0
    t0 = np.zeros(len(p0)); t1 = np.ones(len(p0))
    for ax in range(3):
        lo, hi = c[ax] - h, c[ax] + h
        dd = d[:, ax]; a = p0[:, ax]
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = np.where(dd != 0, (lo - a) / dd, -np.inf)
            tb = np.where(dd != 0, (hi - a) / dd, np.inf)
        tmin = np.minimum(ta, tb); tmax = np.maximum(ta, tb)
        par = dd == 0
        inside_par = (a > lo) & (a < hi)
        tmin = np.where(par, np.where(inside_par, -np.inf, np.inf), tmin)
        tmax = np.where(par, np.where(inside_par, np.inf, -np.inf), tmax)
        t0 = np.maximum(t0, tmin); t1 = np.minimum(t1, tmax)
    t0 = np.clip(t0, 0.0, 1.0); t1 = np.clip(t1, 0.0, 1.0)
    hit = t1 > t0 + 1e-12                      # zero-length tangencies are skipped
    return hit, np.where(hit, t0, 0.0), np.where(hit, t1, 0.0)


def cube_fraction(p0, p1, c, h):
    """Fraction of each chord that lies inside the cube (partial-volume
    membership of a segment in a voxel)."""
    hit, t0, t1 = clip_segment_cube(p0, p1, c, h)
    return np.where(hit, t1 - t0, 0.0)


def clip_segment_sphere(p0, p1, c, R):
    """Clip each chord against the sphere of centre c and radius R.
    Returns (hit, t0, t1), the same convention as clip_segment_cube."""
    p0 = np.atleast_2d(p0); p1 = np.atleast_2d(p1); c = np.asarray(c, float)
    d = p1 - p0; f = p0 - c
    a = np.einsum("ij,ij->i", d, d)
    b = 2 * np.einsum("ij,ij->i", f, d)
    cc = np.einsum("ij,ij->i", f, f) - R ** 2
    disc = b ** 2 - 4 * a * cc
    ok = (a > 0) & (disc > 0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = np.where(ok, (-b - sq) / (2 * a), 0.0)
        t1 = np.where(ok, (-b + sq) / (2 * a), 0.0)
    t0 = np.clip(t0, 0.0, 1.0); t1 = np.clip(t1, 0.0, 1.0)
    hit = ok & (t1 > t0 + 1e-12)
    return hit, np.where(hit, t0, 0.0), np.where(hit, t1, 0.0)


def sphere_fraction(p0, p1, c, R):
    hit, t0, t1 = clip_segment_sphere(p0, p1, c, R)
    return np.where(hit, t1 - t0, 0.0)


def murray_daughters(r, alpha):
    c1 = (1.0 + alpha ** 3) ** (-1.0 / 3.0)
    return r * c1, r * alpha * c1


def murray_angles(rp, r1, r2):
    """Optimal branching angles (Murray 1926) for given radii."""
    def ang(ra, rb):
        c = (rp ** 4 + ra ** 4 - rb ** 4) / (2 * rp ** 2 * ra ** 2)
        return np.arccos(np.clip(c, -1, 1))
    return ang(r1, r2), ang(r2, r1)


def perp_basis(u):
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(u, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return e1, e2


class _Collider:
    """KD-tree over sampled vessel points used only to FIND candidate segments;
    the clearance itself is the exact point-to-segment distance, so sampling
    density does not limit accuracy."""
    def __init__(self, ds=1.0e-4):
        self.ds = ds
        self.pts = np.zeros((0, 3)); self.sid = np.zeros(0, int)
        self.P0 = []; self.P1 = []; self.R = []
        self.tree = None; self.new_pts = []; self.new_sid = []

    def add(self, p0, p1, r):
        k = len(self.R)
        self.P0.append(np.asarray(p0, float)); self.P1.append(np.asarray(p1, float)); self.R.append(float(r))
        n = max(2, int(np.linalg.norm(p1 - p0) / self.ds) + 1)
        P = p0[None, :] + (p1 - p0)[None, :] * np.linspace(0, 1, n)[:, None]
        self.new_pts.append(P); self.new_sid.append(np.full(n, k))
        if len(self.new_pts) >= REBUILD_EVERY:
            self.rebuild()

    def rebuild(self):
        if self.new_pts:
            self.pts = np.vstack([self.pts] + self.new_pts)
            self.sid = np.concatenate([self.sid] + self.new_sid)
            self.new_pts, self.new_sid = [], []
        self.tree = cKDTree(self.pts) if len(self.pts) else None

    def _cands(self, p, rq):
        ids = set()
        if self.tree is not None:
            ids.update(self.sid[self.tree.query_ball_point(p, rq)].tolist())
        for P, S in zip(self.new_pts, self.new_sid):
            if np.any(np.linalg.norm(P - p, axis=1) < rq): ids.add(int(S[0]))
        return ids

    @staticmethod
    def _seg_seg(p0, p1, a, b):
        """Exact minimum distance between segments p0p1 and ab (Ericson 5.1.9)."""
        d1 = p1 - p0; d2 = b - a; r_ = p0 - a
        A = d1 @ d1; E = d2 @ d2; F = d2 @ r_
        if A < 1e-30 and E < 1e-30: return np.linalg.norm(r_)
        if A < 1e-30: s_, t_ = 0.0, np.clip(F / E, 0, 1)
        else:
            C = d1 @ r_
            if E < 1e-30: t_, s_ = 0.0, np.clip(-C / A, 0, 1)
            else:
                B = d1 @ d2; den = A * E - B * B
                s_ = np.clip((B * F - C * E) / den, 0, 1) if den > 1e-30 else 0.0
                t_ = (B * s_ + F) / E
                if t_ < 0: t_, s_ = 0.0, np.clip(-C / A, 0, 1)
                elif t_ > 1: t_, s_ = 1.0, np.clip((B - C) / A, 0, 1)
        return np.linalg.norm((p0 + s_ * d1) - (a + t_ * d2))

    @staticmethod
    def _pt_seg(p, a, b):
        ab = b - a; t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-30), 0, 1)
        return np.linalg.norm(p - (a + t * ab))

    def clearance_chord(self, p0, p1, r, parent=-1, nq=4):
        """Exact min wall-to-wall clearance of the chord p0->p1 (radius r) to
        all existing vessels, excluding ONLY the parent segment.  Siblings ARE
        checked (the second daughter sees the first)."""
        L = np.linalg.norm(p1 - p0)
        rq = 0.5 * L + r + 4e-4                      # covers r_max ~ 300 um
        ids = self._cands(0.5 * (p0 + p1), rq)
        best = np.inf
        for k in ids:
            a, b, rk = self.P0[k], self.P1[k], self.R[k]
            if k == parent: continue
            # a sibling shares the junction: legitimately zero clearance there.
            # exempt it inside a junction zone of 2.5 (r + rk), check beyond.
            sib = np.linalg.norm(a - p0) < 1e-9
            if sib:
                # exempt the junction zone: test the part of the new chord
                # beyond 2.5 (r + rk) from p0 against the part of the sibling
                # beyond the same distance from p0
                z = 2.5 * (r + rk)
                if z >= L or z >= np.linalg.norm(b - a): continue
                q0 = p0 + (p1 - p0) * (z / L)
                a2 = a + (b - a) * (z / np.linalg.norm(b - a))
                d = self._seg_seg(q0, p1, a2, b)
            else:
                d = self._seg_seg(p0, p1, a, b)
            best = min(best, d - r - rk)
            if best < 0: break
        return best


# ------------------------------------------------------------ builder
class _Builder:
    def __init__(self, prm: Params):
        self.prm = prm
        self.rng = np.random.default_rng(prm.seed)
        self.rows = []             # dicts, one per segment
        self.col = _Collider()
        self.pruned = dict(collision=0, domain=0)

    def _add(self, **kw):
        self.rows.append(kw); return len(self.rows) - 1

    def _in_lesion(self, p):
        L = self.prm.lesion
        return bool(L) and np.linalg.norm(np.asarray(p) - np.asarray(L["centre"])) < L["radius"]

    def _local(self, key, p):
        """Parameter value at point p: lesion override if p is in the lesion."""
        L = self.prm.lesion
        if L and key in L and self._in_lesion(p): return L[key]
        return getattr(self.prm, key)

    def _draw_gamma(self, p=None):
        med = self._local("gamma_med", p) if p is not None else self.prm.gamma_med
        return float(np.exp(np.log(med) + GAMMA_SD * self.rng.standard_normal()))

    def _draw_tort(self, p=None):
        mean = self._local("tort_mean", p) if p is not None else self.prm.tort_mean
        return float(np.clip(mean + TORT_SD * self.rng.standard_normal(), 1.05, 2.5))

    def grow(self, frontier, d_term, inside, zone, rve_id):
        """Grow branches in `frontier` (list of segment indices) until every
        branch is below d_term.  `inside(p)` is the confinement test."""
        while frontier:
            self.col.rebuild()
            nxt = []
            for idx in frontier:
                s = self.rows[idx]
                rp = s["r"]
                a = self._local("alpha", s["p1"]) * (1 + ALPHA_JIT * self.rng.standard_normal())
                a = float(np.clip(a, 0.5, 1.0))
                r1, r2 = murray_daughters(rp, a)
                s["r_dau"] = (float(r1), float(r2))     # realized, jittered, lesion-local
                if 2 * r2 < d_term and 2 * r1 < d_term:
                    s["term"] = True; s["why"] = "caliber"; continue
                th1, th2 = murray_angles(rp, r1, r2)
                u = (s["p1"] - s["p0"]) / s["Lchord"]
                e1, e2 = perp_basis(u)
                daughters = [(r1, th1, +1), (r2, th2, -1)]
                # daughter that would fall below d_term is not created; the
                # parent becomes terminal for that branch (no floor, no clip)
                daughters = [(r, th, sg) for r, th, sg in daughters if 2 * r >= d_term]
                if not daughters:
                    s["term"] = True; s["why"] = "caliber"; continue
                phis = self.rng.uniform(0, 2 * np.pi, N_AZ)
                placed = 0
                for r, th, sg in daughters:
                    g = self._draw_gamma(s["p1"]); L = g * 2 * r
                    best, bestv = -np.inf, None
                    for ph in phis:
                        ph = ph + (0 if sg > 0 else np.pi)
                        v = (np.cos(th) * u + np.sin(th) *
                             (np.cos(ph) * e1 + np.sin(ph) * e2))
                        p1 = s["p1"] + L * v
                        if not inside(p1): continue
                        c = self.col.clearance_chord(s["p1"], p1, r, parent=idx)
                        if c < GAP: continue
                        if c > best: best, bestv = c, v
                    if bestv is None:
                        # try shorter chord before giving up
                        for ph in phis:
                            v = (np.cos(th) * u + np.sin(th) *
                                 (np.cos(ph) * e1 + np.sin(ph) * e2))
                            p1 = s["p1"] + 0.5 * L * v
                            if inside(p1) and self.col.clearance_chord(s["p1"], p1, r, parent=idx) >= GAP:
                                bestv, L = v, 0.5 * L; break
                    if bestv is None:
                        self.pruned["collision" if inside(s["p1"] + L * u) else "domain"] += 1
                        continue
                    p1 = s["p1"] + L * bestv
                    j = self._add(p0=s["p1"].copy(), p1=p1, r=float(r), Lchord=float(L),
                                  Lpath=float(L * self._draw_tort(s["p1"])), parent=idx,
                                  gen=s["gen"] + 1, tree=s["tree"], kind=0,
                                  zone=zone, rve=rve_id, term=False, why="",
                                  lesion=self._in_lesion(s["p1"]))
                    self.col.add(s["p1"], p1, r)
                    nxt.append(j); placed += 1
                if placed == 0:
                    s["term"] = True; s["why"] = "pruned"
            frontier = nxt

    def build(self):
        prm = self.prm
        # feeders: n entry points on the gland surface, posterolateral bias
        az = np.deg2rad(np.linspace(0, 360, prm.n_feeders, endpoint=False) + 45)
        roots = []
        for k, a in enumerate(az):
            dirn = np.array([np.cos(a), -0.4, np.sin(a)]); dirn /= np.linalg.norm(dirn)
            # point on ellipsoid surface along dirn
            t = 1.0 / np.sqrt(np.sum((dirn / GLAND_SEMI) ** 2))
            entry = t * dirn
            p0 = entry + FEED_L * dirn
            f = self._add(p0=p0, p1=entry.copy(), r=prm.feed_d / 2, Lchord=FEED_L,
                          Lpath=FEED_L, parent=-1, gen=0, tree=k, kind=0,
                          zone=0, rve=-1, term=False)
            self.col.add(p0, entry, prm.feed_d / 2)
            # intraprostatic root aimed at centre
            L = self._draw_gamma() * prm.d_root
            p1 = entry - L * dirn
            j = self._add(p0=entry.copy(), p1=p1, r=prm.d_root / 2, Lchord=L,
                          Lpath=L * self._draw_tort(), parent=f, gen=1, tree=k,
                          kind=0, zone=0, rve=-1, term=False, why="")
            self.col.add(entry, p1, prm.d_root / 2)
            roots.append(j)
        self.grow(roots, prm.d_term_gland, in_gland, zone=0, rve_id=-1)

        # RVE refinement: gland terminals inside each cube keep growing
        for q, c in enumerate(prm.rve_centres):
            c = np.asarray(c, float)
            front = [i for i, s in enumerate(self.rows)
                     if s["term"] and s["kind"] == 0 and s["zone"] == 0
                     and in_cube(s["p1"], c, prm.rve_half)]
            for i in front:
                self.rows[i]["term"] = False; self.rows[i]["why"] = ""; self.rows[i]["rve"] = q
            self.grow(front, prm.d_term_rve,
                      lambda p, c=c: in_cube(p, c, prm.rve_half), zone=1, rve_id=q)

        # paired veins (venae comitantes), 1.6x caliber.  The offset is
        # defined PER NODE, not per segment, so the venous tree is
        # geometrically continuous: vein segment i runs from
        # p0_i + o(parent node) to p1_i + o(node i).  The offset azimuth is
        # inherited down the tree with a small perturbation.
        n_art = len(self.rows)
        vid, onode, oaz = {}, {}, {}
        for i in range(n_art):
            if self.rows[i]["gen"] > 0: vid[i] = n_art + len(vid)
        for i in range(n_art):                       # rows are in BFS order
            s = self.rows[i]
            if s["gen"] == 0: continue
            u = (s["p1"] - s["p0"]) / s["Lchord"]; e1, e2 = perp_basis(u)
            az = oaz.get(s["parent"], self.rng.uniform(0, 2 * np.pi)) + 0.3 * self.rng.standard_normal()
            oaz[i] = az
            mag = s["r"] + F_VEIN * s["r"] + 3 * GAP
            onode[i] = mag * (np.cos(az) * e1 + np.sin(az) * e2)
        for i in range(n_art):
            s = self.rows[i]
            if s["gen"] == 0: continue
            o1 = onode[i]
            o0 = onode.get(s["parent"], o1)          # root vein starts at its own offset
            vp = vid.get(s["parent"], -1)
            vp0 = s["p0"] + o0; vp1 = s["p1"] + o1
            self._add(p0=vp0, p1=vp1, r=F_VEIN * s["r"],
                      Lchord=float(np.linalg.norm(vp1 - vp0)),
                      Lpath=s["Lpath"] * float(np.linalg.norm(vp1 - vp0)) / s["Lchord"],
                      parent=vp, gen=s["gen"], tree=s["tree"], kind=1, zone=s["zone"],
                      rve=s["rve"], term=s["term"], why=s["why"], twin=i,
                      lesion=s.get("lesion", False))
        return self._pack()

    def _pack(self):
        R = self.rows
        g = lambda k, dt=float: np.array([s[k] for s in R], dtype=dt)
        return Network(p0=np.array([s["p0"] for s in R]), p1=np.array([s["p1"] for s in R]),
                       r=g("r"), Lchord=g("Lchord"), Lpath=g("Lpath"),
                       parent=g("parent", int), gen=g("gen", int), tree=g("tree", int),
                       kind=g("kind", int), zone=g("zone", int), rve=g("rve", int),
                       term=g("term", bool),
                       why=np.array([s.get("why", "") for s in R]),
                       twin=np.array([s.get("twin", -1) for s in R], int),
                       lesion=np.array([s.get("lesion", False) for s in R], bool),
                       r_dau=np.array([s.get("r_dau", (np.nan, np.nan)) for s in R], float),
                       params=self.prm, pruned=dict(self.pruned))


def build(prm: Params | None = None) -> Network:
    return _Builder(prm or Params()).build()
