"""Paired study infrastructure: provenance, sampling design and pairing.

Every study writes to a new directory holding a manifest (configuration, seed
list, calibrated bed constants, code hash, package versions, schema version,
timestamps) and one record per sampling volume.  Records are never appended to
an existing file, missing values are written as null rather than NaN, and a
completion marker is written only after every expected record has been produced.

Pairing.  An intervention is compared with a baseline built from the same seed,
the same sampling-volume centres, the same input voxels and the same refinement
order, with the capillary-bed constant calibrated once on the baseline and held
fixed in the intervention, so that a geometric change is not accompanied by a
hidden change of distal resistance.  For an intervention that changes the
branching draws the two graphs still diverge downstream of the first difference;
such an arm is a seed-paired, spatially matched regional comparison, not a
voxelwise counterfactual, and the records say so.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass

import numpy as np

from .geometry import network as N
from .physics import flow as F

SCHEMA = 5          # 3: pre-repair archive; 4: repaired records with one estimator setting;
                    # 5: tf and rule per estimate, arrival_margin_s, dV over the region


# ------------------------------------------------------------- provenance
def code_hash():
    """SHA-256 over the package sources, so a record can be traced to code."""
    h = hashlib.sha256()
    root = os.path.dirname(os.path.abspath(__file__))
    for dirpath, _, names in sorted(os.walk(root)):
        for nm in sorted(names):
            if nm.endswith(".py"):
                with open(os.path.join(dirpath, nm), "rb") as f:
                    h.update(f.read())
    return h.hexdigest()


def versions():
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


class Run:
    """One study directory: manifest, records, completion marker."""

    def __init__(self, out, kind, cfg, seeds, expected, extra=None):
        if os.path.isdir(out) and os.listdir(out):
            raise SystemExit("%s is not empty; write to a new directory" % out)
        os.makedirs(out, exist_ok=True)
        self.out = out
        self.path = os.path.join(out, "records.jsonl")
        self.n_expected = expected
        self.n_written = 0
        self.failed = 0
        self.keys = set()
        self.manifest = dict(schema=SCHEMA, kind=kind, config=cfg, seeds=list(seeds),
                             expected_records=expected, code_sha256=code_hash(),
                             versions=versions(), started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                             **(extra or {}))
        self._save_manifest()

    def _save_manifest(self):
        tmp = os.path.join(self.out, "manifest.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=1, allow_nan=False, default=_jsonable)
        os.replace(tmp, os.path.join(self.out, "manifest.json"))

    def note(self, **kw):
        self.manifest.update(kw); self._save_manifest()

    def write(self, rec):
        key = (rec.get("arm"), rec.get("seed"), rec.get("rve"), rec.get("condition"))
        if key in self.keys:
            raise SystemExit("duplicate record %s" % (key,))
        self.keys.add(key)
        if "error" in rec:
            self.failed += 1
        with open(self.path, "a") as f:
            f.write(json.dumps(_clean(rec), allow_nan=False, default=_jsonable) + "\n")
        self.n_written += 1

    def _records_digest(self):
        """sha256 of the record file, so that a later read can tell whether the
        archive still holds what this run wrote."""
        h = hashlib.sha256()
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
        return h.hexdigest()

    def finish(self):
        self.note(finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                  records=self.n_written, failed=self.failed,
                  records_sha256=self._records_digest(),
                  records_bytes=os.path.getsize(self.path)
                  if os.path.exists(self.path) else 0)
        if self.failed or self.n_written != self.n_expected:
            with open(os.path.join(self.out, "FAILED"), "w") as f:
                f.write("%d of %d records, %d failed\n" % (self.n_written, self.n_expected, self.failed))
            return False
        with open(os.path.join(self.out, "COMPLETE"), "w") as f:
            f.write(self.manifest["finished"] + "\n")
        return True


def _jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return _num(float(o))
    if isinstance(o, np.ndarray): return _clean(o.tolist())
    if isinstance(o, (set, tuple)): return list(o)
    return str(o)


def _num(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def _clean(o):
    """Replace every non-finite number by null, recursively."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return _num(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


# ---------------------------------------------------------- sampling design
def _corners(c, h):
    c = np.asarray(c, float)
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return c + h * s


def cubes_overlap(c1, c2, h):
    return bool(np.all(np.abs(np.asarray(c1) - np.asarray(c2)) < 2 * h))


_DIRS = None
def _unit_directions(n=600):
    global _DIRS
    if _DIRS is None or len(_DIRS) != n:
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
        _DIRS = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    return _DIRS


def sphere_inside_gland(centre, R, margin=0.01):
    """True when the whole sphere of radius R about `centre` lies inside the
    gland ellipsoid.  The surface of the sphere is sampled in 600 directions,
    with the radius enlarged by `margin` to cover the gaps between samples; a
    sphere that touches the capsule is rejected."""
    c = np.asarray(centre, float)
    pts = c + R * (1 + margin) * _unit_directions()
    return bool(np.all(np.sum((pts / N.GLAND_SEMI) ** 2, axis=1) < 1.0))


def support_inside_gland(centre, half, R_support):
    """Containment of every support attached to a sampling volume: the eight
    corners of its cube and the largest sphere used on it (the compression
    sphere, or the lesion sphere of a lesion study)."""
    c = np.asarray(centre, float)
    if np.any(np.sum((_corners(c, half) / N.GLAND_SEMI) ** 2, axis=1) >= 1.0):
        return False
    return R_support <= 0 or sphere_inside_gland(c, R_support)


def rve_centres(n_rve, rng, half, min_sep=0.0, R_support=0.0, tries=200000):
    """Sampling-volume centres that do not overlap, whose supports lie inside
    the gland, and whose centres are at least `min_sep` apart.

    `R_support` is the largest sphere that will be attached to a volume: the
    largest compression radius of any arm of the study, or the lesion radius,
    whichever is larger.  `min_sep` is what a lesion study needs: the lesion
    radius plus the circumscribed radius of a sampling cube plus the acquisition
    kernel, so that no part of the lesion sphere can reach the support of the
    reference volume."""
    out = []
    for _ in range(tries):
        if len(out) == n_rve:
            break
        p = rng.uniform(-1, 1, 3) * N.GLAND_SEMI
        if not support_inside_gland(p, half, R_support):
            continue
        if any(cubes_overlap(p, q, half) for q in out):
            continue
        if min_sep > 0 and any(np.linalg.norm(np.asarray(p) - np.asarray(q)) < min_sep for q in out):
            continue
        out.append(tuple(p))
    if len(out) < n_rve:
        raise RuntimeError("could not place %d sampling volumes with separation %.1f mm"
                           % (n_rve, min_sep * 1e3))
    return tuple(out)


def lesion_separation(lesion_radius, half, R_max, vox_half):
    """Center separation that keeps a lesion sphere clear of the acquisition
    support of another sampling volume: the lesion radius plus the circumscribed
    radius of the cube plus the kernel radius and half a voxel."""
    return float(lesion_radius + np.sqrt(3) * half + R_max + vox_half)


def input_voxels(rng, centre, half, n, vox_half, R_max):
    """Input voxel centres inside a sampling volume, far enough from its faces
    that the whole acquisition kernel of radius R_max fits inside."""
    m = half - R_max - vox_half
    if m <= 0:
        raise ValueError("sampling volume too small for kernel radius %g" % R_max)
    return [tuple(np.asarray(centre) + rng.uniform(-m, m, 3)) for _ in range(n)]


def segment_box_distance(p0, p1, centre, half):
    """Exact distance from each segment to an axis-aligned box (zero when the
    segment meets the box).

    Along the segment, x_a(t) = p0_a - c_a + t d_a, and the squared distance is
    sum_a max(|x_a(t)| - h, 0)^2.  Each term is piecewise linear with breaks
    where x_a(t) = +/- h, so between consecutive breaks the squared distance is
    a quadratic in t; it is minimized on every sub-interval of [0, 1], at the
    vertex when that lies inside, else at an end point."""
    p0 = np.atleast_2d(np.asarray(p0, float)); p1 = np.atleast_2d(np.asarray(p1, float))
    c = np.asarray(centre, float)
    out = np.zeros(len(p0))
    for i in range(len(p0)):
        x0 = p0[i] - c; d = p1[i] - p0[i]
        breaks = [0.0, 1.0]
        for a in range(3):
            if d[a] != 0:
                for sgn in (-1.0, 1.0):
                    tb = (sgn * half - x0[a]) / d[a]
                    if 0 < tb < 1: breaks.append(tb)
        breaks = np.unique(breaks)
        best = np.inf
        for ta, tb in zip(breaks[:-1], breaks[1:]):
            tm = 0.5 * (ta + tb)
            xm = x0 + tm * d
            # on this interval each axis term is either 0 or linear: (s (x0 + t d) - h)
            active = np.abs(xm) > half
            if not active.any():
                best = 0.0; break
            sgn = np.sign(xm[active])
            A = (sgn * d[active]); B = sgn * x0[active] - half     # g_a(t) = A t + B
            aa = np.sum(A * A); bb = np.sum(A * B); cc = np.sum(B * B)
            cand = [ta, tb]
            if aa > 0:
                tv = -bb / aa
                if ta < tv < tb: cand.append(tv)
            for tt in cand:
                best = min(best, aa * tt * tt + 2 * bb * tt + cc)
        out[i] = np.sqrt(max(best, 0.0))
    return out


def support_clearance(net, centre, half, vox_half, R_max):
    """Smallest distance from a lesion-modified segment to the acquisition
    support of a sampling volume, that is its cube grown by the kernel radius
    and half a voxel.  Zero means a modified segment reaches inside the support;
    +inf means the build has no lesion."""
    if getattr(net, "lesion", None) is None or not np.any(net.lesion):
        return float("inf")
    idx = np.where(net.lesion)[0]
    g = half + R_max + vox_half
    return float(segment_box_distance(net.p0[idx], net.p1[idx], centre, g).min())


def fingerprints(net):
    """Two hashes, so that "the same graph" is verified rather than inferred.

    topology: the connectivity, the arterial-venous pairing, the terminal flags,
    the radii, the vessel end points and the realized daughter radii, that is
    everything the flow solve depends on except path length.  It is unchanged by
    an intervention that only alters path length, such as tortuosity, and
    changes as soon as the branching draws differ.
    path: the same, plus the path length of every segment, which tortuosity
    does change."""
    h = hashlib.sha256()
    for a in (net.parent, net.kind, net.twin, net.term):
        h.update(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes())
    r_dau = np.asarray(getattr(net, "r_dau", np.zeros((net.n, 2))), float)
    for a in (net.r, net.p0, net.p1, np.nan_to_num(r_dau, nan=-1.0)):
        h.update(np.ascontiguousarray(np.round(np.asarray(a, float), 12)).tobytes())
    topo = h.hexdigest()[:16]
    h.update(np.ascontiguousarray(np.round(np.asarray(net.Lpath, float), 12)).tobytes())
    return topo, h.hexdigest()[:16]


# ------------------------------------------------------------------ pairing
@dataclass
class Arm:
    """One condition of a paired comparison."""
    name: str
    cfg: dict
    lesion: dict | None = None
    recalibrate: bool = False       # the intervention IS the bed calibration


def build_arm(arm: Arm, seed, centres):
    prm = N.Params(n_feeders=arm.cfg["n_feeders"], feed_d=arm.cfg["feed_d"], d_root=arm.cfg["d_root"],
                   alpha=arm.cfg["alpha"], gamma_med=arm.cfg["gamma_med"], tort_mean=arm.cfg["tort_mean"],
                   d_term_gland=arm.cfg["d_term_gland"], d_term_rve=arm.cfg["d_term_rve"],
                   rve_half=arm.cfg["rve_half"], seed=seed, rve_centres=centres, lesion=arm.lesion)
    return N.build(prm)


def solve_arm(net, arm: Arm, G_bed0):
    """Flow of an arm.  The bed constant is the one calibrated on the baseline,
    unless this arm's intervention is the calibration target itself."""
    if arm.recalibrate:
        return F.solve(net, R_lat=arm.cfg["R_LAT"])
    return F.solve(net, G_bed0=G_bed0, R_lat=arm.cfg["R_LAT"])
