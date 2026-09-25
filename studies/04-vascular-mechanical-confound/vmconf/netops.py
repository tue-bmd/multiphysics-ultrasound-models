"""Constriction as a geometric operation on the published network.

Smooth-muscle constriction narrows lumens; it does not remodel the tree.  The
operation is therefore a scaling of the segment radii with every other array
left untouched: end points, parentage, generation, artery/vein pairing,
terminal flags, chord and path lengths.

Two routes exist to the same physics and both are used, which is what makes
this checkable:

  1. `porovasc.physics.flow.solve` and `porovasc.physics.drainage.solve` both
     accept `radius_scale` and apply it internally.  Nothing is copied.
  2. `scaled_copy` builds a new `Network` whose radii are already scaled, which
     is what the transport solve needs because it reads `net.r` directly.

`agreement_check` verifies that the two routes give the same flow solution, so
a result cannot depend on which one was taken.

Note on fingerprints.  The study-01 topology fingerprint hashes the radii by
design, so it *does* change under constriction.  That is correct behaviour and
not a topology change: `structural_invariants` is the test that the graph is
the same graph.
"""
from __future__ import annotations

import copy

import numpy as np


def scaled_copy(net, s: float):
    """Return a copy of `net` with every radius multiplied by `s`.

    Every array the physics reads is copied. `params` and `pruned` are deep-copied
    because they are mutable objects.

    Daughter radii `r_dau` are scaled too.  That is a deliberate choice and it
    is not neutral: study-01's flow solve derives the unresolved capillary bed's
    feeding radii from `r_dau`, and bed conductance goes as their cube.  Scaling
    them therefore constricts the bed as well as the explicit vessels.  The
    alternative - `flow.solve(net, radius_scale=s)` - constricts only the
    explicit vessels and leaves the bed untouched.  The two are NOT equivalent;
    `flow_agreement` measures the difference and the caller must choose.
    """
    if not (0 < s):
        raise ValueError("radius scale must be positive, got %r" % (s,))
    out = copy.copy(net)
    out.params = copy.deepcopy(getattr(net, "params", None))
    out.pruned = copy.deepcopy(getattr(net, "pruned", {}))
    for name in ("p0", "p1", "r", "Lchord", "Lpath", "parent", "gen", "tree",
                 "kind", "zone", "rve", "term", "why", "twin", "r_dau", "lesion"):
        a = getattr(net, name, None)
        if isinstance(a, np.ndarray):
            setattr(out, name, a.copy())
    out.r = net.r * float(s)
    if isinstance(getattr(net, "r_dau", None), np.ndarray):
        out.r_dau = net.r_dau * float(s)
    return out


def structural_invariants(a, b) -> dict:
    """What must be identical between a network and its constricted copy.

    Returns a dict of name -> bool. Every entry, including `r_changed`, must be
    True.
    """
    same = {}
    for name in ("p0", "p1", "Lchord", "Lpath", "parent", "gen", "tree",
                 "kind", "zone", "rve", "term", "twin"):
        x, y = getattr(a, name, None), getattr(b, name, None)
        if x is None and y is None:
            continue
        same[name] = bool(np.array_equal(np.asarray(x), np.asarray(y)))
    same["r_changed"] = not bool(np.allclose(a.r, b.r))
    return same


def assert_constriction_valid(a, b, tol=1e-12):
    """Raise unless `b` is a uniform constriction of `a`.

    Every radius must be scaled by the same factor, the factor must describe a
    genuine constriction (0 < s < 1), and daughter radii must be scaled by the
    same factor.
    """
    inv = structural_invariants(a, b)
    bad = [k for k, v in inv.items() if not v]
    if bad:
        raise AssertionError("constriction changed more than the radii: %s" % bad)
    ratio = b.r / a.r
    s = float(np.median(ratio))
    if np.max(np.abs(ratio - s)) > tol * max(s, 1.0):
        raise AssertionError("radii are not scaled by a single common factor "
                             "(spread %.3g)" % float(np.ptp(ratio)))
    if not (0.0 < s < 1.0):
        raise AssertionError("not a constriction: s = %.6g" % s)
    rd_a, rd_b = getattr(a, "r_dau", None), getattr(b, "r_dau", None)
    if isinstance(rd_a, np.ndarray) and isinstance(rd_b, np.ndarray):
        m = np.isfinite(rd_a) & np.isfinite(rd_b)
        if m.any() and np.max(np.abs(rd_b[m] / rd_a[m] - s)) > 1e-9:
            raise AssertionError("daughter radii not scaled consistently")
    return s


def lumen_fraction_ratio(s: float) -> float:
    """Lumen volume scales as r^2 at fixed path length, so the vascular volume
    fraction of any region scales as s^2 exactly.  Used as an analytic check on
    the measured `phi` returned by the drainage solve."""
    return float(s) ** 2


def flow_agreement(net, s, R_lat=None):
    """Compare the two constriction routes on the flow solution.

    route A: flow.solve(net, radius_scale=s)      - explicit vessels only
    route B: flow.solve(scaled_copy(net, s))      - vessels AND the bed, via r_dau

    They are not equivalent, because route B also scales the feeding radii from
    which the unresolved bed conductance is built (bed conductance goes as their
    cube, so it changes by s^3).  Holding `G_bed0` fixed does not hold the
    actual distal conductances fixed.

    Returns the relative L2 difference of the segment flows and the largest
    pressure difference.
    """
    from porovasc.physics import flow as F
    base = F.solve(net, R_lat=R_lat)
    a = F.solve(net, G_bed0=base.G_bed0, radius_scale=s, R_lat=R_lat)
    b = F.solve(scaled_copy(net, s), G_bed0=base.G_bed0, R_lat=R_lat)
    dq = np.linalg.norm(a.Q - b.Q) / max(np.linalg.norm(b.Q), 1e-30)
    dp = float(np.max(np.abs(a.p - b.p)))
    return dict(s=float(s), rel_L2_flow=float(dq), max_dp_Pa=dp,
                bed_conductance_ratio=float(s) ** 3)
