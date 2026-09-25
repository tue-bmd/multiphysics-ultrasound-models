"""CEUS observables computed with the study-01 bubble-transport model.

The primary observable is obtained from propagated time-intensity curves on
the same network used by the mechanical model. It therefore has its own
acquisition dependence and is not inferred from lumen-volume scaling alone.

Observable choice
-----------------
The transport observable is the area under the CEUS time-intensity curve
(AUC) in a voxel, taken directly from the propagation.  No
contrast-kinetic estimator is involved: no transfer function, no shell fit, no
choice of causality rule.  The AUC is monotone over the evaluated radius-scale
range.

The AUC is a *calibrated-amplitude* observable. It is informative about radius scale only
when the input concentration and the acoustic sensitivity are calibrated, which
is the condition established for vascular fraction in study 02.

The shell estimator can also be run with `estimator=True`; its velocity and
dispersion outputs are diagnostic only and are not used in inference.

Cost.  One evaluation is a flow solve plus a propagation over the shell
kernels; it is seconds, not minutes, because only the input voxel and its shell
are propagated, not a filled block.  The (v, D) response to constriction is
therefore tabulated over a grid of s once and interpolated afterwards, which is
what makes inference affordable.
"""
from __future__ import annotations

import numpy as np


def observables(net, fl, centre, dz=1.5e-3, vox_half=0.375e-3, n_dir=48,
                poiseuille=False, tf="old", rule="front20",
                n_inputs=1, jitter=0.9e-3, seed=0, estimator=False):
    """Simulated CEUS descriptors at one location.

    Returned descriptors:

    v    shell-estimated transport velocity.  Responds strongly to
         constriction: Poiseuille flow gives v ~ r^2, so v ~ s^2.
    auc  area under the input-voxel time-intensity curve.
         Proportional to the lumen volume in the kernel, so also ~ s^2.  This
         is the calibrated-amplitude case: it is informative only when the
         input concentration and the acoustic sensitivity are calibrated.

    D    shell-estimated dispersion, recorded for diagnostics but not used in
         the reported inference.
    """
    from porovasc.physics import transport as TR
    # Use the same set of nearby input-voxel locations at every radius scale
    # and summarize their AUC values by the median.
    rng = np.random.default_rng(seed)
    c0 = np.asarray(centre, float)
    cs = [c0] + [c0 + rng.uniform(-jitter, jitter, 3) for _ in range(n_inputs - 1)]
    _area = (lambda y, t: float(np.trapezoid(y, t))) if hasattr(np, "trapezoid") \
        else (lambda y, t: float(np.trapz(y, t)))

    if not estimator:
        kernels = [(c, vox_half) for c in cs]
        tr = TR.propagate(net, fl, kernels, poiseuille=poiseuille)
        aucs = [_area(np.asarray(tr.tic[k]), tr.t) for k in range(len(cs))]
        vs, Ds, r2s, nus = [np.nan], [np.nan], [np.nan], [0]
    else:
        vs, Ds, aucs, r2s, nus = [], [], [], [], []
        for c in cs:
            kernels = TR.shell_kernels(c, dz, vox_half, n_dir=n_dir)
            tr = TR.propagate(net, fl, kernels, poiseuille=poiseuille)
            tic_in = np.asarray(tr.tic[0])
            tic_outs = [np.asarray(tr.tic[k]) for k in range(1, len(kernels))]
            vi, Di, ni, ri = TR.identify_shell(tr.t, tic_in, tic_outs, dz, tf=tf, rule=rule)
            vs.append(vi); Ds.append(Di); r2s.append(ri); nus.append(ni)
            aucs.append(_area(tic_in, tr.t))
    v, D = float(np.median(vs)), float(np.median(Ds))
    n_used, r2 = int(np.median(nus)), float(np.median(r2s))
    auc_all = np.array(aucs)
    D_railed = bool(np.isfinite(D) and D >= 0.99e-4)
    return dict(auc=float(np.median(auc_all)),
                estimator_used=bool(estimator),
                v_NOT_USED=float(v), D_NOT_USED=float(D), v=float(v),
                v_iqr=float(np.subtract(*np.percentile(vs, [75, 25]))) if len(vs) > 2 else 0.0,
                auc_iqr=float(np.subtract(*np.percentile(auc_all, [75, 25])))
                if len(auc_all) > 2 else 0.0,
                n_inputs=len(cs), D=float(D), D_railed=D_railed,
                n_used=int(n_used), r2=float(r2))


def response_table(net, s_values, centre, G_bed0=None, **kw):
    """CEUS AUC against radius scale on one network.

    Radius scaling is applied through `flow.solve(radius_scale=s)` to the
    explicit vessels while the unresolved capillary-bed conductance remains at
    its baseline value.
    """
    from porovasc.physics import flow as F
    from porovasc.physics import transport as TR

    base = F.solve(net, R_lat=None) if G_bed0 is None else None
    G0 = base.G_bed0 if base is not None else G_bed0
    rows = []
    original_t_end = TR.T_END
    t_end_used = original_t_end
    try:
        for s in s_values:
            fl = F.solve(net, G_bed0=G0, radius_scale=float(s))
            # Constriction slows transport, so the record must be long enough
            # for the slowest route; otherwise the transform wraps. The bound
            # is recomputed per radius scale and the record is extended when needed.
            bound = TR.travel_time_bound(net, fl, poiseuille=kw.get("poiseuille", False))
            if bound > 0.9 * TR.T_END:
                TR.T_END = float(2 ** np.ceil(np.log2(bound / 0.9 / 0.02)) * 0.02)
                t_end_used = max(t_end_used, TR.T_END)
            o = observables(net, fl, centre, **kw)
            o["s"] = float(s)
            o["travel_time_bound_s"] = float(bound)
            o["t_end_s"] = float(TR.T_END)
            rows.append(o)
    finally:
        TR.T_END = original_t_end
    return dict(G_bed0=float(G0), t_end_max_s=float(t_end_used), rows=rows)


class Interpolant:
    """CEUS AUC as a smooth function of s from a tabulated response.

    Returns the CEUS AUC only.  Log-linear, because it is positive and
    varies over a wide range.  The shell-estimated velocity is not used; see the
    module docstring.
    Outside the tabulated range the value is held at the end point and a flag
    is raised, so an inference that wanders outside the simulated domain cannot
    silently extrapolate.
    """

    def __init__(self, table):
        rows = sorted(table["rows"], key=lambda r: r["s"])
        self.s = np.array([r["s"] for r in rows])
        auc = np.array([r["auc"] for r in rows])
        if len(self.s) < 2 or np.any(~np.isfinite(self.s)) or np.any(np.diff(self.s) <= 0):
            raise ValueError("CEUS response table needs at least two distinct, finite s values")
        if np.any(~np.isfinite(auc)) or np.any(auc <= 0):
            raise ValueError("CEUS response table requires positive, finite AUC values")
        self.logA = np.log(auc)

    def in_range(self, s):
        return (np.asarray(s) >= self.s[0]) & (np.asarray(s) <= self.s[-1])

    def __call__(self, s):
        ss = np.clip(np.asarray(s, float), self.s[0], self.s[-1])
        return np.exp(np.interp(ss, self.s, self.logA))
