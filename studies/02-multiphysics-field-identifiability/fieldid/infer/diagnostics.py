"""Sensitivity, rank, Fisher information and profile likelihood.

All diagnostics are computed in **prior-standardized coordinates**

    z_i = (x_i - prior_mean_i) / prior_sd_i,

with x the transformed (logarithmic for positive quantities) parameter.  In
these coordinates the prior is a standard normal, the Fisher information F is
dimensionless, and for a Gaussian likelihood the posterior precision is I + F.
An eigenvalue lambda of F therefore has a direct reading: along its eigenvector
the posterior is narrower than the prior by the factor

    contraction = 1 / sqrt(1 + lambda),

so lambda near zero means the data say nothing about that combination and large
lambda means they determine it.  Eigenvectors are reported as products of powers
of the parameters, which is what the logarithmic transform buys.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def standardize(inv):
    """(mean, sd) of the transformed coordinates under the marginal priors."""
    return inv.space.prior_mean(), inv.space.prior_sd()


def z_to_x(inv, z):
    m, s = standardize(inv)
    return m + s * np.asarray(z, float)


def x_to_z(inv, x):
    m, s = standardize(inv)
    return (np.asarray(x, float) - m) / s


def prior_precision(inv):
    """Prior precision matrix in standardized coordinates.

    Independent normal marginals give the identity.  A uniform marginal has
    variance (3.29^2 / 12) in these units, because the standardization divides by
    a 90 percent interval rather than by the standard deviation.  A correlated
    prior contributes the inverse of its correlation matrix on its own block."""
    from ..params import Z90
    d = len(inv.free)
    lam = np.eye(d)
    for i, n in enumerate(inv.free):
        if inv.space.entry(n).kind == "uniform":
            lam[i, i] = 12.0 / (2 * Z90) ** 2
    if inv.corr is not None and all(n in inv.free for n in inv.corr.names):
        idx = [inv.space.index(n) for n in inv.corr.names]
        block = np.linalg.inv(inv.corr.corr)
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                lam[ia, ib] = block[a, b]
    return lam


def jacobian(inv, z=None, step=1e-3, gen=None):
    """Central-difference Jacobian d(predict) / dz  at ``z`` (default the
    truth).  Returns (n_obs, n_par)."""
    z0 = x_to_z(inv, inv.space.x0(inv.truth)) if z is None else np.asarray(z, float)
    cols = []
    for i in range(len(z0)):
        zp, zm = z0.copy(), z0.copy()
        zp[i] += step
        zm[i] -= step
        cols.append((inv.predict(z_to_x(inv, zp), gen)
                     - inv.predict(z_to_x(inv, zm), gen)) / (2 * step))
    return np.column_stack(cols)


def jacobian_convergence(inv, z=None, steps=(4e-3, 2e-3, 1e-3, 5e-4)):
    """Relative change of the Jacobian as the difference step is halved."""
    js = [jacobian(inv, z, s) for s in steps]
    ref = js[-1]
    scale = np.linalg.norm(ref)
    return [float(np.linalg.norm(j - ref) / scale) for j in js]


def scaled_jacobian(inv, z=None, step=1e-3, gen=None):
    """Jacobian divided by the observation noise: J / sigma."""
    return jacobian(inv, z, step, gen) / inv.sigma()[:, None]


def sensitivity_table(inv, z=None, step=1e-3):
    """Per-parameter sensitivity, as the norm of the noise-scaled derivative.

    The entry for parameter i is || d y / d z_i || / sigma, that is, how many
    noise standard deviations the whole observation vector moves when the
    parameter moves by one prior standard deviation.  This is a marginal
    quantity and is reported alongside, never instead of, the eigenvector
    analysis."""
    js = scaled_jacobian(inv, z, step)
    return {n: float(np.linalg.norm(js[:, i])) for i, n in enumerate(inv.free)}


def window_sensitivity(inv, z=None, step=1e-3):
    """Sensitivity table split by observation window."""
    js = scaled_jacobian(inv, z, step)
    out = {}
    for w, sl in inv.window_slices().items():
        out[w] = {n: float(np.linalg.norm(js[sl, i]))
                  for i, n in enumerate(inv.free)}
    return out


def fisher(inv, z=None, step=1e-3, gen=None):
    """Fisher information in standardized coordinates, F = J^T Sigma^-1 J."""
    js = scaled_jacobian(inv, z, step, gen)
    return js.T @ js


def spectrum(inv, z=None, step=1e-3, gen=None):
    """Eigen-decomposition of the Fisher information, ascending.

    Returns a dict with the eigenvalues, the eigenvectors as columns, the
    per-direction contraction 1 / sqrt(1 + lambda), and a text rendering of each
    eigenvector as a product of powers of the parameters."""
    f = fisher(inv, z, step, gen)
    lam = prior_precision(inv)
    # eigen-decompose the Fisher information in the metric of the prior, so that
    # lambda is the information the data add relative to the information the
    # prior already carries along the same direction
    li = np.linalg.cholesky(np.linalg.inv(lam))
    f = li.T @ f @ li
    w, v = np.linalg.eigh(f)
    v = li @ v
    v = v / np.linalg.norm(v, axis=0, keepdims=True)
    w = np.clip(w, 0.0, None)
    order = np.argsort(w)
    w, v = w[order], v[:, order]
    _, sd = standardize(inv)
    vlog = v * sd[:, None]
    vlog = vlog / np.linalg.norm(vlog, axis=0, keepdims=True)
    return dict(eigenvalues=w, eigenvectors=v, eigenvectors_log=vlog,
                contraction=1.0 / np.sqrt(1.0 + w),
                labels=[combination_label(v[:, i], inv.free) for i in range(len(w))],
                labels_log=[combination_label(vlog[:, i], inv.free) for i in range(len(w))])


def combination_label(vec, names, tol=0.15):
    """Render an eigenvector as a product of powers, keeping components whose
    absolute weight exceeds ``tol``."""
    v = np.asarray(vec, float)
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    keep = np.where(np.abs(v) > tol)[0]
    if keep.size == 0:
        keep = [int(np.argmax(np.abs(v)))]
    return " ".join("%s^%+.2f" % (names[i], v[i]) for i in keep)


def structural_rank(inv, z=None, step=1e-3, rtol=1e-8):
    """Rank of the noise-free scaled Jacobian, and its singular spectrum.

    This is the local structural-identifiability diagnostic: a singular value
    that is zero to machine precision is an exact null direction of the map from
    parameters to ideal observations, independent of noise."""
    js = scaled_jacobian(inv, z, step)
    s = np.linalg.svd(js, compute_uv=False)
    _, _, vt = np.linalg.svd(js)
    rank = int(np.sum(s > rtol * s.max()))
    null = vt[rank:].T if rank < len(s) else np.zeros((len(inv.free), 0))
    _, sd = standardize(inv)
    nlog = null * sd[:, None]
    if nlog.size:
        nlog = nlog / np.linalg.norm(nlog, axis=0, keepdims=True)
    return dict(singular_values=s, rank=rank, n_par=len(inv.free),
                null_space=null, null_space_log=nlog,
                null_labels=[combination_label(null[:, i], inv.free)
                             for i in range(null.shape[1])],
                null_labels_log=[combination_label(nlog[:, i], inv.free)
                                 for i in range(nlog.shape[1])])


PROFILE_LEVELS = ("imaging", "imaging_plus_auxiliary", "posterior")

#: Roles whose prior enters the middle profile level.  The two are *not* the
#: same kind of information and are deliberately named apart:
#:
#:   ``calibrated``  measured independently of the imaging observations, so it
#:                   is data in its own right and would appear in a likelihood
#:                   if the auxiliary experiment were written out.
#:   ``matched``     no measurement at all.  A marginal deliberately set equal
#:                   to the one another configuration induces, so that two arms
#:                   differ only in the structure under test.  It is prior
#:                   information, and calling it calibrated would claim an
#:                   auxiliary experiment that does not exist.
#:
#: They share a profile level because the comparison requires the same
#: information to be counted at the same level in both arms, not because they
#: are the same thing.  The level is named for what it contains.
AUXILIARY_ROLES = ("calibrated", "matched")

#: Hard physical domains, in natural units, for quantities that have one.
#:
#: Positivity is already enforced by the logarithmic transform, so only
#: quantities with a further limit appear here.  ``phi`` is a volume fraction
#: and cannot exceed one; the two drift angles are angles.  Everything absent
#: from this table has no hard upper limit that the model itself imposes, and
#: is bounded by the padded catalogue range alone.
PHYSICAL_DOMAIN = {
    "phi": (None, 1.0),
    "vth": (0.0, np.pi),
    "vaz": (-np.pi, np.pi),
    "t0": (0.0, None),
}


def physical_bounds(inv, pad_decades=2.0):
    """Optimizer bounds that do not depend on how tight a prior is.

    The profile is taken in prior-standardized coordinates, so bounding the
    optimizer at plus or minus a few of those units silently bounds it at a few
    times the prior width.  For a quantity calibrated to ten percent that is a
    ten percent box, and a "likelihood only" profile computed inside it is not
    likelihood only: the calibration is doing the work through the bounds.

    These bounds come from the **catalogue** range of each quantity, widened by
    ``pad_decades``, and ignore any prior override.  They are expressed in the
    current standardized coordinates so the optimizer can use them, but their
    width in physical units is the same whatever the prior is.

    Widening is then **clipped to the physical domain** (`PHYSICAL_DOMAIN`).
    Independence from the prior is the property that matters here, and it does
    not require admitting values the model cannot mean: two decades above the
    catalogue range of the vascular volume fraction is a volume fraction of ten,
    and a profile that used one would be reporting a fit outside the model, not
    a flat likelihood.  Whether the flat directions in fact reach the bounds is
    recorded by `profile_likelihood` rather than assumed either way."""
    from ..params import CATALOGUE
    m, sd = standardize(inv)
    lo, hi = [], []
    for i, n in enumerate(inv.free):
        e = CATALOGUE[n]                       # catalogue, never the override
        a_, b_ = e.to_x(e.lo), e.to_x(e.hi)
        if e.transform == "log":
            pad = pad_decades * np.log(10.0)
            a_, b_ = a_ - pad, b_ + pad
        else:
            span = b_ - a_
            a_, b_ = a_ - 0.5 * span, b_ + 0.5 * span
        dlo, dhi = PHYSICAL_DOMAIN.get(n, (None, None))
        if dlo is not None:
            a_ = max(a_, e.to_x(dlo))
        if dhi is not None:
            b_ = min(b_, e.to_x(dhi))
        if not b_ > a_:
            raise ValueError("empty bound for %r after clipping to its "
                             "physical domain" % n)
        lo.append((a_ - m[i]) / sd[i])
        hi.append((b_ - m[i]) / sd[i])
    return np.array(lo), np.array(hi)


def prior_residual_at(inv, z, level):
    """Prior residual for a profile level.

    ``imaging``                no prior at all
    ``imaging_plus_auxiliary``  the priors of the quantities in
                               `AUXILIARY_ROLES`: those measured independently
                               of the imaging observations, and those whose
                               marginal is matched to another configuration so
                               that the two differ only in the structure under
                               test.  See `AUXILIARY_ROLES` for why the two are
                               named apart.
    ``posterior``              every prior
    """
    from .sampling import prior_residual
    if level == "imaging":
        return np.zeros(0)
    r = prior_residual(inv, z)
    if level == "posterior":
        return r
    keep = [i for i, n in enumerate(inv.free)
            if inv.space.entry(n).role in AUXILIARY_ROLES]
    return r[keep]


def profile_likelihood(inv, y, name, grid=None, n=21, span=2.5, step=1e-3,
                       level="posterior", include_prior=None):
    """Profile of -2 log of the chosen objective over one standardized
    coordinate, re-optimizing the others.

    ``level`` selects what is being profiled; see `prior_residual_at`.  The
    optimizer bounds are physical (see `physical_bounds`) and do not tighten
    when a prior does, so an "imaging" profile is not secretly a calibrated one.

    At every grid point the inner problem is solved twice, from the previous
    grid point and from the truth, and the better of the two is kept.  A single
    warm-started sweep is path dependent, and two configurations that share a
    likelihood were producing different profiles because of it.

    The minimizing point is returned with the profile, together with a flag per
    parameter for sitting on a bound and the solver's termination status.  A
    flat profile is a claim that the likelihood does not change over a range,
    and it is only that if the minimizer stayed inside the admissible region and
    the inner problems converged; both are now recorded rather than assumed
    (see `profile_audit`)."""
    from scipy.optimize import least_squares
    if include_prior is not None:               # backwards-compatible shim
        level = "posterior" if include_prior else "imaging"
    if level not in PROFILE_LEVELS:
        raise ValueError("unknown level %r" % level)
    i = inv.space.index(name)
    z0 = x_to_z(inv, inv.space.x0(inv.truth))
    g = np.linspace(z0[i] - span, z0[i] + span, n) if grid is None else np.asarray(grid)
    others = [j for j in range(len(z0)) if j != i]
    blo, bhi = physical_bounds(inv)
    # a coordinate counts as on a bound when it is within this fraction of the
    # bound interval of it; the solver stops short of the bound by design
    tol_bound = 1e-6

    def solve(gi, start):
        def res(zo):
            z = z0.copy()
            z[i] = gi
            z[others] = zo
            r = (y - inv.predict(z_to_x(inv, z))) / inv.sigma()
            return np.concatenate([r, prior_residual_at(inv, z, level)])
        st = np.clip(start, blo[others] + 1e-9, bhi[others] - 1e-9)
        r = least_squares(res, st, bounds=(blo[others], bhi[others]), method="trf",
                          xtol=1e-12, ftol=1e-12, gtol=1e-10,
                          max_nfev=50 * len(others))
        return float(2.0 * r.cost), r.x, int(r.status), int(r.nfev)

    mid = int(np.argmin(np.abs(g - z0[i])))
    d = len(z0)
    out = np.empty(len(g))
    zhat = np.empty((len(g), d))
    status = np.empty(len(g), int)
    nfev = np.empty(len(g), int)
    truth = z0[others].copy()

    def record(j, gi, cost, zo, st, nf):
        out[j] = cost
        zhat[j] = z0
        zhat[j, i] = gi
        zhat[j, others] = zo
        status[j], nfev[j] = st, nf

    c0, warm0, s0, f0 = solve(g[mid], truth)
    record(mid, g[mid], c0, warm0, s0, f0)
    for direction in (range(mid + 1, len(g)), range(mid - 1, -1, -1)):
        w = warm0.copy()
        for j in direction:
            c1, x1, s1, f1 = solve(g[j], w)
            c2, x2, s2, f2 = solve(g[j], truth)
            if c2 < c1:
                c1, x1, s1, f1 = c2, x2, s2, f2
            record(j, g[j], c1, x1, s1, f1)
            w = x1

    width = bhi - blo
    at_bound = ((zhat - blo) <= tol_bound * width) | ((bhi - zhat) <= tol_bound * width)
    at_bound[:, i] = False                      # the profiled coordinate is set, not fitted
    return dict(grid=g, nll=out, delta=out - out.min(), level=level,
                names=list(inv.free), profiled=name,
                z_hat=zhat, at_bound=at_bound, status=status, nfev=nfev,
                bounds_lo=blo, bounds_hi=bhi)


#: A tighter ceiling than the mathematical domain, used only for the
#: sensitivity check below.  No tissue has a vascular volume fraction of thirty
#: percent; the catalogue range tops out at ten.  This is not the bound the
#: profile uses, because a bound chosen for plausibility would put prior belief
#: back into a likelihood-only calculation.  It is the number the profile is
#: *tested* against.
PLAUSIBLE_CEILING = {"phi": 0.30}


def profile_domain_sensitivity(inv, y, name, ceiling=None, base=None, **kw):
    """Does a flat profile depend on the optimizer reaching the domain edge?

    A profile that is flat over the whole range is a claim that the objective
    does not distinguish the values, and it is worth nothing if the optimizer
    only kept it flat by running some other parameter to a value the model
    cannot mean.  This recomputes the profile with a stated tighter ceiling and
    returns both, so the claim rests on the difference rather than on the
    absence of a check.

    The tighter ceiling is a diagnostic, never the bound used for the reported
    profile: choosing optimizer bounds by plausibility would put prior belief
    back into a calculation whose point is to exclude it."""
    ceiling = PLAUSIBLE_CEILING if ceiling is None else ceiling
    # ``base`` may be supplied when the caller has already computed the profile
    # at the full domain; it must be the same profile, on the same grid
    base = profile_likelihood(inv, y, name, **kw) if base is None else base
    saved = {n: PHYSICAL_DOMAIN.get(n) for n in ceiling}
    try:
        for n, hi in ceiling.items():
            lo = (PHYSICAL_DOMAIN.get(n) or (None, None))[0]
            PHYSICAL_DOMAIN[n] = (lo, hi)
        tight = profile_likelihood(inv, y, name, **kw)
    finally:
        for n, v in saved.items():
            if v is None:
                PHYSICAL_DOMAIN.pop(n, None)
            else:
                PHYSICAL_DOMAIN[n] = v
    db, dt = np.asarray(base["delta"]), np.asarray(tight["delta"])
    return dict(profiled=name, level=base["level"],
                ceiling={n: float(v) for n, v in ceiling.items()},
                delta_max=float(db.max()), delta_max_tightened=float(dt.max()),
                max_abs_difference=float(np.abs(db - dt).max()),
                audit=profile_audit(inv, base),
                audit_tightened=profile_audit(inv, tight))


def profile_audit(inv, prof):
    """What the profile optimizer actually did, per parameter.

    Reports the range of physical values the minimizer visited, whether it ever
    sat on a bound, and whether any inner problem failed to converge.  This is
    the check that a flat profile is flatness of the likelihood and not the
    optimizer running out of admissible room or out of iterations."""
    z = np.asarray(prof["z_hat"], float)
    ab = np.asarray(prof["at_bound"], bool)
    names = list(prof["names"])
    vals = {}
    for j, n in enumerate(names):
        e = inv.space.entry(n)
        v = np.array([e.to_value(xi) for xi in z_to_x(inv, z)[:, j]], float)
        dlo, dhi = PHYSICAL_DOMAIN.get(n, (None, None))
        vals[n] = dict(min=float(v.min()), max=float(v.max()),
                       at_bound=int(ab[:, j].sum()),
                       outside_domain=int(((dlo is not None) and (v < dlo).sum())
                                          + ((dhi is not None) and (v > dhi).sum())))
    bad = [int(s) for s in np.asarray(prof["status"], int) if s <= 0]
    return dict(profiled=prof["profiled"], level=prof["level"], per_parameter=vals,
                n_points=int(len(prof["grid"])),
                n_points_with_a_bound=int(ab.any(axis=1).sum()),
                n_not_converged=len(bad),
                parameters_on_a_bound=sorted(
                    n for j, n in enumerate(names) if ab[:, j].any()))
