"""Posterior sampling, Laplace approximation and posterior summaries.

The Laplace approximation uses the Gauss-Newton form of the Hessian,
``Lambda_prior + J^T Sigma^-1 J``, which is exact for a linear model with
Gaussian noise and accurate near the optimum for a smooth one.  It is validated
against the sampler wherever both are run.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares, minimize

from . import diagnostics as DG


# --------------------------------------------------------------- priors -----

def sample_prior(inv, rng, n):
    """Draw ``n`` prior samples in standardized coordinates."""
    d = len(inv.free)
    z = rng.standard_normal((n, d))
    if inv.corr is not None and all(nm in inv.free for nm in inv.corr.names):
        idx = [inv.space.index(nm) for nm in inv.corr.names]
        lch = np.linalg.cholesky(inv.corr.corr)
        z[:, idx] = z[:, idx] @ lch.T
    from ..params import Z90
    for i, nm in enumerate(inv.free):
        if inv.space.entry(nm).kind == "uniform":
            z[:, i] = rng.uniform(-Z90, Z90, n)
    return z


# ---------------------------------------------------------------- MAP -------

def prior_residual(inv, z):
    """Residual vector r with 0.5 ||r||^2 equal to the negative log prior.

    Independent normal marginals contribute z itself, since the coordinates are
    prior-standardized.  A uniform marginal contributes zero inside its interval
    and the soft wall outside.  A correlated block contributes the Cholesky
    factor of its precision applied to the block."""
    from ..params import Z90, Entry
    z = np.asarray(z, float)
    out = z.copy()
    for i, n in enumerate(inv.free):
        if inv.space.entry(n).kind == "uniform":
            w = Entry.WALL * 2 * Z90
            out[i] = 0.0 if abs(z[i]) <= Z90 else (abs(z[i]) - Z90) / w * np.sign(z[i])
    if inv.corr is not None and all(n in inv.free for n in inv.corr.names):
        idx = [inv.space.index(n) for n in inv.corr.names]
        lch = np.linalg.cholesky(np.linalg.inv(inv.corr.corr))
        out[idx] = lch.T @ z[idx]
    return out


def _residuals(inv, z, y):
    r = (y - inv.predict(DG.z_to_x(inv, z))) / inv.sigma()
    return np.concatenate([r, prior_residual(inv, z)])


def find_map(inv, y, z_start=None, restarts=2, rng=None):
    """Maximum a posteriori point, by Levenberg-Marquardt on the residual
    vector.

    The posterior is a sum of squares by construction, so a least-squares
    solver converges in tens of iterations where a general quasi-Newton method
    needs hundreds; with a numerical Jacobian that is the difference between
    seconds and minutes for the larger configurations."""
    z0 = DG.x_to_z(inv, inv.space.x0(inv.truth)) if z_start is None else np.asarray(z_start)
    rng = rng or np.random.default_rng(0)
    best = None
    for j in range(restarts):
        s0 = z0 if j == 0 else z0 + rng.normal(0, 0.3, len(z0))
        try:
            r = least_squares(lambda z: _residuals(inv, z, y), s0,
                              bounds=(-6.0, 6.0), method="trf",
                              xtol=1e-11, ftol=1e-11, gtol=1e-9,
                              max_nfev=60 * len(s0))
            cand = (r.x, float(0.5 * np.dot(r.fun, r.fun)))
        except Exception:
            continue
        if best is None or cand[1] < best[1]:
            best = cand
    if best is None:
        r = minimize(lambda z: -inv.log_post(DG.z_to_x(inv, z), y), z0,
                     method="Nelder-Mead", options=dict(maxiter=20000))
        best = (r.x, float(r.fun))
    return best[0], best[1]


def laplace(inv, y, z_map=None, step=1e-3):
    """Gaussian approximation of the posterior in standardized coordinates."""
    if z_map is None:
        z_map, _ = find_map(inv, y)
    f = DG.fisher(inv, z_map, step)
    lam = DG.prior_precision(inv)
    prec = lam + f
    cov = np.linalg.inv(prec)
    return dict(mean=np.asarray(z_map, float), cov=cov, precision=prec,
                fisher=f, prior_precision=lam)


# ------------------------------------------------------- adaptive Metropolis -

def adaptive_metropolis(inv, y, n_steps=40000, n_chains=4, seed=0, burn=None,
                        z_start=None, cov0=None, thin=5, adapt_after=200,
                        spread=0.25, spread_cov=None, blend=0.0):
    """Haario adaptive Metropolis in standardized coordinates.

    Returns the pooled post-burn samples plus per-parameter Gelman-Rubin and
    effective sample size.

    ``cov0`` is the initial proposal covariance and ``spread_cov`` the
    covariance used to disperse the chain starts; both are proposal choices and
    cannot change the stationary distribution, only how quickly it is reached.
    ``blend`` keeps a fraction of ``cov0`` in the adapted proposal, which
    stabilizes the empirical covariance in high dimension."""
    rng = np.random.default_rng(seed)
    d = len(inv.free)
    burn = n_steps // 2 if burn is None else burn
    if cov0 is None:
        cov0 = np.eye(d) * 0.05
    cov0 = np.asarray(cov0, float)
    sd = 2.38 ** 2 / d
    eps = 1e-8

    z0 = DG.x_to_z(inv, inv.space.x0(inv.truth)) if z_start is None else np.asarray(z_start)
    chains, accepts = [], []
    for c in range(n_chains):
        if spread_cov is None:
            z = z0 + rng.normal(0, spread, d)
        else:
            z = rng.multivariate_normal(z0, np.asarray(spread_cov, float))
        lp = inv.log_post(DG.z_to_x(inv, z), y)
        keep = np.empty((n_steps, d))
        cov = np.array(cov0, float)
        mean = z.copy()
        n_acc = 0
        for i in range(n_steps):
            prop_cov = (1.0 - blend) * cov + blend * cov0
            prop = z + rng.multivariate_normal(np.zeros(d),
                                               sd * prop_cov + eps * np.eye(d))
            lq = inv.log_post(DG.z_to_x(inv, prop), y)
            if np.log(rng.random()) < lq - lp:
                z, lp = prop, lq
                n_acc += 1
            keep[i] = z
            # running mean and covariance; adaptation starts after adapt_after
            delta = z - mean
            mean += delta / (i + 2)
            cov = (cov * (i + 1) + np.outer(delta, z - mean)) / (i + 2)
            if i < adapt_after:
                cov = np.array(cov0, float)
        chains.append(keep)
        accepts.append(n_acc / n_steps)
    chains = np.array(chains)
    post = chains[:, burn::thin, :]
    return dict(samples=post.reshape(-1, d), chains=post,
                rhat=gelman_rubin(post), ess=effective_size(post),
                accept=float(np.mean(accepts)), n_steps=int(n_steps),
                burn=int(burn), thin=int(thin), n_chains=int(n_chains))


def preconditioned_metropolis(inv, y, n_steps=40000, n_chains=4, seed=0,
                              thin=5, lap=None, z_map=None, over=2.0,
                              blend=0.1, **kw):
    """Adaptive Metropolis started at the mode and scaled by the Laplace
    covariance.

    In sixteen dimensions with posterior standard deviations spanning three
    orders of magnitude, an isotropic proposal moves along the well-determined
    directions and almost never along the confounded ones, so a chain of any
    affordable length reports the prior width for the flat directions and calls
    it a posterior.  Scaling this line of work by the Gaussian approximation removes
    that artefact.

    The approximation being tested therefore appears in the **proposal** and in
    the dispersion of the chain starts.  Neither enters the accept ratio, so
    neither can change what the chain converges to; if the true posterior is not
    the Gaussian, the chain still finds the difference, it just starts looking
    in a sensible place.  Chain starts are drawn with ``over`` times the Laplace
    standard deviation so that they are overdispersed relative to the target
    when the approximation is right, which is what makes Gelman-Rubin able to
    detect it when it is wrong."""
    if z_map is None:
        z_map, _ = find_map(inv, y)
    if lap is None:
        lap = laplace(inv, y, z_map)
    cov = np.asarray(lap["cov"], float)
    out = adaptive_metropolis(inv, y, n_steps=n_steps, n_chains=n_chains,
                              seed=seed, thin=thin, z_start=z_map, cov0=cov,
                              spread_cov=over ** 2 * cov, blend=blend, **kw)
    out["z_map"] = np.asarray(z_map, float)
    out["preconditioned"] = True
    return out


class AMChain:
    """One adaptive Metropolis chain that can be advanced in pieces.

    The same recursion as `adaptive_metropolis`, exposed so that a long chain
    can be run as a sequence of chunks with its state written to disk between
    them.  A chain advanced in ten pieces is the same chain as one advanced in
    one piece: this line of work depends only on the running mean and covariance,
    both of which are carried in the state, and the random stream is carried
    with them."""

    def __init__(self, inv, y, z0, cov0, seed, blend=0.1, adapt_after=200):
        self.inv, self.y = inv, y
        self.d = len(z0)
        self.cov0 = np.array(cov0, float)
        self.blend = float(blend)
        self.adapt_after = int(adapt_after)
        self.sd = 2.38 ** 2 / self.d
        self.eps = 1e-8
        self.rng = np.random.default_rng(seed)
        self.z = np.array(z0, float)
        self.lp = inv.log_post(DG.z_to_x(inv, self.z), y)
        self.mean = self.z.copy()
        self.cov = np.array(cov0, float)
        self.n = 0
        self.n_acc = 0

    def advance(self, n_steps, thin=1):
        keep = []
        for _ in range(n_steps):
            prop_cov = (1.0 - self.blend) * self.cov + self.blend * self.cov0
            step = self.rng.multivariate_normal(
                np.zeros(self.d), self.sd * prop_cov + self.eps * np.eye(self.d))
            prop = self.z + step
            lq = self.inv.log_post(DG.z_to_x(self.inv, prop), self.y)
            if np.log(self.rng.random()) < lq - self.lp:
                self.z, self.lp = prop, lq
                self.n_acc += 1
            delta = self.z - self.mean
            self.mean += delta / (self.n + 2)
            self.cov = (self.cov * (self.n + 1)
                        + np.outer(delta, self.z - self.mean)) / (self.n + 2)
            if self.n < self.adapt_after:
                self.cov = np.array(self.cov0, float)
            self.n += 1
            if self.n % thin == 0:
                keep.append(self.z.copy())
        return np.array(keep) if keep else np.zeros((0, self.d))

    # ---- state, for checkpointing ---------------------------------------
    def get_state(self):
        return dict(z=self.z, lp=float(self.lp), mean=self.mean, cov=self.cov,
                    n=int(self.n), n_acc=int(self.n_acc),
                    rng=self.rng.bit_generator.state)

    def set_state(self, s):
        self.z = np.array(s["z"], float)
        self.lp = float(s["lp"])
        self.mean = np.array(s["mean"], float)
        self.cov = np.array(s["cov"], float)
        self.n = int(s["n"])
        self.n_acc = int(s["n_acc"])
        self.rng.bit_generator.state = s["rng"]
        return self

    @property
    def accept(self):
        return self.n_acc / max(self.n, 1)


def directional_diagnostics(inv, chains, z_map, n_dir=4, step=1e-3):
    """Mixing along the least-informed directions of the Fisher information.

    Per-parameter Gelman-Rubin can look healthy while the chain crawls along a
    confounded combination, because that combination is spread over many
    parameters and each of them is also moved by better-determined directions.
    The chains are projected onto the eigenvectors of the Fisher information,
    smallest eigenvalue first, and the same diagnostics are computed there."""
    sp = DG.spectrum(inv, z_map, step)
    v = np.asarray(sp["eigenvectors"], float)[:, :n_dir]
    proj = np.einsum("cnd,dk->cnk", np.asarray(chains, float), v)
    return dict(eigenvalues=np.asarray(sp["eigenvalues"], float)[:n_dir].tolist(),
                labels=[sp["labels_log"][i] for i in range(v.shape[1])],
                rhat=gelman_rubin(proj).tolist(),
                ess=effective_size(proj).tolist())


def gelman_rubin(chains):
    m, n, d = chains.shape
    if m < 2 or n < 4:
        return np.full(d, np.nan)
    cm = chains.mean(axis=1)
    b = n * cm.var(axis=0, ddof=1)
    w = chains.var(axis=1, ddof=1).mean(axis=0)
    w = np.where(w > 0, w, np.nan)
    var = (n - 1) / n * w + b / n
    return np.sqrt(var / w)


def effective_size(chains):
    m, n, d = chains.shape
    out = np.empty(d)
    for j in range(d):
        x = chains[:, :, j]
        v = x.var(axis=1, ddof=1).mean()
        if not np.isfinite(v) or v <= 0:
            out[j] = np.nan
            continue
        rho_sum, t = 0.0, 1
        while t < n // 2:
            r = np.mean([np.corrcoef(c[:-t], c[t:])[0, 1] for c in x])
            if not np.isfinite(r) or r < 0.05:
                break
            rho_sum += r
            t += 1
        out[j] = m * n / (1 + 2 * rho_sum)
    return out


# ------------------------------------------------------------- summaries ----

def quantities_from_z(inv, z):
    """Physical values of every catalogue quantity for each standardized draw."""
    z = np.atleast_2d(z)
    rows = [inv.physical(DG.z_to_x(inv, zi)) for zi in z]
    return rows


def summarize(inv, z_samples, z_prior, names, truth=None):
    """Posterior median, 90 percent interval, contraction against the prior, and
    bias against the synthetic truth, for each requested physical quantity."""
    truth = truth or inv.truth_physical()
    post = quantities_from_z(inv, z_samples)
    pri = quantities_from_z(inv, z_prior)
    out = {}
    for n in names:
        a = np.array([r[n] for r in post], float)
        b = np.array([r[n] for r in pri], float)
        lo, med, hi = np.percentile(a, [5, 50, 95])
        plo, phi_ = np.percentile(b, [5, 95])
        wpost = np.log(hi) - np.log(lo) if n != "t0" else hi - lo
        wpri = np.log(phi_) - np.log(plo) if n != "t0" else phi_ - plo
        out[n] = dict(median=float(med), lo=float(lo), hi=float(hi),
                      prior_lo=float(plo), prior_hi=float(phi_),
                      width_post=float(wpost), width_prior=float(wpri),
                      contraction=float(wpost / wpri) if wpri > 0 else np.nan,
                      truth=float(truth[n]),
                      bias_log=float(np.log(med / truth[n])) if truth[n] > 0
                      else float(med - truth[n]),
                      covered=bool(lo <= truth[n] <= hi))
    return out


def laplace_summary(inv, lap, names, n=20000, seed=1, truth=None):
    """Summaries from the Gaussian approximation, by drawing from it."""
    rng = np.random.default_rng(seed)
    d = len(inv.free)
    zs = rng.multivariate_normal(lap["mean"], lap["cov"], n)
    zp = sample_prior(inv, rng, n)
    return summarize(inv, zs, zp, names, truth)
