"""Sampling validation of the Gaussian posterior approximation.

Every contraction in `RESULTS.md` comes from a Laplace approximation: the
posterior precision is taken as ``Lambda_prior + J^T Sigma^-1 J`` at the mode.
That is exact for a linear model and it is the natural thing to use across a
large experiment matrix, but the configurations that matter most here are the
ones with a near-flat ridge, and a curvature at a point is exactly the quantity
a ridge can make misleading.  This module samples the three configurations the
main claim rests on and asks whether the approximation holds where it is used.

The three, all with the same synthetic truth and the same underlying noise
draw:

  E8_relax_independent_matched        relaxation and transport, independent
                                      parameters, storage marginal matched
  E8_relax_constitutive_Cvcal         the coupled model, ``S_v = phi C_v``,
                                      compliance calibrated
  E8_relax_constitutive_Cvcal_norm    the same, with normalized contrast curves

The first two are the comparison the reported width reduction is taken from.
The third is the control that is expected to lose it, because normalization
removes the amplitude and with it the ``phi`` that the coupling transfers.

What is reported per configuration: the sampled 90 percent interval against the
Gaussian one for every quantity, the shift in location in units of the sampled
standard deviation, Gelman-Rubin and effective sample size per parameter and
along the least-informed directions of the Fisher information, and the
acceptance rate.  What is reported across configurations: the sampled
counterpart of the width reduction the coupling is credited with.

The noise realization is shared by construction: each configuration draws its
observation from a generator seeded identically, so the standard normal
sequence behind the three is the same.  The normalized arm scales those draws by
its own noise model, which is a different observation of the same draw and is
the same convention used everywhere else in this package.
"""
from __future__ import annotations
import json
import os

import numpy as np

from . import experiments as EX
from .coupling import CorrelatedPrior
from .infer import diagnostics as DG
from .infer import sampling as SA

#: the configurations to sample, in reporting order
CONFIGS = ("E8_relax_independent_matched",
           "E8_relax_constitutive_Cvcal",
           "E8_relax_constitutive_Cvcal_norm")

#: the pair whose ratio is the reported effect of the coupling
EFFECT_PAIR = ("E8_relax_independent_matched", "E8_relax_constitutive_Cvcal")

#: quantities compared between the sampler and the approximation
NAMES = ("k", "S_v", "phi", "M", "mu", "eta_s", "D", "vmag")


def _summaries(inv, z_post, z_lap, z_prior, names):
    """Sampled and Gaussian summaries against the *same* prior draw, so that
    the two contractions differ only in the posterior."""
    s_mc = SA.summarize(inv, z_post, z_prior, names)
    s_la = SA.summarize(inv, z_lap, z_prior, names)
    out = {}
    for n in names:
        if n not in s_mc or n not in s_la:
            continue
        a, b = s_mc[n], s_la[n]
        # location, in units of the sampled spread: the interval is 3.29 sampled
        # standard deviations wide for a normal, which is how the width is
        # converted before dividing
        sd = a["width_post"] / (2 * 1.6448536269514722)
        shift = (np.log(b["median"]) - np.log(a["median"])) if n != "t0" \
            else (b["median"] - a["median"])
        out[n] = dict(
            sampled=dict(median=a["median"], lo=a["lo"], hi=a["hi"],
                         width=a["width_post"], contraction=a["contraction"],
                         covered=a["covered"]),
            gaussian=dict(median=b["median"], lo=b["lo"], hi=b["hi"],
                          width=b["width_post"], contraction=b["contraction"],
                          covered=b["covered"]),
            width_ratio_gaussian_over_sampled=float(b["width_post"] / a["width_post"])
            if a["width_post"] > 0 else float("nan"),
            location_shift_in_sampled_sd=float(shift / sd) if sd > 0 else float("nan"),
            truth=a["truth"])
    return out


def _ckpt(work, name):
    return os.path.join(work, "mcmc_%s_state.pkl" % name)


def sample_chunk(name, work, corr=None, seed=0, add_steps=4000, n_chains=4,
                 thin=5, over=2.0, blend=0.1):
    """Advance one configuration's chains by ``add_steps``, from a checkpoint.

    A long chain run as a sequence of chunks is the same chain as one run in a
    single process: the adaptive proposal depends only on the running mean and
    covariance, and both, with the random stream, are carried in the
    checkpoint.  This exists because the useful chain length here is longer than
    any one process is allowed to live, not to change the sampler."""
    import pickle
    cfg = {c.name: c for c in EX.matrix()}[name]
    inv = cfg.build(corr, seed)
    y = inv.simulate(np.random.default_rng(seed))
    os.makedirs(work, exist_ok=True)
    path = _ckpt(work, name)

    if os.path.exists(path):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        z_map, cov = blob["z_map"], blob["cov"]
        chains = [SA.AMChain(inv, y, z_map, cov, seed + 11 + i, blend=blend)
                  .set_state(s) for i, s in enumerate(blob["states"])]
        draws = [np.asarray(d, float) for d in blob["draws"]]
    else:
        z_map, _ = SA.find_map(inv, y)
        lap = SA.laplace(inv, y, z_map)
        cov = np.asarray(lap["cov"], float)
        rng = np.random.default_rng(seed + 101)
        chains, draws = [], []
        for i in range(n_chains):
            # overdispersed starts, so that Gelman-Rubin can see a posterior
            # wider than the approximation used to scale this line of work
            z0 = rng.multivariate_normal(z_map, over ** 2 * cov)
            chains.append(SA.AMChain(inv, y, z0, cov, seed + 11 + i, blend=blend))
            draws.append(np.zeros((0, len(inv.free))))
        blob = dict(z_map=z_map, cov=cov, thin=int(thin))

    for i, ch in enumerate(chains):
        draws[i] = np.vstack([draws[i], ch.advance(add_steps, thin=thin)])

    blob.update(z_map=z_map, cov=cov, thin=int(thin),
                states=[ch.get_state() for ch in chains],
                draws=[d.astype(np.float32) for d in draws],
                accept=float(np.mean([ch.accept for ch in chains])),
                n_steps=int(chains[0].n))
    with open(path, "wb") as fh:
        pickle.dump(blob, fh)
    print("%s: %d steps per chain, %d kept, accept %.3f"
          % (name, chains[0].n, draws[0].shape[0], blob["accept"]))
    return blob


def finish(name, work, out_dir, corr=None, seed=0, n_prior=20000):
    """Turn a checkpoint into the reported record, discarding the first half of
    each chain as burn-in."""
    import pickle
    with open(_ckpt(work, name), "rb") as fh:
        blob = pickle.load(fh)
    cfg = {c.name: c for c in EX.matrix()}[name]
    inv = cfg.build(corr, seed)
    y = inv.simulate(np.random.default_rng(seed))
    z_map = np.asarray(blob["z_map"], float)

    d = np.array([np.asarray(x, float) for x in blob["draws"]])  # (chains, n, d)
    n_keep = d.shape[1] // 2
    post = d[:, n_keep:, :]
    mc = dict(chains=post, samples=post.reshape(-1, post.shape[-1]),
              rhat=SA.gelman_rubin(post), ess=SA.effective_size(post),
              accept=float(blob["accept"]), burn=int(blob["n_steps"] // 2),
              n_steps=int(blob["n_steps"]))
    return _record(name, cfg, inv, y, z_map, blob["cov"], mc, blob["thin"],
                   n_prior, out_dir, seed)


def run_one(name, corr=None, seed=0, n_steps=60000, n_chains=4, thin=10,
            n_prior=20000, out_dir=None):
    """Sample one configuration and compare it with its Gaussian approximation."""
    cfg = {c.name: c for c in EX.matrix()}[name]
    inv = cfg.build(corr, seed)
    y = inv.simulate(np.random.default_rng(seed))

    z_map, _ = SA.find_map(inv, y)
    lap = SA.laplace(inv, y, z_map)
    mc = SA.preconditioned_metropolis(inv, y, n_steps=n_steps, n_chains=n_chains,
                                      seed=seed + 11, thin=thin, lap=lap,
                                      z_map=z_map)
    return _record(name, cfg, inv, y, z_map, lap["cov"], mc, thin, n_prior,
                   out_dir, seed)


def _record(name, cfg, inv, y, z_map, lap_cov, mc, thin, n_prior, out_dir, seed):
    lap = dict(mean=np.asarray(z_map, float), cov=np.asarray(lap_cov, float))
    rng = np.random.default_rng(seed + 3)
    z_prior = SA.sample_prior(inv, rng, n_prior)
    z_lap = rng.multivariate_normal(lap["mean"], lap["cov"], n_prior)

    names = [n for n in NAMES
             if n in inv.free or (n == "S_v" and cfg.level != "independent")]
    rec = dict(
        name=name, note=cfg.note, level=cfg.level, normalized=bool(cfg.normalized),
        free=list(inv.free), seed=int(seed),
        sampler=dict(n_steps=int(mc["n_steps"]), n_chains=int(mc["chains"].shape[0]),
                     burn=int(mc["burn"]), thin=int(thin),
                     n_samples=int(mc["samples"].shape[0]),
                     accept=float(mc["accept"]), preconditioned=True),
        rhat={n: float(v) for n, v in zip(inv.free, mc["rhat"])},
        ess={n: float(v) for n, v in zip(inv.free, mc["ess"])},
        directions=SA.directional_diagnostics(inv, mc["chains"], z_map),
        comparison=_summaries(inv, mc["samples"], z_lap, z_prior, names),
    )
    rec["worst_rhat"] = float(np.nanmax(mc["rhat"]))
    rec["worst_rhat_direction"] = float(np.nanmax(rec["directions"]["rhat"]))
    rec["min_ess"] = float(np.nanmin(mc["ess"]))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        phys = SA.quantities_from_z(inv, mc["samples"])
        np.savez_compressed(
            os.path.join(out_dir, "mcmc_%s_samples.npz" % name),
            z=mc["samples"].astype(np.float32),
            names=np.array(list(inv.free)),
            log_k=np.log([p["k"] for p in phys]).astype(np.float32),
            log_Sv=np.log([p["S_v"] for p in phys]).astype(np.float32),
            log_phi=np.log([p["phi"] for p in phys]).astype(np.float32),
            z_map=np.asarray(z_map, float),
            lap_cov=np.asarray(lap["cov"], float),
            truth_log_k=np.log(inv.truth_physical()["k"]),
            truth_log_Sv=np.log(inv.truth_physical()["S_v"]))
        with open(os.path.join(out_dir, "mcmc_%s.json" % name), "w") as fh:
            json.dump(rec, fh, indent=2)
    return rec


def effect_ratio(records, quantity="k"):
    """The reported effect of the coupling, as the sampler sees it.

    The narrowing credited to ``S_v = phi C_v`` is the ratio of the matched
    independent contraction to the coupled one.  It is formed here from both
    the sampled and the Gaussian posteriors so that the two can be quoted side
    by side."""
    by = {r["name"]: r for r in records}
    a, b = EFFECT_PAIR
    if a not in by or b not in by:
        return None
    ca = by[a]["comparison"].get(quantity)
    cb = by[b]["comparison"].get(quantity)
    if not ca or not cb:
        return None
    return dict(
        quantity=quantity, independent_matched=a, coupled=b,
        sampled=dict(independent=ca["sampled"]["contraction"],
                     coupled=cb["sampled"]["contraction"],
                     ratio=float(ca["sampled"]["contraction"]
                                 / cb["sampled"]["contraction"])),
        gaussian=dict(independent=ca["gaussian"]["contraction"],
                      coupled=cb["gaussian"]["contraction"],
                      ratio=float(ca["gaussian"]["contraction"]
                                  / cb["gaussian"]["contraction"])))


def run(out_dir, prior=None, seed=0, n_steps=60000, n_chains=4, thin=10,
        configs=None):
    """Sample the requested configurations and write one record per
    configuration, then collect whatever records are present.

    Each configuration is written to its own file so that the three can be run
    as separate processes; `collect` then assembles the comparison from
    whichever are on disk.  The cross-configuration numbers are only written
    when every configuration they need is present."""
    corr = CorrelatedPrior.load(prior) if prior and os.path.exists(prior) else None
    for name in (configs or CONFIGS):
        r = run_one(name, corr, seed=seed, n_steps=n_steps, n_chains=n_chains,
                    thin=thin, out_dir=out_dir)
        path = os.path.join(out_dir, "mcmc_%s.json" % name)
        with open(path, "w") as fh:
            json.dump(r, fh, indent=2)
        print("sampled %s: accept %.2f, worst R-hat %.4f (%.4f along the least "
              "informed directions), smallest effective size %.0f -> %s"
              % (name, r["sampler"]["accept"], r["worst_rhat"],
                 r["worst_rhat_direction"], r["min_ess"], path))
    return collect(out_dir)


def collect(out_dir):
    """Assemble `mcmc_validation.json` from the per-configuration records."""
    recs = []
    for name in CONFIGS:
        p = os.path.join(out_dir, "mcmc_%s.json" % name)
        if os.path.exists(p):
            with open(p) as fh:
                recs.append(json.load(fh))
    if not recs:
        raise SystemExit("no per-configuration records in %s" % out_dir)
    out = dict(configurations=recs,
               present=[r["name"] for r in recs],
               missing=[n for n in CONFIGS if n not in {r["name"] for r in recs}],
               effect=effect_ratio(recs),
               normalized_control=_norm_control(recs))
    path = os.path.join(out_dir, "mcmc_validation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path)
    return out


def _norm_control(records):
    """Whether the normalized arm loses the benefit, as the sampler sees it."""
    by = {r["name"]: r for r in records}
    a, n = EFFECT_PAIR[0], "E8_relax_constitutive_Cvcal_norm"
    if a not in by or n not in by:
        return None
    ca, cn = by[a]["comparison"].get("k"), by[n]["comparison"].get("k")
    if not ca or not cn:
        return None
    return dict(matched_independent=ca["sampled"]["contraction"],
                coupled_normalized=cn["sampled"]["contraction"],
                ratio=float(ca["sampled"]["contraction"]
                            / cn["sampled"]["contraction"]),
                note=("a ratio near one is the control behaving as expected: "
                      "with the amplitude normalized away there is no phi for "
                      "the storage relation to carry"))
