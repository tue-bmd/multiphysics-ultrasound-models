"""Sampling validation along the confounded direction.

Every interval in the main table comes from a Laplace approximation, a curvature
at the mode.  The whole study is about a pair that is exactly degenerate in one
window, `sig_e` and its point-spread width, so the posterior in that plane is a
ridge and a curvature is exactly the quantity a ridge can misrepresent.  The
main case is therefore sampled directly in the independent and shared arms.

The sampler is adaptive Metropolis in the prior-standardized coordinates,
started at the mode with this line of work covariance taken from the Laplace
approximation and the chain starts overdispersed relative to it.  The
approximation under test enters this line of work and not the accept ratio, so it
cannot change the stationary distribution, only how fast the chain reaches it.

Convergence is reported per parameter and along the least informed eigenvectors
of the Fisher information, because a confounded combination is spread over
several parameters and can crawl while each of them looks healthy.
"""
from __future__ import annotations
import json
import os

import numpy as np

from . import experiments as EX
from .constants import psf_widths
from .infer import diagnostics as DG
from .infer import sampling as SA
from .params import REPORT

NAMES = ("sig_e", "R", "c_s", "alpha0", "v", "D")


def _checkpoint_config(case, arm, seed, n_chains, thin):
    """What a checkpoint must match to be resumable.  Anything here that
    changes makes an old checkpoint a different study."""
    return dict(case=case.to_dict(), arm=arm, seed=int(seed),
                n_chains=int(n_chains), thin=int(thin))


def sample_chunk(case, arm, work, seed=0, add_steps=20000, n_chains=4, thin=5,
                 over=2.0, blend=0.1, allow_code_change=False):
    """Advance one arm's chains by ``add_steps`` from a checkpoint.

    A chain advanced in pieces is the same chain: the adaptive proposal depends
    only on the running mean and covariance, and those, with the random stream,
    are carried in the checkpoint.  The checkpoint also carries the source and
    configuration fingerprints and refuses to resume when either has moved, so
    a chain cannot silently span two versions of the model."""
    from .provenance import load_checkpoint, save_checkpoint
    y, sigma = EX.generate(case, seed)
    inv = case.build(arm, sigma=sigma)
    cfg = _checkpoint_config(case, arm, seed, n_chains, thin)
    path = os.path.join(work, "chain_%s_%s.pkl" % (case.name, arm))
    blob = load_checkpoint(path, cfg, allow_code_change=allow_code_change)

    if blob is None:
        z_map, _ = SA.find_map(inv, y)
        lap = SA.laplace(inv, y, z_map)
        cov = np.asarray(lap["cov"], float)
        rng = np.random.default_rng(seed + 101)
        chains, draws = [], []
        for i in range(n_chains):
            z0 = rng.multivariate_normal(z_map, over ** 2 * cov)
            chains.append(SA.AMChain(inv, y, z0, cov, seed + 11 + i, blend=blend))
            draws.append(np.zeros((0, len(inv.free))))
        blob = dict(z_map=z_map, cov=cov)
    else:
        z_map, cov = blob["z_map"], blob["cov"]
        chains = [SA.AMChain(inv, y, z_map, cov, seed + 11 + i, blend=blend)
                  .set_state(s) for i, s in enumerate(blob["states"])]
        draws = [np.asarray(d, float) for d in blob["draws"]]

    for i, ch in enumerate(chains):
        draws[i] = np.vstack([draws[i], ch.advance(add_steps, thin=thin)])
    blob.update(states=[ch.get_state() for ch in chains],
                draws=[d.astype(np.float32) for d in draws],
                accept=float(np.mean([ch.accept for ch in chains])),
                n_steps=int(chains[0].n))
    save_checkpoint(path, blob, cfg)
    print("%s %s: %d steps per chain, %d kept, accept %.3f"
          % (case.name, arm, chains[0].n, draws[0].shape[0], blob["accept"]))
    return blob


def finish(case, arm, work, out_dir, seed=0, n_chains=4, thin=5, n_draw=20000,
           allow_code_change=False):
    """Turn a checkpoint into the reported record, discarding the first half of
    each chain as burn-in."""
    from .provenance import load_checkpoint
    cfg = _checkpoint_config(case, arm, seed, n_chains, thin)
    path = os.path.join(work, "chain_%s_%s.pkl" % (case.name, arm))
    blob = load_checkpoint(path, cfg, allow_code_change=allow_code_change)
    if blob is None:
        raise SystemExit("no checkpoint at %s" % path)
    d = np.array([np.asarray(x, float) for x in blob["draws"]])
    keep = d.shape[1] // 2
    post = d[:, keep:, :]
    mc = dict(chains=post, samples=post.reshape(-1, post.shape[-1]),
              rhat=SA.gelman_rubin(post), ess=SA.effective_size(post),
              accept=float(blob["accept"]), burn=int(blob["n_steps"] // 2),
              n_steps=int(blob["n_steps"]))
    return _record(case, arm, seed, np.asarray(blob["z_map"], float),
                   np.asarray(blob["cov"], float), mc, thin, n_draw, out_dir)


def run_one(case, arm, seed=0, n_steps=20000, n_chains=4, thin=5, n_draw=20000):
    y, sigma = EX.generate(case, seed)
    inv = case.build(arm, sigma=sigma)
    z_map, _ = SA.find_map(inv, y)
    lap = SA.laplace(inv, y, z_map)
    mc = SA.preconditioned_metropolis(inv, y, n_steps=n_steps, n_chains=n_chains,
                                      seed=seed + 11, thin=thin, lap=lap,
                                      z_map=z_map)
    return _record(case, arm, seed, z_map, np.asarray(lap["cov"], float), mc,
                   thin, n_draw, None)[0], inv, mc


def _record(case, arm, seed, z_map, lap_cov, mc, thin, n_draw, out_dir):
    y, sigma = EX.generate(case, seed)
    inv = case.build(arm, sigma=sigma)
    lap = dict(mean=np.asarray(z_map, float), cov=np.asarray(lap_cov, float))
    rng = np.random.default_rng(seed + 3)
    zp = SA.sample_prior(inv, rng, n_draw)
    zl = rng.multivariate_normal(lap["mean"], lap["cov"], n_draw)
    names = [n for n in NAMES if n in inv.free]

    s_mc = SA.summarize(inv, mc["samples"], zp, names)
    s_la = SA.summarize(inv, zl, zp, names)
    # the contraction denominator is the **exact** prior 90 percent width, the
    # same number in every arm.  Estimating it separately by Monte Carlo in each
    # arm moves the ratio between arms by a fraction of a percent for no reason
    exact = {n: float(2 * 1.6448536269514722 * inv.space.entry(n).sd)
             for n in names if inv.space.entry(n).transform == "log"}
    cmp_ = {}
    for n in names:
        a, b = s_mc[n], s_la[n]
        if n in exact:
            a = dict(a, width_prior=exact[n],
                     contraction=a["width_post"] / exact[n])
            b = dict(b, width_prior=exact[n],
                     contraction=b["width_post"] / exact[n])
        sd = a["width_post"] / (2 * 1.6448536269514722)
        cmp_[n] = dict(
            sampled=dict(median=a["median"], lo=a["lo"], hi=a["hi"],
                         width=a["width_post"], contraction=a["contraction"],
                         covered=a["covered"]),
            gaussian=dict(median=b["median"], lo=b["lo"], hi=b["hi"],
                          width=b["width_post"], contraction=b["contraction"],
                          covered=b["covered"]),
            width_ratio_gaussian_over_sampled=float(b["width_post"] / a["width_post"]),
            location_shift_in_sampled_sd=float(
                (np.log(b["median"]) - np.log(a["median"])) / sd) if sd > 0 else float("nan"),
            truth=a["truth"])

    rec = dict(case=case.name, arm=arm, seed=int(seed), free=list(inv.free),
               sampler=dict(n_steps=int(mc["n_steps"]),
                            n_chains=int(mc["chains"].shape[0]),
                            burn=int(mc["burn"]), thin=int(thin),
                            n_samples=int(mc["samples"].shape[0]),
                            accept=float(mc["accept"]), preconditioned=True),
               rhat={n: float(v) for n, v in zip(inv.free, mc["rhat"])},
               ess={n: float(v) for n, v in zip(inv.free, mc["ess"])},
               rhat_rank={n: float(v) for n, v in
                          zip(inv.free, SA.rhat_rank(mc["chains"]))},
               ess_bulk={n: float(v) for n, v in
                         zip(inv.free, SA.ess_bulk(mc["chains"]))},
               ess_tail={n: float(v) for n, v in
                         zip(inv.free, SA.ess_tail(mc["chains"]))},
               per_chain_width=_per_chain_width(inv, mc["chains"], names),
               directions=SA.directional_diagnostics(inv, mc["chains"], z_map),
               comparison=cmp_)
    rec["worst_rhat"] = float(np.nanmax(mc["rhat"]))
    rec["worst_rhat_rank"] = float(np.nanmax(list(rec["rhat_rank"].values())))
    rec["min_ess_bulk"] = float(np.nanmin(list(rec["ess_bulk"].values())))
    rec["min_ess_tail"] = float(np.nanmin(list(rec["ess_tail"].values())))
    rec["worst_rhat_direction"] = float(np.nanmax(rec["directions"]["rhat"]))
    rec["min_ess"] = float(np.nanmin(mc["ess"]))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        w = np.array([inv.quantity(DG.z_to_x(inv, z), "w_bmode")
                      for z in mc["samples"]])
        se = np.array([inv.quantity(DG.z_to_x(inv, z), "sig_e")
                       for z in mc["samples"]])
        np.savez_compressed(
            os.path.join(out_dir, "mcmc_%s_samples.npz" % arm),
            log_sig_e=np.log(se).astype(np.float32),
            log_w_bmode=np.log(w).astype(np.float32),
            truth_log_sig_e=np.log(inv.truth["sig_e"]),
            truth_log_w_bmode=np.log(float(psf_widths(inv.truth["w0"],
                                             inv.gamma)["bmode"])))
        with open(os.path.join(out_dir, "mcmc_%s.json" % arm), "w") as fh:
            json.dump(rec, fh, indent=2)
    return rec, inv, mc


def run(out_dir, seed=0, n_steps=20000, n_chains=4, thin=5, arms=("independent", "shared")):
    case = {c.name: c for c in EX.matrix()}["main"]
    os.makedirs(out_dir, exist_ok=True)
    for arm in arms:
        rec = _finish_print(run_one_to_disk(case, arm, out_dir, seed, n_steps,
                                            n_chains, thin))
    return collect(out_dir)


def run_one_to_disk(case, arm, out_dir, seed, n_steps, n_chains, thin):
    y, sigma = EX.generate(case, seed)
    inv = case.build(arm, sigma=sigma)
    z_map, _ = SA.find_map(inv, y)
    lap = SA.laplace(inv, y, z_map)
    mc = SA.preconditioned_metropolis(inv, y, n_steps=n_steps, n_chains=n_chains,
                                      seed=seed + 11, thin=thin, lap=lap,
                                      z_map=z_map)
    return _record(case, arm, seed, z_map, np.asarray(lap["cov"], float), mc,
                   thin, 20000, out_dir)[0]


def _finish_print(rec):
    print("sampled %s: %d steps per chain, accept %.2f, worst R-hat %.4f "
          "(%.4f along the least informed directions), smallest effective "
          "size %.0f" % (rec["arm"], rec["sampler"]["n_steps"],
                         rec["sampler"]["accept"], rec["worst_rhat"],
                         rec["worst_rhat_direction"], rec["min_ess"]))
    return rec


def collect(out_dir):
    """Assemble the comparison from whichever arm records are on disk, so the
    two arms can be sampled in separate processes."""
    recs = []
    for arm in ("independent", "shared"):
        p = os.path.join(out_dir, "mcmc_%s.json" % arm)
        if os.path.exists(p):
            with open(p) as fh:
                recs.append(json.load(fh))
    if not recs:
        raise SystemExit("no arm records in %s" % out_dir)
    out = dict(configurations=recs,
               present=[r["arm"] for r in recs],
               effect=_effect(recs))
    path = os.path.join(out_dir, "mcmc_validation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path)
    return out


def _per_chain_width(inv, chains, names):
    """The 90 percent log-width each chain would report on its own.

    A headline quoted to two significant figures is only meaningful if the
    chains agree on it, so the spread between them is reported rather than
    hidden behind a pooled number."""
    out = {}
    for n in names:
        if inv.space.entry(n).transform != "log":
            continue
        w = []
        for c in np.asarray(chains, float):
            v = np.array([inv.quantity(DG.z_to_x(inv, z), n) for z in c])
            lo, hi = np.percentile(v, [5, 95])
            w.append(float(np.log(hi) - np.log(lo)))
        out[n] = dict(per_chain=w, min=float(min(w)), max=float(max(w)),
                      spread_factor=float(max(w) / min(w)))
    return out


def _effect(recs, quantity="sig_e"):
    by = {r["arm"]: r for r in recs}
    if "independent" not in by or "shared" not in by:
        return None
    a = by["independent"]["comparison"].get(quantity)
    b = by["shared"]["comparison"].get(quantity)
    if not a or not b:
        return None
    return dict(
        quantity=quantity,
        sampled=dict(independent=a["sampled"]["contraction"],
                     shared=b["sampled"]["contraction"],
                     ratio=float(a["sampled"]["contraction"]
                                 / b["sampled"]["contraction"])),
        gaussian=dict(independent=a["gaussian"]["contraction"],
                      shared=b["gaussian"]["contraction"],
                      ratio=float(a["gaussian"]["contraction"]
                                  / b["gaussian"]["contraction"])))
