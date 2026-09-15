"""Command-line entry point.  Every reported result is produced here.

    fieldid ensemble-prior --ensemble results/network_ensemble.csv \
                           --out results/network_prior.json
    fieldid config         --out configs/default.json
    fieldid e0             --out results
    fieldid matrix         --out results [--mcmc 40000] [--no-relax]
    fieldid discrepancy    --out results
    fieldid coverage       --out results [--n-rep 100]
    fieldid truth-sweep    --out results/truth_sweep [--dry-run]
    fieldid figures        --out figures --results results
    fieldid all            --out results
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys

import numpy as np

from . import experiments as EX
from . import screening as SC
from .coupling import CorrelatedPrior, NETWORK_NAMES
from .infer import sampling as SA


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def _write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, cls=_Enc)
    print("wrote", path)


def _read_ensemble(path):
    cols = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v) if v else np.nan)
    return {k: np.array(v) for k, v in cols.items()}


# ------------------------------------------------------------------ actions -

#: the fit bounds of the shell estimator in the vascular-network package; a
#: value sitting on one of them is censored, not measured, and cannot enter a
#: correlation estimate
CENSOR = {"D": (1.01e-9, 0.99e-4), "vmag": (1.01e-4, 0.099)}


def _drop_censored(tab):
    keep = np.ones(len(tab["phi"]), bool)
    for n, (lo, hi) in CENSOR.items():
        if n in tab:
            keep &= (tab[n] > lo) & (tab[n] < hi)
    return {k: v[keep] for k, v in tab.items()}, int((~keep).sum())


def cmd_ensemble_prior(a):
    raw = _read_ensemble(a.ensemble)
    tab, n_cens = _drop_censored(raw)
    cp = CorrelatedPrior.from_ensemble(tab, NETWORK_NAMES,
                                       source=os.path.basename(a.ensemble))
    d = cp.to_dict()
    d["n_rows_raw"] = int(len(raw["phi"]))
    d["n_censored_dropped"] = n_cens
    d["censor_bounds"] = {k: list(v) for k, v in CENSOR.items()}
    d["marginal_note"] = ("marginals are those of the catalogue; only the "
                          "correlation of the logarithms comes from the ensemble")
    d["conditional_sd_given_phi"] = {
        n: cp.conditional_sd(n, "phi") for n in NETWORK_NAMES if n != "phi"}
    d["ensemble_log_sd"] = {n: float(np.std(np.log(tab[n]), ddof=1))
                            for n in NETWORK_NAMES}
    d["n_seeds"] = int(len(set(tab["seed"].tolist())))
    d["ensemble_median"] = {n: float(np.median(tab[n])) for n in NETWORK_NAMES}
    _write(a.out, d)


def cmd_config(a):
    """Dump the effective configuration: every parameter with its transform,
    prior interval and role, every fixed quantity, and the acquisition."""
    from .acquisition import Acquisition
    from .constants import FIXED, UNITS
    from .params import CATALOGUE, TRUTH
    from . import experiments as EXP
    d = dict(
        version=__import__("fieldid").__version__,
        parameters={n: dict(transform=e.transform, nominal=e.nominal,
                            prior_lo=e.lo, prior_hi=e.hi, role=e.role,
                            kind=e.kind, unit=UNITS.get(n))
                    for n, e in CATALOGUE.items()},
        truth=TRUTH,
        fixed={n: dict(value=v, unit=UNITS.get(n)) for n, v in FIXED.items()},
        acquisition=Acquisition().to_dict(),
        prior_overrides={
            "A_calibrated": {k: dict(lo=v.lo, hi=v.hi, role=v.role)
                             for k, v in EXP.A_CAL.items()},
            "C_v_calibrated": {k: dict(lo=v.lo, hi=v.hi, role=v.role)
                               for k, v in EXP.CV_CAL.items()},
            "P_nuisance": {k: dict(lo=v.lo, hi=v.hi, role=v.role)
                           for k, v in EXP.P_NUI.items()},
            # prior information matched to what the coupled arm induces, not a
            # measurement; the role says so
            "S_v_matched_to_calibrated_compliance": {
                k: dict(lo=v.lo, hi=v.hi, role=v.role)
                for k, v in EXP.SV_MATCHED_CVCAL.items()}},
        roles={n: d for n, d in [
            ("free", "estimated from the observations"),
            ("nuisance", "estimated, not reported, not measured elsewhere"),
            ("calibrated", "measured independently of the imaging observations"),
            ("matched", "not measured; a marginal set equal to the one another "
                        "configuration induces, so that two arms differ only in "
                        "the structure under test"),
            ("fixed", "held at a constant, with its consequence screened in E0"),
            ("derived", "computed from others by a coupling relation")]},
        configurations=[dict(name=c.name, windows=list(c.windows), level=c.level,
                             normalized=c.normalized, note=c.note,
                             free=list(c.free()),
                             overrides=sorted(c.overrides))
                        for c in EXP.matrix() + EXP.discrepancy_matrix()],
    )
    _write(a.out if a.out.endswith(".json")
           else os.path.join(a.out, "config.json"), d)


def cmd_e0(a):
    _write(os.path.join(a.out, "E0_screening.json"), SC.run())


def cmd_matrix(a):
    corr = CorrelatedPrior.load(a.prior) if a.prior and os.path.exists(a.prior) else None
    rows = []
    for cfg in EX.matrix(include_relax=not a.no_relax):
        if corr is None and cfg.level == "network":
            print("skipping %s: no network prior available" % cfg.name)
            continue
        r = EX.run_config(cfg, corr, seed=a.seed, mcmc=a.mcmc,
                          profiles=a.profile or ())
        _write(os.path.join(a.out, "%s.json" % cfg.name), r)
        rows.append(r)
    _summary_csv(os.path.join(a.out, "summary_table.csv"), rows)


def cmd_discrepancy(a):
    corr = CorrelatedPrior.load(a.prior) if a.prior and os.path.exists(a.prior) else None
    rows = []
    for cfg in EX.discrepancy_matrix():
        r = EX.run_config(cfg, corr, seed=a.seed)
        _write(os.path.join(a.out, "%s.json" % cfg.name), r)
        rows.append(r)
    _summary_csv(os.path.join(a.out, "discrepancy_table.csv"), rows)


def cmd_coverage(a):
    corr = CorrelatedPrior.load(a.prior) if a.prior and os.path.exists(a.prior) else None
    want = a.configs or ["E1_swe", "E2_ceus_Acal", "E5_joint_network",
                         "E8_relax_constitutive_Cvcal"]
    pool = list(EX.matrix())
    if a.richer:
        pool += list(EX.discrepancy_matrix())
        want = want + [c.name for c in EX.discrepancy_matrix()]
    path = os.path.join(a.out, "coverage.json")
    out = []
    if os.path.exists(path):                      # merge, so the study can be
        with open(path) as fh:                    # completed in stages
            out = [r for r in json.load(fh) if r["name"] not in want]
    for cfg in pool:
        if cfg.name not in want:
            continue
        out.append(EX.coverage(cfg, corr, n_rep=a.n_rep, seed=a.seed))
        print("coverage done: %s  (%d repetitions)" % (cfg.name, out[-1]["n_rep"]))
        _write(path, out)


def cmd_profiles(a):
    """Profile likelihoods for the configurations where the question is whether
    a direction is flat or merely correlated."""
    corr = CorrelatedPrior.load(a.prior) if a.prior and os.path.exists(a.prior) else None
    want = a.configs or ["E8_relax_independent", "E8_relax_constitutive_Cvcal",
                         "E5_joint_network", "E3_joint_independent"]
    names = a.profile or ["k"]
    # a separate file per invocation lets configurations run as parallel
    # processes without clobbering each other; `profiles-merge` joins them
    path = os.path.join(a.out, a.file)
    out = {}
    if os.path.exists(path):                      # merge, so the sweep can be
        with open(path) as fh:                    # completed in stages
            out = json.load(fh)
    for cfg in EX.matrix():
        if cfg.name not in want:
            continue
        inv = cfg.build(corr, a.seed)
        y = inv.simulate(np.random.default_rng(a.seed))
        out.setdefault(cfg.name, {})
        for n in names:
            if n not in inv.free:
                continue
            rec = out[cfg.name].get(n, {})     # merge, so levels can be run apart
            imaging = None
            for level in (a.level or EX.DG.PROFILE_LEVELS):
                pr = EX.DG.profile_likelihood(inv, y, n, n=a.n_grid, span=3.0,
                                              level=level)
                rec["grid"] = np.asarray(pr["grid"]).tolist()
                rec["delta_" + level] = np.asarray(pr["delta"]).tolist()
                # what the optimizer did, so that a flat profile can be checked
                # against the possibility that it ran out of admissible room
                rec["audit_" + level] = EX.DG.profile_audit(inv, pr)
                if level == "imaging":
                    imaging = pr
            if a.domain_check:
                rec["domain_sensitivity_imaging"] = EX.DG.profile_domain_sensitivity(
                    inv, y, n, base=imaging, n=a.n_grid, span=3.0, level="imaging")
            out[cfg.name][n] = rec
            print("profile done: %s %s (%s)"
                  % (cfg.name, n, ", ".join(a.level or EX.DG.PROFILE_LEVELS)
                     if not a.domain_check else "domain check"))
        _write(path, out)


def cmd_profiles_merge(a):
    """Join per-invocation profile files into one, deep-merging by
    configuration and quantity."""
    out = {}
    for p in sorted(a.inputs):
        with open(p) as fh:
            d = json.load(fh)
        for cfg, qs in d.items():
            for q, rec in qs.items():
                out.setdefault(cfg, {}).setdefault(q, {}).update(rec)
    _write(os.path.join(a.out, a.file), out)


def cmd_mcmc(a):
    """Sampling validation of the Gaussian posterior approximation.

    Three modes.  Without options it samples each configuration in one process.
    ``--add-steps`` advances the chains of the named configurations by that many
    steps from a checkpoint and stops, so that a chain longer than one process
    may live can be built up in pieces; ``--finish`` then turns the checkpoints
    into records.  ``--collect-only`` assembles the comparison from records
    already on disk.
    """
    from . import mcmc_study as MS
    corr = (CorrelatedPrior.load(a.prior)
            if a.prior and os.path.exists(a.prior) else None)
    work = a.work or os.path.join(a.out, "mcmc_work")
    names = a.configs or list(MS.CONFIGS)
    if a.collect_only:
        MS.collect(a.out)
        return
    if a.add_steps:
        for n in names:
            MS.sample_chunk(n, work, corr=corr, seed=a.seed,
                            add_steps=a.add_steps, n_chains=a.chains,
                            thin=a.thin)
        return
    if a.finish:
        for n in names:
            r = MS.finish(n, work, a.out, corr=corr, seed=a.seed)
            print("%s: %d steps per chain, accept %.2f, worst R-hat %.4f "
                  "(%.4f along the least informed directions), "
                  "smallest effective size %.0f"
                  % (n, r["sampler"]["n_steps"], r["sampler"]["accept"],
                     r["worst_rhat"], r["worst_rhat_direction"], r["min_ess"]))
        MS.collect(a.out)
        return
    MS.run(a.out, prior=a.prior, seed=a.seed, n_steps=a.steps,
           n_chains=a.chains, thin=a.thin, configs=a.configs)


def cmd_figures(a):
    from .figures import make_all
    make_all(a.results, a.out)


def cmd_truth_sweep(a):
    from . import truth_sweep as TS
    spec = TS.load_spec(a.spec)
    p = TS.plan(spec, configs=a.configs, n_rep=a.n_rep)
    print("truth sweep: %(n_scenarios)d scenarios x %(n_configs)d configurations "
          "x %(n_replicates)d replicates = %(n_fits)d fits" % p)
    if a.dry_run:
        return
    corr = (CorrelatedPrior.load(a.prior)
            if a.prior and os.path.exists(a.prior) else None)
    TS.run(spec, a.out, corr=corr, seed=a.seed, configs=a.configs,
           n_rep=a.n_rep, max_cases=a.max_cases, force=a.force,
           recenter_calibrations=not a.no_recenter_calibrations)


def cmd_all(a):
    cmd_e0(a)
    cmd_matrix(a)
    cmd_discrepancy(a)
    cmd_coverage(a)


def _summary_csv(path, rows):
    fields = ["config", "windows", "level", "normalized", "richer", "quantity",
              "truth", "median", "lo", "hi", "width_post", "width_prior",
              "contraction", "bias_log", "covered"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            for q, s in r["summary"].items():
                w.writerow(dict(config=r["name"], windows="+".join(r["windows"]),
                                level=r["level"], normalized=r["normalized"],
                                richer=r["richer"], quantity=q,
                                truth=s["truth"], median=s["median"],
                                lo=s["lo"], hi=s["hi"],
                                width_post=s["width_post"],
                                width_prior=s["width_prior"],
                                contraction=s["contraction"],
                                bias_log=s["bias_log"], covered=s["covered"]))
    print("wrote", path)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--out", default="results")
    common.add_argument("--prior", default="results/network_prior.json")

    ap = argparse.ArgumentParser(prog="fieldid", description=__doc__,
                                 parents=[common],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name):
        return sub.add_parser(name, parents=[common])

    p = add("ensemble-prior")
    p.add_argument("--ensemble", required=True)
    p.set_defaults(func=cmd_ensemble_prior)

    p = add("config"); p.set_defaults(func=cmd_config)

    p = add("e0"); p.set_defaults(func=cmd_e0)

    p = add("matrix")
    p.add_argument("--mcmc", type=int, default=0)
    p.add_argument("--no-relax", action="store_true")
    p.add_argument("--profile", action="append")
    p.set_defaults(func=cmd_matrix)

    p = add("discrepancy"); p.set_defaults(func=cmd_discrepancy)

    p = add("coverage")
    p.add_argument("--n-rep", type=int, default=100)
    p.add_argument("--configs", action="append")
    p.add_argument("--richer", action="store_true")
    p.set_defaults(func=cmd_coverage)

    p = add("profiles")
    p.add_argument("--configs", action="append")
    p.add_argument("--profile", action="append")
    p.add_argument("--n-grid", type=int, default=13)
    p.add_argument("--level", action="append",
                   help="profile level, repeatable; default all three.  Levels "
                        "merge into the existing file, so a sweep can be run "
                        "one level at a time")
    p.add_argument("--domain-check", action="store_true",
                   help="recompute the imaging profile with a tighter ceiling, "
                        "to test whether flatness depends on the optimizer "
                        "reaching the edge of the physical domain")
    p.add_argument("--file", default="profiles.json")
    p.set_defaults(func=cmd_profiles)

    p = add("profiles-merge")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--file", default="profiles.json")
    p.set_defaults(func=cmd_profiles_merge)

    p = add("mcmc")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--thin", type=int, default=10)
    p.add_argument("--configs", action="append")
    p.add_argument("--collect-only", action="store_true",
                   help="assemble mcmc_validation.json from records already "
                        "on disk, without sampling")
    p.add_argument("--add-steps", type=int, default=0,
                   help="advance the chains by this many steps from a "
                        "checkpoint and stop")
    p.add_argument("--finish", action="store_true",
                   help="turn the checkpoints into reported records")
    p.add_argument("--work", default=None,
                   help="checkpoint directory (default <out>/mcmc_work)")
    p.set_defaults(func=cmd_mcmc)

    p = add("figures")
    p.add_argument("--results", default="results")
    p.set_defaults(func=cmd_figures)

    p = add("truth-sweep")
    p.add_argument("--spec", default="configs/truth_sweep.json")
    p.add_argument("--configs", action="append",
                   help="configuration to run; repeatable and overrides the spec")
    p.add_argument("--n-rep", type=int, default=None,
                   help="noise replicates per truth/configuration (overrides spec)")
    p.add_argument("--max-cases", type=int, default=500,
                   help="safety limit on total fits")
    p.add_argument("--force", action="store_true",
                   help="replace records in the output directory")
    p.add_argument("--dry-run", action="store_true",
                   help="validate and report sweep size without fitting")
    p.add_argument("--no-recenter-calibrations", action="store_true",
                   help="do not recenter calibrated/matched marginals on each truth; "
                        "this intentionally mixes robustness with miscalibration")
    p.set_defaults(func=cmd_truth_sweep, out="results/truth_sweep")

    p = add("all"); p.set_defaults(func=cmd_all)
    p.add_argument("--mcmc", type=int, default=0)
    p.add_argument("--no-relax", action="store_true")
    p.add_argument("--profile", action="append")
    p.add_argument("--n-rep", type=int, default=100)
    p.add_argument("--configs", action="append")
    p.add_argument("--richer", action="store_true")

    a = ap.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
