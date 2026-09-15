"""Command-line entry point.  Every reported number is produced here.

    operatorid config    --out configs/default.json
    operatorid screening --out results
    operatorid matrix    --out results [--profile sig_e]
    operatorid coverage  --out results [--n-rep 40]
    operatorid mcmc      --out results [--steps 20000]
    operatorid figures   --out figures --results results
    operatorid all       --out results
    python scripts/verify_results.py
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
from .provenance import fingerprint


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def _write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, cls=_Enc)
    print("wrote", path)


def _config_dict():
    from .acquisition import Acquisition
    from .constants import (FIXED, GAMMA_NOMINAL, TRANSMIT_HZ,
                            TRACK_KERNEL_M, UNITS, psf_widths)
    from .params import REPORT, ROLES, full_catalogue, TRUTH
    import operatorid
    cat = full_catalogue()
    return dict(
        version=operatorid.__version__,
        parameters={n: dict(transform=e.transform, nominal=e.nominal,
                            prior_lo=e.lo, prior_hi=e.hi, role=e.role,
                            kind=e.kind, unit=UNITS.get(n))
                    for n, e in cat.items()},
        roles={
            "free": "estimated from the observations",
            "nuisance": "estimated, not reported as a result, not measured elsewhere",
            "operator": "a nuisance acting inside the observation operator; the "
                        "study is about one of them",
            "calibrated": "measured independently of these observations",
            "fixed": "held at a constant, with its consequence screened"},
        truth=TRUTH,
        reported=list(REPORT),
        fixed={n: dict(value=v, unit=UNITS.get(n)) for n, v in FIXED.items()},
        gamma_nominal=dict(GAMMA_NOMINAL),
        psf_widths_at_nominal={m: float(v) for m, v in
                               psf_widths(0.35e-3).items()},
        transmit_hz=dict(TRANSMIT_HZ),
        track_kernel_m=TRACK_KERNEL_M,
        acquisition=Acquisition().to_dict(),
        arms=list(EX.ARMS),
        cases=[c.to_dict() for c in EX.matrix()],
        coverage_truths=[dict(name=n, overrides=o) for n, o in EX.COVERAGE_TRUTHS],
    )


def cmd_config(a):
    d = _config_dict()
    d["fingerprint"] = fingerprint(d)
    _write(a.out if a.out.endswith(".json")
           else os.path.join(a.out, "config.json"), d)


def cmd_screening(a):
    _write(os.path.join(a.out, "screening.json"), SC.run())


def cmd_matrix(a):
    rows = []
    want = set(a.cases) if a.cases else None
    for case in EX.matrix():
        if want and case.name not in want:
            continue
        data = EX.generate(case, seed=a.seed)   # once, for every arm
        for arm in EX.ARMS:
            r = EX.run_arm(case, arm, seed=a.seed, data=data,
                           profiles=tuple(a.profile or ()))
            _write(os.path.join(a.out, "%s__%s.json" % (case.name, arm)), r)
            rows.append(r)
    _summary_csv(os.path.join(a.out, "summary_table.csv"), rows)


def cmd_summary(a):
    """Rebuild the summary table from whatever arm files are on disk, so that
    the matrix can be run in stages without leaving a partial table."""
    rows = []
    for case in EX.matrix():
        for arm in EX.ARMS:
            p = os.path.join(a.out, "%s__%s.json" % (case.name, arm))
            if os.path.exists(p):
                with open(p) as fh:
                    rows.append(json.load(fh))
    if not rows:
        raise SystemExit("no arm files in %s" % a.out)
    _summary_csv(os.path.join(a.out, "summary_table.csv"), rows)


def cmd_coverage(a):
    path = os.path.join(a.out, "coverage.json")
    out = []
    if os.path.exists(path):                 # merge, so it can be run in stages
        with open(path) as fh:
            out = json.load(fh)
    cases = {c.name: c for c in EX.matrix()}
    want = a.cases or ["main", "wrong_ratio"]
    done = {(r["case"], r["arm"], r["truth"]) for r in out}
    for cname in want:
        case = cases[cname]
        for arm in EX.ARMS:
            for tname, tover in EX.COVERAGE_TRUTHS:
                if (cname, arm, tname) in done:
                    continue
                r = EX.coverage(case, arm, tname, tover, n_rep=a.n_rep,
                                seed=a.seed)
                out.append(r)
                print("coverage %s %s %s: %d repetitions" %
                      (cname, arm, tname, r["n_rep"]))
                _write(path, out)


def cmd_mcmc(a):
    from . import validate_posterior as VP
    arms = tuple(a.arms) if a.arms else ("independent", "shared")
    work = a.work or os.path.join(a.out, "mcmc_work")
    case = {c.name: c for c in EX.matrix()}["main"]
    if a.collect_only:
        VP.collect(a.out)
        return
    if a.add_steps:
        for arm in arms:
            VP.sample_chunk(case, arm, work, seed=a.seed,
                            add_steps=a.add_steps, n_chains=a.chains,
                            thin=a.thin,
                            allow_code_change=a.allow_code_change)
        return
    if a.finish:
        for arm in arms:
            VP._finish_print(VP.finish(
                case, arm, work, a.out, seed=a.seed, n_chains=a.chains,
                thin=a.thin, allow_code_change=a.allow_code_change)[0])
        VP.collect(a.out)
        return
    VP.run(a.out, seed=a.seed, n_steps=a.steps, n_chains=a.chains, thin=a.thin,
           arms=arms)


def cmd_figures(a):
    from .figures import make_all
    make_all(a.results, a.out)


def cmd_all(a):
    cmd_config(argparse.Namespace(out="configs/default.json"))
    cmd_screening(a)
    cmd_matrix(a)
    cmd_coverage(a)


def _summary_csv(path, rows):
    fields = ["case", "arm", "windows", "n_frames_used", "quantity", "truth",
              "median", "lo", "hi", "width_post", "width_prior", "contraction",
              "bias_log", "covered"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            for q, s in r["summary"].items():
                w.writerow(dict(case=r["case"], arm=r["arm"],
                                windows="+".join(r["windows"]),
                                n_frames_used=r["n_frames_used"], quantity=q,
                                truth=s.get("truth"), median=s.get("median"),
                                lo=s.get("lo"), hi=s.get("hi"),
                                width_post=s.get("width_post"),
                                width_prior=s.get("width_prior"),
                                contraction=s.get("contraction"),
                                bias_log=s.get("bias_log"),
                                covered=s.get("covered")))
    print("wrote", path)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--out", default="results")

    ap = argparse.ArgumentParser(prog="operatorid", description=__doc__,
                                 parents=[common],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name):
        return sub.add_parser(name, parents=[common])

    p = add("config"); p.set_defaults(func=cmd_config)
    p = add("screening"); p.set_defaults(func=cmd_screening)

    p = add("matrix")
    p.add_argument("--profile", action="append")
    p.add_argument("--cases", action="append")
    p.set_defaults(func=cmd_matrix)

    p = add("summary"); p.set_defaults(func=cmd_summary)

    p = add("coverage")
    p.add_argument("--n-rep", type=int, default=40)
    p.add_argument("--cases", action="append")
    p.set_defaults(func=cmd_coverage)

    p = add("mcmc")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--thin", type=int, default=5)
    p.add_argument("--arms", action="append",
                   help="sample only these arms; repeatable, so the two can be "
                        "run in separate processes and collected afterwards")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--add-steps", type=int, default=0,
                   help="advance the chains by this many steps from a "
                        "checkpoint and stop")
    p.add_argument("--finish", action="store_true",
                   help="turn the checkpoints into reported records")
    p.add_argument("--work", default=None,
                   help="checkpoint directory (default <out>/mcmc_work)")
    p.add_argument("--allow-code-change", action="store_true",
                   help="resume a checkpoint written by a different version of "
                        "the package.  Only when the change provably cannot "
                        "reach the chain; the configuration fingerprint is "
                        "still enforced")
    p.set_defaults(func=cmd_mcmc)

    p = add("figures")
    p.add_argument("--results", default="results")
    p.set_defaults(func=cmd_figures)

    p = add("all")
    p.add_argument("--profile", action="append")
    p.add_argument("--cases", action="append")
    p.add_argument("--n-rep", type=int, default=40)
    p.set_defaults(func=cmd_all)

    a = ap.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
