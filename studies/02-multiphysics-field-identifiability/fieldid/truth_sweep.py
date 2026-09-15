"""Robustness sweeps over synthetic truth combinations.

The published experiment matrix uses one nominal synthetic truth.  This module
repeats selected configurations at explicitly declared parameter combinations
without changing their priors, acquisition or forward models.  It is a
robustness analysis, not a physiological population simulation.

All configurations in a scenario receive the same physical truth and the same
standard-normal noise realization.  Calibrated and matched marginals are
recentered on the scenario truth while retaining their stated uncertainty; a
truth sweep must not turn an otherwise correct calibration into an accidental
miscalibration experiment.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from itertools import product
import csv
import json
import os

import numpy as np

from . import experiments as EX
from .params import CATALOGUE, TRUTH, Entry


def load_spec(path):
    with open(path) as fh:
        spec = json.load(fh)
    if not isinstance(spec, dict):
        raise ValueError("truth-sweep specification must be a JSON object")
    return spec


def _validate_value(name, value):
    if name not in CATALOGUE:
        raise ValueError("unknown truth parameter %r" % name)
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("truth value for %s is not finite" % name)
    entry = CATALOGUE[name]
    if entry.transform == "log" and value <= 0:
        raise ValueError("truth value for positive parameter %s must exceed zero" % name)
    if name == "phi" and value > 1.0:
        raise ValueError("phi is a volume fraction and cannot exceed one")
    if name == "vth" and not 0.0 <= value <= np.pi:
        raise ValueError("vth must lie in [0, pi]")
    if name == "vaz" and not -np.pi <= value <= np.pi:
        raise ValueError("vaz must lie in [-pi, pi]")
    if name == "t0" and value < 0.0:
        raise ValueError("t0 cannot be negative")
    return value


def _complete_truth(values, derive_storage=True):
    truth = dict(TRUTH)
    for name, value in values.items():
        truth[name] = _validate_value(name, value)
    if derive_storage:
        truth["S_v"] = truth["phi"] * truth["C_v"]
    else:
        truth["S_v"] = _validate_value("S_v", truth["S_v"])
    return truth


def expand_scenarios(spec):
    """Return named truth scenarios from either an explicit list or a grid."""
    has_scenarios = "scenarios" in spec
    has_grid = "grid" in spec
    if has_scenarios == has_grid:
        raise ValueError("specification must contain exactly one of 'scenarios' or 'grid'")
    derive = bool(spec.get("derive_storage", True))
    out = []
    if has_scenarios:
        raw = spec["scenarios"]
        if not isinstance(raw, list) or not raw:
            raise ValueError("'scenarios' must be a non-empty list")
        for i, rec in enumerate(raw):
            if not isinstance(rec, dict) or "name" not in rec:
                raise ValueError("scenario %d needs a name" % i)
            values = rec.get("values", {})
            if not isinstance(values, dict):
                raise ValueError("values of scenario %r must be an object" % rec["name"])
            out.append(dict(name=str(rec["name"]), values=dict(values),
                            truth=_complete_truth(values, derive)))
    else:
        grid = spec["grid"]
        if not isinstance(grid, dict) or not grid:
            raise ValueError("'grid' must be a non-empty object")
        names = list(grid)
        axes = []
        for name in names:
            values = grid[name]
            if not isinstance(values, list) or not values:
                raise ValueError("grid axis %r must be a non-empty list" % name)
            axes.append([_validate_value(name, v) for v in values])
        for i, combination in enumerate(product(*axes)):
            values = dict(zip(names, combination))
            out.append(dict(name="grid_%04d" % i, values=values,
                            truth=_complete_truth(values, derive)))
    names = [r["name"] for r in out]
    if len(names) != len(set(names)):
        raise ValueError("scenario names must be unique")
    return out


def _recenter(entry, truth):
    """Keep a calibrated/matched interval width but center it at new truth."""
    if entry.transform == "log":
        half = 0.5 * (np.log(entry.hi) - np.log(entry.lo))
        lo, hi = truth * np.exp(-half), truth * np.exp(half)
    else:
        half = 0.5 * (entry.hi - entry.lo)
        lo, hi = truth - half, truth + half
    return Entry(entry.name, entry.transform, float(truth), float(lo), float(hi),
                 entry.role, entry.kind)


def recentered_config(cfg, truth, enabled=True):
    """Recenter only independently calibrated and matched quantities."""
    if not enabled:
        return cfg
    overrides = dict(cfg.overrides)
    for name, entry in list(overrides.items()):
        if entry.role in ("calibrated", "matched"):
            overrides[name] = _recenter(entry, truth[name])
    return replace(cfg, overrides=overrides)


def select_configs(names):
    pool = {c.name: c for c in EX.matrix()}
    missing = [n for n in names if n not in pool]
    if missing:
        raise ValueError("unknown configurations: %s" % ", ".join(missing))
    return [pool[n] for n in names]


def plan(spec, configs=None, n_rep=None):
    scenarios = expand_scenarios(spec)
    names = list(configs or spec.get("configs", []))
    if not names:
        raise ValueError("declare at least one configuration")
    select_configs(names)
    repetitions = int(n_rep if n_rep is not None else spec.get("n_replicates", 1))
    if repetitions < 1:
        raise ValueError("n_replicates must be positive")
    return dict(n_scenarios=len(scenarios), n_configs=len(names),
                n_replicates=repetitions,
                n_fits=len(scenarios) * len(names) * repetitions,
                scenario_names=[s["name"] for s in scenarios], configs=names)


def _signature(spec, config_names, n_rep, seed, report, recenter):
    payload = dict(spec=spec, configs=list(config_names), n_replicates=int(n_rep),
                   seed=int(seed), report=list(report),
                   recenter_calibrations=bool(recenter))
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(raw).hexdigest()


def _write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def _read_records(directory, signature):
    out = []
    if not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(directory, name)) as fh:
            rec = json.load(fh)
        if rec.get("sweep_signature") == signature:
            out.append(rec)
    return out


def _record_path(directory, si, cfg, rep):
    return os.path.join(directory, "s%04d__%s__r%03d.json" % (si, cfg, rep))


def _write_detail_csv(path, records):
    fields = ["scenario", "configuration", "replicate", "seed", "windows",
              "level", "quantity", "truth", "median", "lo", "hi",
              "contraction", "bias_log", "covered", "rank", "n_par",
              "least_fisher_eigenvalue", "least_informed_contraction"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            eig = rec["eigenvalues"]
            directional = rec["contraction"]
            for quantity, summary in rec["summary"].items():
                writer.writerow(dict(
                    scenario=rec["scenario"], configuration=rec["name"],
                    replicate=rec["replicate"], seed=rec["seed"],
                    windows="+".join(rec["windows"]), level=rec["level"],
                    quantity=quantity, truth=summary["truth"],
                    median=summary["median"], lo=summary["lo"], hi=summary["hi"],
                    contraction=summary["contraction"], bias_log=summary["bias_log"],
                    covered=summary["covered"], rank=rec["rank"], n_par=rec["n_par"],
                    least_fisher_eigenvalue=eig[0],
                    least_informed_contraction=directional[0]))


def summarize(records):
    groups = {}
    for rec in records:
        for quantity, values in rec["summary"].items():
            groups.setdefault((rec["scenario"], rec["name"], quantity), []).append((rec, values))
    rows = []
    for (scenario, config, quantity), items in sorted(groups.items()):
        vals = [x[1] for x in items]
        recs = [x[0] for x in items]
        contraction = np.array([v["contraction"] for v in vals], float)
        bias = np.array([v["bias_log"] for v in vals], float)
        rows.append(dict(
            scenario=scenario, configuration=config, quantity=quantity,
            n_replicates=len(items), truth=float(vals[0]["truth"]),
            contraction_median=float(np.median(contraction)),
            contraction_min=float(np.min(contraction)),
            contraction_max=float(np.max(contraction)),
            coverage=float(np.mean([v["covered"] for v in vals])),
            absolute_bias_log_median=float(np.median(np.abs(bias))),
            rank_min=int(min(r["rank"] for r in recs)),
            n_par=int(recs[0]["n_par"]),
            least_fisher_eigenvalue_median=float(np.median(
                [r["eigenvalues"][0] for r in recs])),
            least_informed_contraction_median=float(np.median(
                [r["contraction"][0] for r in recs]))))
    return rows


def _write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparisons(records, declarations):
    """Width gains, formed **within scenario and replicate** before summarizing.

    Replicate ``r`` uses the same standardized noise draw for every scenario and
    configuration, so the two configurations of a comparison see the same noise
    at the same truth.  The gain is therefore a paired quantity and has to be
    computed pair by pair; a ratio of two separately computed medians discards
    the pairing and is not the median of the gains.

    Returned per scenario: the five paired gains, their median, and their range.
    The range over five paired replicates is reported as the interval; it is a
    descriptive spread, not a confidence interval, and no distribution is
    assumed.
    """
    lookup = {}
    for rec in records:
        for quantity, values in rec["summary"].items():
            lookup[(rec["scenario"], rec["name"], quantity,
                    int(rec["replicate"]))] = values
    scenarios, out = sorted({r["scenario"] for r in records}), []
    for comp in declarations or []:
        needed = ("name", "numerator", "denominator", "quantity")
        if any(k not in comp for k in needed):
            raise ValueError("every comparison needs %s" % ", ".join(needed))
        reps = sorted({int(r["replicate"]) for r in records})
        for scenario in scenarios:
            gains, nums, dens = [], [], []
            for rep in reps:
                a = lookup.get((scenario, comp["numerator"], comp["quantity"], rep))
                b = lookup.get((scenario, comp["denominator"], comp["quantity"], rep))
                if a is None or b is None:
                    continue
                num, den = float(a["contraction"]), float(b["contraction"])
                nums.append(num)
                dens.append(den)
                gains.append(float(den / num) if num > 0 else float("inf"))
            if not gains:
                continue
            g = np.array(gains, float)
            out.append(dict(
                scenario=scenario, comparison=comp["name"],
                quantity=comp["quantity"], numerator=comp["numerator"],
                denominator=comp["denominator"],
                n_replicates=len(gains),
                numerator_contraction_median=float(np.median(nums)),
                denominator_contraction_median=float(np.median(dens)),
                width_gain_median=float(np.median(g)),
                width_gain_min=float(np.min(g)),
                width_gain_max=float(np.max(g)),
                width_gain_per_replicate=";".join("%.6g" % v for v in gains)))
    return out


def run(spec, out, corr=None, seed=0, configs=None, n_rep=None,
        max_cases=500, force=False, recenter_calibrations=True):
    scenarios = expand_scenarios(spec)
    config_names = list(configs or spec.get("configs", []))
    cfgs = select_configs(config_names)
    repetitions = int(n_rep if n_rep is not None else spec.get("n_replicates", 1))
    report = list(spec.get("report", EX.REPORT))
    bad_report = [n for n in report if n not in CATALOGUE]
    if bad_report:
        raise ValueError("unknown reported quantities: %s" % ", ".join(bad_report))
    n_cases = len(scenarios) * len(cfgs) * repetitions
    if n_cases > int(max_cases):
        raise ValueError("sweep requests %d fits; increase --max-cases deliberately" % n_cases)
    if any(c.level == "network" for c in cfgs) and corr is None:
        raise ValueError("a network configuration requires --prior")

    signature = _signature(spec, config_names, repetitions, seed, report,
                           recenter_calibrations)
    records_dir = os.path.join(out, "records")
    os.makedirs(records_dir, exist_ok=True)
    for si, scenario in enumerate(scenarios):
        for cfg0 in cfgs:
            cfg = recentered_config(cfg0, scenario["truth"], recenter_calibrations)
            for rep in range(repetitions):
                path = _record_path(records_dir, si, cfg.name, rep)
                if os.path.exists(path) and not force:
                    with open(path) as fh:
                        old = json.load(fh)
                    if old.get("sweep_signature") == signature:
                        continue
                    raise ValueError("existing record belongs to another sweep: %s; "
                                     "use --force or a new output directory" % path)
                run_seed = int(seed) + rep
                rec = EX.run_config(cfg, corr, seed=run_seed,
                                    truth=scenario["truth"], report=report)
                rec.update(scenario=scenario["name"], scenario_index=si,
                           scenario_values=scenario["values"],
                           truth_full=scenario["truth"], replicate=rep,
                           sweep_signature=signature,
                           calibrations_recentered=bool(recenter_calibrations))
                _write_json(path, rec)
                print("truth sweep: %s / %s / replicate %d" %
                      (scenario["name"], cfg.name, rep + 1))

    records = _read_records(records_dir, signature)
    expected = n_cases
    if len(records) != expected:
        raise RuntimeError("found %d records for this sweep, expected %d" %
                           (len(records), expected))
    records.sort(key=lambda r: (r["scenario_index"], r["name"], r["replicate"]))
    return _write_outputs(out, spec, scenarios, records, signature, seed,
                          len(cfgs), repetitions, n_cases, config_names,
                          report, recenter_calibrations)


def _env_versions():
    import platform
    import numpy
    out = dict(python=platform.python_version(), numpy=numpy.__version__)
    try:
        import scipy
        out["scipy"] = scipy.__version__
    except Exception:                                  # pragma: no cover
        out["scipy"] = None
    return out


def code_sha256():
    """sha256 over the package sources, in path order."""
    root = os.path.dirname(os.path.abspath(__file__))
    h = sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for f in sorted(filenames):
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            h.update(os.path.relpath(p, root).encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def records_checksum(records):
    """sha256 over the ordered record set, on the fields that define a fit.

    Taken over the canonical serialization of each record in sweep order, so a
    reordering, an insertion or an edited number all change it."""
    h = sha256()
    for r in records:
        h.update(json.dumps(r, sort_keys=True, separators=(",", ":"),
                            default=float).encode())
    return h.hexdigest()


def _write_outputs(out, spec, scenarios, records, signature, seed, n_cfg,
                   repetitions, n_cases, config_names, report,
                   recenter_calibrations):
    """Derived files only.  Never touches the records."""
    from datetime import datetime, timezone
    summary = summarize(records)
    comp = comparisons(records, spec.get("comparisons", []))
    manifest = dict(
        sweep_signature=signature, seed=int(seed),
        n_scenarios=len(scenarios), n_configurations=int(n_cfg),
        n_replicates=int(repetitions), n_fits=int(n_cases),
        n_records_found=len(records),
        configurations=list(config_names), report=list(report),
        derive_storage=bool(spec.get("derive_storage", True)),
        calibrations_recentered=bool(recenter_calibrations),
        noise_pairing=("replicate r uses seed+r for every scenario and "
                       "configuration, isolating truth/configuration effects"),
        coverage_is_diagnostic=(
            "the same standardized noise draws are reused across scenarios, so "
            "the scenario-by-replicate cells are not independent coverage "
            "experiments; 'covered' is a diagnostic field, not a coverage "
            "estimate"),
        comparison_aggregation=(
            "gains are formed within scenario and replicate, then summarized "
            "over the paired replicates as median and range"),
        code_sha256=code_sha256(),
        spec_sha256=sha256(json.dumps(spec, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest(),
        records_sha256=records_checksum(records),
        # The records do not carry an environment, so this describes the
        # process that wrote these derived files, which is not necessarily the
        # process that performed the fits.  When the summaries are regenerated
        # from stored records the two differ, and saying so is the point of
        # naming the field this way.
        environment_deriving=_env_versions(),
        environment_note=("the stored records carry no interpreter or library "
                          "versions, so the generating environment cannot be "
                          "recovered from them; 'environment_deriving' is the "
                          "environment that produced these derived files"),
        derived_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    _write_json(os.path.join(out, "manifest.json"), manifest)
    _write_json(os.path.join(out, "truth_sweep.json"),
                dict(manifest=manifest, scenarios=scenarios,
                     summary=summary, comparisons=comp))
    _write_detail_csv(os.path.join(out, "truth_sweep_long.csv"), records)
    _write_rows(os.path.join(out, "truth_sweep_summary.csv"), summary)
    _write_rows(os.path.join(out, "truth_sweep_comparisons.csv"), comp)
    return dict(manifest=manifest, summary=summary, comparisons=comp)


def regenerate(out, spec, seed=0, configs=None, n_rep=None,
               recenter_calibrations=True):
    """Rebuild every derived file from the stored records, fitting nothing.

    The records are the experiment; the summaries, the comparison table and the
    manifest are derivations of them.  This exists so that an aggregation can be
    corrected without rerunning 360 fits, and it refuses to run if the records
    on disk are not the ones the specification describes."""
    scenarios = expand_scenarios(spec)
    config_names = list(configs or spec.get("configs", []))
    cfgs = select_configs(config_names)
    repetitions = int(n_rep if n_rep is not None else spec.get("n_replicates", 1))
    report = list(spec.get("report", EX.REPORT))
    signature = _signature(spec, config_names, repetitions, seed, report,
                           recenter_calibrations)
    records = _read_records(os.path.join(out, "records"), signature)
    n_cases = len(scenarios) * len(cfgs) * repetitions
    if len(records) != n_cases:
        raise RuntimeError(
            "found %d records carrying this sweep signature in %s, expected %d; "
            "the records do not match the specification, so the derived files "
            "would not describe them" % (len(records), out, n_cases))
    records.sort(key=lambda r: (r["scenario_index"], r["name"], r["replicate"]))
    return _write_outputs(out, spec, scenarios, records, signature, seed,
                          len(cfgs), repetitions, n_cases, config_names,
                          report, recenter_calibrations)
