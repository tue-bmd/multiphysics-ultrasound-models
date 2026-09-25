"""Run inference over generating matrix properties and radius scales.

The CEUS response tables must already exist. The Cartesian product of the
requested mu, eta, and s values is evaluated for every supplied network. Each
run retains the noisy observations and repetition-level posterior summaries.
The relation-offset sweep is disabled because this script varies the generating
parameters rather than the assumed cross-modality relationship.
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ceus",
    nargs="+",
    help="CEUS tables to use; defaults to results/ceus_d30_seed*.json",
)
parser.add_argument("--mu-values", type=float, nargs="+", default=[2000.0])
parser.add_argument("--eta-values", type=float, nargs="+", default=[1.0])
parser.add_argument("--s-values", type=float, nargs="+", default=[0.70, 0.80, 0.90])
parser.add_argument("--reps", type=int, default=40)
parser.add_argument("--out-dir", default="results/true_value_sweep")
parser.add_argument("--relation-sd", type=float, default=0.02)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()


def label(value):
    """Filesystem-safe compact representation of a numerical value."""
    return f"{value:g}".replace("-", "m").replace(".", "p").replace("+", "")


study_dir = Path(__file__).resolve().parent
ceus_files = [Path(path) for path in args.ceus] if args.ceus else sorted(
    (study_dir / "results").glob("ceus_d30_seed*.json")
)
if not ceus_files:
    raise FileNotFoundError("no CEUS response tables found")

out_dir = Path(args.out_dir)
if not out_dir.is_absolute():
    out_dir = study_dir / out_dir
if not args.dry_run:
    out_dir.mkdir(parents=True, exist_ok=True)

combinations = list(itertools.product(args.mu_values, args.eta_values, args.s_values))
print(f"{len(ceus_files)} networks x {len(combinations)} parameter combinations "
      f"= {len(ceus_files) * len(combinations)} inference runs")

for ceus_path in ceus_files:
    if not ceus_path.is_absolute():
        ceus_path = study_dir / ceus_path
    with open(ceus_path) as handle:
        blob = json.load(handle)
    settings = blob["provenance"]["settings"]
    seed = int(settings["seed"])
    dterm_um = int(round(1e6 * float(settings["dterm"])))
    available = [float(row["s"]) for row in blob["table"]["rows"]]

    for mu_true, eta_true, s_true in combinations:
        if not min(available) <= s_true <= max(available):
            raise ValueError(
                f"s={s_true:g} is outside the CEUS table for seed {seed}: "
                f"[{min(available):g}, {max(available):g}]"
            )
        output = out_dir / (
            f"inference_d{dterm_um}_seed{seed}"
            f"_mu{label(mu_true)}_eta{label(eta_true)}_s{label(s_true)}.json"
        )
        command = [
            sys.executable,
            str(study_dir / "run_inference.py"),
            "--ceus",
            str(ceus_path),
            "--reps",
            str(args.reps),
            "--mu",
            str(mu_true),
            "--eta",
            str(eta_true),
            "--s-true",
            str(s_true),
            "--relation-sd",
            str(args.relation_sd),
            "--with-alpha",
            "--offset-sweep",
            "--out",
            str(output),
        ]
        print(f"seed {seed}: mu={mu_true:g} Pa, eta={eta_true:g} Pa s, s={s_true:g}")
        if args.dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, cwd=study_dir, check=True)
