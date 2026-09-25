"""Generate the figures used in RESULTS.md from stored result files."""
import argparse

from vmconf.figures import make_all

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results")
parser.add_argument("--out", default="figures")
args = parser.parse_args()

make_all(args.results, args.out)

