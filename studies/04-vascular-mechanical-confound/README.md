# Vascular-mechanical confounding (v0)

This study tests whether a vascular transport observation can reduce ambiguity
between a change in the tissue matrix and a change in vessel radius. It combines
a reduced shear-wave model with contrast-ultrasound transport calculated on the
vascular networks from [study 01](../01-vascular-network-coupling/README.md).

The mechanical model assigns each vessel segment a relaxation time determined by
its radius and path length, following the microchannel-flow framework
([Parker, 2014](https://doi.org/10.1088/0031-9155/59/15/4443);
[Parker et al., 2016](https://doi.org/10.1088/0031-9155/61/13/4890)).
Uniform constriction changes both this spectrum and the vascular volume fraction,
consistent with the vasoconstriction experiments of
[Poul et al. (2020)](https://doi.org/10.3390/fluids5040228). Matrix elasticity,
matrix viscosity, and vascular constriction can consequently produce similar
shear-wave observations. The CEUS area under the time-intensity curve (AUC)
depends on the radius scale but not on the matrix parameters, giving the joint
model complementary information.

Across three network realizations and 40 noise repetitions per realization, the
coupled model reduced the matrix-elasticity interval width by 17-23% and the
matrix-viscosity interval width by 8-12% relative to SWE alone. Coverage was
maintained for the specified coupling. Displacing the coupling relation produced
increasing bias and loss of coverage. The result is conditional on the reduced
mechanical mapping and the network-specific AUC-to-radius-scale relationship
described in [`METHODS.md`](METHODS.md).

The numerical findings and figures are reported in [`RESULTS.md`](RESULTS.md).

## Installation

Study 01 supplies the vascular-network, flow, and bubble-transport package.
Install both studies from the repository checkout:

```bash
cd ../01-vascular-network-coupling
pip install -e .
cd ../04-vascular-mechanical-confound
pip install -e .
pytest -q
```

Python 3.9 or later is required. Study 01 must be installed in the same Python
environment because it supplies the `porovasc` package.

## Reproducing the main analyses

For each network seed, tabulate the CEUS response and run the inference study:

```bash
python run_ceus.py --dterm 30e-6 --seed 1 --n-s 16 --s-min 0.65 \
  --n-inputs 8 --out results/ceus_d30_seed1.json
python run_inference.py --ceus results/ceus_d30_seed1.json --reps 40 \
  --with-alpha --offset-sweep 0 0.02 0.05 0.10 0.15 0.20 \
  --out results/inference_d30_seed1.json
```

Repeat with seeds 2 and 3. Generate the validity and resolution diagnostics and
figures with:

```bash
python run_diagnostics.py --dterms 60e-6 40e-6 30e-6 \
  --out results/diagnostics.json
python run_figures.py --results results --out figures
```

Each inference file retains the noisy observations and the posterior median,
90% interval, and interval width for every repetition, in addition to the
aggregate summary. After generating the three CEUS tables, inference can be run
over combinations of generating matrix properties and radius scales with:

```bash
python run_true_value_sweep.py \
  --mu-values 1500 2000 2500 \
  --eta-values 0.5 1.0 1.5 \
  --s-values 0.70 0.80 0.90 \
  --reps 40
```

The three lists are combined as a Cartesian product for every network. The
example therefore runs $3\times3\times3\times3=81$ inference analyses. Use
`--dry-run` to inspect the commands first. Exploratory outputs are written to
`results/true_value_sweep/`. Summarize the completed sweep and generate its
precision heatmaps with:

```bash
python run_sweep_figures.py \
  --results results/true_value_sweep \
  --summary results/true_value_sweep_summary.json \
  --out figures
```

## Comparison models

Here, $s$ is the factor multiplying every vessel radius: $s=1$ is the baseline
network and $s<1$ represents uniform constriction.

| Model | Data used | Treatment of $s$ |
|---|---|---|
| `swe` | SWE only | one $s$ enters the mechanical model; CEUS supplies no constraint |
| `ceus` | CEUS only | one $s$ enters the transport model; matrix properties remain unconstrained |
| `joint_free` | SWE and CEUS | independent $s_{\mathrm{SWE}}$ and $s_{\mathrm{CEUS}}$; no information is transferred between modalities |
| `joint_shared` | SWE and CEUS | the same radius scale $s$ links both models, with uncertainty in the relation |
| `joint_wrong` | SWE and CEUS | the linked relation is deliberately displaced to test misspecification |
| `perfect_model` | SWE only | $s$ is fixed at its generating value as a reference case |

## Main contents

| Path | Contents |
|---|---|
| `vmconf/mech.py` | reduced vascular relaxation and shear-wave model |
| `vmconf/ceus.py` | network-based CEUS observable and interpolation |
| `vmconf/inference.py` | grid posterior and comparison models |
| `vmconf/provenance.py` | source, configuration, and network provenance |
| `results/` | CEUS tables, inference summaries, and diagnostics |
| `figures/` | figures used in the result summary |

## Limitations

The vascular contribution to shear is assumed from a reduced microchannel-flow
model rather than derived from a fluid-solid boundary-value problem on the
network. The mechanical model uses the explicit vascular tree, whereas transport
also uses the unresolved distal bed. CEUS inference assumes calibrated input
concentration and acoustic sensitivity. Noise is relative Gaussian, with the
reported comparison using independent errors across frequencies. Quantitative
interval reductions apply to the stated simulation settings. Removing vascular
contributions outside the quasi-static Poiseuille criterion changes predicted
phase speed by up to 2.9% over the evaluated 50-200 Hz band; the criterion follows
the transition analysis of
[Parker (2017)](https://doi.org/10.1088/1361-6560/aa62b2).
