# Shared-operator identifiability (v0)

This study examines whether information about a shared measurement effect can
reduce its confounding with a tissue parameter. It uses reduced image-domain
models for B-mode, shear-wave, and contrast observations of the same region.

The B-mode profile depends on intrinsic lesion-margin width and image blur
through their quadrature sum. These quantities are therefore structurally
unidentifiable from B-mode alone. A contrast time series separates transport
dispersion from its image blur. A calibrated relation between the
sequence-specific image widths then transfers this information to the B-mode
model.

In the principal simulation, sharing the operator reduced the sampled interval
for lesion-margin width by approximately one order of magnitude. A deliberately
misspecified relation produced a narrow but biased posterior, showing the need
to estimate or calibrate sequence-specific corrections.

The formulation is given in [`METHODS.md`](METHODS.md). The numerical findings
are summarized in [`RESULTS.md`](RESULTS.md).

## Installation

```bash
pip install -e .
pytest -q
```

## Reproducing the main analyses

```bash
operatorid config    --out configs/default.json
operatorid screening --out results
operatorid matrix    --out results --profile sig_e
operatorid coverage  --out results --n-rep 40
operatorid mcmc      --out results --steps 60000
operatorid figures   --out figures --results results
```

## Comparison models

| Model | Image-width relation |
|---|---|
| independent | separate width for each observation |
| shared | one aperture component with fixed sequence corrections |
| shared calibrated | one aperture component with estimated sequence corrections |
| perfect | generating operator supplied as a reference |

The tissue parameters of the three observations are disjoint. The comparison
therefore isolates information transfer through the measurement operator.

## Main contents

| Path | Contents |
|---|---|
| `operatorid/models/` | reduced B-mode, shear-wave, and contrast models |
| `operatorid/infer/` | identifiability diagnostics and Bayesian inference |
| `configs/` | parameter, prior, and acquisition settings |
| `results/` | numerical outputs and posterior samples |
| `figures/` | figures used in the result summary |

## Limitations

The measurement operator is a Gaussian spatial convolution. Acoustic
propagation, speckle, displacement tracking, microbubble acoustics, and
depth-dependent point-spread functions are not included. Noise samples are
independent, and the numerical posterior contraction depends on this
assumption. The study establishes a mechanism within the reduced model, not the
accuracy of a shared operator for experimental ultrasound data.
