# Multiphysics field identifiability (v0)

This study examines parameter identifiability from idealized mechanical and
contrast-transport fields. It combines a shear-wave model, a
convection-dispersion model, and an auxiliary consolidation-relaxation model
within a Bayesian inverse problem.

The analysis distinguishes three questions:

1. Which parameters are identifiable from each physical field?
2. Does joint analysis reduce uncertainty through shared parameters or coupling
   relations?
3. How sensitive are the conclusions to the parameter regime and to model
   discrepancy?

The principal results are:

- Absolute contrast constrains the vascular volume fraction when the input
  concentration is independently calibrated.
- Effective permeability remains unresolved under the tested SWE-CEUS model.
- A relaxation observation and the relation `S_v = phi C_v` can reduce
  uncertainty in permeability when the required auxiliary quantities are
  constrained.
- A misspecified network-informed relation can produce a narrow but strongly
  biased posterior.

The formulation is given in [`METHODS.md`](METHODS.md). The numerical findings
are summarized in [`RESULTS.md`](RESULTS.md).

## Installation

```bash
pip install -e .
pytest -q
```

## Reproducing the main analyses

```bash
fieldid config         --out configs/default.json
fieldid ensemble-prior --ensemble results/network_ensemble.csv \
                       --out results/network_prior.json
fieldid matrix         --out results --profile k --profile phi --profile S_v
fieldid discrepancy    --out results
fieldid profiles       --out results --profile k --domain-check
fieldid mcmc           --out results --steps 14100
fieldid figures        --out figures --results results
```

The robustness analysis repeats eight inference configurations at nine
parameter settings with five paired noise realizations:

```bash
fieldid truth-sweep --out results/truth_sweep_r5 --n-rep 5
```


## Main contents

| Path | Contents |
|---|---|
| `fieldid/models/` | mechanical and transport models |
| `fieldid/infer/` | identifiability diagnostics and Bayesian inference |
| `configs/` | parameter, prior, and acquisition settings |
| `results/` | numerical outputs and posterior samples |
| `figures/` | figures used in the result summary |

## Relation to study 01

The network-informed prior is estimated from an ensemble generated with the
vascular-network model in study 01. The studies remain separate in code; the
ensemble is imported through `results/network_ensemble.csv`.

## Limitations

The observations are idealized physical fields with additive noise. Ultrasound
signal formation and image reconstruction are not modeled. Parameters are
spatially uniform, dispersion is isotropic, and the relaxation experiment is a
reduced confined-slab model. The effective permeability used here should not be
interpreted as a validated vascular or interstitial tissue coefficient.
