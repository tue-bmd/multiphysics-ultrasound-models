# Computational studies toward multiphysics ultrasound

Multiparametric ultrasound combines measurements of tissue structure,
mechanical response, and vascular transport. Measurements can be linked in two ways: 
they may depend on the same tissue property and they may carry the same acquisition effect. 
Joint parameter estimation requires a model of each form of coupling. 
This repository contains three independent simulation studies addressing these 
two forms of coupling in isolation.

The studies are initial (`v0`) implementations of separate components of a
future integrated framework.

![Framework](docs/framework.svg)

*Proposed framework. The colored boxes indicate the three studies included in
this repository. The dashed arrows indicate interfaces that are not yet
implemented.*

## Studies

| Directory | Aim | Main result |
|---|---|---|
| [`01-vascular-network-coupling`](studies/01-vascular-network-coupling/RESULTS.md) | Determine whether vascular architecture affects mechanical relaxation and contrast transport in the same vascularized tissue model. | Increasing the modeled tortuosity parameter by 16% lengthened vascular-compliance relaxation by approximately 33% and reduced estimated transport drift and dispersion by approximately 22% and 28%, respectively, in five paired prostate-gland realizations. |
| [`02-multiphysics-field-identifiability`](studies/02-multiphysics-field-identifiability/RESULTS.md) | Assess parameter identifiability from idealized mechanical and contrast-transport fields. | Absolute contrast constrained the vascular volume fraction when the input concentration was calibrated. Effective permeability remained unresolved by the tested SWE-CEUS model. A network-informed coupling narrowed its posterior but produced large bias when the relation was misspecified. |
| [`03-shared-operator-identifiability`](studies/03-shared-operator-identifiability/RESULTS.md) | Test whether a shared measurement parameter can reduce operator-tissue confounding. | A calibrated relation between sequence-specific image widths transferred information from a contrast time series to B-mode and substantially reduced uncertainty in the intrinsic lesion-margin width. |

Each study contains a short overview (`README.md`), the implemented formulation
(`METHODS.md`), and the main findings (`RESULTS.md`). Symbols used across studies
are summarized in [`docs/terminology.md`](docs/terminology.md).

## Installation

Each study is a self-contained Python package.

```bash
cd studies/01-vascular-network-coupling          && pip install -e . && pytest -q
cd studies/02-multiphysics-field-identifiability && pip install -e . && pytest -q
cd studies/03-shared-operator-identifiability    && pip install -e . && pytest -q
```

Python 3.9 or later is required. Commands for reproducing the reported analyses
and figures are provided in each study directory.

## Limitations

The three studies are reduced and are not integrated in software. Study 01
simulates an explicit vascular network without a deformable tissue continuum.
Tissue mechanics are represented as lumped parameters.
Study 02 observes idealized physical fields. Study 03 uses reduced image-domain
measurement operators. None implements complete ultrasound signal formation,
including acoustic propagation, beamforming, displacement tracking, and
microbubble acoustics. Quantitative results apply to the stated models and
parameter settings.

## AI statement

Generative AI tools were used to assist with code organization, polishing, and debugging, 
and with structuring and polishing the documentation. All model choices, analyses, reported 
results, code, and final text were reviewed and verified by the author. 

## License and citation

The code is distributed under the BSD 3-Clause license. Citation metadata is
provided in `CITATION.cff`.
