# Vascular-network coupling (v0)

This study examines whether vascular organization affects mechanical relaxation
and contrast transport in the same model. The network-generation framework is
general. The realization studied here represents a prostate gland by its size
and arterial organization: a 22 mL ellipsoid is supplied by four 600 µm
extracapsular feeding arteries representing the superior and inferior arterial
pedicles from each hemipelvis
([García-Mónaco et al., 2014](https://doi.org/10.1016/j.jvir.2013.10.026)).
The selected diameter lies within the 0.5–1.5 mm angiographic range reported by
[Zhang et al. (2015)](https://doi.org/10.1371/journal.pone.0132678).
Pressure-driven flow, intravascular tracer transport, and vascular-compliance
relaxation under localized compression are computed on the same network.

The main analysis uses paired realizations. Changing the modeled tortuosity
parameter from 1.25 to 1.45 lengthened the relaxation time and reduced the fitted
transport drift and dispersion in all five prostate-gland realizations. Separate
changes in matrix modulus and arterial inlet pressure were used as controls.

The implemented equations and analysis are described in
[`METHODS.md`](METHODS.md). Numerical findings and figures are reported in
[`RESULTS.md`](RESULTS.md).

## Installation

```bash
pip install -e .
pytest -q
```

Python 3.9 or later is required. NumPy and SciPy are used for the simulations;
Matplotlib is required for the figures.

## Reproducing the reported analysis

```bash
python -m porovasc.validate runs/coupling
python -m porovasc.analyse_coupling runs/coupling
python -m porovasc.figures --out figures
```

The stored coupling run contains 90 records from five prostate-gland
realizations, two sampling volumes per realization, and nine conditions.

## Main contents

| Path | Contents |
|---|---|
| `porovasc/geometry/` | vascular-network construction |
| `porovasc/physics/` | flow, transport, and compression-relaxation models |
| `porovasc/homogenise/` | network permeability calculation |
| `runs/coupling/` | stored paired simulations |
| `figures/` | figures used in the result summary |

## Limitations

The tissue matrix is rigid and enters the vascular compliance through a single
modulus. The model does not include tissue deformation, interstitial flow, or
acoustic image formation. The venous tree is constructed as an offset copy of
the arterial tree, and the capillary bed is represented by a lumped terminal
element. The study tests shared dependence on vascular organization, not whether 
the associated parameters are jointly identifiable. Identifiability under a 
reduced model of ideal mechanical and contrast-transport fields is examined 
separately in [Study 02](../02-multiphysics-field-identifiability/README.md).

## Citation and license

Citation metadata is provided in `CITATION.cff`. The code is distributed under
the BSD 3-Clause license.
