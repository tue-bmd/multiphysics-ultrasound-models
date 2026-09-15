# Terminology

The table summarizes the main model parameters and observed physical fields used
in the three studies. Intermediate variables are defined where their equations
are introduced.

| Symbol | Quantity | Unit | Study |
|---|---|---|---|
| $\phi$ | vascular volume fraction | 1 | 01, 02 |
| $D$ | effective transport-dispersion coefficient | m²/s | 01, 02, 03 |
| $\mathbf{v}$ | effective transport drift | m/s | 01, 02, 03 |
| $c$ | intravascular microbubble concentration | arbitrary concentration unit | 02 |
| $b=\phi c$ | microbubble concentration per unit tissue volume; simulated contrast field | arbitrary | 02 |
| $\mathbf{u}$ | solid-displacement field; simulated mechanical observation | m | 02 |
| $\mathbf{F}_m$ | ideal physical field for modality $m$, before measurement-operator effects | modality-dependent | 03 |
| $\mathbf{y}_m$ | simulated observation for modality $m$ | modality-dependent | 03 |
| $\mu$ | shear modulus | Pa | 02 |
| $\eta_s$ | solid viscosity | Pa·s | 02 |
| $k_v$ | vascular-network permeability | m² | 01 |
| $k$ | effective permeability in the reduced field model | m² | 02 |
| $A$ | microbubble input-concentration amplitude | arbitrary | 02 |
| $C_v$ | fractional vascular compliance | Pa⁻¹ | 01, 02 |
| $S_v$ | vascular storage | Pa⁻¹ | 02 |
| $L$ | drainage length | m | 02 |
| $\sigma_e$ | intrinsic lesion-margin width | m | 03 |
| $w_0$ | shared aperture-limited image width | m | 03 |

## Study-specific definitions

In study 01, $\phi$ is calculated from the lumen volume of the simulated
network. In study 02, it is an inferred parameter of the analytic contrast
model.

In study 01, $D$ and $\mathbf{v}$ are effective descriptors obtained by
spatiotemporal analysis of the simulated contrast field. In studies 02 and 03,
they are coefficients of an assumed convection-dispersion model. Study 01
demonstrates that the recovered descriptors depend on the estimator and analysis
settings. No quantitative mapping between the network-derived descriptors and
the continuum coefficients is assumed.

Study 01 calculates a vascular-network permeability $k_v$ by imposing a pressure
difference across a sampling volume. Study 02 uses $k$ as the effective
permeability entering its reduced mechanical model. The network-informed prior
in study 02 tests a possible statistical relation between vascular properties;
it does not define a deterministic tissue law.

In study 01, $C_v$ is prescribed through the vessel-compliance relation. In
study 02, $C_v$ may be inferred or independently constrained, and
$S_v=\phi C_v$ is evaluated as a candidate coupling relation.

## Inference measures

**Structural identifiability** describes whether distinct parameters can produce
the same noise-free observations under the model.

**Posterior contraction** is the ratio of the posterior 90% interval width to
the corresponding prior interval width. A value of one indicates no reduction
in uncertainty.

**Coverage** is the fraction of repeated simulations in which a credible
interval contains the generating parameter value.

Effects in study 01 are reported as base-10 logarithmic ratios for positive
quantities. Positive parameters in studies 02 and 03 are represented in natural
logarithmic coordinates during inference.
