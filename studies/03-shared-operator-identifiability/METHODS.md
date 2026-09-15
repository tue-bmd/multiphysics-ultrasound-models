# Methods

The study isolates coupling through the measurement operator. Tissue parameters
are separate for the B-mode, shear-wave, and contrast observations.

## 1. Formulation

For observation $m$,

$$
\mathbf y_m =\mathcal H_m\!\left[ \mathbf F_m(\boldsymbol\theta_m); \boldsymbol\psi_{\mathrm{shared}}, \boldsymbol\psi_m \right] +\boldsymbol\varepsilon_m,
$$

where $\mathbf F_m$ is an ideal tissue or transport field,
$\boldsymbol\theta_m$ contains the corresponding tissue parameters,
$\boldsymbol\psi_{\mathrm{shared}}$ is a shared operator parameter,
$\boldsymbol\psi_m$ contains sequence-specific corrections, and
$\boldsymbol\varepsilon_m$ is independent Gaussian noise.

No tissue parameter is shared across the three observations. Differences
between the comparison models therefore arise from the operator relation.

## 2. Shared image-width model

All sequences are represented by one aperture-limited width $w_0$. Their
effective image widths are

$$
w_B=w_0, \qquad w_C=\gamma_C\frac{f_B}{f_C}w_0, \qquad w_S=\sqrt{ \left(\gamma_S\frac{f_B}{f_S}w_0\right)^2+w_{\mathrm{track}}^2 }.
$$

The transmit frequencies are 7.0 MHz for B-mode, 5.0 MHz for shear-wave
tracking, and 2.8 MHz for contrast imaging. The tracking-kernel width is
0.40 mm. The factors $\gamma_C$ and $\gamma_S$ represent
sequence-specific deviations from nominal frequency scaling.

## 3. Observation models

### 3.1 B-mode

The intrinsic echogenicity profile contains a circular lesion of radius $R$ and
margin width $\sigma_e$. Gaussian convolution with width $w_B$ gives an
observed margin proportional to

$$
\sqrt{\sigma_e^2+w_B^2}.
$$

Consequently, $\sigma_e$ and $w_B$ are structurally unidentifiable from the
B-mode profile alone.

### 3.2 Contrast ultrasound imaging

The contrast field is a Gaussian solution of the convection-dispersion
equation, convolved with width $w_C$. Its spatial variance is

$$
\operatorname{var}_{CEUS}(t)=2D(t-t_0)+w_C^2.
$$

The time-dependent term separates $D$ from $w_C$ when multiple frames are
available. A single frame does not provide this separation.

### 3.3 Shear-wave imaging

The shear observation is a broadband pulse that propagates across the region
and attenuates with distance. Spatial convolution with $w_S$ acts as a fixed
wavenumber filter, whereas attenuation varies with propagation distance.

## 4. Comparison models

| Model | Operator parameters |
|---|---|
| independent | $w_B$, $w_C$, and $w_S$ are unrelated |
| shared | $w_0$ is estimated; $\gamma_C$ and $\gamma_S$ are fixed |
| shared calibrated | $w_0$, $\gamma_C$, and $\gamma_S$ are estimated with calibration priors |
| perfect | all image widths are fixed at their generating values |

The marginal priors for $w_B$, $w_C$, and $w_S$ in the independent model match
those induced by the shared model. All comparison models receive the same
synthetic observations and noise levels.

The following controls are used: removing the contrast time series, replacing
it with one contrast frame, removing the shear observation, introducing a
15% error in $\gamma_C$, and adding uncertainty in the reconstructed-length
scale.

## 5. Inference

Structural identifiability is evaluated from the singular values and null
directions of the noise-scaled Jacobian. Profile likelihoods examine the
intrinsic margin width before and after including calibration information.

Posterior modes and local covariance are obtained with nonlinear least squares
and a Laplace approximation. The principal comparison is also sampled with
adaptive Metropolis. Posterior contraction is the ratio of the posterior 90%
interval width to the corresponding prior width. Coverage is evaluated across
repeated noise realizations.

## 6. Limitations

All observations are defined on one reconstructed image plane with one fixed
probe position. The point-spread function is spatially invariant and represented
by one Gaussian width per sequence. The model omits acoustic propagation,
scattering, speckle, envelope detection, displacement tracking, and microbubble
acoustics. Noise is independent across samples. Quantitative contraction and
coverage therefore apply only to this reduced observation model.

## 7. Reproduction

```bash
pip install -e .
pytest -q
operatorid config    --out configs/default.json
operatorid screening --out results
operatorid matrix    --out results --profile sig_e
operatorid coverage  --out results --n-rep 40
operatorid mcmc      --out results --steps 60000
operatorid figures   --out figures --results results
```
