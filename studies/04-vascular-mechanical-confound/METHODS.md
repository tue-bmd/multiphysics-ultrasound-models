# Methods

The study asks whether a transport observation can reduce confounding between
matrix viscoelasticity and vascular constriction in a reduced mechanical model.
The vascular geometry and contrast transport are supplied by
[study 01](../01-vascular-network-coupling/METHODS.md).

## 1. Vascular network and sampling region

Each realization is a study-01 prostate vascular network with four feeders and
explicit terminal diameter 30 µm. One 6 mm cubic sampling region is centered at
$(0,-5,0)$ mm. Segment intersections with the cube are calculated by the same
partial-volume clipping operation used by the transport model.

The uniform radius scale $s$ multiplies every explicit lumen radius, with $s=1$
denoting baseline and $s<1$ denoting constriction. Let $r_{i,0}$ and $\phi_0$
denote the radius of segment $i$ and the vascular volume fraction at baseline
($s=1$), respectively. Segment endpoints, path lengths, parentage, and topology
remain unchanged. The radius and vascular volume fraction therefore scale as

$$
r_i(s)=s\,r_{i,0},\qquad \phi(s)=s^2\phi_0.
$$

The transport calculation constricts the explicit vessels while retaining the
baseline conductance of the unresolved distal bed.

## 2. Matrix and vascular mechanical response

The matrix is represented by a Kelvin-Voigt complex shear modulus, one of the
classical linear viscoelastic models summarized by
[Parker et al. (2019)](https://doi.org/10.1088/1361-6560/ab453d2):

$$
G_m^*(\omega)=\mu+i\omega\eta,
$$

where $\mu$ is the matrix shear modulus and $\eta$ is the matrix shear viscosity.

The vascular response follows the reduced microchannel-flow construction of
[Parker (2014)](https://doi.org/10.1088/0031-9155/59/15/4443) and its
frequency-domain extension for shear-wave dispersion
([Parker et al., 2016](https://doi.org/10.1088/0031-9155/61/13/4890)). The
baseline relaxation time of segment $i$, evaluated at $s=1$, is denoted by
$\tau_{i,0}$:

$$
\tau_{i,0}=\frac{\eta_b L_i^2}{\mu r_{i,0}^2},
$$

where $\eta_b$ is blood viscosity, $L_i$ is path length, and $r_{i,0}$ is the
baseline lumen radius. Lumen-volume fractions within the sampling region define
normalized weights $w_i$.

During constriction, vascular resistance scales as $r^{-4}$ while the compliance
is assigned to the surrounding tissue element, whose volume is fixed. Thus,

$$
\tau_i(s)=\tau_{i,0}s^{-4}.
$$

The vascular contribution is a discrete Maxwell sum,

$$
G_v^*(\omega)=\Delta G\sum_i w_i
\frac{i\omega\tau_i}{1+i\omega\tau_i}.
$$

The reference mixture is parallel addition,

$$
G^*(\omega)=G_m^*(\omega)+G_v^*(\omega).
$$

The amplitude $\Delta G$ is set by `visc_share`, the vascular fraction of the
total loss modulus at 200 Hz in the baseline configuration. It is held fixed during the
reference constriction experiment. A lumen-volume amplitude law and a series
mixture are implemented as sensitivity models.

This mechanical mapping is an assumption informed by the microchannel-flow
model and the vasoconstriction experiment of
[Poul et al. (2020)](https://doi.org/10.3390/fluids5040228). It is not obtained
from a deformation or fluid-solid coupling solve on the simulated network.

## 3. Shear-wave observation

For density $\rho=1050\ \mathrm{kg\,m^{-3}}$, the complex-wavenumber convention follows
the viscoelastic shear-wave relations summarized by
[Parker et al. (2019)](https://doi.org/10.1088/1361-6560/ab453d2):

$$
k^*(\omega)=\omega\sqrt{\frac{\rho}{G^*(\omega)}}.
$$

Under the convention $u=\exp[i(\omega t-kx)]$, phase velocity and attenuation
are

$$
c_p=\frac{\omega}{\operatorname{Re}k^*},\qquad
\alpha=-\operatorname{Im}k^*.
$$

The reported inference uses 24 logarithmically spaced frequencies from 50 to
200 Hz and includes both phase velocity and attenuation. Independent relative
Gaussian noise with standard deviation 3% is applied to each value.

## 4. Contrast-ultrasound observation

[Study 01](../01-vascular-network-coupling/METHODS.md) supplies the flow and
frequency-domain bubble-transport calculation. At radius scale $s$, the
transport model returns a contrast-concentration curve $C_j(t;s)$ at each
input-voxel location $j$. A
standard AUC is first calculated separately for every curve,

$$
\mathrm{AUC}_j(s)=\int C_j(t;s)\,dt.
$$

The AUC is a standard signal-related CEUS parameter associated primarily with
blood volume ([Dietrich et al., 2024](https://doi.org/10.1055/a-2157-2587)). Because
the simulated input bolus has unit time integral, each AUC is proportional to
the vascular volume sampled during the bolus passage. Curves are propagated at
eight nearby input-voxel locations, and their median AUC is used as the single
noise-free simulated CEUS value:

$$
f_{\mathrm{CEUS}}(s)=\operatorname{median}_{j=1,\ldots,8}
\left[\mathrm{AUC}_j(s)\right].
$$

This aggregation reduces sensitivity to the precise voxel location. The
absolute-amplitude observation is informative about the radius scale only when
input concentration and acoustic sensitivity are calibrated. Only AUC is used
as the CEUS observation; the curve shape, arrival time, velocity, and dispersion
are not fitted. The response $f_{\mathrm{CEUS}}(s)$ is tabulated for
$0.65\leq s\leq1.05$ and interpolated in log amplitude. Relative Gaussian noise
has standard deviation 10%.

## 5. Inference models

In the nominal analysis, each inference repetition uses two synthetic datasets generated at
$\mu=2$ kPa, $\eta=1$ Pa s, and $s=0.8$. The SWE dataset is a vector of 48
values: phase velocity and attenuation at 24 frequencies. Its components are

$$
y_{\mathrm{SWE},m}
=f_{\mathrm{SWE},m}(\boldsymbol\theta_0)(1+\sigma_{\mathrm{SWE}}z_m),
\qquad z_m\sim\mathcal N(0,1),\quad m=1,\ldots,48,
$$

where $\boldsymbol\theta_0=(\mu_0,\eta_0,s_0)$ contains the generating
parameters, $f_{\mathrm{SWE},m}(\boldsymbol\theta_0)$ is the $m$th noise-free
simulated SWE value, and $y_{\mathrm{SWE},m}$ is its noisy value. The CEUS
dataset contains one noisy AUC:

$$
y_{\mathrm{CEUS}}
=f_{\mathrm{CEUS}}(s_0)(1+\sigma_{\mathrm{CEUS}}z_C),
\qquad z_C\sim\mathcal N(0,1).
$$

Thus $\mathbf y_{\mathrm{SWE}}$ denotes the 48-value SWE dataset and
$y_{\mathrm{CEUS}}$ denotes the single CEUS datum. All normal draws are
independent.

The quantities inferred are the matrix shear modulus $\mu$, matrix shear
viscosity $\eta$, and radius scale $s$. Here $f_{\mathrm{CEUS}}(s)$ is the
noise-free simulated AUC and $y_{\mathrm{CEUS}}$ is the corresponding synthetic
measured value after noise is added. For every candidate value of $s$, the CEUS
lookup table gives $f_{\mathrm{CEUS}}(s)$; inference compares it with
$y_{\mathrm{CEUS}}$. The radius scale is included because it changes the SWE
response and can therefore be confused with changes in $\mu$ and $\eta$. CEUS
constrains this nuisance parameter through the lookup table, allowing the
matrix parameters to be estimated more precisely.

Consequently, any inferred value or interval for $s$ is conditional on assuming
that the calibrated, network-specific relationship between AUC and $s$ is known
and applicable, up to the relation uncertainty described below. It is not a
direct measurement of vessel radius. The main results therefore focus on $\mu$
and $\eta$; results involving $s$ are used only to diagnose the assumed
cross-modality relationship.

The relative noise levels are fixed simulation assumptions rather than values
estimated from a particular acquisition. We set $\sigma_{\mathrm{SWE}}=0.03$
to represent the lower end of controlled shear-wave-speed repeatability;
phantom coefficients of variation of 0.5-6.8% have been reported
([Dillman et al., 2015](https://doi.org/10.1007/s00247-014-3150-6)). This does
not validate 3% independent errors for frequency-resolved phase velocity and
attenuation. We set $\sigma_{\mathrm{CEUS}}=0.10$ as an idealized calibrated-
amplitude case. It is optimistic relative to a QIBA phantom study, which found
AUC coefficients of variation as high as 50% even with one scanner and analysis
package ([Averkiou et al., 2020](https://doi.org/10.1097/RLI.0000000000000702)).
The reported uncertainty reductions are therefore conditional on these assumed
noise levels.

Fitting is performed by testing candidate values of $(\mu,\eta,s)$ on a regular
$61\times61\times61$ grid; no optimizer or Markov chain is used. At every grid
point $\boldsymbol\theta=(\mu,\eta,s)$, the mechanical model evaluates
$\mathbf f_{\mathrm{SWE}}(\boldsymbol\theta)$, the simulated SWE values for
that candidate parameter combination, and the CEUS lookup table supplies
$f_{\mathrm{CEUS}}(s)$ by interpolation in log amplitude. The SWE
likelihood compares all 48 candidate and observed SWE values, whereas the CEUS
likelihood compares the candidate and observed AUC. In each case the residual
is divided by the corresponding assumed relative standard deviation.

The observations compared and parameters inferred in each model are:

| Model | Observations compared | Parameters inferred |
|---|---|---|
| `swe` | 48 SWE values | $\mu$, $\eta$, and nuisance parameter $s$ |
| `ceus` | one AUC | $s$; $\mu$ and $\eta$ remain at their priors |
| `joint_free` | 48 SWE values and one AUC, with separate radius scales | $\mu$, $\eta$, and $s_{\mathrm{SWE}}$; the independent CEUS scale is integrated out and cannot update the mechanical parameters |
| `joint_shared` | 48 SWE values and one AUC, linked by the same $s$ | $\mu$, $\eta$, and $s$ |
| `joint_wrong` | as above, but the AUC is evaluated at $s+\delta$ | $\mu$, $\eta$, and $s$ under a deliberately displaced relationship |
| `perfect_model` | 48 SWE values, with $s=s_0$ fixed | $\mu$ and $\eta$ |

Uniform priors are represented by the grid bounds: $0.5$-to-$1.6$ times the
generating $\mu$, $0.3$-to-$2.2$ times the generating $\eta$, and the tabulated
CEUS range for $s$. For `joint_shared`, relation uncertainty is applied by
convolving the CEUS likelihood along $s$ with a Gaussian of standard deviation
0.02 before multiplying it by the SWE likelihood. This is equivalent to
allowing the radius scale linked to the CEUS model to differ from the radius
scale in the mechanical model with a standard deviation of 0.02. The vascular
contribution is 20% of the baseline loss modulus at 200 Hz.

The SWE and CEUS likelihoods are multiplied because their measurement errors
are assumed conditionally independent given $(\mu,\eta,s)$. Equivalently,
$p(\mathbf y_{\mathrm{SWE}},y_{\mathrm{CEUS}}\mid\mu,\eta,s)$ is assumed to
factor into the two modality-specific likelihoods. This excludes shared
calibration errors and other cross-modality error correlations.

The posterior is normalized by dividing each unnormalized posterior value by
the sum of all unnormalized posterior values over the grid. Marginal
distributions are obtained by summing over the other parameters; their medians
and central 90% intervals are reported. Forty paired noise realizations are used
for each network seed, with the same noisy dataset supplied to every comparison
model. Bias, mean interval width, and empirical coverage are reported. Binomial
standard errors describe the Monte Carlo uncertainty of coverage. The noisy
observations and the posterior median, interval bounds, and interval width from
every repetition are retained in the result files before aggregate summaries
are calculated.

Robustness to the generating parameters is evaluated on the Cartesian product
of $\mu_0\in\{1.5,2.0,2.5\}$ kPa,
$\eta_0\in\{0.5,1.0,1.5\}$ Pa s, and
$s_0\in\{0.7,0.8,0.9\}$. The 27 combinations are evaluated on each of the
three networks with 40 paired noise repetitions. For each network and parameter
combination, precision gain is calculated as

$$
100\,\frac{W_{\mathrm{SWE}}-W_{\mathrm{joint}}}{W_{\mathrm{SWE}}},
$$

where $W$ is the mean 90% posterior-interval width over the 40 repetitions.
The values shown for each parameter combination are means of the percentage
reductions calculated separately for the three networks. At every combination,
the grid limits for $\mu$ and $\eta$ are rescaled using the same multipliers of
their generating values as in the nominal analysis. The amplitude $\Delta G$ is
also recalculated so that the vascular contribution remains 20% of the baseline
loss modulus at 200 Hz.

## 6. Validity and resolution diagnostics

The reduced mechanical model uses the Poiseuille resistance scaling
$R\propto r^{-4}$. This assumes that viscous momentum has enough time to diffuse
across the vessel radius during each oscillation, so the velocity profile can be
treated as quasi-static and approximately parabolic. As frequency or vessel
radius increases, the profile cannot adjust within one cycle and unsteady
Womersley effects become important. The relaxation model used here does not
represent those effects.

The quasi-static approximation is screened using
$\omega<7\nu/r^2$, where $\nu$ is the kinematic viscosity of blood, following
the transition from Poiseuille to dynamic Womersley flow analyzed by
[Parker (2017)](https://doi.org/10.1088/1361-6560/aa62b2). Because the validity
limit depends on both radius and frequency, a segment may satisfy it at the low
end of the SWE band but not at the high end. The sensitivity calculation does
not replace the invalid contribution with a Womersley model. Instead, it removes
the vascular amplitude carried by segments outside the quasi-static range,
without redistributing that amplitude among the retained segments. This gives a
conservative measure of how strongly those segments affect the simulated phase
velocity. At 30 µm resolution, their removal changed phase velocity by 0.76% at
100 Hz, 2.20% at 158 Hz, and 2.88% at 200 Hz.

Resolution sensitivity is evaluated at terminal diameters of 60, 40, and 30 µm.
Results are reported under both a fixed vascular share and a fixed dimensionless
coupling calibrated on the coarsest network. Fixing the vascular share
recalculates the amplitude at every resolution and therefore absorbs some of the
resolution dependence by construction. Fixing the coupling leaves that
dependence visible. From 60 to 30 µm, the vascular volume fraction increased by
29.4%; phase velocity at 100 Hz changed by 0.04% with fixed share and by 1.89%
with fixed coupling.

## 7. Limitations

The network was developed for transport rather than mechanics. The model omits
a resolved tissue deformation field, fluid-solid boundary conditions, acoustic
propagation, displacement tracking, and microbubble acoustics. The mechanical
calculation excludes blood volume in the unresolved distal bed. Uniform radius
scaling is an idealization of a spatially heterogeneous vascular response. The
independent-frequency noise model may overstate information when errors arise
from a common estimator or acquisition.

## 8. Reproduction

```bash
pip install -e ../01-vascular-network-coupling
pip install -e .
pytest -q
python run_ceus.py --dterm 30e-6 --seed 1 --n-s 16 --s-min 0.65 --n-inputs 8 \
  --out results/ceus_d30_seed1.json
python run_inference.py --ceus results/ceus_d30_seed1.json --reps 40 \
  --with-alpha --offset-sweep 0 0.02 0.05 0.10 0.15 0.20 \
  --out results/inference_d30_seed1.json
python run_true_value_sweep.py \
  --mu-values 1500 2000 2500 \
  --eta-values 0.5 1.0 1.5 \
  --s-values 0.70 0.80 0.90 --reps 40
python run_diagnostics.py --dterms 60e-6 40e-6 30e-6 \
  --out results/diagnostics.json
python run_figures.py --results results --out figures
python run_sweep_figures.py --results results/true_value_sweep \
  --summary results/true_value_sweep_summary.json --out figures
```
