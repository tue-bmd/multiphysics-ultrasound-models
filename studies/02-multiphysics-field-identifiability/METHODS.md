# Methods

This study examines joint parameter identifiability in locally homogeneous,
effectively isotropic vascularized tissue. The forward models predict physical
fields rather than ultrasound signals. All calculations use SI units.

## Principal symbols

| Symbol | Meaning | Unit |
|---|---|---|
| $\mu$ | shear modulus | Pa |
| $\eta_s$ | shear viscosity | Pa s |
| $\phi$ | vascular volume fraction | dimensionless |
| $k$ | effective hydraulic permeability in the poroelastic model | m$^2$ |
| $D$ | effective isotropic microbubble-dispersion coefficient | m$^2$ s$^{-1}$ |
| $\mathbf v$ | effective microbubble-transport drift | m s$^{-1}$ |
| $S_v$ | vascular storage per unit tissue volume | Pa$^{-1}$ |
| $C_v$ | fractional vascular compliance | Pa$^{-1}$ |
| $M$ | effective constrained modulus | Pa |
| $F_0,T_p$ | shear-force amplitude and duration | N m$^{-1}$, s |
| $A,t_0$ | microbubble input concentration amplitude and arrival time | arbitrary units, s |
| $P,L$ | applied compressive stress and slab thickness | Pa, m |

The bulk tissue density $\rho$, blood density $\rho_f$, blood viscosity
$\eta_b$, Biot coefficient $\alpha$, and inertial tortuosity $\alpha_\infty$
are fixed. A hat denotes a temporal Fourier transform and $\omega$ is angular
frequency.

## 1. Mechanical model

The tissue is represented as a poro-viscoelastic medium following Biot-type
theory ([Biot, 1956](https://doi.org/10.1121/1.1908239)), including the
relative-fluid inertia used in poroelastic shear-wave models
([Aichele and Catheline, 2021](https://doi.org/10.3389/fphy.2021.697990)):

$$
\nabla\cdot\hat{\boldsymbol\sigma} =-\omega^2(\rho\hat{\mathbf u}+\rho_f\hat{\mathbf w}), \qquad -\nabla\hat p =-\omega^2(\rho_f\hat{\mathbf u}+\tilde\rho\hat{\mathbf w}),
$$

$$
\hat{\boldsymbol\sigma} =2G^*(\omega)\hat{\boldsymbol\varepsilon} +\lambda(\omega)(\nabla\cdot\hat{\mathbf u})\mathbf I -\alpha\hat p\mathbf I, \qquad \hat\zeta=\alpha\nabla\cdot\hat{\mathbf u}+S_v\hat p,
$$

$$
G^*(\omega)=\mu+i\omega\eta_s, \qquad \tilde\rho(\omega)=\frac{\alpha_\infty\rho_f}{\phi} -\frac{i\eta_b}{\omega k}.
$$

Here, $\mathbf u$ is solid displacement, $\mathbf w$ is fluid displacement
relative to the solid, $p$ is the mechanically induced pressure perturbation,
$\boldsymbol\sigma$ is total stress, $\boldsymbol\varepsilon$ is strain,
$\zeta$ is the change in fluid content, $\lambda$ is the first Lamé parameter,
and $\mathbf I$ is the identity tensor. $G^*$ is the complex shear modulus and
$\tilde\rho$ is the frequency-dependent fluid-inertia term. The relative flux
$i\omega\mathbf w$ belongs only to the mechanical experiment; it is not used
as the baseline microbubble drift in the contrast model. This assumes that the
two fields are probed sequentially rather than simultaneously.

### 1.1 Shear-wave observation

The divergence-free displacement satisfies

$$
G^*(\omega)\nabla^2\hat{\mathbf u}_S +\rho_{\mathrm{eff}}(\omega)\omega^2\hat{\mathbf u}_S =-\hat{\mathbf f}_S, \qquad \rho_{\mathrm{eff}}=\rho-\frac{\rho_f^2}{\tilde\rho},
$$

where $\mathbf f_S$ is the shear forcing and $\rho_{\mathrm{eff}}$ is the
effective dynamic density. For a line source, the cylindrical Green function
is

$$
\hat u_S(x,\omega) =\hat F(\omega)\frac{i}{4G^*(\omega)}H_0^{(2)}(k_sx), \qquad k_s=\omega\sqrt{\frac{\rho_{\mathrm{eff}}}{G^*}}, \qquad \hat F(\omega)=F_0\operatorname{sinc}\!\left(\frac{\omega T_p}{2}\right).
$$

$H_0^{(2)}$ is a Hankel function and $k_s$ is the complex shear wavenumber.
Displacement is sampled at 21 lateral positions from 2 to 12 mm, at 8 kHz for
20 ms. A smooth 40--800 Hz band-pass response is applied to the prediction and
synthetic observation. Boundaries, reflections, and the finite lateral width of
the acoustic-radiation-force push are omitted from the reduced inversion model.

### 1.2 Consolidation-relaxation observation

The relaxation model follows one-dimensional consolidation
([Biot, 1941](https://doi.org/10.1063/1.1712886)). A slab of thickness $L$ is
drained at $z=0$ and sealed at $z=L$, and a compressive stress $P$ is applied at
$t=0$:

$$
\frac{\partial p}{\partial t} =c_v\frac{\partial^2p}{\partial z^2}, \qquad c_v=\frac{k}{\eta_b S_{\mathrm{tot}}}, \qquad S_{\mathrm{tot}}=S_v+\frac{\alpha^2}{M}.
$$

Here, $c_v$ is the consolidation coefficient and $S_{\mathrm{tot}}$ is total
storage. $M$ is treated as an independent effective constrained modulus. The
initial and boundary conditions are

$$
p(z,0)=\frac{\alpha P}{MS_v+\alpha^2}, \qquad p(0,t)=0, \qquad \left.\frac{\partial p}{\partial z}\right|_{z=L}=0.
$$

The observed surface compaction is

$$
u(t)=\frac{-PL+\alpha\int_0^L p(z,t)\,dz}{M}.
$$

Relaxation is sampled at 20 Hz for 30 s. In the principal comparisons, $P$ is
an unknown nuisance parameter and $L$ is assumed to be known within 10% from 
an independent geometric measurement. This idealized observation is evaluated 
as a possible addition when the current shear and contrast observations leave 
an unresolved parameter direction.

## 2. Contrast-transport model

Microbubble transport is represented by a three-dimensional
convection–dispersion equation, consistent with established models of
contrast-agent kinetics in contrast-enhanced ultrasound
([Turco et al., 2020](https://doi.org/10.1016/j.ultrasmedbio.2019.11.008)):

$$
\frac{\partial(\phi c)}{\partial t} +\nabla\cdot\left(\phi c\mathbf v-\phi D\nabla c\right) =\phi A\,\delta(\mathbf r-\mathbf r_s)\delta(t-t_0),
$$

where $c$ is intravascular microbubble concentration, $\mathbf r_s$ is the
source position, and $\delta$ is the Dirac delta. The predicted contrast
concentration per unit tissue volume, $b=\phi c$, is

$$
b(\mathbf r,t) =\phi A[4\pi D\tau]^{-3/2} \exp\!\left[-\frac{|\mathbf r-\mathbf r_s-\mathbf v\tau|^2}{4D\tau}\right], \qquad \tau=t-t_0>0.
$$

The contrast field is sampled on a $3\times3\times3$ voxel grid at 2 Hz for
90 s. Dividing each curve by its temporal integral is included as a control. It
removes the amplitude factor $\phi A$ while retaining the timing and spatial
shape of the curve.

## 3. Parameter coupling

Three models specify how parameters are shared between the mechanical and
contrast observations.

| Model | Definition | Purpose |
|---|---|---|
| independent | $S_v$ and $\phi$ are unrelated parameters | control with no transfer of information between mechanical storage and contrast concentration |
| constitutive | $S_v=\phi C_v$ | links storage per unit tissue volume to vascular volume fraction and fractional vascular compliance |
| network-informed | $S_v=\phi C_v$, with a joint prior on $\phi$, $k$, $D$, and $\|\mathbf v\|$ | additionally represents their shared dependence on vascular organization |

The constitutive model treats $S_v=\phi C_v$ as a candidate reduced relation:
vascular storage per unit tissue volume is the vascular volume fraction
multiplied by compliance per unit vascular volume. It therefore allows an
absolute contrast measurement that informs $\phi$ to constrain mechanical
storage, provided $C_v$ is known sufficiently well. The network-informed model
adds correlations among $\phi$, $k$, $D$, and $|\mathbf v|$ estimated from six
vascular-network realizations in Study 01. This prior represents dependence on
vascular geometry that is not explicit in the reduced field equations; it does
not impose a Darcy relation between baseline contrast drift and $k$.

For a fair comparison, the independent control assigns $S_v$ the same marginal
prior induced by the product $\phi C_v$ in the constitutive model. This
**matched storage prior** is not an independent measurement: it prevents a
difference in prior width from being mistaken for an effect of parameter
coupling. The network-informed prior is implemented as a Gaussian copula, which
preserves the specified marginal priors while introducing correlations among
the logarithms of the four network-related parameters.

## 4. Observations and noise

| Observation | Sampling | Noise standard deviation |
|---|---|---|
| shear displacement | 21 positions, 8 kHz, 20 ms | 0.2 µm |
| contrast concentration | 27 voxels, 2 Hz, 90 s | 2% of peak concentration |
| surface relaxation | 20 Hz, 30 s | 1 µm |

Each synthetic observation is modeled as

$$
\mathbf y=\mathcal F(\boldsymbol\theta)+\boldsymbol\varepsilon, \qquad \boldsymbol\varepsilon\sim\mathcal N(\mathbf 0,\boldsymbol\Sigma),
$$

where $\mathcal F$ is the relevant forward model, $\boldsymbol\theta$ is its
parameter vector, and $\boldsymbol\Sigma$ contains fixed noise variances. The
likelihood is therefore assumed Gaussian, with known variance. The same
standardized noise draw is used in paired comparisons.

## 5. Identifiability and inference

Local structural identifiability is assessed from singular values and null
directions of the noise-scaled Jacobian: an exact null direction is a parameter
combination that leaves the ideal observations unchanged. Practical
identifiability is examined after including measurement noise and prior
information, using prior-scaled Fisher information and parameter profiles
([Raue et al., 2009](https://doi.org/10.1093/bioinformatics/btp358)).

Positive parameters are represented in natural-logarithmic coordinates to
enforce positivity and express changes on a relative scale. All coordinates are
then standardized by their prior widths to place parameters with different
units and ranges on comparable numerical scales.

A parameter profile fixes one parameter successively across a grid and
re-optimizes all remaining parameters at every point. It shows whether a change
in that parameter can be compensated by changes elsewhere. Profiles are
computed at three information levels:

1. **imaging:** the observation likelihood only;
2. **imaging plus auxiliary information:** the likelihood plus independent
   constraints on selected quantities and, where required for comparison, the
   matched storage prior;
3. **posterior:** the likelihood plus all priors.

Independent auxiliary constraints represent measurements or prior knowledge
obtained outside the modeled imaging observations. Depending on the analysis,
these constrain the microbubble input amplitude $A$, slab thickness $L$, or
fractional vascular compliance $C_v$. The applied stress $P$ remains a nuisance
parameter in the principal relaxation comparisons and is constrained only in a
separate control.

The posterior mode is the parameter combination that maximizes the posterior
density; it is obtained here by nonlinear least squares. Local posterior
covariance is estimated with a Laplace approximation about this mode. The
principal coupled and independent comparisons are also sampled using adaptive
Metropolis ([Haario et al., 2001](https://doi.org/10.2307/3318737)), a Markov
chain Monte Carlo method that updates its proposal covariance from the sampled
history to improve exploration of correlated parameter directions. Convergence
is assessed for individual parameters and along the least-informed Fisher
directions, including between-chain diagnostics ([Gelman and Rubin,
1992](https://doi.org/10.1214/ss/1177011136)).

Posterior contraction is the 90% posterior interval width divided by the
corresponding prior interval width. Coverage is the fraction of repeated noisy
synthetic datasets for which the nominal 90% posterior interval contains the
generating parameter value. It assesses interval calibration at the tested
synthetic parameter value, not across a patient population.

## 6. Parameter combinations and robustness analysis

Eight analysis configurations combine different observation sets, coupling
models, and auxiliary information. Each is evaluated at nine parameter
combinations: the reference combination and an eight-point two-level design
over $\mu$, $\eta_s$, $\phi$, $k$, $D$, $|\mathbf v|$, and $C_v$. Five paired
noise realizations are used at each combination, giving 360 fits. Within each
replicate, all configurations receive the same standardized noise draw.

For each synthetic parameter combination, the auxiliary constraints on $A$,
$L$, or $C_v$ are centered on the values used to generate the observations and
retain their stated uncertainty. When $\phi$ or $C_v$ changes, the generating
$S_v$ is recomputed as $\phi C_v$. The robustness analysis therefore evaluates
different parameter regimes under correctly centered auxiliary information; it
is not a test of auxiliary-measurement bias.

## 7. Model-discrepancy analysis

A separate [model-discrepancy analysis](RESULTS.md#5-model-discrepancy)
generates synthetic observations with a finite-width shear push, two contrast-
transport pathways, or a distribution of relaxation times, separately and in
combination. These observations are then fitted with the reduced models above.
This tests whether an incomplete forward model can bias inferred parameters
without producing an obvious residual.

## 8. Limitations

The parameters are uniform within the sampled region. The shear model omits
boundaries and uses an idealized line source. Contrast transport has one
pathway, no recirculation, and isotropic dispersion. The relaxation model is a
confined-slab surrogate rather than a model of probe compression. Ultrasound
signal formation, spatially correlated noise, and uncertainty in the learned
network-informed relation are not included. The alternatives in the
[model-discrepancy analysis](RESULTS.md#5-model-discrepancy) address selected
departures only and do not constitute complete tissue or ultrasound models.

## 9. Reproduction

```bash
pip install -e .
pytest -q
fieldid config         --out configs/default.json
fieldid ensemble-prior --ensemble results/network_ensemble.csv \
                       --out results/network_prior.json
fieldid matrix         --out results --profile k --profile phi --profile S_v
fieldid discrepancy    --out results
fieldid profiles       --out results --profile k --domain-check
fieldid mcmc           --out results --steps 14100
fieldid truth-sweep    --out results/truth_sweep_r5 --n-rep 5
fieldid figures        --out figures --results results
```
