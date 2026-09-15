# Methods

All calculations use SI units. Displayed values are converted where indicated.

## 1. Vascular network

The network-generation model is general. The realization used in this study
represents a prostate gland as a 22 mL ellipsoid with semi-axes of 20 × 15 ×
17.5 mm. Four 600 µm-diameter, 25 mm-long extracapsular feeding arteries
represent the superior and inferior arterial pedicles from each hemipelvis.
Cadaveric anatomy identifies these four pedicles consistently
([García-Mónaco et al., 2014](https://doi.org/10.1016/j.jvir.2013.10.026));
the chosen feeder diameter lies within the 0.5–1.5 mm range reported by
angiography ([Zhang et al., 2015](https://doi.org/10.1371/journal.pone.0132678)).

Arterial segments branch according to Murray's law
([Murray, 1926](https://doi.org/10.1073/pnas.12.3.207)),

$$
d_{\mathrm{parent}}^3=d_1^3+d_2^3.
$$

with the daughter-diameter asymmetry parameterized following nonsymmetrical
branching theory ([Zamir, 1978](https://doi.org/10.1085/jgp.72.6.837)) and
varied randomly between bifurcations. Segment length and path-length tortuosity
are lognormally distributed. Arterial intersections are rejected. Each artery
is paired with an offset vein of larger diameter.

The explicit gland-scale tree extends to a terminal diameter of 30 µm. Within
6 mm sampling cubes used for analysis, the tree is continued to 20 µm. 
The remaining microvascular bed is lumped to one conductance and one residence-time
distribution per terminal arteriole.

## 2. Blood flow

Steady Poiseuille flow is solved with pressures of 70 mmHg at the arterial
feeders and 8 mmHg at the venous outlets. The terminal-bed conductance is
calibrated once per prostate-gland realization to obtain an arteriolar-end
pressure of 35 mmHg.
Pressures and flows then determine segment velocities and transit times.

Kirchhoff conservation is enforced at internal nodes, and pressure decreases
along each arterial-to-venous path.

## 3. Reduced vascularized-tissue relaxation

Vascularized tissue is represented as a compliant vascular network mechanically
supported by a spatially unresolved elastic matrix. For each vessel segment,
the resistance of the vessel wall and surrounding tissue to a change in lumen
volume is represented by the effective stiffness $H+K_{\mathrm{wall}}$, giving

$$
C_i=\frac{V_i}{H+K_{\mathrm{wall}}},
$$

where $V_i$ is the segment volume, $H$ the effective stiffness of the
surrounding matrix, and $K_{\mathrm{wall}}$ the vessel-wall stiffness.

A step in external pressure is applied over a spherical region. Vascular
pressure initially rises in the compressed segments, after which blood
redistributes through the resistive network. The observable is the volume
leaving the compressed region as the vascularized tissue approaches its new
equilibrium.

The relaxation time $\tau_{\mathrm{rc}}$ is the time at which the drained
fraction reaches $1-1/e$. The model does not resolve matrix deformation or
interstitial flow; the tissue contribution is limited to its mechanical
resistance to vascular volume change.

## 4. Intravascular transport

A lognormal microbubble concentration curve is prescribed at the arterial
inlets and transformed to the frequency domain. In each vessel segment, the
inlet spectrum is multiplied by a transit-time transfer function,

$$
C_{\mathrm{out},i}(\omega)=C_{\mathrm{in},i}(\omega)H_i(\omega).
$$

For plug flow, $H_i$ represents the delay determined by segment length and
blood velocity. At arterial bifurcations, concentration is passed to the
daughter vessels while microbubble flux divides according to blood flow.
Terminal arteries connect to the venous network through a gamma-distributed
residence-time kernel representing the unresolved microvascular bed; inputs
are flow-weighted where venous paths merge. Inverse Fourier transformation
then gives the number of microbubbles within each vessel segment and voxel as
a function of time.

Two estimators are applied to the simulated curves:

- The shell-kernel system-identification approach proposed by
  [van Sloun et al. (2017)](https://doi.org/10.1016/j.media.2016.09.010)
  fits scalar drift and dispersion between a central time-intensity curve and
  curves sampled on a surrounding kernel. It is implemented here in 3D using a
  spherical shell.
- The 3D partial-differential-equation approach proposed by
  [Wildeboer et al. (2018)](https://doi.org/10.1109/TMI.2018.2843396)
  fits a locally constant velocity vector and symmetric dispersion tensor
  directly to the convection-dispersion equation over a voxel grid.

The estimators differ in spatial support, dimensionality, derivative scale, and
regularization. Their dependence on these settings is evaluated on unchanged
simulated curves.

## 5. Network permeability

Permeability is calculated along each coordinate direction by imposing a
pressure difference across opposite faces of a sampling cube. The result is set
to zero when no vascular path connects the selected faces.

The calculated permeability is compared with the parallel-bundle approximation

$$
k_v=\frac{\phi d_{\mathrm{perm}}^2}{32\langle T^2\rangle},
$$

where $\phi$ is the vascular volume fraction, $d_{\mathrm{perm}}$ the
length-weighted root-mean-square diameter, and $\langle T^2\rangle$ the
length-weighted mean squared tortuosity.
This quantity describes hydraulic transport through the vascular
network.

## 6. Paired comparisons

Five prostate-gland realizations are generated from seeds 100 to 104. Two
non-overlapping sampling volumes are selected in each realization. Each 
simulation in which vascular tortuosity, matrix modulus, or arterial inlet 
pressure is changed is compared with a matched baseline. The same random seed, 
sampling volumes, input voxels, and terminal-bed calibration are used in each 
comparison, so that only the specified model parameter differs.

The primary comparison tests an increase in the path-to-chord tortuosity
parameter from 1.25 to 1.45. Increased vascular tortuosity is supported by
tumor corrosion casts ([Konerding et al., 1999](https://doi.org/10.1038/sj.bjc.6690416))
and prostate super-resolution ultrasound
([Huang et al., 2025](https://doi.org/10.1186/s40644-024-00819-z)), although
neither study reports this particular path-to-chord parameter. Control
comparisons change only the matrix modulus or only the arterial inlet pressure.

Effects for each positive quantity $x$ are calculated as

$$
\Delta\log_{10}x
=
\log_{10}\left(
\frac{x_{\mathrm{perturbed}}}{x_{\mathrm{baseline}}}
\right).
$$

Voxel-level effects are summarized by the median within each sampling volume,
then by the median across volumes within each realization. The final result is
the median and range across the five realizations. Transport effects use voxels
for which the estimator returns a valid result in both paired conditions.

## 7. Limitations

The capillary bed has no explicit geometry. Veins follow offset copies of the
arterial paths. Compression changes vascular volume but does not generate a
tissue-displacement field. The simulated contrast curves do not include
acoustic propagation, microbubble scattering, beamforming, or image-processing
effects. Consequently, the fitted transport quantities are estimator-dependent
descriptors rather than known continuum coefficients.
