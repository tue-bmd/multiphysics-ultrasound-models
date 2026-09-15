"""Three-dimensional convective-dispersion fitting on a voxel grid.

This is the second of the two contrast-kinetic estimators in this package, and
it differs from the shell estimator of `physics.transport` in every respect
that an inverse problem can differ: dimensionality, spatial support,
regularisation and the scale at which derivatives are taken.  Running both on
the same simulated tissue is the point: it separates what the estimator
reports from what the tissue does.

    shell estimator   one input voxel and a spherical shell of output voxels;
                      a scalar (v, D) fitted through a one-dimensional
                      transfer function along the chord; no regularisation;
                      no explicit derivative scale.

    this estimator    a solid spherical kernel of voxels; the local partial
                      differential equation fitted directly, giving a full
                      dispersion tensor and velocity vector; ridge
                      regularisation; derivatives taken by convolution with
                      Gaussian derivatives at a chosen scale.

Method, after Wildeboer et al., "Convective-Dispersion Modeling in Three-
Dimensional Contrast-Ultrasound Imaging for the Localization of Prostate
Cancer" (see also R. R. Wildeboer, PhD thesis, TU/e, 2019, chapter 7).

The transport of the contrast concentration C(x, y, z, t) is taken to obey

    dC/dt = div(D grad C) - v . grad C

with a locally constant symmetric dispersion tensor D and velocity vector v.
Written out, with the six independent tensor elements and three velocity
components as unknowns,

    dC/dt = Dxx Cxx + 2 Dxy Cxy + 2 Dxz Cxz + Dyy Cyy + 2 Dyz Cyz + Dzz Czz
            - vx Cx - vy Cy - vz Cz.

Every derivative is taken by convolving the data with the corresponding
derivative of a Gaussian, of standard deviation sigma_x in space and sigma_t
in time, which sets the scale the estimate refers to.  Within a spherical
kernel of diameter s voxels the nine unknowns are then fitted by ridge
regression,

    beta = argmin ||y - Z beta||^2 + l ||beta||^2,    l = l0 n,

where n is the number of rows of Z (kernel voxels times frames), so that l0 is
independent of the kernel size and of the length of the recording.

Two properties of this estimator are worth stating, because they are part of
what is being compared rather than defects to be fixed.  The ridge penalty is
applied to all nine coefficients alike although they carry different units
(mm^2/s and mm/s), so the shrinkage is not the same for the tensor and for the
velocity.  And the reported scalars are contractions of the fit,

    D_CD = (Dxx + Dyy + Dzz) / 3        (the apparent dispersion coefficient,
                                         the analogue of the ADC in diffusion
                                         weighted MRI)
    v_CD = |v|

so an anisotropic tensor and an isotropic one with the same trace are reported
identically.

Sign convention: the velocity columns of the design matrix carry the minus
sign of the equation above, so that a bolus travelling towards +y returns
vy > 0.  The source reports |v| only, for which the convention is immaterial.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve1d

# the nine unknowns, in the order they appear in beta
TERMS = ("Dxx", "Dxy", "Dxz", "Dyy", "Dyz", "Dzz", "vx", "vy", "vz")


@dataclass
class Fit:
    beta: np.ndarray         # (9,) the coefficients, in the order of TERMS
    D: np.ndarray            # (3, 3) the symmetric dispersion tensor [m^2/s]
    v: np.ndarray            # (3,) the velocity vector [m/s]
    D_CD: float              # trace(D) / 3 [m^2/s]
    v_CD: float              # |v| [m/s]
    n_rows: int              # kernel voxels times frames
    residual: float          # ||y - Z beta|| / ||y||, unregularised residual
    condition: float         # condition number of Z'Z before the ridge term


def _gauss_weights(sigma, spacing, order, truncate=4.0):
    """Samples of the `order`-th derivative of a unit-area Gaussian, scaled by
    the sample spacing so that a discrete convolution approximates the
    continuous one."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    half = int(np.ceil(truncate * sigma / spacing))
    u = np.arange(-half, half + 1) * spacing
    g = np.exp(-u ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    if order == 0:
        w = g
    elif order == 1:
        w = -u / sigma ** 2 * g
    elif order == 2:
        w = (u ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
    else:
        raise ValueError("order must be 0, 1 or 2")
    return w * spacing


def _modes(mode):
    """One boundary rule, or a (space, time) pair.  They are worth separating:
    a field can be periodic in space and not in time, and an implementation
    that convolves by FFT, as the source does, is effectively "wrap" in both."""
    if isinstance(mode, str):
        return mode, mode
    space, time = mode
    return space, time


def _smooth(C, orders, dx, dt, sigma_x, sigma_t, mode):
    """C convolved with d^orders of the 4D Gaussian.  `orders` gives the
    derivative order along x, y, z and t.  The Gaussian is separable, so this
    is four one-dimensional convolutions rather than one four-dimensional
    one."""
    space_mode, time_mode = _modes(mode)
    out = C
    for axis, order in enumerate(orders):
        spacing = dt if axis == 3 else dx
        sigma = sigma_t if axis == 3 else sigma_x
        m = time_mode if axis == 3 else space_mode
        out = convolve1d(out, _gauss_weights(sigma, spacing, order), axis=axis, mode=m)
    return out


def derivative_fields(C, dx, dt, sigma_x, sigma_t, mode="reflect"):
    """The ten smoothed fields the fit needs, as a dict.

    C has shape (nx, ny, nz, nt); dx is the isotropic voxel spacing and dt the
    frame interval.  `mode` is the boundary rule of the convolution: "reflect"
    keeps the estimate usable near the edge of the volume, "wrap" reproduces
    the circular convolution of an FFT implementation."""
    o = {"t": (0, 0, 0, 1),
         "x": (1, 0, 0, 0), "y": (0, 1, 0, 0), "z": (0, 0, 1, 0),
         "xx": (2, 0, 0, 0), "yy": (0, 2, 0, 0), "zz": (0, 0, 2, 0),
         "xy": (1, 1, 0, 0), "xz": (1, 0, 1, 0), "yz": (0, 1, 1, 0)}
    return {k: _smooth(C, v, dx, dt, sigma_x, sigma_t, mode) for k, v in o.items()}


def kernel_offsets(s):
    """Voxel offsets of a spherical kernel of diameter s voxels."""
    r = s / 2.0
    h = int(np.floor(r))
    g = np.arange(-h, h + 1)
    di, dj, dk = np.meshgrid(g, g, g, indexing="ij")
    keep = (di ** 2 + dj ** 2 + dk ** 2) <= r ** 2
    return np.stack([di[keep], dj[keep], dk[keep]], axis=1)


def fit_at(fields, idx, offsets, l0):
    """Fit the nine coefficients in the kernel centred on the voxel `idx`.

    The fields must already be differentiated in millimetres and seconds, so
    that beta comes out in mm^2/s and mm/s; see `fit_volume`, which converts.
    Working in millimetres matters for two reasons.  It keeps the second
    derivative and first derivative columns within a few orders of magnitude
    of each other rather than a million, and it is the unit in which the
    source's regularisation parameter was chosen.

    The ridge penalty is applied after dividing both sides by the root mean
    square of y.  Without that, l0 would not be a property of the analysis at
    all: it would depend on the arbitrary amplitude of the concentration data,
    and the same l0 that regularises intensity in arbitrary units would be
    inert on a bubble count.  The unregularised solution is unchanged by this
    scaling; only the strength of the shrinkage is made comparable.

    Returns None when the kernel does not fit inside the volume, so that the
    caller can tell an edge voxel from a failed fit."""
    i, j, k = idx
    nx, ny, nz, nt = fields["t"].shape
    ii = i + offsets[:, 0]; jj = j + offsets[:, 1]; kk = k + offsets[:, 2]
    if ii.min() < 0 or jj.min() < 0 or kk.min() < 0 or \
       ii.max() >= nx or jj.max() >= ny or kk.max() >= nz:
        return None
    take = lambda f: f[ii, jj, kk, :].reshape(-1)
    y = take(fields["t"])
    Z = np.stack([take(fields["xx"]), 2 * take(fields["xy"]), 2 * take(fields["xz"]),
                  take(fields["yy"]), 2 * take(fields["yz"]), take(fields["zz"]),
                  -take(fields["x"]), -take(fields["y"]), -take(fields["z"])], axis=1)
    n = Z.shape[0]
    scale = float(np.sqrt(np.mean(y ** 2)))
    if not scale > 0:
        return None                                   # no signal in this kernel
    y = y / scale; Z = Z / scale
    ZtZ = Z.T @ Z
    beta = np.linalg.solve(ZtZ + l0 * n * np.eye(9), Z.T @ y)
    res = float(np.linalg.norm(y - Z @ beta) / max(np.linalg.norm(y), 1e-300))
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = float(np.linalg.cond(ZtZ))
    D = np.array([[beta[0], beta[1], beta[2]],
                  [beta[1], beta[3], beta[4]],
                  [beta[2], beta[4], beta[5]]])
    v = beta[6:9]
    return Fit(beta=beta, D=D, v=v, D_CD=float(np.trace(D) / 3.0),
               v_CD=float(np.linalg.norm(v)), n_rows=int(n), residual=res, condition=cond)


def fit_volume(C, dx, dt, sigma_x, sigma_t, s=7, l0=0.1, centres=None, mode="reflect",
               t_slice=None):
    """Fit every centre of interest in one volume.

    C        (nx, ny, nz, nt) concentration, linear in the bubble count
    dx, dt   voxel spacing [m] and frame interval [s]
    sigma_x  derivative scale in space [m]; the estimate refers to this scale
    sigma_t  derivative scale in time [s]
    s        kernel diameter in voxels
    l0       size-independent ridge parameter
    centres  iterable of (i, j, k) voxel indices, or None for every voxel whose
             kernel fits inside the volume
    mode     boundary rule of the convolution, one rule or a (space, time)
             pair; "wrap" reproduces an FFT implementation
    t_slice  frames to fit on, after differentiating on the whole record.  The
             temporal convolution needs about four sigma_t of record on either
             side, so the first and last frames of any finite recording carry
             a boundary error; fitting on the interior removes it instead of
             letting it into the estimate.

    Returns a dict from (i, j, k) to Fit, with edge voxels absent rather than
    silently filled in.  D and v come back in SI units (m^2/s and m/s) like
    everything else in this package, although the fit itself is done in
    millimetres; see `fit_at` for why."""
    MM = 1e3
    fields = derivative_fields(C, dx * MM, dt, sigma_x * MM, sigma_t, mode=mode)
    if t_slice is not None:
        fields = {k: f[:, :, :, t_slice] for k, f in fields.items()}
    offsets = kernel_offsets(s)
    nx, ny, nz, nt = C.shape
    if centres is None:
        h = int(np.floor(s / 2.0))
        centres = [(i, j, k)
                   for i in range(h, nx - h) for j in range(h, ny - h) for k in range(h, nz - h)]
    out = {}
    for idx in centres:
        f = fit_at(fields, tuple(idx), offsets, l0)
        if f is None:
            continue
        # mm^2/s -> m^2/s and mm/s -> m/s
        out[tuple(idx)] = Fit(beta=f.beta, D=f.D * 1e-6, v=f.v * 1e-3,
                              D_CD=f.D_CD * 1e-6, v_CD=f.v_CD * 1e-3,
                              n_rows=f.n_rows, residual=f.residual, condition=f.condition)
    return out
