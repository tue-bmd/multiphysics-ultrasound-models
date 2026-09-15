"""A dense voxel grid of contrast curves, as an imaging acquisition would give.

The shell estimator of `physics.transport` needs a handful of curves per fit:
one input voxel and the voxels of a shell around it.  The grid estimator of
`physics.cdi3d` needs a filled block of them, because it takes spatial
derivatives.  This module builds that block.

Two things make it affordable.

The grid is propagated in chunks.  A curve on the internal time grid is 4096
samples, so a 30 x 30 x 30 block of them is about 0.9 GB, and the frequency
domain intermediates as much again.  Propagating a few thousand voxels at a
time keeps that within tens of megabytes.

Each chunk is reduced to the frame rate of the acquisition before it is
stored.  The internal grid is 50 Hz because the transport has to be resolved;
a 3D contrast acquisition runs at about 0.25 Hz.  Keeping 4096 samples per
voxel would be storing a temporal resolution that no acquisition delivers and
that the estimator never uses.  The reduction is a mean over each frame
interval, which is what an imaging frame does, rather than a bare subsample,
which would alias.

The margin matters and is easy to get wrong.  To evaluate the grid estimator
at one voxel, the fit needs the voxels of its kernel, and each of those needs
the Gaussian derivative support around it, which reaches about four sigma_x.
So the propagated block has to extend beyond the region of interest by

    ceil(truncate * sigma_x / dx) + floor(s / 2)

voxels on every side, and at sigma_x = 1.5 mm with 0.75 mm voxels and a
7-voxel kernel that is 11 voxels, or 8.25 mm, on each side of a 6 mm cube.
`margin_voxels` computes it; `grid_for` builds a block that satisfies it.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .physics import transport as TR


@dataclass
class Grid:
    C: np.ndarray            # (nx, ny, nz, n_frames) bubbles present per voxel
    t: np.ndarray            # (n_frames,) frame times [s]
    origin: np.ndarray       # (3,) position of voxel (0, 0, 0) [m]
    dx: float                # voxel pitch [m]
    frame_dt: float          # frame interval [s]
    vox_half: float          # half-width of a voxel [m]

    def index_of(self, p):
        """Voxel index nearest the position p, and the offset that was taken."""
        p = np.asarray(p, float)
        idx = np.rint((p - self.origin) / self.dx).astype(int)
        offset = p - (self.origin + idx * self.dx)
        return tuple(int(i) for i in idx), float(np.linalg.norm(offset))

    def position_of(self, idx):
        return self.origin + np.asarray(idx, float) * self.dx

    def contains(self, idx, margin=0):
        n = self.C.shape[:3]
        return all(margin <= i < d - margin for i, d in zip(idx, n))


def margin_voxels(sigma_x, dx, s, truncate=4.0):
    """Voxels of padding a fit at one voxel needs on every side."""
    return int(np.ceil(truncate * sigma_x / dx)) + int(np.floor(s / 2.0))


def frame_times(frame_dt, t_end=None):
    """Frame times of the acquisition, and the internal samples of each frame."""
    t_end = TR.T_END if t_end is None else t_end
    per = int(round(frame_dt / TR.DT))
    if per < 1:
        raise ValueError("frame_dt %g s is shorter than the internal step %g s" % (frame_dt, TR.DT))
    n_frames = int(len(TR.time_grid()[0]) // per)
    # a frame is labelled by the centre of the interval it integrates
    t = (np.arange(n_frames) + 0.5) * per * TR.DT
    return t, per, n_frames


def _reduce(tic, per, n_frames):
    """Mean of the curve over each frame interval, which is what an imaging
    frame integrates.  A bare subsample would alias whatever the curve does
    between frames."""
    usable = per * n_frames
    return tic[:usable].reshape(n_frames, per).mean(axis=1)


def grid_for(centre, half, dx, sigma_x, s, truncate=4.0):
    """Indices and positions of a block centred on `centre` that covers the
    cube of half-width `half` with enough margin for the fit."""
    m = margin_voxels(sigma_x, dx, s, truncate)
    n_in = int(np.ceil(2 * half / dx))
    n = n_in + 2 * m + 1
    centre = np.asarray(centre, float)
    origin = centre - (n - 1) / 2.0 * dx
    return n, origin, m


def apply_psf(g: Grid, fwhm) -> Grid:
    """The grid seen through an imaging system of the given resolution.

    A Gaussian blur of the bubble distribution, frame by frame, with fwhm the
    full width at half maximum of the point spread function, either one number
    for an isotropic blur or three for (x, y, z).  This is not an acoustic
    simulation: there is no propagation, no beamforming and no nonlinearity.
    It is the minimum needed to compare an estimator that takes spatial
    derivatives against one that does not, because without it the
    concentration changes by about 90 % between neighbouring voxels and the
    derivatives are of a structure no scanner could resolve.  The source's own
    validation applied a 1.1 mm point spread function for the same reason.

    An isotropic blur is a convenience, not a description of a scanner.  A
    real three-dimensional contrast point spread function is strongly
    anisotropic: at 4 MHz the wavelength is about 0.39 mm, so the axial extent
    is a few tenths of a millimetre, the lateral extent is of order 1 mm (1.1
    mm at 25 mm depth for the system the grid estimator was published on), and
    the elevational extent, set by a fixed elevation focus on a mechanically
    swept probe, is commonly two to three times the lateral.  That anisotropy
    is worth carrying because one of the two estimators fits a tensor: an
    anisotropic blur inflates one diagonal element preferentially, and the
    reported trace then mixes an instrumental anisotropy with a real one.

    fwhm = 0 returns the grid unchanged, so that "no acquisition blur" is one
    of the settings rather than a separate code path."""
    from scipy.ndimage import gaussian_filter
    f = np.atleast_1d(np.asarray(fwhm, float))
    if f.size == 1:
        f = np.repeat(f, 3)
    if f.size != 3:
        raise ValueError("fwhm must be one number or three, got %r" % (fwhm,))
    if not np.any(f > 0):
        return g
    sigma = f / 2.3548200450309493 / g.dx             # FWHM -> sigma, in voxels
    C = gaussian_filter(g.C, sigma=(sigma[0], sigma[1], sigma[2], 0), mode="nearest")
    return Grid(C=C, t=g.t, origin=g.origin, dx=g.dx, frame_dt=g.frame_dt, vox_half=g.vox_half)


def rebin(g: Grid, frame_dt) -> Grid:
    """The same grid at a coarser frame rate, by averaging whole frames.

    The frame rate is an acquisition setting: a 3D contrast acquisition runs an
    order of magnitude slower than a 2D one, and what an estimator can resolve
    follows from it."""
    k = int(round(frame_dt / g.frame_dt))
    if k < 1 or not np.isclose(k * g.frame_dt, frame_dt, rtol=1e-6):
        raise ValueError("frame_dt %g s is not a multiple of the grid's %g s" % (frame_dt, g.frame_dt))
    if k == 1:
        return g
    n = g.C.shape[3] // k
    C = g.C[:, :, :, :n * k].reshape(g.C.shape[:3] + (n, k)).mean(axis=4)
    t = g.t[:n * k].reshape(n, k).mean(axis=1)
    return Grid(C=C, t=t, origin=g.origin, dx=g.dx, frame_dt=frame_dt, vox_half=g.vox_half)


def sample_at(g: Grid, positions):
    """Curves at arbitrary positions, by trilinear interpolation of the grid.

    Both estimators are given their curves this way, so that the only thing
    that differs between them is the estimator: they see one acquisition, with
    one voxel grid, one frame rate and one point spread function.  Positions
    outside the block raise, rather than being silently clamped."""
    P = np.atleast_2d(np.asarray(positions, float))
    f = (P - g.origin) / g.dx
    i0 = np.floor(f).astype(int)
    w = f - i0
    n = np.array(g.C.shape[:3])
    if np.any(i0 < 0) or np.any(i0 + 1 >= n):
        raise ValueError("a requested position lies outside the propagated block")
    out = np.zeros((len(P), g.C.shape[3]))
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                wt = ((w[:, 0] if cx else 1 - w[:, 0]) *
                      (w[:, 1] if cy else 1 - w[:, 1]) *
                      (w[:, 2] if cz else 1 - w[:, 2]))
                out += wt[:, None] * g.C[i0[:, 0] + cx, i0[:, 1] + cy, i0[:, 2] + cz, :]
    return out


def propagate_grid(net, fl, centre, half, dx=0.75e-3, vox_half=None, sigma_x=1.5e-3, s=7,
                   frame_dt=4.0, poiseuille=False, chunk=2000, truncate=4.0, progress=False):
    """Contrast curves on a dense voxel grid around `centre`.

    The block covers the cube of half-width `half` plus the margin the fit
    needs.  Curves are propagated `chunk` voxels at a time and reduced to the
    acquisition frame rate before being stored, so peak memory stays at a few
    tens of megabytes whatever the size of the block."""
    vox_half = dx / 2 if vox_half is None else vox_half
    n, origin, m = grid_for(centre, half, dx, sigma_x, s, truncate)
    t, per, n_frames = frame_times(frame_dt)
    idx = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]
    C = np.empty((n, n, n, n_frames), float)
    for a in range(0, len(idx), chunk):
        block = idx[a:a + chunk]
        kernels = [(origin + np.array(b, float) * dx, vox_half) for b in block]
        tr = TR.propagate(net, fl, kernels, poiseuille=poiseuille)
        for q, b in enumerate(block):
            C[b[0], b[1], b[2], :] = _reduce(tr.tic[q], per, n_frames)
        del tr
        if progress:
            print("    grid %d/%d voxels" % (min(a + chunk, len(idx)), len(idx)), flush=True)
    return Grid(C=C, t=t, origin=origin, dx=dx, frame_dt=frame_dt, vox_half=vox_half)
