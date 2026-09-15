"""Sampling grids and noise levels of the three windows.

One probe, one imaging plane, one fixed position, three sequences run one after
another.  The probe is fixed, so the geometry does not change between them; the
sequences differ, so the point spread does, by the stated ratios.

Noise is independent between windows and within each window.  Sharing an
operator parameter is not sharing a noise realization, and the two are kept
apart on purpose: the mechanism under test is a shared systematic effect, not a
shared disturbance.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class BModeWindow:
    x0: float = 0.0            # m, start of the profile
    x1: float = 30.0e-3        # m, end
    n_x: int = 301
    sigma: float = 0.02        # additive noise sd, arbitrary intensity units
    enabled: bool = True

    def positions(self):
        return np.linspace(self.x0, self.x1, self.n_x)


@dataclass
class SWEWindow:
    x0: float = 2.0e-3         # m, first tracked position
    x1: float = 22.0e-3        # m, last
    n_x: int = 41
    t_end: float = 20.0e-3     # s
    n_t: int = 200             # 10 kHz frame rate
    f0: float = 200.0          # Hz, center of the source spectrum
    bandwidth: float = 120.0   # Hz, one standard deviation
    sigma: float = 0.10e-6     # m, displacement noise sd
    enabled: bool = True

    def positions(self):
        return np.linspace(self.x0, self.x1, self.n_x)

    def times(self):
        return np.arange(self.n_t) * (self.t_end / self.n_t)


@dataclass
class CEUSWindow:
    x0: float = 0.0            # m
    x1: float = 30.0e-3        # m
    n_x: int = 61
    # The first frame is two seconds after the nominal arrival, not at it.  A
    # frame at the arrival time sits exactly on the boundary where the bolus
    # switches on, which makes the prediction non-differentiable in t0 and
    # destroys the convergence of every finite-difference diagnostic.
    t_first: float = 3.0       # s, first frame
    t_last: float = 22.0       # s, last frame
    n_t: int = 20
    sigma_rel: float = 0.02    # noise sd as a fraction of the peak occupancy
    enabled: bool = True
    #: number of frames actually used.  The information-removal control sets
    #: this to one: with a single frame the observed variance is one number and
    #: the dispersion cannot be separated from the point spread.
    n_frames_used: int = 0     # 0 means all

    def positions(self):
        return np.linspace(self.x0, self.x1, self.n_x)

    def times(self):
        t = np.linspace(self.t_first, self.t_last, self.n_t)
        if self.n_frames_used and self.n_frames_used < self.n_t:
            return t[-self.n_frames_used:]
        return t


@dataclass
class Acquisition:
    bmode: BModeWindow = field(default_factory=BModeWindow)
    swe: SWEWindow = field(default_factory=SWEWindow)
    ceus: CEUSWindow = field(default_factory=CEUSWindow)

    def to_dict(self):
        return {k: asdict(v) for k, v in
                dict(bmode=self.bmode, swe=self.swe, ceus=self.ceus).items()}
