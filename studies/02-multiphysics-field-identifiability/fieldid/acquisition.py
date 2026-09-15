"""Acquisition settings: sampling grids and noise levels.

The defaults describe the acquisition that is already available, not a proposed
one.  The relaxation window is switched off by default and is enabled only in
the contingency experiment.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class Swe:
    """Shear-wave window."""
    x_min: float = 2.0e-3          # first lateral offset from the push [m]
    x_max: float = 12.0e-3         # last lateral offset [m]
    n_x: int = 21                  # lateral positions
    frame_rate: float = 8.0e3      # tracking frame rate [Hz]
    duration: float = 20.0e-3      # record length [s]
    f_band: tuple = (40.0, 800.0)  # analysis band [Hz]
    geometry: str = "cylindrical"  # cylindrical | plane
    sigma: float = 2.0e-7          # displacement noise, one sample [m]

    def positions(self):
        return np.linspace(self.x_min, self.x_max, self.n_x)

    def times(self):
        n = int(round(self.duration * self.frame_rate))
        return np.arange(n) / self.frame_rate


@dataclass
class Ceus:
    """Contrast window.  A small three-dimensional block of voxels downstream of
    the delivery point, over the duration of the current acquisition."""
    center: tuple = (6.0e-3, 0.0, 0.0)   # block center relative to delivery [m]
    spacing: tuple = (2.0e-3, 2.0e-3, 2.0e-3)
    shape: tuple = (3, 3, 3)
    frame_rate: float = 2.0        # [Hz]
    duration: float = 90.0         # [s]
    sigma_rel: float = 0.02        # noise as a fraction of the peak occupancy
    normalized: bool = False       # unit-area curves instead of absolute
    delivery: str = "concentration"  # or "amount"; see models.transport

    def positions(self):
        ax = [(np.arange(n) - (n - 1) / 2) * d + c
              for n, d, c in zip(self.shape, self.spacing, self.center)]
        g = np.meshgrid(*ax, indexing="ij")
        return np.column_stack([a.ravel() for a in g])

    def times(self):
        n = int(round(self.duration * self.frame_rate))
        return np.arange(1, n + 1) / self.frame_rate


@dataclass
class Relax:
    """Consolidation-relaxation window.  Contingency observation, off by
    default."""
    enabled: bool = False
    frame_rate: float = 20.0       # [Hz]
    duration: float = 30.0         # [s]
    sigma: float = 1.0e-6          # compaction noise, one sample [m]

    def times(self):
        n = int(round(self.duration * self.frame_rate))
        return np.arange(n) / self.frame_rate


@dataclass
class Acquisition:
    swe: Swe = field(default_factory=Swe)
    ceus: Ceus = field(default_factory=Ceus)
    relax: Relax = field(default_factory=Relax)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        return cls(swe=Swe(**d.get("swe", {})),
                   ceus=Ceus(**d.get("ceus", {})),
                   relax=Relax(**d.get("relax", {})))
