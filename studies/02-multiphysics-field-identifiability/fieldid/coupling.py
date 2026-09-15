"""Cross-modal coupling levels.

1. ``independent``   - S_v is a free parameter with the marginal prior induced by
   phi * C_v, so that this configuration and the coupled ones share marginal
   priors and differ only in correlation.  Nothing links the modalities.  This is
   the matched control: any contraction seen here comes from adding data.
2. ``constitutive``  - S_v = phi * C_v is imposed, C_v free.  The candidate
   vascular-storage relation, and the only route by which a CEUS observation can
   reach a mechanical parameter.
3. ``network``       - as ``constitutive``, and in addition {phi, k, D, |v|} carry
   a joint prior whose correlation is taken from vascular-network ensembles while
   the marginals stay those of the catalogue.  Equal phi is allowed to
   correspond to different k, D and v: the conditional spread is preserved and is
   reported.

In the vascular-network simulator the fractional vascular compliance is
C_v = 1 / (H + K_wall) by construction, so the ensemble carries no information
about C_v beyond the prior on the matrix modulus.  C_v is therefore left
independent at level 3, and this is stated wherever level 3 is reported.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np

from .params import CATALOGUE

#: parameters the network ensemble can inform
NETWORK_NAMES = ("phi", "k", "D", "vmag")

LEVELS = ("independent", "constitutive", "network")


def extra_free(level):
    """The storage-related parameter that is free at this level."""
    if level == "independent":
        return ["S_v"]
    if level in ("constitutive", "network"):
        return ["C_v"]
    raise ValueError("unknown coupling level %r" % level)


def apply(level, p):
    """Complete the physical parameter dictionary for this coupling level."""
    p = dict(p)
    if level in ("constitutive", "network"):
        p["S_v"] = p["phi"] * p["C_v"]
    return p


@dataclass
class CorrelatedPrior:
    """Gaussian copula in log coordinates: catalogue marginals, ensemble
    correlation."""
    names: tuple
    corr: np.ndarray
    source: str = "unset"
    n_ensemble: int = 0

    def __post_init__(self):
        self.corr = np.asarray(self.corr, float)
        w, _ = np.linalg.eigh(self.corr)
        if w.min() <= 0:
            raise ValueError("correlation matrix is not positive definite")

    # -- density -----------------------------------------------------------
    def log_pdf(self, values, entry=None):
        """values: dict name -> physical value; entry: name -> Entry lookup."""
        entry = entry or (lambda n: CATALOGUE[n])
        m = np.array([entry(n).mean for n in self.names])
        s = np.array([entry(n).sd for n in self.names])
        z = (np.array([np.log(values[n]) for n in self.names]) - m) / s
        ci = np.linalg.inv(self.corr)
        sign, logdet = np.linalg.slogdet(self.corr)
        return float(-0.5 * z @ ci @ z - 0.5 * logdet)

    def conditional_sd(self, target, given):
        """Standard deviation of log(target) given log(given), in units of its
        own marginal standard deviation.  One means the conditioning removed
        nothing; zero would mean the prior made target a function of given."""
        i = self.names.index(target)
        j = self.names.index(given)
        return float(np.sqrt(1.0 - self.corr[i, j] ** 2))

    def to_dict(self):
        return dict(names=list(self.names), corr=self.corr.tolist(),
                    source=self.source, n_ensemble=int(self.n_ensemble))

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(d["names"]), np.array(d["corr"], float),
                   d.get("source", "unset"), int(d.get("n_ensemble", 0)))

    @classmethod
    def from_ensemble(cls, table, names=NETWORK_NAMES, source="unset"):
        """Correlation of the logarithms of the ensemble columns."""
        x = np.column_stack([np.log(np.asarray(table[n], float)) for n in names])
        good = np.all(np.isfinite(x), axis=1)
        x = x[good]
        if len(x) < len(names) + 2:
            raise ValueError("ensemble too small: %d usable rows" % len(x))
        c = np.corrcoef(x, rowvar=False)
        # shrink towards the identity just enough to stay positive definite
        w = np.linalg.eigvalsh(c)
        if w.min() <= 1e-6:
            lam = 0.05
            c = (1 - lam) * c + lam * np.eye(len(names))
        return cls(tuple(names), c, source, len(x))

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
