"""Small compatibility helpers so the package runs on numpy 1.x and 2.x."""
from __future__ import annotations
import numpy as np

trapezoid = getattr(np, "trapezoid", None) or np.trapz
