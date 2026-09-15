"""this line of work ideal-field identifiability simulator.

A proof-of-principle study of identifiability and posterior contraction in joint
parameter inversion from ideal shear-wave and contrast-transport physical
fields.  Not a physiological model and not an ultrasound simulator.

See METHODS.md for the equations, units, idealizations and claim
boundaries.
"""
__version__ = "0.1.0"

from .acquisition import Acquisition, Swe, Ceus, Relax
from .model import Inversion
from .params import CATALOGUE, Space, TRUTH

__all__ = ["Acquisition", "Swe", "Ceus", "Relax", "Inversion",
           "CATALOGUE", "Space", "TRUTH", "__version__"]
