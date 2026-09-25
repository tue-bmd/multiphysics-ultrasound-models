"""Vascular-mechanical confounding in joint SWE-CEUS inference.

Modules
    netops    constriction as a geometric operation on the network, and the
              structural invariants that must survive it
    mech      relaxation spectrum -> complex shear modulus -> shear-wave
              dispersion; the one piece of physics that is new here
    forward   the mechanical forward map from network and tissue parameters to SWE
    ceus      network-based CEUS observables
    inference posterior comparison models for separate and coupled radius scales
"""
__version__ = "0.1.0"
