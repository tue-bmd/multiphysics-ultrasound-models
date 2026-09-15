"""Default configuration and the one-at-a-time sensitivity arms.

BASE holds every design constant of the default gland; SENS maps an arm name
to the constants that the arm changes.  `apply_globals` pushes the pressures
and the transport constants into the physics modules, which read them at call
time.  The values are documented, with the source of each, in README.md.
"""
from __future__ import annotations
import numpy as np

from .physics import flow as F, transport as TR

BASE = dict(
    # geometry
    n_feeders=4, feed_d=600e-6, d_root=500e-6, alpha=0.8, gamma_med=10.0,
    tort_mean=1.25, d_term_gland=30e-6, d_term_rve=20e-6, rve_half=3e-3,
    # flow
    P_ART_mmHg=70.0, P_ART_END_mmHg=35.0, R_LAT=0.5e-3,
    # drainage
    H=5e3, R_comp=3e-3,
    # transport
    TAU_CAP=1.0, CV_CAP=0.5, AIF_MU_s=10.0, AIF_SIG=0.55, RHO_MAX=0.7,
    # length of the transform record.  The default holds every arm whose
    # travel-time bound stays under it; arms that lengthen the paths (a small
    # Zamir asymmetry above all) need a longer one, and `run_paired --t-end`
    # sets it.  It is recorded in the manifest, because an arm run with a
    # different record is not directly comparable with the others.
    T_END_s=81.92,
    # estimator grid: run_paired runs every combination of kernel, shell
    # radius, transfer function and causality rule on the same propagated
    # curves, at one shell sampling density
    shell_radii_mm=(0.75, 1.0, 1.5, 2.0), n_dir=48, vox_half_mm=0.375,
    n_inputs=24, poiseuille=(False, True), tfs=("new", "old"),
    causal_rules=("front20", "front10", "moment"),
)

# one-at-a-time perturbations: name -> {key: value}
SENS = {
    "alpha_0.6": dict(alpha=0.6), "alpha_0.95": dict(alpha=0.95),
    "gamma_6": dict(gamma_med=6.0), "gamma_15": dict(gamma_med=15.0),
    "tort_1.1": dict(tort_mean=1.1), "tort_1.45": dict(tort_mean=1.45),
    "feeders_8x600": dict(n_feeders=8), "feed_500_root_400": dict(feed_d=500e-6, d_root=400e-6),
    "dterm_gland_40": dict(d_term_gland=40e-6),
    "P_ART_60": dict(P_ART_mmHg=60.0), "P_ART_90": dict(P_ART_mmHg=90.0),
    "P_END_30": dict(P_ART_END_mmHg=30.0), "P_END_40": dict(P_ART_END_mmHg=40.0),
    "R_LAT_0": dict(R_LAT=0.0), "R_LAT_1mm": dict(R_LAT=1.0e-3),
    "H_3k": dict(H=3e3), "H_10k": dict(H=10e3),
    "Rcomp_1.5": dict(R_comp=1.5e-3), "Rcomp_6": dict(R_comp=6e-3),
    "taucap_0.5": dict(TAU_CAP=0.5), "taucap_2": dict(TAU_CAP=2.0),
    "aif_narrow": dict(AIF_SIG=0.35), "aif_wide": dict(AIF_SIG=0.8),
    "aif_early": dict(AIF_MU_s=6.0), "aif_late": dict(AIF_MU_s=15.0),
    "rho_0.5": dict(RHO_MAX=0.5), "rho_0.9": dict(RHO_MAX=0.9),
}


def apply_globals(cfg):
    """Push the configuration into the module constants.

    The pressures live in physics.flow; physics.drainage reads them from that
    module at call time rather than importing them by value, so both see the
    same value here."""
    F.P_ART = cfg["P_ART_mmHg"] * F.MMHG
    F.P_ART_END = cfg["P_ART_END_mmHg"] * F.MMHG
    F.R_LAT = cfg["R_LAT"]
    TR.TAU_CAP = cfg["TAU_CAP"]; TR.CV_CAP = cfg["CV_CAP"]
    TR.AIF_MU = np.log(cfg["AIF_MU_s"]); TR.AIF_SIG = cfg["AIF_SIG"]
    TR.RHO_MAX = cfg["RHO_MAX"]
    TR.T_END = cfg.get("T_END_s", TR.T_END)


#: Arms of the shared-dependence experiment (`porovasc.run_coupling`).
#:
#: The point of the selection is that the arms do not all act at the same place.
#: Four act on the vascular geometry, two act only on the matrix modulus, which
#: enters the compliance law and nothing else, and two act only on the driving
#: pressure, which changes the flow without touching the geometry.  The last two
#: groups are controls that make the first interpretable.
#:
#: An arm belongs here only if it does something under the settings this study
#: uses.  Two candidates were removed after a numerical check showed they do
#: nothing at all, and the check is now a test (`tests/test_coupling.py`):
#:
#:   rho_0.5    changes RHO_MAX, the radial cutoff on the microbubble position
#:              used to build the Poiseuille residence-time distribution.  This
#:              study propagates with the plug kernel, where RHO_MAX is never
#:              read: geometry, flow and every contrast curve are bit-identical
#:              to baseline.  It is also not a terminal-density parameter, as an
#:              earlier version of this list wrongly described it.
#:
#:   P_END_30   changes P_ART_END, which is the target of the bed-conductance
#:              calibration and is read only inside that calibration.  This study
#:              holds the conductance calibrated on the baseline and never
#:              recalibrates, so P_ART_END is never read: the flow solution is
#:              bit-identical to baseline.  Recalibrating it would change the bed
#:              conductance itself, which is a different intervention needing a
#:              different interpretation.
COUPLING_ARMS = {
    "tort_1.45":      "geometry: path tortuosity raised, lumen volume nearly held",
    "gamma_6":        "geometry: shorter segments per diameter, denser tree",
    "alpha_0.6":      "geometry: branching asymmetry lowered (the Zamir daughter "
                      "ratio, not the Murray exponent)",
    "dterm_gland_40": "geometry: coarser explicit terminals, 30 to 40 um.  This is "
                      "a resolution change as much as a structural one - it moves "
                      "where the explicit tree stops and the lumped bed begins - "
                      "and is labelled as such wherever it is reported",
    "H_3k":           "mechanics only: matrix modulus, enters the compliance law alone",
    "H_10k":          "mechanics only: matrix modulus, enters the compliance law alone",
    "P_ART_60":       "flow only: arterial inlet pressure lowered, geometry untouched",
    "P_ART_90":       "flow only: arterial inlet pressure raised, geometry untouched",
}

#: where each arm acts, used to label the report and to choose which invariance
#: the control checks assert
COUPLING_LOCUS = {
    "tort_1.45": "geometry", "gamma_6": "geometry", "alpha_0.6": "geometry",
    "dterm_gland_40": "geometry",
    "H_3k": "mechanics", "H_10k": "mechanics",
    "P_ART_60": "flow", "P_ART_90": "flow",
}

#: arms that were tried and found to do nothing under this study's settings
COUPLING_REJECTED = {
    "rho_0.5": "RHO_MAX is read only by the Poiseuille kernel; this study is plug",
    "P_END_30": "P_ART_END is read only by the bed calibration; this study holds "
                "the baseline conductance",
}
