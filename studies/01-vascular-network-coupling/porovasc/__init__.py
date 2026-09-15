"""porovasc: an explicit vascular-network simulator of a perfused organ, from
which contrast-ultrasound, strain-relaxation and permeability quantities are
computed on the same network and compared with the parameters that generate
them.  The interstitium is not modeled; see the scope statement in README.md.

Modules
    geometry.network     Murray tree in a gland ellipsoid, adaptive resolution
                         (RVE refinement), collision-free arteries; veins as
                         offset copies, not collision checked.
    physics.flow         pressure-driven Poiseuille flow with a lumped
                         capillary bed calibrated to a capillary pressure drop.
    physics.drainage     compliant RC drainage under a compression step;
                         relaxation time and spectrum.
    physics.transport    microbubble transport (frequency domain, plug or
                         Poiseuille kernels, explicit veins, gamma bed) and
                         the CEUS estimators (two-kernel transfer functions,
                         spherical-shell CUDI, mLDRW kappa).
    physics.cdi3d        the convection-dispersion equation fitted directly on
                         a voxel grid (tensor and velocity, Gaussian
                         derivatives, ridge), the second contrast estimator.
    homogenise.darcy     permeability measured on the network (face-to-face
                         Darcy) and the capillary-bundle-law check.
    debug.check_*        one checker per module, each asserting named
                         invariants; run them before trusting a result.
    config               default constants (BASE) and sensitivity arms (SENS).
    study                paired-arm design: sampling volumes, lesion placement,
                         fingerprints, containment, provenance of a run.
    run_paired           runner for the baseline, a sensitivity arm or a
                         lesion, each paired with its baseline gland.
    analyse_paired       hierarchical summary (voxel, volume, gland) of a run.
"""
__version__ = "0.1.0"
