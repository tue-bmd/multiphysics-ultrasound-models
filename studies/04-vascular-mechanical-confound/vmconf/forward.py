"""Forward map: a network realization and tissue parameters -> observables.

Tissue parameters
    mu          matrix shear modulus                    [Pa]
    eta         matrix shear viscosity                  [Pa s]
    s           constriction, radii scaled by s (1.0 = baseline)
    visc_share  fraction of the tissue loss modulus at the reference frequency
                that the vasculature accounts for

The amplitude of the vascular contribution cannot be derived from storage: in
Biot poroelasticity the shear modulus is the same drained and undrained. It is
therefore parameterized through an explicit, sweepable `visc_share`. See
`vmconf.mech` for the model assumptions.

Coupling between the two sides of the model is not imposed: the relaxation
times depend on mu, so a change in the matrix alters the vascular contribution
as well. Constriction changes the relaxation times as s^-4 and the lumen
fraction as s^2 under the fixed-tissue-element convention.
"""
from __future__ import annotations

import numpy as np

from . import mech

ETA_BLOOD = 3.6e-3


def region_weights(net, centre, half):
    """Lumen volume of each segment inside an axis-aligned sampling cube, and
    the vascular volume fraction of that cube.

    Uses the study-01 clipping primitive, so membership is the same partial
    volume used by the transport and permeability calculations.
    """
    from porovasc.geometry.network import cube_fraction
    frac = cube_fraction(net.p0, net.p1, np.asarray(centre, float), half)
    vol = np.pi * net.r ** 2 * net.Lpath * frac
    phi = float(vol.sum() / (2 * half) ** 3)
    return vol, phi


def amplitude_fixed_coupling(k, phi, mu):
    """Vascular increment under a fixed PHYSICAL coupling constant.

    dG = k * phi * mu, with k dimensionless and held constant.  This is the
    alternative to fixing `visc_share`, and the two answer different questions:

      fixed share     the vasculature is declared to contribute a stated
                      fraction of the measured tissue viscosity.  Refining the
                      network changes the spectrum, and the amplitude is
                      re-derived to keep that fraction, so the change is
                      absorbed.  Resolution-independence of the observable is
                      then partly a property of the normalization.

      fixed coupling  k is held, so a finer network with a larger phi produces
                      a larger increment.  Resolution dependence is then visible
                      rather than absorbed.

    Both are reported side by side in `run_diagnostics.py`, because presenting
    only the first would let a reader take resolution-independence for a
    property of the tissue.
    """
    return float(k) * float(phi) * float(mu)


def amplitude_scale(s, law, phi, phi0):
    """How the vascular modulus increment changes under constriction.

    Two candidate laws, differing in which element carries the stress:

    tissue_element : the surrounding tissue volume does not shrink when the
        lumen narrows, so the increment is unchanged.  This is the same choice
        that gives tau ~ s^-4, and it is the one the published observation
        selects: it predicts that constriction STIFFENS the tissue, which is
        what Poul et al. (Fluids 5:228, 2020) measured in perfused placenta
        under a vasoactive agent.

    lumen : the increment scales with lumen volume, dG ~ phi ~ s^2. It predicts
        softening under the tested constriction and is retained as the
        alternative disfavored by the reported experimental sign.

    The sign of the measured effect is therefore a test that discriminates the
    two, rather than an assumption imported into the model.
    """
    if law == "tissue_element":
        return 1.0
    if law == "lumen":
        return (phi / phi0) if phi0 > 0 else 0.0
    raise ValueError("unknown amplitude law %r" % (law,))


def swe_curve(net, centre, half, mu, eta, visc_share, s=1.0, omega=None,
              mode="parallel", eta_b=ETA_BLOOD, rho=mech.RHO, f_ref=200.0,
              dG=None, amplitude_law="tissue_element"):
    """Shear-wave phase velocity and attenuation for one parameter combination.

    Constriction is applied analytically. Lumen volume and `phi` scale as s^2.
    Relaxation times scale as s^-4 because the compliance is assigned to a fixed
    surrounding tissue element while vascular resistance scales as r^-4.
    """
    omega = mech.band() if omega is None else np.asarray(omega, float)
    vol, phi0 = region_weights(net, centre, half)
    # exact per-segment times, no binning; constriction scales them by s^-4
    # because the tissue element a vessel drains does not shrink with the lumen
    w, tau = mech.segment_times(net.r, net.Lpath, vol, mu, eta_b=eta_b, s=s)
    phi = phi0 * s ** 2
    # Amplitude convention.  `visc_share` is defined for the UNCONSTRICTED network
    # of the network passed in, and the lumen-fraction dependence then enters as
    # dG ~ phi.  This matters: if the share were re-derived on the constricted
    # configuration, constriction would silently rescale the assumption as well as the
    # geometry, and the two would no longer be separable.  Pass `dG` explicitly
    # to bypass the convention, which is what the cross-check between the
    # analytic and rebuilt routes does.
    if dG is None:
        w0, tau0 = mech.segment_times(net.r, net.Lpath, vol, mu, eta_b=eta_b)
        dG0 = mech.amplitude_for_share(w0, tau0, eta, visc_share, f_ref=f_ref)
        dG = dG0 * amplitude_scale(s, amplitude_law, phi, phi0)
    gm = mech.kelvin_voigt(omega, mu, eta)
    gv = mech.vascular_modulus(w, tau, omega, dG)
    g = mech.combine(gm, gv, mode, phase_fraction=phi)
    c, a = mech.shear_wave(g, omega, rho)
    wb, tb = mech.bin_spectrum(w, tau)
    return dict(omega=omega, c=c, alpha=a, phi=phi, dG=dG,
                visc_share=visc_share, amplitude_law=amplitude_law, spectrum=w, tau=tau,
                spectrum_binned=wb, tau_binned=tb,
                curvature=mech.spectrum_curvature(w, tau, omega),
                invalid_weight=mech.poiseuille_validity(net.r[vol > 0],
                                                       vol[vol > 0], omega),
                summary=mech.spectrum_summary(w, tau))
