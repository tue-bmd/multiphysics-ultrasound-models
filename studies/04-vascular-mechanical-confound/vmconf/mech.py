"""Reduced vascular contribution to a shear-wave dispersion curve.

The model follows the microchannel-flow hypothesis that shear deformation drives
fluid along vessels and produces a relaxing contribution to the complex shear
modulus. No elasticity or poroelasticity problem is solved on the network.

The network supplies segment radii, path lengths, lumen-volume weights, and the
resulting relaxation-time distribution. For each segment at baseline,

    tau_i = eta_b L_i^2 / (G_ref r_i^2).

During constriction, the compliance is assigned to the surrounding tissue
element, whose volume is fixed, while flow resistance scales as r^-4. Therefore
`tau_i(s) = tau_i(1) s^-4`. The vascular modulus is represented by a discrete
Maxwell sum,

    G*_v(omega) = dG sum_i w_i (i omega tau_i) / (1 + i omega tau_i).

The existence, Maxwell form, mixing rule, and amplitude of this contribution are
model assumptions. The default amplitude is expressed as `visc_share`, the
vascular share of the measured loss modulus at a reference frequency. Parallel
addition to a Kelvin-Voigt matrix is the reference model; a series mixture and a
lumen-volume amplitude law are retained as sensitivity models.

The CEUS calculation is more explicit: study 01 solves flow and bubble transport
on the same network. Conclusions about mechanical-transport coupling are thus
conditional on the reduced mechanical mapping stated above.
"""
from __future__ import annotations

import numpy as np

RHO = 1050.0        # tissue density [kg m^-3]


# ------------------------------------------------------------ moduli
def vascular_modulus(spectrum, tau_grid, omega, dG):
    """G*_v(omega) from a normalised relaxation spectrum (discrete Maxwell)."""
    w = np.asarray(spectrum, float)
    tau = np.asarray(tau_grid, float)
    om = np.atleast_1d(np.asarray(omega, float))
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(om, dtype=complex)
    w = w / s
    x = 1j * om[:, None] * tau[None, :]
    return dG * np.sum(w[None, :] * x / (1.0 + x), axis=1)


def kelvin_voigt(omega, mu, eta):
    """G*(omega) = mu + i omega eta."""
    om = np.atleast_1d(np.asarray(omega, float))
    return mu + 1j * om * eta


def springpot(omega, G0, beta, omega0=2 * np.pi * 100.0):
    """Power-law (fractional) matrix, G*(omega) = G0 (i omega / omega0)^beta.

    Used only as the *generating* model in the misspecification arm, because a
    Kelvin-Voigt inversion cannot represent power-law behaviour over a wide band
    (Parker, Szabo & Holm, Phys. Med. Biol. 64:215012, 2019).
    """
    om = np.atleast_1d(np.asarray(omega, float))
    return G0 * (1j * om / omega0) ** beta


def combine(g_matrix, g_vasc, mode="parallel", **kwargs):
    """Combine matrix and vascular contributions.

    parallel (Voigt): G* = G_m + G_v.
    series (Reuss):   the vascular element carries the matrix in series over the
        volume fraction it occupies, so that G* -> G_m when the vascular
        contribution vanishes.  Written with an explicit phase fraction `f`:

            1/G* = (1 - f)/G_m + f/(G_m + G_v)

        This form recovers the matrix modulus when the vascular contribution
        vanishes.
    """
    if mode == "parallel":
        return g_matrix + g_vasc
    if mode == "series":
        f = kwargs.get("phase_fraction", None)
        if f is None:
            raise ValueError("series mixing needs phase_fraction")
        return 1.0 / ((1.0 - f) / g_matrix + f / (g_matrix + g_vasc))
    raise ValueError("unknown combination mode %r" % (mode,))


# ------------------------------------------------------------ wave
def shear_wave(g_star, omega, rho=RHO):
    """Phase velocity and attenuation of a shear wave in a viscoelastic medium.

    Convention: u = exp(i(omega t - k x)), which is the one that pairs with a
    passive modulus written as G* = G' + i G'' with G'' >= 0.  Substituting
    gives k* = omega sqrt(rho / G*), and for G'' > 0 the principal root has
    Re(k*) > 0 and Im(k*) < 0, so exp(-i k* x) decays in the direction of
    travel.  Phase velocity is omega / Re(k*) and attenuation is -Im(k*) Np/m.

    The implementation checks the sign of the imaginary wavenumber explicitly.
    """
    om = np.atleast_1d(np.asarray(omega, float))
    g = np.asarray(g_star, complex)
    k = om * np.sqrt(rho / g)
    k = np.where(k.real < 0, -k, k)            # forward-travelling branch
    if np.any(k.imag > 1e-12 * np.abs(k)):
        raise ValueError("non-passive modulus: Im(k) > 0 under exp(i(wt - kx))")
    c = np.where(np.abs(k.real) > 0, om / k.real, np.nan)
    alpha = -np.minimum(k.imag, 0.0)
    return c, alpha


def dispersion(spectrum, tau_grid, omega, dG, mu, eta,
               mode="parallel", matrix="kv", rho=RHO, **matrix_kw):
    """Convenience: spectrum + matrix parameters -> (phase velocity, attenuation)."""
    gm = kelvin_voigt(omega, mu, eta) if matrix == "kv" else springpot(omega, **matrix_kw)
    gv = vascular_modulus(spectrum, tau_grid, omega, dG)
    return shear_wave(combine(gm, gv, mode), omega, rho)


def band(f_lo=50.0, f_hi=200.0, n=24):
    """Angular-frequency grid over a shear-wave band, log spaced.

    The default upper edge is 200 Hz, not 500 Hz.  Above roughly 200 Hz the
    quasi-static Poiseuille assumption fails for a large share of the regional
    lumen weight (`poiseuille_validity`), so the model is outside its stated
    range there.
    """
    return 2 * np.pi * np.logspace(np.log10(f_lo), np.log10(f_hi), n)


# ------------------------------------------------- spectrum from the network
def segment_times(r, Lpath, weight, G_ref, eta_b=3.6e-3, s=1.0):
    """Exact per-segment relaxation times and weights.  No binning.

    Baseline law.  A segment of radius r and path length L drains a tissue
    element whose volume is proportional to that segment's share of the region.
    Flow resistance goes as eta_b L / r^4 and the element compliance as its
    volume over the modulus, giving

        tau = eta_b L^2 / (G_ref r^2)                                   (1)

    Constriction.  When the lumen narrows the surrounding tissue element does
    NOT shrink with it: the vessel constricts inside the same piece of tissue.
    The compliance therefore stays fixed while the resistance rises as r^-4, so

        tau(s) = tau(1) s^-4                                            (2)

    which is the scaling of the published microchannel-flow model
    (Poul et al., Fluids 5:228, 2020, equations 5 and 9).

    Binning is applied only for display (`bin_spectrum`); the forward model uses
    these exact times, because rounding times onto a coarse log grid makes the
    forward map discontinuous in G_ref.
    """
    r = np.asarray(r, float); L = np.asarray(Lpath, float)
    w = np.asarray(weight, float)
    ok = (w > 0) & (r > 0) & (L > 0)
    if not ok.any():
        return np.zeros(0), np.zeros(0)
    tau = eta_b * L[ok] ** 2 / (G_ref * r[ok] ** 2) * float(s) ** -4
    ww = w[ok] / w[ok].sum()
    return ww, tau


def bin_spectrum(weights, tau, n_bins=40, tau_lo=1e-6, tau_hi=1e0):
    """Bin exact times onto a log grid FOR DISPLAY ONLY."""
    edges = np.logspace(np.log10(tau_lo), np.log10(tau_hi), n_bins + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    idx = np.clip(np.searchsorted(edges, tau) - 1, 0, n_bins - 1)
    out = np.zeros(n_bins)
    np.add.at(out, idx, weights)
    return out, centres


def microchannel_spectrum(r, Lpath, weight, G_ref, eta_b=3.6e-3, n_bins=40,
                          tau_lo=1e-6, tau_hi=1e0):
    """Volume-weighted relaxation spectrum of fluid motion inside the vessels.

    For a segment of radius r and path length L embedded in a matrix of shear
    modulus G_ref, fluid driven along the segment by a deformation of the
    surrounding solid relaxes with

        tau = eta_b L^2 / (G_ref r^2)

    the ratio of a viscous flow resistance to an elastic restoring stiffness.
    Because Murray branching keeps L proportional to diameter, tau is only
    weakly dependent on caliber within one network; its spread comes from the
    realized length-to-diameter distribution. This function describes the
    baseline spectrum. Intervention scaling is applied through the `s` argument
    of `segment_times`, using the fixed-tissue-element s^-4 law.

    This is the local, channel-scale process that acts in the shear-wave band.
    It is a different quantity from the drainage relaxation of study 01, which
    is set by the size of the compressed region and is roughly fifty times
    slower; both are computed on the same network and reported together.

    Returns (weights, tau_grid) with weights summing to one, binned on a
    log-spaced grid so that the spectrum has the same form as the one returned
    by the drainage solve and the two can be plotted together.
    """
    w, tau = segment_times(r, Lpath, weight, G_ref, eta_b=eta_b)
    if len(w) == 0:
        return np.zeros(n_bins), np.logspace(np.log10(tau_lo), np.log10(tau_hi), n_bins)
    return bin_spectrum(w, tau, n_bins, tau_lo, tau_hi)


def spectrum_summary(weights, tau_grid, f_lo=50.0, f_hi=200.0):
    """Where a spectrum sits relative to a measurement band."""
    w = np.asarray(weights, float); t = np.asarray(tau_grid, float)
    s = w.sum()
    if s <= 0:
        return dict(tau_gm=np.nan, below=np.nan, inband=np.nan, above=np.nan)
    w = w / s
    lo, hi = 1 / (2 * np.pi * f_hi), 1 / (2 * np.pi * f_lo)
    return dict(tau_gm=float(np.exp(np.sum(w * np.log(t)))),
                below=float(w[t < lo].sum()),
                inband=float(w[(t >= lo) & (t <= hi)].sum()),
                above=float(w[t > hi].sum()))


# ------------------------------------------------- amplitude and shape
def amplitude_for_share(spectrum, tau_grid, eta_matrix, share,
                        f_ref=200.0, dG_hi=1e6):
    """Modulus increment dG giving the vasculature a chosen share of the loss.

    `share` is G''_v / (G''_v + omega_ref * eta_matrix) at `f_ref`, i.e. the
    fraction of the measured tissue viscosity attributable to the vasculature.
    The relation is linear in dG, so it inverts in closed form.
    """
    if not (0.0 <= share < 1.0):
        raise ValueError("share must be in [0, 1), got %r" % (share,))
    if share == 0.0:
        return 0.0
    om = 2 * np.pi * f_ref
    unit = vascular_modulus(spectrum, tau_grid, [om], 1.0)[0].imag
    if unit <= 0:
        return 0.0
    target = share / (1.0 - share) * om * eta_matrix     # G''_v required
    dG = float(target / unit)
    return min(dG, dG_hi)


def power_law_exponent(spectrum, tau_grid, omega=None):
    """Slope of log|G*_v| against log omega over the band.

    The microchannel flow model derives a power-law response from a branching
    vessel tree; this measures whether the spectrum the network actually
    produces behaves that way, and with what exponent, rather than assuming it.
    """
    om = band() if omega is None else np.asarray(omega, float)
    g = vascular_modulus(spectrum, tau_grid, om, 1.0)
    m = np.abs(g) > 0
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(om[m]), np.log(np.abs(g[m])), 1)[0])


def poiseuille_validity(r, weight, omega, rho=RHO, eta_b=3.6e-3):
    """Fraction of the lumen weight for which quasi-static Poiseuille flow fails.

    The microchannel construction assumes fully developed viscous flow.  That
    holds while the viscous diffusion time across the lumen is short compared
    with the period, approximately omega < 7 nu / r^2 with nu = eta_b / rho
    (Parker, Phys. Med. Biol. 62:1046, 2017).  Beyond it the response becomes
    inertial and the present model is out of its range of validity.

    Returns the weight fraction violating the criterion at each frequency; this
    is a screening quantity, not a corrected modulus.
    """
    r = np.asarray(r, float); w = np.asarray(weight, float)
    om = np.atleast_1d(np.asarray(omega, float))
    ok = w > 0
    if not ok.any():
        return np.zeros_like(om)
    w = w[ok] / w[ok].sum(); rr = r[ok]
    nu = eta_b / rho
    om_max = 7.0 * nu / rr ** 2
    return np.array([float(w[om_i > om_max].sum()) for om_i in om])


def spectrum_curvature(weights, tau, omega=None):
    """Is the response actually a power law over the band, or a crossover?

    Returns the global fitted slope, the local slope at the two ends, the phase
    at the two ends, and the worst relative error of the fitted power law.  A
    single Maxwell element gives a fitted slope of exactly 0.5 on a symmetric
    log grid, so the slope alone carries no information about the vessel tree;
    the curvature is what distinguishes them.
    """
    om = band() if omega is None else np.asarray(omega, float)
    g = vascular_modulus(weights, tau, om, 1.0)
    mag = np.abs(g)
    m = mag > 0
    if m.sum() < 4:
        return {}
    lx, ly = np.log(om[m]), np.log(mag[m])
    slope, icpt = np.polyfit(lx, ly, 1)
    fit = np.exp(icpt) * om[m] ** slope
    k = max(3, len(lx) // 4)
    lo = float(np.polyfit(lx[:k], ly[:k], 1)[0])
    hi = float(np.polyfit(lx[-k:], ly[-k:], 1)[0])
    ph = np.degrees(np.angle(g[m]))
    return dict(slope=float(slope), slope_low=lo, slope_high=hi,
                phase_low=float(ph[0]), phase_high=float(ph[-1]),
                max_rel_error=float(np.max(np.abs(fit - mag[m]) / mag[m])),
                is_power_law=bool(abs(hi - lo) < 0.1))


# ------------------------------------------------- fast forward evaluation
class VascularResponse:
    """Precomputed vascular modulus, exact in the segment times.

    The Maxwell sum over every segment is too slow to evaluate inside an
    inference loop.  It does not have to be: with tau_i = a * t_i, where t_i are
    the segment times at a reference modulus and `a` collects the dependence on
    the matrix modulus and on constriction,

        sum_i w_i (i omega tau_i) / (1 + i omega tau_i)  =  F(omega * a)

    so the whole family is one function of a single variable.  F is tabulated
    once on a log grid and interpolated, which is exact in the segment times -
    unlike binning the times themselves, which makes the forward map
    discontinuous in the modulus.
    """

    def __init__(self, weights, tau_ref, mu_ref, n=2048, pad=4.0):
        self.w = np.asarray(weights, float)
        self.tau_ref = np.asarray(tau_ref, float)
        self.mu_ref = float(mu_ref)
        lo = np.log10(1.0 / (self.tau_ref.max() * 10 ** pad))
        hi = np.log10(1.0 / (self.tau_ref.min() / 10 ** pad))
        self.x = np.logspace(lo, hi, n)
        z = 1j * self.x[:, None] * self.tau_ref[None, :]
        f = np.sum(self.w[None, :] * z / (1.0 + z), axis=1)
        self.re, self.im = f.real, f.imag

    def scale(self, mu, s):
        """The single variable: tau ~ 1/mu (elastic restoring) and s^-4."""
        return (self.mu_ref / float(mu)) * float(s) ** -4

    def __call__(self, omega, mu, s, dG):
        x = np.asarray(omega, float) * self.scale(mu, s)
        xc = np.clip(x, self.x[0], self.x[-1])
        return dG * (np.interp(xc, self.x, self.re)
                     + 1j * np.interp(xc, self.x, self.im))

    def unit_loss(self, omega, mu, s):
        return self(omega, mu, s, 1.0).imag
