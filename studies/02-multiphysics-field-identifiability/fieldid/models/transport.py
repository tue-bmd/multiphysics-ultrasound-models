"""Microbubble occupancy: tracer conservation in a perfused bed.

    d(phi c)/dt + div J = s_c,      J = phi c v - phi D grad c

with c the intravascular concentration, b = phi c the occupancy per unit tissue
volume, v an effective tracer drift, D an effective network dispersion and s_c
delivery not imposed through boundary conditions.

v is an effective drift, not a mean vessel speed.  The closure
q_0 = -(k / eta_b) grad p_0, v = q_0 / phi is not imposed and does not appear:
there is no pressure, no permeability and no blood viscosity in this module.

**Delivery is a modeling choice, and it decides whether phi appears at all.**
For spatially uniform phi the transport operator does not contain phi: it
cancels from d(phi c)/dt + div(phi c v - phi D grad c) = s_c.  Whatever phi
dependence the occupancy has therefore comes from the source, and two deliveries
are implemented:

  ``concentration`` (default)
      s_c = phi A delta(r - r_s) delta(t - t_0): the tracer arrives dissolved in
      blood at concentration amplitude A, so a voxel receives an amount in
      proportion to the vascular volume it holds.  Then

          b(r, t) = phi A (4 pi D tau)^(-3/2) exp( - |d - v tau|^2 / (4 D tau) )

      and the spatial integral is phi A: absolute occupancy constrains the
      product, and phi alone only if A is known.

  ``amount``
      s_c = A delta(r - r_s) delta(t - t_0): a fixed quantity of tracer is
      delivered regardless of how much vasculature is present.  Then the
      occupancy carries no phi at all and the spatial integral is A, so absolute
      occupancy says nothing whatever about phi.

Neither is derived here.  The first is the one the rest of the study uses, and
every statement about the vascular volume fraction is conditional on it.
"""
from __future__ import annotations
import numpy as np


def drift_vector(vmag, vth, vaz):
    """v = |v| (cos th, sin th cos az, sin th sin az) [m s^-1].

    The polar angle is measured from the x axis, which is the nominal
    downstream direction fixed by anatomy in the B-mode frame."""
    st = np.sin(vth)
    return np.array([vmag * np.cos(vth), vmag * st * np.cos(vaz),
                     vmag * st * np.sin(vaz)], float)


DELIVERIES = ("concentration", "amount")


def occupancy(r, t, p, r_src=(0.0, 0.0, 0.0), delivery="concentration"):
    """Occupancy b(r, t) [arb m^-3] on positions ``r`` (n, 3) [m] and times
    ``t`` (m,) [s].  Returns shape (n, m).  Zero before arrival.

    ``delivery`` selects the source: ``concentration`` scales it by phi, so the
    spatial integral is phi A; ``amount`` does not, so the integral is A and the
    occupancy is independent of phi."""
    if delivery not in DELIVERIES:
        raise ValueError("unknown delivery %r" % delivery)
    r = np.atleast_2d(np.asarray(r, float))
    t = np.asarray(t, float)
    v = drift_vector(p["vmag"], p["vth"], p["vaz"])
    d = r - np.asarray(r_src, float)[None, :]
    tau = t[None, :] - p["t0"]
    ok = tau > 0
    taus = np.where(ok, tau, 1.0)
    disp = d[:, :, None] - v[None, :, None] * taus[:, None, :]
    r2 = np.sum(disp ** 2, axis=1)
    scale = p["phi"] * p["A"] if delivery == "concentration" else p["A"]
    amp = scale * (4 * np.pi * p["D"] * taus) ** -1.5
    return np.where(ok, amp * np.exp(-r2 / (4 * p["D"] * taus)), 0.0)


def normalize(b, t):
    """Normalize every curve to unit time integral.

    This removes phi and A exactly and simultaneously, which is the point of the
    normalized-curve comparison: the two are degenerate in the absolute data and
    both absent from the normalized data."""
    area = np.trapezoid(b, t, axis=-1)
    area = np.where(area > 0, area, 1.0)
    return b / area[..., None]


def total_mass(b, grid_spacing):
    """Riemann sum of b over a regular grid, for the conservation test."""
    return b.sum(axis=0) * float(np.prod(grid_spacing))


def moments(b, t):
    """Zeroth, first and second central time moments of each curve."""
    m0 = np.trapezoid(b, t, axis=-1)
    safe = np.where(m0 > 0, m0, 1.0)
    m1 = np.trapezoid(b * t, t, axis=-1) / safe
    m2 = np.trapezoid(b * (t - m1[..., None]) ** 2, t, axis=-1) / safe
    return m0, m1, m2
