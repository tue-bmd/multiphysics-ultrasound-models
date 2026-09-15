"""Drainage under compression: compliant RC network.

Each segment i owns lumen volume V_i = pi r_i^2 Lpath_i and compliance
    C_i = V_i / (H + K_wall)          [m^3 / Pa]
(vessel embedded in an elastic matrix of modulus H; K_wall adds wall
stiffness, default 0).

Compression: an external pressure step applied at t = 0 to the part of each
vessel that lies inside a sphere of radius R_comp about `centre`.  A segment
half inside the sphere is pressurised over half of its length, so the support of
the compression is the same clipped support on which the descriptors are
measured, and no volume is counted twice.
Undrained response: node pressures jump by dP_ext there.  Then

    C dp/dt = -(L p - b)          L, b : conductance Laplacian and Dirichlet
                                          terms from physics.flow (G fixed)

integrated by backward Euler on log-spaced steps.  The drained-volume time
course is  dV(t) = sum_i C_i [ (p_i(t) - Pext_i) - p_i^rest ]  over the
compressed segments only (those with a part inside the sphere), normalized by
its long-time value.  Uncompressed vessels also change volume during the
transient, because the expelled blood is stored in them before it reaches the
boundary reservoirs; that storage is not part of the local observable, which is
the volume that leaves the compressed region.  The sum over the whole network
would instead be the efflux into the reservoirs.

tau_rc is the time at which f reaches 1 - 1/e = 0.632, that is the e-folding
time of the residual 1 - f; it is an output, and the spectrum shows how far the
relaxation is from a single exponential.

The relaxation spectrum is recovered by non-negative least squares on a
log-spaced tau grid (Parker-style), so a distribution of time constants is
reported rather than one number.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.optimize import nnls
from dataclasses import dataclass
from ..geometry.network import Network, sphere_fraction
from ..physics import flow as _flow
from ..physics.flow import Flow, ETA


@dataclass
class Drainage:
    t: np.ndarray            # time grid [s]
    f: np.ndarray            # drained fraction 0 -> 1
    dV_inf: float            # total volume expelled from the compressed region [m^3]
    V_blood: float           # blood volume in the compressed region at rest [m^3]
    V_tissue: float          # tissue volume of the compressed region [m^3]
    phi: float               # V_blood / V_tissue
    tau_rc: float            # time to f = 1 - 1/e (e-folding time of the residual 1 - f)
    tau_grid: np.ndarray     # spectrum support
    spectrum: np.ndarray     # NNLS weights, sum = 1
    in_region: np.ndarray    # mask of compressed segments
    dP_ext: float; H: float
    frac: np.ndarray = None  # fraction of each segment inside the compressed sphere


def _laplacian(net: Network, fl: Flow, radius_scale=None):
    n = net.n; N = n + 2
    r = net.r if radius_scale is None else net.r * radius_scale
    G = np.pi * r ** 4 / (8 * ETA * net.Lpath)
    Gb = fl.G_bed0 * (2 * fl.bed_r / net.params.d_term_rve) ** 3 * fl.bed_w
    rows = np.concatenate([fl.start, np.arange(n), fl.bed_a, fl.bed_v])
    cols = np.concatenate([np.arange(n), fl.start, fl.bed_v, fl.bed_a])
    vals = np.concatenate([G, G, Gb, Gb])
    A = sp.coo_matrix((-vals, (rows, cols)), shape=(N, N)).tocsr()
    A = A + sp.diags(-np.asarray(A.sum(axis=1)).ravel())
    return A


def solve(net: Network, fl: Flow, centre, R_comp, dP_ext=500.0, H=5e3, K_wall=0.0,
          t_max=2.0, n_t=80, radius_scale=None) -> Drainage:
    n = net.n; N = n + 2
    r = net.r if radius_scale is None else net.r * radius_scale
    Lfull = _laplacian(net, fl, radius_scale)
    inlet, outlet = fl.inlet, fl.outlet
    keep = np.ones(N, bool); keep[[inlet, outlet]] = False
    idx = np.where(keep)[0]
    L = Lfull[idx][:, idx].tocsc()
    pbc = np.zeros(N); pbc[inlet] = _flow.P_ART; pbc[outlet] = _flow.P_VEN
    b = -(Lfull[idx][:, [inlet, outlet]] @ pbc[[inlet, outlet]])

    V = np.pi * r ** 2 * net.Lpath                       # segment volumes
    C = V / (H + K_wall)                                 # compliance per node
    frac = sphere_fraction(net.p0, net.p1, np.asarray(centre), R_comp)   # part inside
    region = frac > 0
    Pext = dP_ext * frac                                 # partial-volume loading

    # rest state: the steady solution of THIS system, L p_rest = b.  It equals
    # the flow solution unless the radii are rescaled, in which case fl.p is
    # not a steady state and the relaxation would drift instead of converging.
    p_rest = fl.p[idx] if radius_scale is None else spl.spsolve(L, b)
    # undrained jump
    p = p_rest + Pext[idx]
    Cd = sp.diags(C[idx])
    t = np.concatenate([[0.0], np.logspace(-4, np.log10(t_max), n_t)])
    dV = np.zeros(len(t))
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        # backward Euler: (C + dt L) p_new = C p + dt b
        M = (Cd + dt * L).tocsc()
        p = spl.spsolve(M, Cd @ p + dt * b)
        # volume change of the compressed segments relative to rest; the
        # loading fraction is already in Pext, so it must not be applied again,
        # and the uncompressed segments (transient storage) are excluded
        dV[k] = np.sum((C[idx] * ((p - Pext[idx]) - p_rest))[region[idx]])
    dV_inf = -np.sum(C * Pext)                           # equilibrium of the same sum
    V_blood = float((V * frac).sum()); V_tissue = 4 / 3 * np.pi * R_comp ** 3
    tau_grid = np.logspace(-4, np.log10(t_max), 40)
    # No segment has any part inside the sphere (can happen for a small
    # R_comp in a sparse pocket of the gland): frac and Pext are identically
    # zero, so dV and dV_inf are both exactly zero and f = 0/0.  dV_inf is
    # otherwise a well-scaled, sign-definite quantity (-V_blood * dP_ext / H
    # exactly, however small V_blood is), never floating-point noise, so the
    # test is on the region itself, not on a magnitude threshold of the sum.
    if not region.any():
        # There is nothing to drain, so phi and V_blood are correctly zero,
        # but the relaxation time and spectrum are undefined, not zero or
        # infinite; report them as such instead of dividing 0/0 and feeding
        # NaNs into the NNLS spectrum fit, which raises.
        f = np.full(len(t), np.nan)
        tau_rc = np.nan
        w = np.full(len(tau_grid), np.nan)
    else:
        f = dV / dV_inf
        # time to f = 1 - 1/e
        k = np.searchsorted(f, 1 - np.exp(-1))
        tau_rc = float(np.interp(1 - np.exp(-1), f[max(k - 1, 0):k + 1], t[max(k - 1, 0):k + 1])) if k < len(t) else np.nan
        # spectrum by NNLS: f(t) = sum_j w_j (1 - exp(-t / tau_j))
        Amat = 1 - np.exp(-t[:, None] / tau_grid[None, :])
        w, _ = nnls(Amat, f)
        w = w / max(w.sum(), 1e-30)
    return Drainage(t=t, f=f, dV_inf=float(dV_inf), V_blood=float(V_blood), V_tissue=float(V_tissue),
                    phi=float(V_blood / V_tissue), tau_rc=tau_rc, tau_grid=tau_grid, spectrum=w,
                    in_region=region, frac=frac, dP_ext=dP_ext, H=H)
