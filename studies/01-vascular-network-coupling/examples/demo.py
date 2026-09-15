"""porovasc demo: build a gland, check it, compress it, perfuse it with a bolus,
and apply the clinical two-kernel CEUS estimator at two kernel sizes.

    python examples/demo.py            (about 1 to 2 minutes; writes demo_result.png)

The gland tree is stopped at 60 um here so that the demo runs quickly; the
study configuration (config.BASE) uses 30 um in the gland and 20 um in the
sampling volumes.
"""
import time, io, contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from porovasc.geometry import network as N
from porovasc.physics import flow as F, drainage as DR, transport as TR
from porovasc.homogenise import darcy as DA
from porovasc.debug import check_geometry as CG, check_flow as CF

t0 = time.time()
centre = np.array([-5e-3, -8e-3, 0.0])                 # one 6 mm sampling volume
prm = N.Params(seed=1, d_term_gland=60e-6, rve_centres=(tuple(centre),))
net = N.build(prm)
print("network: %d segments in %.0f s" % (net.n, time.time() - t0))

with contextlib.redirect_stdout(io.StringIO()):
    ok_geom = CG.check(net)
fl = F.solve(net)
with contextlib.redirect_stdout(io.StringIO()):
    ok_flow = CF.check(net, fl)
Q = np.abs(fl.Q[net.gen == 0]).sum() * 6e7
mass = 4 / 3 * np.pi * np.prod(N.GLAND_SEMI) * 1e6 * 1.05
print("invariants: geometry %s, flow %s; perfusion %.1f mL/min/100 g (not imposed)" % (ok_geom, ok_flow, 100 * Q / mass))

# compression of a 3 mm sphere at the sampling volume: relaxation amplitude and time
dr = DR.solve(net, fl, centre, R_comp=3e-3, dP_ext=100.0, H=5e3)
print("drainage: phi_region %.2f %%, fractional relaxation / applied strain = %.4f (= phi), tau_rc %.1f ms"
      % (100 * dr.phi, -dr.dV_inf / dr.V_tissue / (100.0 / 5e3), dr.tau_rc * 1e3))

# permeabilities measured on the cube
da = DA.measure(net, fl, centre, prm.rve_half)
k_rc_index = F.ETA * dr.phi * (3e-3) ** 2 / (dr.tau_rc * 5e3)
print("permeability descriptors: bundle %.2e, k_rc_index %.2e, network face-to-face mean %.2e m^2; c_k_rc %.0f (bundle 32<T^2> = %.0f)"
      % (da.phi * da.d_perm ** 2 / 32, k_rc_index, da.k_mean, da.phi * da.d_perm ** 2 / k_rc_index, da.c_k_bundle))

# CEUS: eight input voxels in the sampling volume, each with a shell of output
# voxels at 1 and 2 mm; the clinical two-kernel estimator on each
vox = 0.375e-3
rng = np.random.default_rng(0)
inputs = [centre + rng.uniform(-2.2e-3, 2.2e-3, 3) for _ in range(8)]
res = {}
for R in (1e-3, 2e-3):
    allk, index = [], []
    for c in inputs:
        ks = TR.shell_kernels(c, R, vox, n_dir=48); index.append(len(allk)); allk += ks
    tr = TR.propagate(net, fl, allk, poiseuille=False)
    est = []
    for i0 in index:
        if tr.tic[i0].sum() == 0: continue
        outs = [tr.tic[k] for k in range(i0 + 1, i0 + 49)]
        v, D, n, r2 = TR.identify_shell(tr.t, tr.tic[i0], outs, R, tf="new")
        if np.isfinite(v) and 1.5e-4 < v < 0.09 and 1.5e-9 < D < 0.9e-4: est.append((v, D))
    est = np.array(est); res[R] = (tr, index)
    print("CEUS kernel %.0f mm: median v = %.2f mm/s, D = %.2f mm^2/s over %d voxels with a finite, in-bounds fit"
          % (R * 1e3, np.median(est[:, 0]) * 1e3, np.median(est[:, 1]) * 1e6, len(est)))

# figure: vessels in a 1 mm slab through the sampling volume, and the curves
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
mid = 0.5 * (net.p0 + net.p1)
m = (np.abs(mid[:, 2] - centre[2]) < 0.5e-3) & np.all(np.abs(mid[:, :2] - centre[:2]) < prm.rve_half, axis=1)
for i in np.where(m)[0]:
    ax[0].plot([net.p0[i, 0] * 1e3, net.p1[i, 0] * 1e3], [net.p0[i, 1] * 1e3, net.p1[i, 1] * 1e3],
               color="#c0392b" if net.kind[i] == 0 else "#7b9fd6", lw=max(0.3, net.d[i] * 1e6 / 80))
ax[0].set_aspect("equal"); ax[0].set_title("1 mm slab through the sampling volume", fontsize=9)
ax[0].set_xlabel("mm"); ax[0].set_ylabel("mm")
tr, index = res[1e-3]
i0 = max(index, key=lambda i: tr.tic[i].max())
shell = np.mean([tr.tic[k] for k in range(i0 + 1, i0 + 49)], axis=0)
ax[1].plot(tr.t, tr.tic[i0] / tr.tic[i0].max(), "k", label="input voxel")
ax[1].plot(tr.t, shell / shell.max(), color="#c0392b", label="shell at 1 mm (mean)")
ax[1].set_xlim(0, 60); ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("bubbles present (normalized)")
ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("time-intensity curves", fontsize=9)
plt.tight_layout(); plt.savefig("demo_result.png")
print("figure written to demo_result.png (%.0f s total)" % (time.time() - t0))
