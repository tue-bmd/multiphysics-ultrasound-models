import sys, json, pickle
from pathlib import Path
RES = Path(__file__).resolve().parents[2] / 'results'      # the study's stored outputs
OUT = RES / 'overview_figure'; OUT.mkdir(exist_ok=True)   # intermediates and figure data
import numpy as np
from porovasc.geometry import network as N
from porovasc.geometry.network import cube_fraction
seed=int(sys.argv[1]) if len(sys.argv)>1 else 1
blob=json.load(open(RES / f'ceus_d30_seed{seed}.json')); cfg=blob['provenance']['settings']
centre=(0.0,-5e-3,0.0); half=3e-3
net=N.build(N.Params(n_feeders=4, d_term_gland=cfg['dterm'], d_term_rve=max(cfg['dterm']-10e-6,20e-6), rve_centres=(centre,), rve_half=half, seed=cfg['seed']))
frac=cube_fraction(net.p0,net.p1,np.asarray(centre),half)
from porovasc.geometry.network import GLAND_SEMI
np.savez(OUT / f'net_seed{seed}.npz', p0=net.p0, p1=net.p1, r=net.r, Lpath=net.Lpath, frac=frac, centre=np.array(centre), half=half, kind=net.kind, gen=net.gen, gland_semi=np.asarray(GLAND_SEMI))
print('saved', net.p0.shape, 'in cube:', int((frac>0).sum()))
