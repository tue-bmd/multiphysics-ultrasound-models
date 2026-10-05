import sys, pickle, json
from pathlib import Path
RES = Path(__file__).resolve().parents[2] / 'results'      # the study's stored outputs
OUT = RES / 'overview_figure'; OUT.mkdir(exist_ok=True)   # intermediates and figure data
import numpy as np
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 1
from vmconf import mech, ceus as CE
from vmconf import inference as INF
P=pickle.load(open(OUT / f'fwd_seed{SEED}.pkl','rb'))
d=json.load(open(RES / f'inference_d30_seed{SEED}.json')); st=d['provenance']['settings']
resp=mech.VascularResponse(P['w'],P['tau_ref'],P['mu']); interp=CE.Interpolant(P['tab'])
grid=INF.Grid3(np.linspace(0.5*P['mu'],1.6*P['mu'],st['n_mu']), np.linspace(0.3*P['eta'],2.2*P['eta'],st['n_eta']), np.linspace(interp.s[0],interp.s[-1],st['n_s']))
sp=INF.SwePredictor(grid,resp,P['omega'],P['dG'],with_alpha=st['with_alpha'])
acc={'swe':0,'joint_shared':0}; inside={'swe':0,'joint_shared':0}
for rep,data in enumerate(d['observations']):
    for arm in acc:
        lp=INF.log_posterior(grid,arm,data,sp,st['sigma_swe'],st['sigma_ceus'],ceus_interp=interp,s_true=st['s_true'],relation_sd=st['relation_sd'],relation_offset=0.0)
        p=np.exp(lp-lp.max()); p/=p.sum(); acc[arm]=acc[arm]+p.sum(axis=1)/len(d['observations'])
G=dict(np.load(OUT / f'post_seed{SEED}.npz'))
G['swe_mean']=acc['swe']; G['cpl_mean']=acc['joint_shared']
np.savez(OUT / f'post_seed{SEED}.npz', **G)
# where does the truth sit in the mean posteriors? HPD level containing the truth
for arm in acc:
    p=acc[arm]; i=np.argmin(abs(grid.mu-2000)); j=np.argmin(abs(grid.s-0.8)); pt=p[i,j]
    print(arm,'truth lies inside the HPD region of mass %.2f'%(p[p>=pt].sum()))
