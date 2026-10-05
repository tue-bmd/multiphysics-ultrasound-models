import sys, pickle, json, time
from pathlib import Path
RES = Path(__file__).resolve().parents[2] / 'results'      # the study's stored outputs
OUT = RES / 'overview_figure'; OUT.mkdir(exist_ok=True)   # intermediates and figure data
import numpy as np
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 1
from vmconf import mech, ceus as CE
from vmconf import inference as INF
P=pickle.load(open(OUT / f'fwd_seed{SEED}.pkl','rb'))
d=json.load(open(RES / f'inference_d30_seed{SEED}.json')); st=d['provenance']['settings']
resp=mech.VascularResponse(P['w'],P['tau_ref'],P['mu'])
interp=CE.Interpolant(P['tab'])
grid=INF.Grid3(np.linspace(0.5*P['mu'],1.6*P['mu'],st['n_mu']), np.linspace(0.3*P['eta'],2.2*P['eta'],st['n_eta']), np.linspace(interp.s[0],interp.s[-1],st['n_s']))
t0=time.time(); sp=INF.SwePredictor(grid,resp,P['omega'],P['dG'],with_alpha=st['with_alpha']); print('pred tensor',sp.pred.shape,'%.1fs'%(time.time()-t0))
# pick a representative repetition: the one whose swe-only mu median is closest to the 40-rep median of medians
meds=np.array([r['mu']['median'] for r in d['posterior_by_repetition']['swe']])
rep=int(np.argmin(np.abs(meds-np.median(meds)))); print('rep',rep,'swe mu median',meds[rep])
data=d['observations'][rep]
out={}
for arm,off in [('swe',0.0),('joint_shared',0.0),('joint_wrong',0.10),('joint_free',0.0)]:
    lp=INF.log_posterior(grid,arm,data,sp,st['sigma_swe'],st['sigma_ceus'],ceus_interp=interp,s_true=st['s_true'],relation_sd=st['relation_sd'],relation_offset=off)
    p=np.exp(lp-lp.max()); p/=p.sum()
    out[arm]=p.sum(axis=1)   # marginal over eta -> (mu, s)
    print(arm, 'mu summary', grid.summary(lp,0)['lo'], grid.summary(lp,0)['hi'], 's', grid.summary(lp,2)['lo'], grid.summary(lp,2)['hi'])
np.savez(OUT / f'post_seed{SEED}.npz', mu=grid.mu, s=grid.s, rep=rep, **out)
