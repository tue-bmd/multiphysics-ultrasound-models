"""Collect every array the Study-04 figure plots into one npz (no study code needed to render)."""
import sys, pickle, json
from pathlib import Path
RES = Path(__file__).resolve().parents[2] / 'results'      # the study's stored outputs
OUT = RES / 'overview_figure'; OUT.mkdir(exist_ok=True)   # intermediates and figure data
import numpy as np
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 1
from scipy.optimize import least_squares
from vmconf import mech
from vmconf import inference as INF
P=pickle.load(open(OUT / f'fwd_seed{SEED}.pkl','rb'))
resp=mech.VascularResponse(P['w'],P['tau_ref'],P['mu'])
om=P['omega']; f=om/(2*np.pi)
def pred(mu,eta,s): return INF.swe_predict(resp,om,mu,eta,s,P['dG'],with_alpha=True)
y0=pred(P['mu'],P['eta'],P['s_true'])
states=[(P['s_true'],P['mu'],P['eta'])]
for s_alt in (0.73,0.83):
    r=least_squares(lambda p:(pred(p[0],p[1],s_alt)-y0)/y0,x0=[P['mu'],P['eta']])
    states.append((s_alt,float(r.x[0]),float(r.x[1])))
curves=np.array([pred(mu,eta,s) for s,mu,eta in states])   # (3,48)
for (s,mu,eta),y in zip(states,curves):
    rel=(y-y0)/y0; print("s=%.2f mu=%.0f eta=%.3f  max|dc|=%.2f%% max|da|=%.2f%%"%(s,mu,eta,100*abs(rel[:24]).max(),100*abs(rel[24:]).max()))
tab=P['tab']['rows']; s_tab=np.array([r['s'] for r in tab]); auc=np.array([r['auc'] for r in tab])
# posterior marginals and sweep summaries (3 networks)
G=np.load(OUT / f'post_seed{SEED}.npz')
sweep={}
for seed in (1,2,3):
    d=json.load(open(RES / f'inference_d30_seed{seed}.json'))
    sweep[seed]=dict(off=[o['offset'] for o in d['offset_sweep']],
                     cov=[o['mu']['coverage'] for o in d['offset_sweep']],
                     bias=[o['mu']['bias_pct'] for o in d['offset_sweep']],
                     width=[o['mu']['width_pct'] for o in d['offset_sweep']],
                     w_swe=d['summary']['swe']['mu']['width_pct'], w_free=d['summary']['joint_free']['mu']['width_pct'],
                     w_cpl=d['summary']['joint_shared']['mu']['width_pct'],
                     we_swe=d['summary']['swe']['eta']['width_pct'], we_cpl=d['summary']['joint_shared']['eta']['width_pct'])
off=np.array(sweep[1]['off']); cov=np.array([sweep[k]['cov'] for k in (1,2,3)]); bias=np.array([sweep[k]['bias'] for k in (1,2,3)]); width=np.array([sweep[k]['width'] for k in (1,2,3)])
print('widths mu swe/free/coupled per network:', [(round(sweep[k]['w_swe'],2),round(sweep[k]['w_free'],2),round(sweep[k]['w_cpl'],2)) for k in (1,2,3)])
np.savez(OUT / 'fig_study04_data.npz', f=f, curves=curves, states=np.array(states), y0=y0, s_tab=s_tab, auc=auc, s_true=P['s_true'],
         mu_grid=G['mu'], s_grid=G['s'], post_swe=G['swe'], post_cpl=G['joint_shared'], post_wrong=G['joint_wrong'], rep=int(G['rep']),
         swe_mean=G['swe_mean'], cpl_mean=G['cpl_mean'],
         off=off, cov=cov, bias=bias, width=width,
         w_mu=np.array([[sweep[k]['w_swe'],sweep[k]['w_free'],sweep[k]['w_cpl']] for k in (1,2,3)]),
         w_eta=np.array([[sweep[k]['we_swe'],sweep[k]['we_swe'],sweep[k]['we_cpl']] for k in (1,2,3)]),
         mu_true=P['mu'], eta_true=P['eta'], phi0=P['phi0'], dG=P['dG'], seed=SEED)
print('saved fig_study04_data.npz')
