import sys, json, time, pickle
from pathlib import Path
RES = Path(__file__).resolve().parents[2] / 'results'      # the study's stored outputs
OUT = RES / 'overview_figure'; OUT.mkdir(exist_ok=True)   # intermediates and figure data
import numpy as np
from porovasc.geometry import network as N
from vmconf import ceus as CE, forward, mech
from vmconf import inference as INF

seed=int(sys.argv[1]) if len(sys.argv)>1 else 1
blob=json.load(open(RES / f'ceus_d30_seed{seed}.json'))
tab=blob['table']; cfg=blob['provenance']['settings']
centre=(0.0,-5e-3,0.0)
t0=time.time()
net=N.build(N.Params(n_feeders=4, d_term_gland=cfg['dterm'], d_term_rve=max(cfg['dterm']-10e-6,20e-6),
                     rve_centres=(centre,), rve_half=3e-3, seed=cfg['seed']))
print('network built in %.0f s, segments %d'%(time.time()-t0, len(net.r)))
inf=json.load(open(RES / f'inference_d30_seed{seed}.json'))
st=inf['provenance']['settings']
mu,eta,s_true,share=st['mu'],st['eta'],st['s_true'],st['share']
omega=mech.band(st['f_lo'],st['f_hi'],24)
vol,phi0=forward.region_weights(net,centre,3e-3)
w,tau_ref=mech.segment_times(net.r,net.Lpath,vol,mu)
dG=mech.amplitude_for_share(w,tau_ref,eta,share,f_ref=200.0)
print('phi0 %.4f dG %.1f (stored %.1f)'%(phi0,dG,st['dG']))
resp=mech.VascularResponse(w,tau_ref,mu)
pickle.dump(dict(w=w,tau_ref=tau_ref,mu=mu,eta=eta,s_true=s_true,dG=dG,omega=omega,phi0=phi0,
                 with_alpha=st['with_alpha'], tab=tab), open(OUT / f'fwd_seed{seed}.pkl','wb'))
y=INF.swe_predict(resp,omega,mu,eta,s_true,dG,with_alpha=st['with_alpha'])
print('swe_true first 5:', y[:5], 'stored obs0 first 5:', inf['observations'][0]['swe'][:5])
