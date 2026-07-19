#!/usr/bin/env python3
"""Chunked stochastic run (state save/resume) so a full T=1e5 trajectory fits the
45s/call cap.  Usage: python3 mdp_stoch.py <alpha> <chunk_idx> <n_chunks>.
Runs HT-FTRL-OM (known P), HT-FTRL-UOB (unknown P) and the bounded-loss CONTROL
(unknown P, no skipping) on the SAME genuine layered MDP, fixed gap GST."""
import os, sys, json, pickle
import numpy as np, mdp_core as m
from mdp_run import slope, fits, peak_fall, H, S, A, GST, NST, TMAX, NBIS, CKPTS, ASY_T0
def main():
    alpha=float(sys.argv[1]); ci=int(sys.argv[2]); nc=int(sys.argv[3]); ns=NST
    sf='_cache/st_%s.pkl'%str(alpha); t_end=int(round(TMAX*(ci+1)/nc))
    if ci==0:
        is_uob=np.r_[np.zeros(ns,bool),np.ones(ns,bool),np.ones(ns,bool)]
        is_skip=np.r_[np.ones(ns,bool),np.ones(ns,bool),np.zeros(ns,bool)]
        P,lb,V,adv,nS=m.build_mdp(H,S,A,GST)
        o,st=m.run_episodes(alpha,P,lb,adv,nS,is_uob,is_skip,GST,20260717,t_end,CKPTS,nbis=NBIS)
        out=o
    else:
        out,st,is_uob,is_skip,P,lb,adv,nS=pickle.load(open(sf,'rb'))
        o,st=m.run_episodes(alpha,P,lb,adv,nS,is_uob,is_skip,GST,0,t_end,CKPTS,nbis=NBIS,state=st)
        out=out+o
    pickle.dump((out,st,is_uob,is_skip,P,lb,adv,nS),open(sf,'wb'))
    print('chunk',ci,'/',nc,'t_end',t_end,'reg_uob@end',round(float(out[ns:2*ns,-1].mean()),1))
    if ci==nc-1:
        sd_om=out[:ns]; sd_uob=out[ns:2*ns]; sd_ctl=out[2*ns:]
        reg_om=sd_om.mean(0); reg_uob=sd_uob.mean(0); reg_ctl=sd_ctl.mean(0)
        widx=[i for i,T in enumerate(CKPTS) if T>=ASY_T0]; wck=[CKPTS[i] for i in widx]
        fom=fits(CKPTS,reg_om); fuo=fits(CKPTS,reg_uob)
        pf_om,_,rs_om=peak_fall(CKPTS,reg_om); pf_uo,_,rs_uo=peak_fall(CKPTS,reg_uob)
        p90u=np.percentile(sd_uob,90,axis=0); p90c=np.percentile(sd_ctl,90,axis=0)
        fu,_,_=peak_fall(CKPTS,p90u); fc,_,_=peak_fall(CKPTS,p90c)
        stu=float(sd_uob[:,-1].std()); stc=float(sd_ctl[:,-1].std()); ratio=stc/stu if stu>0 else 1e9
        mom,var=m.empirical_moment(alpha)
        ev=dict(regime='stoch',alpha=alpha,H=H,S=S,A=A,gap=GST,tmax=TMAX,ckpts=CKPTS,asy_t0=ASY_T0,nst=ns,
            reg_om=[float(v) for v in reg_om],reg_uob=[float(v) for v in reg_uob],reg_ctl=[float(v) for v in reg_ctl],
            slope_om_full=slope(CKPTS,reg_om),slope_om_win=slope(wck,reg_om[widx]),
            slope_uob_full=slope(CKPTS,reg_uob),slope_uob_win=slope(wck,reg_uob[widx]),
            r2_om=fom,r2_uob=fuo,om_falls=bool(pf_om),uob_falls=bool(pf_uo),regsqrt_om=rs_om,regsqrt_uob=rs_uo,
            ctl=dict(uob_p90_falls=bool(fu),ctl_p90_falls=bool(fc),std_uob=stu,std_ctl=stc,std_ratio=ratio,
                non_robust=bool((not fc) or ratio>=2.0)),noise_moment=mom,noise_var=var,numpy=np.__version__)
        json.dump(ev,open('_cache/stoch_%s.json'%str(alpha),'w'),indent=1)
        print('WROTE _cache/stoch_%s.json OMwin=%.3f UOBwin=%.3f uobfalls=%s ctl_nonrobust=%s ratio=%.2f'%(
            str(alpha),ev['slope_om_win'],ev['slope_uob_win'],ev['uob_falls'],ev['ctl']['non_robust'],ratio))
if __name__=='__main__': main()
