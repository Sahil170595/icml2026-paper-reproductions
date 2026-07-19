#!/usr/bin/env python3
"""Staged driver: ONE (regime, alpha) of the genuine layered-MDP reproduction of
Theorems 4.1 (HT-FTRL-OM, known P) & 5.1 (HT-FTRL-UOB, unknown P).
Usage: python3 mdp_run.py {adv|stoch} <alpha>  -> _cache/{regime}_{alpha}.json"""
import os, sys, json, time
import numpy as np
import mdp_core as m
H=int(os.environ.get('H','3')); S=int(os.environ.get('S','3')); A=int(os.environ.get('A','3'))
C0=float(os.environ.get('C0','2.0')); GST=float(os.environ.get('GST','0.9'))
NADV=int(os.environ.get('NADV','10')); NST=int(os.environ.get('NST','8'))
TMAX=int(os.environ.get('TMAX','100000')); NBIS=int(os.environ.get('NBIS','5'))
ADV_H=[int(x) for x in os.environ.get('ADV_H','1000,2000,4000,8000,16000,32000,64000').split(',')]
CKPTS=[int(x) for x in os.environ.get('CKPTS','1000,2000,4000,8000,16000,24000,32000,48000,64000,80000,100000').split(',')]
ASY_T0=int(os.environ.get('ASY_T0','24000'))
def r2(y,yh):
    y=np.asarray(y,float); yh=np.asarray(yh,float); sr=np.sum((y-yh)**2); st=np.sum((y-y.mean())**2)
    return float(1.0-sr/st) if st>0 else 0.0
def slope(T,reg):
    T=np.asarray(T,float); reg=np.asarray(reg,float); mk=reg>0
    return float(np.polyfit(np.log(T[mk]),np.log(reg[mk]),1)[0]) if mk.sum()>=2 else float('nan')
def fits(T,reg):
    T=np.asarray(T,float); reg=np.asarray(reg,float); lnT=np.log(T); sq=np.sqrt(T)
    Al=np.vstack([np.ones_like(lnT),lnT]).T; A2=np.vstack([np.ones_like(lnT),lnT,lnT**2]).T; As=np.vstack([np.ones_like(sq),sq]).T
    cl,*_=np.linalg.lstsq(Al,reg,rcond=None); c2,*_=np.linalg.lstsq(A2,reg,rcond=None); cs,*_=np.linalg.lstsq(As,reg,rcond=None)
    return dict(r2_log=r2(reg,Al@cl),r2_log2=r2(reg,A2@c2),r2_sqrt=r2(reg,As@cs))
def peak_fall(T,reg):
    T=np.asarray(T,float); reg=np.asarray(reg,float); rs=reg/np.sqrt(T); pk=int(np.argmax(rs))
    falls=bool(pk<=len(rs)-2 and rs[-1]<rs[pk] and rs[-1]<rs[-2]); return falls,pk,[float(v) for v in rs]
def do_adv(alpha):
    ns=NADV; is_uob=np.r_[np.zeros(ns,bool),np.ones(ns,bool)]; is_skip=np.ones(2*ns,bool); ro=[]; ru=[]
    for T in ADV_H:
        G=C0*(float(T)**(1.0/alpha-1.0)); P,lb,V,adv,nS=m.build_mdp(H,S,A,G)
        o,_=m.run_episodes(alpha,P,lb,adv,nS,is_uob,is_skip,G,seed=4000+T,tmax=T,ckpts=[T],nbis=NBIS)
        ro.append(float(o[:ns,0].mean())); ru.append(float(o[ns:,0].mean()))
    s_om=slope(ADV_H,ro); s_uob=slope(ADV_H,ru); tgt=1.0/alpha
    return dict(regime='adv',alpha=alpha,H=H,S=S,A=A,C0=C0,horizons=ADV_H,reg_om=ro,reg_uob=ru,
        slope_om=s_om,slope_uob=s_uob,target=tgt,match_om=bool(abs(s_om-tgt)<=0.12),
        match_uob=bool(abs(s_uob-tgt)<=0.15),uob_minus_om=[ru[i]-ro[i] for i in range(len(ADV_H))])
def do_stoch(alpha):
    ns=NST
    is_uob=np.r_[np.zeros(ns,bool),np.ones(ns,bool),np.ones(ns,bool)]
    is_skip=np.r_[np.ones(ns,bool),np.ones(ns,bool),np.zeros(ns,bool)]
    P,lb,V,adv,nS=m.build_mdp(H,S,A,GST)
    o,_=m.run_episodes(alpha,P,lb,adv,nS,is_uob,is_skip,GST,seed=20260717,tmax=TMAX,ckpts=CKPTS,nbis=NBIS)
    sd_om=o[:ns]; sd_uob=o[ns:2*ns]; sd_ctl=o[2*ns:]
    reg_om=sd_om.mean(0); reg_uob=sd_uob.mean(0); reg_ctl=sd_ctl.mean(0)
    widx=[i for i,T in enumerate(CKPTS) if T>=ASY_T0]; wck=[CKPTS[i] for i in widx]
    fom=fits(CKPTS,reg_om); fuo=fits(CKPTS,reg_uob)
    pf_om,_,rs_om=peak_fall(CKPTS,reg_om); pf_uo,_,rs_uo=peak_fall(CKPTS,reg_uob)
    p90_uob=np.percentile(sd_uob,90,axis=0); p90_ctl=np.percentile(sd_ctl,90,axis=0)
    fu,_,_=peak_fall(CKPTS,p90_uob); fc,_,_=peak_fall(CKPTS,p90_ctl)
    std_u=float(sd_uob[:,-1].std()); std_c=float(sd_ctl[:,-1].std()); ratio=std_c/std_u if std_u>0 else 1e9
    return dict(regime='stoch',alpha=alpha,H=H,S=S,A=A,gap=GST,tmax=TMAX,ckpts=CKPTS,asy_t0=ASY_T0,
        reg_om=[float(v) for v in reg_om],reg_uob=[float(v) for v in reg_uob],reg_ctl=[float(v) for v in reg_ctl],
        slope_om_full=slope(CKPTS,reg_om),slope_om_win=slope(wck,reg_om[widx]),
        slope_uob_full=slope(CKPTS,reg_uob),slope_uob_win=slope(wck,reg_uob[widx]),
        r2_om=fom,r2_uob=fuo,om_falls=bool(pf_om),uob_falls=bool(pf_uo),regsqrt_om=rs_om,regsqrt_uob=rs_uo,
        ctl=dict(p90_uob=[float(v) for v in p90_uob],p90_ctl=[float(v) for v in p90_ctl],
            uob_p90_falls=bool(fu),ctl_p90_falls=bool(fc),std_uob=std_u,std_ctl=std_c,std_ratio=ratio,
            non_robust=bool((not fc) or ratio>=2.0)))
def main():
    regime=sys.argv[1]; alpha=float(sys.argv[2]); t0=time.time(); mom,var=m.empirical_moment(alpha)
    ev=do_adv(alpha) if regime=='adv' else do_stoch(alpha)
    ev['noise_moment']=mom; ev['noise_var']=var; ev['runtime_s']=round(time.time()-t0,1); ev['numpy']=np.__version__
    os.makedirs('_cache',exist_ok=True); fn='_cache/%s_%s.json'%(regime,str(alpha)); json.dump(ev,open(fn,'w'),indent=1)
    print('WROTE',fn,'runtime',ev['runtime_s'],'s noise_mom=%.3f'%mom)
    if regime=='adv': print(' adv OM/UOB slope=%.3f/%.3f tgt=%.3f match=%s/%s'%(ev['slope_om'],ev['slope_uob'],ev['target'],ev['match_om'],ev['match_uob']))
    else: print(' stoch OMwin=%.3f UOBwin=%.3f uob_falls=%s ctl_nonrobust=%s ratio=%.2f'%(ev['slope_om_win'],ev['slope_uob_win'],ev['uob_falls'],ev['ctl']['non_robust'],ev['ctl']['std_ratio']))
if __name__=='__main__': main()
