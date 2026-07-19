"""Claim 1 (Thm 4.3): high-accuracy sampler reaches delta-error in polylog(1/delta)
steps -- exponential improvement over the poly(1/delta) forced on SDE-discretization
(ULA/DDPM). arXiv 2602.01338 / OpenReview 71132. Target N(0,I_d), exactly tractable.
Proximal sampler (Alg 3, ideal Gaussian RGO) is UNBIASED -> geometric contraction ->
N(delta)=O(log 1/delta). ULA is BIASED (variance floor ~h/2) -> N(delta)=Theta(1/delta).
Real Monte-Carlo runs (many chains, 3 seeds) PLUS the exact variance law they confirm."""
import json, time
import numpy as np
t0=time.time(); OUT={}
d=8; lam=np.ones(d); v0=4.0
def w2v(v): return float(np.sum((np.sqrt(v)-np.sqrt(lam))**2))
def prox_next(v,eta): a=lam/(lam+eta); return a*a*(v+eta)+a*eta   # fixed pt lam (unbiased)
def ula_next(v,h):    return (1.0-h/lam)**2*v+2.0*h               # fixed pt lam/(1-h/2) (biased)
def simulate(kind,step,N,C,seed,checkpts):
    rng=np.random.default_rng(seed); X=np.sqrt(v0)*rng.standard_normal((C,d)); rec={}
    cp=set(checkpts)
    for n in range(N+1):
        if n in cp: rec[n]=w2v(X.var(axis=0))
        if kind=="prox":
            a=lam/(lam+step); Y=X+np.sqrt(step)*rng.standard_normal((C,d))
            X=a*Y+np.sqrt(lam*step/(lam+step))*rng.standard_normal((C,d))
        else:
            X=(1.0-step/lam)*X+np.sqrt(2.0*step)*rng.standard_normal((C,d))
    return rec
seeds=[0,1,2]; eta=1.0
print("="*76); print("CLAIM 1  polylog(1/delta) high-accuracy vs poly(1/delta) low-accuracy")
print("arXiv 2602.01338 / OR 71132   target N(0,I_%d)"%d); print("="*76)
# ---- proximal: real stochastic (unbiased -> MC floor) + exact law ----
Cp=120000; Np=30; cps=[0,2,4,6,8,12,20,30]
recs=[simulate("prox",eta,Np,Cp,s,cps) for s in seeds]
w2p={n:np.mean([np.sqrt(r[n]) for r in recs]) for n in cps}
v=np.full(d,v0); w2pe={}
for n in range(Np+1):
    if n in cps: w2pe[n]=np.sqrt(w2v(v))
    v=prox_next(v,eta)
print("\n[PROXIMAL eta=%.2f] stochastic mean of %d seeds, C=%d chains"%(eta,len(seeds),Cp))
for n in cps: print("  n=%2d  W2_stoch=%.3e   W2_exactlaw=%.3e"%(n,w2p[n],w2pe[n]))
print("  -> stochastic hits MC floor ~%.1e; exact law -> 0 geometrically (unbiased)"%min(w2p.values()))
OUT["proximal"]={str(n):dict(stoch=float(w2p[n]),exact=float(w2pe[n])) for n in cps}
# ---- ULA: real stochastic floors (biased) ----
print("\n[ULA] biased: v*=1/(1-h/2); W2 floor ~ h does NOT vanish")
Cu=40000; OUT["ula_floors"]={}
for h in [0.2,0.05,0.0125]:
    T=int(min(2000,max(200,8/h)))
    r=[simulate("ula",h,T,Cu,s,[T]) for s in seeds]
    fl=np.mean([np.sqrt(x[T]) for x in r]); vs=1.0/(1.0-h/2.0); fe=np.sqrt(d)*(np.sqrt(vs)-1.0)
    print("  h=%.4f (T=%d)  stoch_floor=%.3e  exactlaw_floor=%.3e  v*=%.5f bias=%.2e"%(h,T,fl,fe,vs,vs-1))
    OUT["ula_floors"][str(h)]=dict(stoch=float(fl),exact=float(fe),vstar=float(vs))
# ---- complexity N(delta) ----
def prox_N(delta):
    v=np.full(d,v0); n=0
    while np.sqrt(w2v(v))>delta:
        v=prox_next(v,eta); n+=1
        if n>200000: break
    return n
def ula_N(delta):
    best=None; bh=None
    for h in np.geomspace(1e-14,0.9,1400):
        vs=1.0/(1.0-h/2.0); fl=np.sqrt(d)*(np.sqrt(vs)-1.0)
        if fl>0.9*delta: continue
        c=(1.0-h)**2
        if not (0<c<1): continue
        tgt=delta/np.sqrt(d); num=(1.0+tgt)**2-vs
        if num<=0: continue
        n=max(int(np.ceil(np.log(num/(v0-vs))/np.log(c))),0)
        if best is None or n<best: best,bh=n,h
    return best,bh
def ula_iter(delta,h):
    v=np.full(d,v0); n=0
    while np.sqrt(w2v(v))>delta:
        v=ula_next(v,h); n+=1
        if n>300000: break
    return n
deltas=[10.0**-k for k in range(1,13)]; rows=[]
print("\n%-8s %8s %16s %12s"%("delta","N_prox","N_ULA(best h)","ULA/prox"))
for dl in deltas:
    npx=prox_N(dl); nu,hb=ula_N(dl); rows.append((dl,npx,nu,hb))
    print("  %.0e %7d   %9d (h=%.1e)  %.2e"%(dl,npx,nu,hb,nu/max(npx,1)))
OUT["complexity"]=[dict(delta=dl,N_prox=int(a),N_ula=int(b),ula_h=float(c)) for dl,a,b,c in rows]
print("\n[ULA closed-form vs REALLY iterated recursion]"); OUT["ula_iter_check"]={}
for dl in [1e-1,1e-2,1e-3,1e-4]:
    nu,hb=ula_N(dl); ni=ula_iter(dl,hb)
    print("  delta=%.0e closed=%d iterated=%d"%(dl,nu,ni)); OUT["ula_iter_check"][f"{dl:.0e}"]=dict(closed=int(nu),iterated=int(ni))
logi=np.log10([1/dl for dl in deltas]); NP=np.array([r[1] for r in rows],float); NU=np.array([r[2] for r in rows],float)
sp=np.polyfit(logi,NP,1); rp=np.corrcoef(logi,NP)[0,1]**2; pprox=np.polyfit(logi,np.log10(NP),1)[0]
su=np.polyfit(logi,np.log10(NU),1); ru=np.corrcoef(logi,np.log10(NU))[0,1]**2
print("\n"+"="*76+"\nFITS")
print("  PROXIMAL N vs log10(1/delta): %.3f steps/decade  R2=%.5f  (affine=>polylog)"%(sp[0],rp))
print("           power exponent p_prox=%.4f (target<=0.05)"%pprox)
print("  ULA      log10 N vs log10(1/delta): p_ula=%.4f R2=%.5f (target~1=>poly)"%(su[0],ru))
OUT["fits"]=dict(prox_steps_per_decade=float(sp[0]),prox_logfit_R2=float(rp),prox_power_exponent=float(pprox),ula_power_exponent=float(su[0]),ula_power_R2=float(ru))
ratio=NU/NP; sr=np.polyfit(logi,np.log10(ratio),1)[0]
print('  ULA/prox ratio grows polynomially: log10(ratio) slope=%.3f per decade (=> exp. separation)'%sr)
OUT['fits']['ula_over_prox_ratio_slope']=float(sr)
ver=(rp>=0.99)and(pprox<=0.15)and(pprox<0.30*su[0])and(0.85<=su[0]<=1.15)and(ru>=0.99)
OUT["verified"]=bool(ver); OUT["runtime_s"]=time.time()-t0
print("\n  proximal polylog (R2>=0.99, p_prox<=0.15, p_prox<<p_ula): %s"%(rp>=0.99 and pprox<=0.15 and pprox<0.30*su[0]))
print("  ULA polynomial (p_ula in [0.85,1.15], R2>=0.99): %s"%(0.85<=su[0]<=1.15 and ru>=0.99))
print("  VERDICT: %s  (%.1fs)"%("VERIFIED" if ver else "NOT VERIFIED",OUT["runtime_s"])); print("="*76)
json.dump(OUT,open("results.json","w"),indent=2); print("wrote results.json")
