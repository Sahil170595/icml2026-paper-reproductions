"""Claim 1 (Thm 4.3) on a FAITHFUL diffusion-model target: multimodal Gaussian-MIXTURE
data distribution (non-log-concave), OU forward noising, exact scores along the reverse
path. The paper's high-accuracy sampler = reverse chain whose every step samples the
exact reverse conditional q_{t-h|t}(.|y) -- which is EXACTLY an RGO call on -log q_{t-h}
with eta=(e^{2h}-1) (Alg 3 / Sec 4); realized (i) in closed form for the mixture and
(ii) by gradient-only FORS (Alg 1). Baseline: DDPM/Euler-Maruyama discretization of the
reverse SDE with the SAME exact scores. Deterministic law-evolution on a grid (1-D
bimodal mixture marginal) drives W2 to the discretization floor; stochastic d=8 runs on
a 3-mode mixture confirm. arXiv 2602.01338 / OR 71132.
Stages (argv): ha_law ddpm_a ddpm_b sto8 report all   (cache in _cache/)."""
import json, os, sys, time
import numpy as np
from scipy.special import erf

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,"_cache")
os.makedirs(CACHE,exist_ok=True)
def sv(n,o): json.dump(o,open(os.path.join(CACHE,n+".json"),"w"))
def ld(n): return json.load(open(os.path.join(CACHE,n+".json")))
def has(n): return os.path.exists(os.path.join(CACHE,n+".json"))

# ---------------- 1-D bimodal mixture (law-evolution, deterministic) ----------------
W1=np.array([0.6,0.4]); MU=np.array([-3.0,3.0]); SD=np.array([0.8,1.2]); H=0.5; TD=6.0
def mixp(t):
    a=np.exp(-t); return MU*a, SD**2*a*a+1.0-a*a
def pdf1(x,t):
    m,v=mixp(t); p=0.0
    for i in range(2): p=p+W1[i]*np.exp(-(x-m[i])**2/(2*v[i]))/np.sqrt(2*np.pi*v[i])
    return p
def score1(x,t):
    m,v=mixp(t); num=0.0; den=0.0
    for i in range(2):
        g=W1[i]*np.exp(-(x-m[i])**2/(2*v[i]))/np.sqrt(2*np.pi*v[i])
        num=num+g*(m[i]-x)/v[i]; den=den+g
    return num/np.maximum(den,1e-300)
def w2tv(p,x,dx,M=8192):
    q=pdf1(x,0.0); q=q/(q.sum()*dx)
    P=np.cumsum(p); P=P/P[-1]; Q=np.cumsum(q); Q=Q/Q[-1]
    u=(np.arange(M)+0.5)/M
    return (float(np.sqrt(np.mean((np.interp(u,P,x)-np.interp(u,Q,x))**2))),
            float(0.5*np.sum(np.abs(p-q))*dx))

def stage_ha_law():
    t0=time.time(); G=2801; x=np.linspace(-8.0,8.0,G); dx=x[1]-x[0]
    b=np.exp(-H); rows=x[:,None]
    p0=np.exp(-x*x/2)/np.sqrt(2*np.pi); p0=p0/(p0.sum()*dx)
    P=np.zeros((G,0)); starts=[]
    for k in range(28,0,-1):
        t=k*H; tp=t-H
        mm,vv=mixp(t); mp,vp=mixp(tp)
        l0=np.log(W1[0])-0.5*(x-mm[0])**2/vv[0]-0.5*np.log(vv[0])
        l1=np.log(W1[1])-0.5*(x-mm[1])**2/vv[1]-0.5*np.log(vv[1])
        mx=np.maximum(l0,l1); w0=np.exp(l0-mx); w1=np.exp(l1-mx); s=w0+w1; w0/=s; w1/=s
        K=np.zeros((G,G))
        for i,wi in ((0,w0),(1,w1)):
            prec=1.0/vp[i]+b*b/(1.0-b*b); cv=1.0/prec
            cm=cv*(mp[i]/vp[i]+b*x/(1.0-b*b))
            K+=wi[None,:]*np.exp(-(rows-cm[None,:])**2/(2.0*cv))/np.sqrt(cv)
        K/=K.sum(0)[None,:]
        P=np.concatenate([P,p0[:,None]],axis=1); starts.append(k)
        P=K@(P*dx)/dx
        del K
    res=[]
    for j,k in enumerate(starts):
        w2,tv=w2tv(P[:,j],x,dx); res.append(dict(N=k,T=k*H,W2=w2,TV=tv))
    res.sort(key=lambda r:r["N"])
    sv("c1_ha_law",dict(res=res,G=G,h=H,runtime_s=time.time()-t0))
    for r in res: print("  HA  N=%2d (T=%4.1f)  W2=%.3e  TV=%.3e"%(r["N"],r["T"],r["W2"],r["TV"]))

def ddpm_law(N,G=2001):
    x=np.linspace(-8.0,8.0,G); dx=x[1]-x[0]; h=TD/N
    p=pdf1(x,TD); p=p/(p.sum()*dx)
    sig=np.sqrt(2.0*h); bw=max(4,int(np.ceil(6.0*sig/dx))); offs=np.arange(-bw,bw+1)
    for k in range(N):
        t=TD-k*h
        mu=x+h*(x+2.0*score1(x,t))
        ci=np.rint((mu-x[0])/dx).astype(np.int64); r=x[0]+ci*dx-mu
        vals=np.exp(-((offs[:,None]*dx+r[None,:])**2)/(2.0*sig*sig))
        vals/=vals.sum(0)[None,:]
        w=p*dx; out=np.zeros(G)
        for oi in range(offs.size):
            idx=np.clip(ci+offs[oi],0,G-1)
            out+=np.bincount(idx,weights=w*vals[oi],minlength=G)
        p=out/dx
    return w2tv(p,x,dx)

def stage_ddpm(Ns,tag):
    t0=time.time(); res=[]
    for N in Ns:
        w2,tv=ddpm_law(N); res.append(dict(N=N,W2=w2,TV=tv))
        print("  DDPM N=%5d  W2=%.3e  TV=%.3e"%(N,w2,tv))
    sv("c1_ddpm_"+tag,dict(res=res,runtime_s=time.time()-t0))

# ---------------- d=8 three-mode mixture (stochastic) ----------------
D8=8; W8=np.array([0.5,0.3,0.2]); S8=np.array([0.9,1.1,0.8])
M8=np.zeros((3,D8)); M8[0,0]=3.5; M8[1,0]=-3.0; M8[1,1]=2.5; M8[2,0]=0.5; M8[2,1]=-3.0
def m8(t): a=np.exp(-t); return M8*a, S8**2*a*a+1.0-a*a
def logcomp8(X,t):
    m,v=m8(t)
    return np.stack([np.log(W8[i])-0.5*np.sum((X-m[i])**2,1)/v[i]-0.5*D8*np.log(v[i]) for i in range(3)],1)
def score8(X,t):
    m,v=m8(t); L=logcomp8(X,t); L=L-L.max(1,keepdims=True)
    Wp=np.exp(L); Wp/=Wp.sum(1,keepdims=True)
    g=np.zeros_like(X)
    for i in range(3): g+=Wp[:,i:i+1]*(m[i]-X)/v[i]
    return g
def sample8(C,rng,t=0.0):
    idx=rng.choice(3,size=C,p=W8); m,v=m8(t)
    return m[idx]+np.sqrt(v[idx])[:,None]*rng.standard_normal((C,D8))
U3=[np.eye(D8)[0],np.eye(D8)[1],(np.eye(D8)[0]+np.eye(D8)[1])/np.sqrt(2.0)]
def metric8(X):
    best=0.0
    for u in U3:
        pr=np.sort(X@u); C=pr.size; lv=(np.arange(C)+0.5)/C
        mus=M8@u; sds=S8.copy()
        lo=np.full(C,-16.0); hi=np.full(C,16.0)
        for _ in range(52):
            mid=0.5*(lo+hi)
            F=sum(W8[i]*0.5*(1.0+erf((mid-mus[i])/(sds[i]*np.sqrt(2.0)))) for i in range(3))
            lo=np.where(F<lv,mid,lo); hi=np.where(F<lv,hi,mid)
        best=max(best,float(np.sqrt(np.mean((pr-0.5*(lo+hi))**2))))
    return best
def modeerr8(X):
    d2=np.stack([np.sum((X-M8[i])**2,1) for i in range(3)],1)
    f=np.bincount(np.argmin(d2,1),minlength=3)/X.shape[0]
    return float(np.max(np.abs(f-W8)))
def ha8_exact(C,T,h,rng):
    X=rng.standard_normal((C,D8)); t=T; b=np.exp(-h)
    while t>1e-9:
        tp=max(t-h,0.0); mp,vp=m8(tp)
        L=logcomp8(X,t); L=L-L.max(1,keepdims=True); Wp=np.exp(L); Wp/=Wp.sum(1,keepdims=True)
        u=rng.random((C,1)); cs=np.cumsum(Wp,1); idx=(u>cs[:,:-1]).sum(1)
        prec=1.0/vp[idx]+b*b/(1.0-b*b); cv=1.0/prec
        cm=cv[:,None]*(mp[idx]/vp[idx][:,None]+b*X/(1.0-b*b))
        X=cm+np.sqrt(cv)[:,None]*rng.standard_normal((C,D8))
        t=tp
    return X
def ha8_to(C,T,h,rng,tstop):
    X=rng.standard_normal((C,D8)); t=T; b=np.exp(-h)
    while t>tstop+1e-9:
        tp=max(t-h,tstop); bb=np.exp(-(t-tp)); mp,vp=m8(tp)
        L=logcomp8(X,t); L=L-L.max(1,keepdims=True); Wp=np.exp(L); Wp/=Wp.sum(1,keepdims=True)
        u=rng.random((C,1)); cs=np.cumsum(Wp,1); idx=(u>cs[:,:-1]).sum(1)
        prec=1.0/vp[idx]+bb*bb/(1.0-bb*bb); cv=1.0/prec
        cm=cv[:,None]*(mp[idx]/vp[idx][:,None]+bb*X/(1.0-bb*bb))
        X=cm+np.sqrt(cv)[:,None]*rng.standard_normal((C,D8))
        t=tp
    return X
def fors_cond_step(Y,tnext,h,B,rng):
    """One reverse conditional q_{tnext|tnext+h} via FORS (Alg 1): RGO on f=-log q_tnext
    at center Y/b with eta=(1-b^2)/b^2; SCORE queries only."""
    b=np.exp(-h); eta=(1.0-b*b)/(b*b); Yc=Y/b
    def g(z): return -score8(z,tnext)
    xp=Yc.copy()
    for _ in range(16): xp=Yc-eta*g(xp)
    gxp=g(xp); muq=Yc-eta*gxp
    C=Y.shape[0]; out=xp.copy(); done=np.zeros(C,bool); nq=0
    for _ in range(300):
        idx=np.where(~done)[0]
        if idx.size==0: break
        m=idx.size; xq=muq[idx]+np.sqrt(eta)*rng.standard_normal((m,D8))
        J=rng.poisson(2*B,m); Jm=int(J.max()) if m>0 else 0
        logp=np.zeros(m)
        for j in range(Jm):
            act=J>j; r=rng.random((m,1)); pt=r*xq+(1.0-r)*xp[idx]
            Wv=np.sum((xq-xp[idx])*(gxp[idx]-g(pt)),1); nq+=int(act.sum())
            Wv=np.clip(Wv,-B,B)
            logp+=np.where(act,np.log(np.clip((B+Wv)/(2.0*B),1e-300,None)),0.0)
        acc=np.log(rng.random(m))<logp
        out[idx[acc]]=xq[acc]; done[idx[acc]]=True
    return out,nq,int((~done).sum())

def stage_sto8():
    t0=time.time(); C=12000; out={}
    fl=[metric8(sample8(C,np.random.default_rng(100+s))) for s in range(3)]
    fm=[modeerr8(sample8(C,np.random.default_rng(100+s))) for s in range(3)]
    out["floor_W2"]=float(np.mean(fl)); out["floor_mode"]=float(np.mean(fm))
    print("  MC floor (exact q0 samples, C=%d): sliced-W2=%.4f  mode-wt=%.4f"%(C,out["floor_W2"],out["floor_mode"]))
    ha=[]
    for T in [0.5,1.0,1.5,2.0,3.0,4.0,6.0]:
        X=ha8_exact(C,T,H,np.random.default_rng(1))
        ha.append(dict(N=int(round(T/H)),T=T,W2=metric8(X),mode=modeerr8(X)))
        print("  HA-d8 exact-RGO N=%2d (T=%.1f): W2=%.4f mode=%.4f"%(ha[-1]["N"],T,ha[-1]["W2"],ha[-1]["mode"]))
    out["ha8"]=ha
    dd=[]
    for N in [16,64,256,1024]:
        rng=np.random.default_rng(2); X=rng.standard_normal((C,D8)); h=TD/N
        for k in range(N):
            t=TD-k*h
            X=X+h*(X+2.0*score8(X,t))+np.sqrt(2.0*h)*rng.standard_normal((C,D8))
        dd.append(dict(N=N,W2=metric8(X),mode=modeerr8(X)))
        print("  DDPM-d8 N=%5d: W2=%.4f mode=%.4f"%(N,dd[-1]["W2"],dd[-1]["mode"]))
    out["ddpm8"]=dd
    rng=np.random.default_rng(5); hf=0.03
    X=ha8_to(C,TD,H,rng,0.15); nqt=0; nf=0
    for k in range(5):
        tn=0.15-hf*(k+1)
        X,nq,nfail=fors_cond_step(X,max(tn,0.0),hf,1.5,rng); nqt+=nq; nf+=nfail
    out["fors8"]=dict(W2=metric8(X),mode=modeerr8(X),score_q_per_step=nqt/(5.0*C),fail=nf)
    print("  FORS-d8 (score-only, 5 reverse conditionals h=%.2f): W2=%.4f mode=%.4f  q/step=%.2f fail=%d"
          %(hf,out["fors8"]["W2"],out["fors8"]["mode"],out["fors8"]["score_q_per_step"],nf))
    out["runtime_s"]=time.time()-t0; sv("c1_sto8",out)

def stage_report():
    ha=ld("c1_ha_law")["res"]; dd=ld("c1_ddpm_a")["res"]+ld("c1_ddpm_b")["res"]; s8=ld("c1_sto8")
    dd.sort(key=lambda r:r["N"])
    OUT=dict(ha_law=ha,ddpm_law=dd,sto8=s8)
    Nh=np.array([r["N"] for r in ha],float); Wh=np.array([r["W2"] for r in ha])
    flo=Wh.min(); keep=Wh>8*flo
    if keep.sum()<3: keep=np.arange(len(Wh))<max(3,int(0.6*len(Wh)))
    A=np.polyfit(Nh[keep],np.log10(Wh[keep]),1)
    R2h=np.corrcoef(Nh[keep],np.log10(Wh[keep]))[0,1]**2
    spd=-1.0/A[0]
    Nd=np.array([r["N"] for r in dd],float); Wd=np.array([r["W2"] for r in dd])
    kd=Wd>0
    B=np.polyfit(np.log10(Nd[kd]),np.log10(Wd[kd]),1)
    R2d=np.corrcoef(np.log10(Nd[kd]),np.log10(Wd[kd]))[0,1]**2
    def cross(Ns,Ws,dl):
        for i in range(len(Ns)-1):
            if Ws[i]>dl>=Ws[i+1]:
                f=(np.log10(Ws[i])-np.log10(dl))/(np.log10(Ws[i])-np.log10(Ws[i+1]))
                return Ns[i]*(Ns[i+1]/Ns[i])**f if Ns[i+1]>2*Ns[i] else Ns[i]+f*(Ns[i+1]-Ns[i])
        return None
    tab=[]
    for dl in [0.3,0.1,0.03,0.01,3e-3,1e-3,3e-4,1e-4,1e-5]:
        nh=cross(Nh,Wh,dl); nd=cross(Nd,Wd,dl)
        tab.append(dict(delta=dl,N_HA=None if nh is None else float(nh),N_DDPM=None if nd is None else float(nd)))
    OUT["fits"]=dict(HA_steps_per_decade=float(spd),HA_loglin_R2=float(R2h),
                     HA_decades=float(np.log10(Wh.max()/flo)),W2_floor=float(flo),
                     DDPM_slope=float(B[0]),DDPM_R2=float(R2d))
    OUT["N_of_delta"]=tab
    print("="*78); print("CLAIM 1 (diffusion-model target: Gaussian mixture, exact scores)  SUMMARY")
    print("  HA (reverse chain of exact RGO steps): W2 ~ 10^{-N/%.2f}: %.2f steps/decade, R2=%.4f, %.1f decades to floor %.1e"
          %(spd,spd,R2h,OUT["fits"]["HA_decades"],flo))
    print("  DDPM/Euler (same exact scores): log10 W2 vs log10 N slope=%.3f R2=%.4f  => N(delta)~delta^{%.2f}"
          %(B[0],R2d,1.0/B[0]))
    print("  N(delta) (measured first-crossings, 1-D law):")
    for r in tab:
        print("    delta=%7.0e  N_HA=%s  N_DDPM=%s"%(r["delta"],
              "  -" if r["N_HA"] is None else "%6.1f"%r["N_HA"],
              "    -" if r["N_DDPM"] is None else "%9.1f"%r["N_DDPM"]))
    f8=s8["fors8"]
    print("  d=8 mixture: HA hits MC floor %.4f by N=%d (W2=%.4f); DDPM N=1024 W2=%.4f; FORS q/step=%.1f W2=%.4f"
          %(s8["floor_W2"],s8["ha8"][-1]["N"],s8["ha8"][-1]["W2"],s8["ddpm8"][-1]["W2"],
            f8["score_q_per_step"],f8["W2"]))
    haok=(R2h>=0.985)and(spd<=8.0)and(OUT["fits"]["HA_decades"]>=4.0)
    ddok=(-1.35<=B[0]<=-0.70)and(R2d>=0.97)
    d8ok=(s8["ha8"][-1]["W2"]<=1.6*s8["floor_W2"])and(f8["W2"]<=1.7*s8["floor_W2"])and(f8["score_q_per_step"]<=60.0)
    sep=tab[3]["N_DDPM"] is not None and tab[3]["N_HA"] is not None and tab[3]["N_DDPM"]>30*tab[3]["N_HA"]
    ver=haok and ddok and d8ok and sep
    OUT["checks"]=dict(HA_polylog=bool(haok),DDPM_poly=bool(ddok),d8_faithful=bool(d8ok),separation=bool(sep))
    OUT["verified"]=bool(ver)
    print("  checks: HA polylog=%s  DDPM poly=%s  d8 faithful+FORS=%s  separation@1e-2(>30x)=%s"%(haok,ddok,d8ok,sep))
    print("  VERDICT: %s"%("VERIFIED" if ver else "NOT VERIFIED")); print("="*78)
    json.dump(OUT,open(os.path.join(HERE,"results_diffusion.json"),"w"),indent=1)
    print("wrote results_diffusion.json")

if __name__=="__main__":
    stages=sys.argv[1:] or ["all"]
    if "all" in stages: stages=["ha_law","ddpm_a","ddpm_b","sto8","report"]
    for s in stages:
        t0=time.time()
        if s=="ha_law" and not has("c1_ha_law"): stage_ha_law()
        elif s=="ddpm_a" and not has("c1_ddpm_a"): stage_ddpm([8,16,32,64,128,256,512],"a")
        elif s=="ddpm_b" and not has("c1_ddpm_b"): stage_ddpm([1024,2048],"b")
        elif s=="sto8" and not has("c1_sto8"): stage_sto8()
        elif s=="report": stage_report()
        print("[stage %s done %.1fs]"%(s,time.time()-t0))
