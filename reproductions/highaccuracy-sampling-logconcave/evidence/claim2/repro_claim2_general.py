"""Claim 2 (Thm 4.3): total complexity O~(d polylog(1/eps)) -- LINEAR in dimension --
verified on NON-GAUSSIAN targets. arXiv 2602.01338 / OR 71132.
(A) Deterministic law-evolution, d up to 2048: product ANISOTROPIC QUARTIC target
    f(x)=sum_i [x_i^2/(2 a_i) + x_i^4/(4 a_i^2)], a_i in {1/4,1,4} (non-Gaussian,
    log-concave, condition number 16). The proximal chain factorizes per coordinate;
    each 1-D coordinate law is evolved EXACTLY on a grid (sparse banded exact-RGO +
    Gaussian-noising operators), with eta set by condition (16): eta ~ 1/(d log(1/eps)).
    N(d) = steps until every coordinate-type has |var/var_truth - 1| <= eps.
(B) Polylog in 1/eps at fixed d=128 on the same non-Gaussian target.
(C) Stochastic GRADIENT-ONLY FORS (Alg 1 inside Alg 3) on a COUPLED, NON-PRODUCT
    anisotropic quartic chain f(x)=sum_i V_i(x_i) + (g/2) sum (x_{i+1}-x_i)^2,
    d in {32,64,128,256}; ground truth E|x|^2/d from a TRANSFER-OPERATOR quadrature
    (exact for this chain graph); verifies N(d) ~ d and O(1) grad queries/step.
Stages (argv): lawA_small lawA_big laweps truthC coupled_32 coupled_64 coupled_128
               coupled_256 report all       (cache in _cache/)."""
import json, os, sys, time
import numpy as np
import scipy.sparse as sp

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,"_cache")
os.makedirs(CACHE,exist_ok=True)
def sv(n,o): json.dump(o,open(os.path.join(CACHE,n+".json"),"w"))
def ld(n): return json.load(open(os.path.join(CACHE,n+".json")))
def has(n): return os.path.exists(os.path.join(CACHE,n+".json"))

TYPES=[0.25,1.0,4.0]                      # anisotropy scales a (kappa=16)
def eta16(d,eps):
    L=np.log(1.0/eps); return 1.0/(0.5*(d*L+L*L))
def truth_var(a,G=6001):
    x=np.linspace(-7.0*np.sqrt(a),7.0*np.sqrt(a),G)
    w=np.exp(-x*x/(2*a)-x**4/(4*a*a)); w/=w.sum()
    return float((w*x*x).sum())

def coord_ops(a,eta,G=1301):
    """column-stochastic banded operators for one 1-D coordinate of type a:
       Kg: y=x+sqrt(eta)*xi ;  Kr: exact RGO x'~ exp(-V(x')-(x'-y)^2/(2 eta))"""
    R=7.0*np.sqrt(a); x=np.linspace(-R,R,G); dx=x[1]-x[0]
    Vp=x/a+x**3/(a*a); bw=int(np.ceil((7.0*np.sqrt(eta)+eta*np.abs(Vp).max())/dx))+2
    offs=np.arange(-bw,bw+1)
    def build(logker):
        rows=[];cols=[];vals=[]
        base=np.arange(G)
        for o in offs:
            c=base[(base+o>=0)&(base+o<G)]; r=c+o
            rows.append(r);cols.append(c);vals.append(logker(x[r],x[c]))
        rows=np.concatenate(rows);cols=np.concatenate(cols)
        vals=np.exp(np.concatenate(vals))
        K=sp.coo_matrix((vals,(rows,cols)),shape=(G,G)).tocsc()
        K=K@sp.diags(1.0/np.asarray(K.sum(0)).ravel())
        return K.tocsr()
    Kg=build(lambda xr,xc: -(xr-xc)**2/(2*eta))
    Kr=build(lambda xr,xc: -(xr*xr/(2*a)+xr**4/(4*a*a))-(xr-xc)**2/(2*eta))
    p0=np.exp(-x*x/(2*4.0*truth_var(a)))  # cold start: 4x overdispersed Gaussian
    p0/=p0.sum()
    return x,Kg,Kr,p0

def N_of(d,eps,every=25,cap=400000):
    """steps until ALL coordinate types reach |var/truth-1|<=eps, eta from (16)."""
    eta=eta16(d,eps); worst=0; per={}
    for a in TYPES:
        x,Kg,Kr,p=coord_ops(a,eta); vt=truth_var(a); n=0
        while n<cap:
            for _ in range(every): p=Kr@(Kg@p)
            n+=every
            if abs(float((p*x*x).sum())/vt-1.0)<=eps: break
        per[str(a)]=n; worst=max(worst,n)
    return worst,eta,per

def stage_lawA(ds,tag):
    t0=time.time(); eps=1e-3; res=[]
    for d in ds:
        N,eta,per=N_of(d,eps)
        res.append(dict(d=d,N=N,eta=eta,per_type=per))
        print("  lawA d=%5d eta=%.3e  N=%7d  per-type=%s"%(d,eta,N,per))
    sv("c2_lawA_"+tag,dict(eps=eps,res=res,runtime_s=time.time()-t0))

def stage_laweps():
    t0=time.time(); d=128; res=[]
    for eps in [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]:
        N,eta,per=N_of(d,eps)
        res.append(dict(eps=eps,N=N,eta=eta))
        print("  laweps d=%d eps=%.0e eta=%.3e N=%7d"%(d,eps,eta,N))
    sv("c2_laweps",dict(d=d,res=res,runtime_s=time.time()-t0))

# ---------------- coupled non-product chain (stochastic FORS) ----------------
GAM=0.5
def avec(d):
    a=np.ones(d); a[1::2]=2.0; return a
def gradC(X,a):
    g=X/a+X**3/(a*a)
    g[:,1:]+=GAM*(X[:,1:]-X[:,:-1]); g[:,:-1]+=GAM*(X[:,:-1]-X[:,1:])
    return g
def truthC(d,G=401):
    a=avec(d); z=np.linspace(-8.0,8.0,G); dz=z[1]-z[0]
    psi=[np.exp(-z*z/(2*ai)-z**4/(4*ai*ai)) for ai in a]
    phi=np.exp(-GAM*(z[:,None]-z[None,:])**2/2.0)
    mf=[np.ones(G)]
    for i in range(d-1):
        m=phi@(psi[i]*mf[i]); mf.append(m/m.max())
    mb=[np.ones(G) for _ in range(d)]
    for i in range(d-2,-1,-1):
        m=phi@(psi[i+1]*mb[i+1]); mb[i]=m/m.max()
    tot=0.0
    for i in range(d):
        w=psi[i]*mf[i]*mb[i]; w/=w.sum()
        tot+=float((w*z*z).sum())
    return tot/d
def fors_rgo(Y,eta,B,grad,rng):
    # proximal point x* solves x=Y-eta*grad(x); for convex f, |x*-Y|=eta|grad f(x*)|
    # <= eta|grad f(Y)| (gradient monotonicity), so we may project every iterate onto
    # that ball; damped (alpha=1/2) Picard is then stable even for cold-start outliers
    # where plain Picard diverges (eta*L_local>1). Gradient queries only.
    gY=grad(Y); R=eta*np.sqrt(np.sum(gY*gY,1,keepdims=True))+1e-12
    xp=Y-eta*gY
    for _ in range(11):
        t=Y-eta*grad(xp); t=np.where(np.isfinite(t),t,Y)
        xp=xp+0.5*(t-xp)
        dx=xp-Y; nn=np.sqrt(np.sum(dx*dx,1,keepdims=True))
        xp=Y+dx*np.minimum(1.0,R/np.maximum(nn,1e-30))
    gxp=grad(xp); muq=Y-eta*gxp
    C=Y.shape[0]; out=xp.copy(); done=np.zeros(C,bool); nq=0
    for _ in range(200):
        idx=np.where(~done)[0]
        if idx.size==0: break
        m=idx.size; xq=muq[idx]+np.sqrt(eta)*rng.standard_normal((m,Y.shape[1]))
        J=rng.poisson(2*B,m); Jm=int(J.max()) if m>0 else 0
        logp=np.zeros(m)
        for j in range(Jm):
            act=J>j; r=rng.random((m,1)); pt=r*xq+(1.0-r)*xp[idx]
            Wv=np.sum((xq-xp[idx])*(gxp[idx]-grad(pt)),1); nq+=int(act.sum())
            Wv=np.clip(Wv,-B,B)
            logp+=np.where(act,np.log(np.clip((B+Wv)/(2.0*B),1e-300,None)),0.0)
        acc=np.log(rng.random(m))<logp
        out[idx[acc]]=xq[acc]; done[idx[acc]]=True
    return out,nq

def stage_coupled(d,max_steps_call=30):
    t0=time.time(); tag="c2_coupled_%d"%d
    tv=ld("c2_truthC")[str(d)]; a=avec(d)
    eps=0.06; eta=eta16(d,eps)*2.0; B=2.0; C=384
    st=os.path.join(CACHE,tag+"_state.npz")
    if os.path.exists(st):
        z=np.load(st); X=z["X"]; n=int(z["n"]); nq=float(z["nq"]); traj=list(map(list,z["traj"]))
        rng=np.random.default_rng(900+d+7*n)
    else:
        rng=np.random.default_rng(900+d)
        X=np.sqrt(4.0*tv)*rng.standard_normal((C,d)); n=0; nq=0.0; traj=[]
    grad=lambda Z: gradC(Z,a)
    done=False; k0=n
    while n-k0<max_steps_call:
        for _ in range(5):
            Y=X+np.sqrt(eta)*rng.standard_normal((C,d))
            X,q=fors_rgo(Y,eta,B,grad,rng); nq+=q
        n+=5
        e=abs(float(np.mean(X**2))/tv-1.0); traj.append([n,e])
        if len(traj)>=3 and all(t[1]<eps for t in traj[-3:]):
            done=True; break
    if done:
        Ncross=traj[-3][0]
        sv(tag,dict(d=d,N=Ncross,eta=eta,eps=eps,C=C,truth=tv,
                    grad_q_per_step=nq/(n*C),traj=traj[-6:]))
        print("  coupled d=%4d eta=%.3e  N=%5d  grad_q/step/chain=%.1f  err_end=%.3f"
              %(d,eta,Ncross,nq/(n*C),traj[-1][1]))
        if os.path.exists(st): os.remove(st)
    else:
        np.savez(st,X=X,n=n,nq=nq,traj=np.array(traj))
        print("  coupled d=%4d: staged %d steps (err=%.3f), rerun stage to continue"%(d,n,traj[-1][1]))

def stage_truthC():
    t0=time.time(); out={}
    for d in [32,64,128,256]:
        out[str(d)]=truthC(d)
        print("  truthC d=%4d  E|x|^2/d=%.6f"%(d,out[str(d)]))
    sv("c2_truthC",out); print("  (%.1fs)"%(time.time()-t0))

def stage_report():
    la=ld("c2_lawA_small")["res"]+ld("c2_lawA_big")["res"]; la.sort(key=lambda r:r["d"])
    le=ld("c2_laweps")["res"]
    co=[ld("c2_coupled_%d"%d) for d in [32,64,128,256] if has("c2_coupled_%d"%d)]
    OUT=dict(lawA=la,laweps=le,coupled=co)
    dd=np.array([r["d"] for r in la],float); NN=np.array([r["N"] for r in la],float)
    q=np.polyfit(np.log10(dd),np.log10(NN),1); R2=np.corrcoef(np.log10(dd),np.log10(NN))[0,1]**2
    ee=np.array([r["eps"] for r in le]); Ne=np.array([r["N"] for r in le],float)
    pe=np.polyfit(np.log10(1/ee),np.log10(Ne),1)[0]
    Ln=np.log(1/ee); A=np.vstack([Ln*Ln,Ln,np.ones_like(Ln)]).T
    coef,_,_,_=np.linalg.lstsq(A,Ne,rcond=None); pred=A@coef
    R2log=1-np.sum((Ne-pred)**2)/np.sum((Ne-Ne.mean())**2)
    dc=np.array([r["d"] for r in co],float); Nc=np.array([r["N"] for r in co],float)
    qc=np.polyfit(np.log10(dc),np.log10(Nc),1) if len(co)>=3 else [np.nan]
    R2c=np.corrcoef(np.log10(dc),np.log10(Nc))[0,1]**2 if len(co)>=3 else np.nan
    gq=[r["grad_q_per_step"] for r in co]
    print("="*78); print("CLAIM 2 (non-Gaussian targets)  SUMMARY")
    print("  [A] product anisotropic quartic (law, exact): N vs d slope=%.3f R2=%.4f  (d=64..2048)"%(q[0],R2))
    print("  [B] polylog in 1/eps (d=128, quartic): power=%.3f (<<1), deg-2 log-poly R2=%.5f"%(pe,R2log))
    if len(co)>=3:
        print("  [C] coupled NON-PRODUCT chain, gradient-only FORS: N vs d slope=%.3f R2=%.4f; grad q/step=%s"
              %(qc[0],R2c,["%.1f"%g for g in gq]))
    okA=0.85<=q[0]<=1.15 and R2>=0.99
    okB=pe<0.35 and R2log>0.995
    okC=len(co)>=3 and 0.7<=qc[0]<=1.3 and max(gq)<=3.0*min(gq)
    ver=okA and okB and okC
    OUT["fits"]=dict(lawA_slope=float(q[0]),lawA_R2=float(R2),eps_power=float(pe),eps_logpoly_R2=float(R2log),
                     coupled_slope=float(qc[0]) if len(co)>=3 else None,coupled_R2=float(R2c) if len(co)>=3 else None,
                     coupled_gradq=gq)
    OUT["checks"]=dict(linear_d_exact=bool(okA),polylog_eps=bool(okB),coupled_gradient_only=bool(okC))
    OUT["verified"]=bool(ver)
    print("  checks: linear-d(exact)=%s  polylog-eps=%s  coupled-FORS=%s"%(okA,okB,okC))
    print("  VERDICT: %s"%("VERIFIED" if ver else "NOT VERIFIED")); print("="*78)
    json.dump(OUT,open(os.path.join(HERE,"results_general.json"),"w"),indent=1)
    print("wrote results_general.json")

if __name__=="__main__":
    stages=sys.argv[1:] or ["all"]
    if "all" in stages:
        stages=["lawA_small","lawA_big","laweps","truthC","coupled_32","coupled_64",
                "coupled_128","coupled_256","report"]
    for s in stages:
        t0=time.time()
        if s=="lawA_small" and not has("c2_lawA_small"): stage_lawA([64,128,256,512],"small")
        elif s=="lawA_big" and not has("c2_lawA_big"): stage_lawA([1024,2048],"big")
        elif s=="laweps" and not has("c2_laweps"): stage_laweps()
        elif s=="truthC" and not has("c2_truthC"): stage_truthC()
        elif s.startswith("coupled_"):
            d=int(s.split("_")[1])
            if not has("c2_coupled_%d"%d): stage_coupled(d)
        elif s=="report": stage_report()
        print("[stage %s done %.1fs]"%(s,time.time()-t0))
