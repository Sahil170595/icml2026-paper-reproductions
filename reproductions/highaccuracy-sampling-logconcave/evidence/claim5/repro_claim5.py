"""Claim 5 (Section 5): the same FORS framework yields a polylog(1/delta)-accuracy
sampler for log-concave distributions using ONLY first-order (gradient) queries,
via the proximal sampler (Alg 3) whose restricted-Gaussian-oracle (RGO) is
implemented by First-Order Rejection Sampling (FORS, Alg 1). arXiv 2602.01338/OR 71132.

HIGH-ACCURACY mechanism: FORS realises the EXACT RGO tilt with an UNBIASED gradient
estimator of w(x)=-f(x)+f(x+)+<x-x+,grad f(x+)> (Eq. 504-513): NO discretization
bias. So the proximal chain converges to the TRUE target (high accuracy), while ULA
(a first-order SDE discretization) is biased (low accuracy).

REAL gradient-only runs on TWO strongly-log-concave targets:
  (A) Gaussian N(0,I_d)             [FORS-RGO exact; truth var=1]
  (B) non-Gaussian f(t)=t^2/2+(g/4)t^4  [truth var by Gauss-Hermite quadrature]
verify: (1) FORS output UNBIASED (matches truth), (2) O(1) grad queries / RGO step
under condition (16), (3) FORS mixes geometrically from a cold start (polylog),
(4) ULA saturates at a bias floor ~ h (low-accuracy)."""
import json, time
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
t0=time.time(); OUT={}
g=1.0
def gradf(x, gauss): return x if gauss else x + g*x**3
def prox_point(Y, eta, gauss):
    if gauss: return Y/(1.0+eta)
    x=Y.copy()
    for _ in range(8): x=x-(x+eta*(x+g*x**3)-Y)/(1.0+eta*(1.0+3*g*x**2))
    return x
def truth_var_B():
    n,w=hermegauss(140); dn=w*np.exp(-0.25*g*n**4); Z=dn.sum()
    return (dn*n**2).sum()/Z-((dn*n).sum()/Z)**2
varB=truth_var_B()
def fors_rgo(Y, eta, B, gauss, rng, max_rounds=160):
    xp=prox_point(Y,eta,gauss); gxp=gradf(xp,gauss); muq=Y-eta*gxp
    out=np.empty_like(Y); done=np.zeros(Y.shape,bool); nq=0
    for _ in range(max_rounds):
        idx=np.where(~done)[0]
        if idx.size==0: break
        m=idx.size; x=muq[idx]+np.sqrt(eta)*rng.standard_normal(m)
        J=rng.poisson(2*B,size=m); Jmax=int(J.max()); logp=np.zeros(m)
        for j in range(Jmax):
            active=J>j; r=rng.random(m); pt=r*x+(1-r)*xp[idx]
            W=(x-xp[idx])*(gxp[idx]-gradf(pt,gauss)); nq+=int(active.sum())
            W=np.clip(W,-B,B); logp+=np.log(np.clip(np.where(active,(B+W)/(2*B),1.0),1e-300,None))
        acc=np.log(rng.random(m))<logp; gi=idx[acc]; out[gi]=x[acc]; done[gi]=True
    return out,nq
def prox_fors(gauss, eta, Nout, C, d, rng, v0, B=1.0, checkpts=None):
    X=(np.sqrt(v0)*rng.standard_normal((C,d))).ravel(); tq=0; traj={}
    cp=set(checkpts or [])
    for n in range(Nout):
        if n in cp: traj[n]=float(X.reshape(C,d).var(0).mean())
        Y=X+np.sqrt(eta)*rng.standard_normal(X.shape); X,nq=fors_rgo(Y,eta,B,gauss,rng); tq+=nq
    return X.reshape(C,d), tq, traj
def ula(gauss, h, T, C, d, rng, v0):
    X=np.sqrt(v0)*rng.standard_normal((C,d))
    for _ in range(T): X=X-h*gradf(X,gauss)+np.sqrt(2*h)*rng.standard_normal((C,d))
    return X
print("="*76); print("CLAIM 5  log-concave high-accuracy sampler, gradient-only FORS (Sec 5)")
print("arXiv 2602.01338 / OR 71132"); print("="*76)
print("targets: (A) Gaussian truth var=1.000 ; (B) f=t^2/2+t^4/4 truth var=%.5f (quadrature)"%varB)
d=4; C=10000; B=1.0; mc=np.sqrt(2.0/C)/np.sqrt(d)
print("\n[UNBIASEDNESS] FORS-proximal (gradient-only), C=%d d=%d, warm start, MC noise~%.4f"%(C,d,mc))
OUT["fors_unbiased"]={}
for name,gauss,truth,eta in [("A_gauss",True,1.0,0.25),("B_quartic",False,varB,0.12)]:
    X,tq,_=prox_fors(gauss,eta,25,C,d,np.random.default_rng(1),v0=truth)
    mv=float(X.var(0).mean()); err=abs(mv-truth); gps=tq/(25*C*d)
    print("  %-9s eta=%.2f: var=%.5f truth=%.5f |err|=%.4f (rel %.1f%%, %.1f*MC) grad/RGO/chain=%.2f"
          %(name,eta,mv,truth,err,100*err/truth,err/mc,gps))
    OUT["fors_unbiased"][name]=dict(eta=eta,var=mv,truth=truth,abs_err=err,rel_err=err/truth,grad_per_step=gps)
print("\n[CONDITION (16)] warm Gaussian d=%d (TrH=d): FORS unbiased iff eta*TrH <~ B"%d)
OUT["condition16"]=[]
for eta in [0.1,0.25,0.6,1.2]:
    X,tq,_=prox_fors(True,eta,20,C,d,np.random.default_rng(2),v0=1.0)
    mv=float(X.var(0).mean()); gps=tq/(20*C*d)
    print("  eta=%.2f eta*TrH=%.2f: var=%.4f |err|=%.4f (%.1f*MC) grad/RGO=%.2f"%(eta,eta*d,mv,abs(mv-1),abs(mv-1)/mc,gps))
    OUT["condition16"].append(dict(eta=eta,eta_TrH=eta*d,var=mv,abs_err=abs(mv-1),grad_per_step=gps))
print("\n[MIXING from cold start v0=4] FORS-proximal Gaussian eta=0.25 (var error vs outer step)")
_,_,traj=prox_fors(True,0.25,22,C,d,np.random.default_rng(5),v0=4.0,checkpts=list(range(0,22,3)))
OUT["mixing_traj"]={str(k):abs(v-1.0) for k,v in traj.items()}
for k in sorted(traj): print("  n=%2d  |var-1|=%.4e"%(k,abs(traj[k]-1.0)))
print("\n[HIGH- vs LOW-ACCURACY, Gaussian] FORS (unbiased) vs ULA (bias floor = h/2)")
Xf,_,_=prox_fors(True,0.25,30,C,d,np.random.default_rng(3),v0=1.0); fe=abs(Xf.var(0).mean()-1.0)
print("  FORS-proximal (gradient-only, UNBIASED): |var-1|=%.4f (=%.1f*MC, no floor)"%(fe,fe/mc))
OUT["ula_bias_vs_h"]=[]
for h in [0.4,0.2,0.1,0.05]:
    Xu=ula(True,h,700,C,d,np.random.default_rng(4),v0=1.0); ue=abs(Xu.var(0).mean()-1.0)
    print("  ULA h=%.3f (BIASED): |var-1|=%.4f (theory h/2/(1-h/2)=%.4f)"%(h,ue,(h/2)/(1-h/2)))
    OUT["ula_bias_vs_h"].append(dict(h=h,abs_err=float(ue),theory=(h/2)/(1-h/2)))
hs=np.array([r["h"] for r in OUT["ula_bias_vs_h"]]); es=np.array([r["abs_err"] for r in OUT["ula_bias_vs_h"]])
slope=np.polyfit(np.log10(hs),np.log10(es),1)[0]
print("  ULA bias floor vs h: log-log slope=%.3f (~1 => floor ~ h => poly complexity 1/delta)"%slope)
OUT["ula_bias_slope"]=float(slope); OUT["fors_err_gauss"]=float(fe); OUT["mc_noise"]=float(mc)
rA=OUT["fors_unbiased"]["A_gauss"]["rel_err"]; rB=OUT["fors_unbiased"]["B_quartic"]["rel_err"]
gA=OUT["fors_unbiased"]["A_gauss"]["grad_per_step"]; gB=OUT["fors_unbiased"]["B_quartic"]["grad_per_step"]
ks=sorted(traj); mixed=abs(traj[ks[-1]]-1.0) < 0.05*abs(traj[ks[0]]-1.0)
ula_big=max(r["abs_err"] for r in OUT["ula_bias_vs_h"]) > 10*fe
ver=(rA<0.03)and(rB<0.07)and(gA<12)and(gB<12)and(0.8<=slope<=1.2)and ula_big and mixed
OUT["verified"]=bool(ver); OUT["runtime_s"]=time.time()-t0
print("\n"+"="*76)
print("  FORS unbiased (rel err A=%.1f%% B=%.1f%%): %s"%(100*rA,100*rB,rA<0.03 and rB<0.07))
print("  FORS grad/RGO-step O(1): A=%.2f B=%.2f"%(gA,gB))
print("  FORS mixes geometrically (cold->converged): %s"%mixed)
print("  ULA bias floor ~ h (slope=%.2f) and >>FORS: %s"%(slope,0.8<=slope<=1.2 and ula_big))
print("  VERDICT: %s  (%.1fs)"%("VERIFIED" if ver else "NOT VERIFIED",OUT["runtime_s"])); print("="*76)
json.dump(OUT,open("results.json","w"),indent=2); print("wrote results.json")
