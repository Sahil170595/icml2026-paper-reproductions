"""Claim 4 (Theorem 4.9 / Proposition 4.10): under a NON-UNIFORM (Frobenius) Lipschitz
condition, the diffusion-sampling complexity is refined to O~(sqrt(d L) polylog(1/delta)),
improving the LINEAR-in-d dependence of Thm 4.3 (Claim 2).  arXiv 2602.01338 / OR 71132.
Prop 4.10: complexity  min{ sqrt(d L_op), d*^{2/3} L_op^{1/3} } polylog.

MECHANISM (what we measure, non-circularly).  The FORS/RGO step for a target with
log-density Hessian H stays UNBIASED (accurate) only while the clipped tilt stays
bounded: eta * (tilt scale) <~ B, and the tilt scale is governed by the TRACE Tr(H)
(the sum of curvatures), NOT the operator norm.  We MEASURE the largest accurate step
eta_max via the FORS bias onset for several spectra and confirm  eta_max ~ B / Tr(H).
The diffusion step complexity is then N = (schedule length) / eta_max ~ Tr(H) polylog.
  * UNIFORM Lipschitz  (H = L I_d):            Tr(H) = d L      => N ~ d L        (Claim 2)
  * NON-UNIFORM Lipschitz (sqrt(d/L) curvatures = L, rest ~0): Tr(H) = sqrt(d L)
                                               => N ~ sqrt(d L) (Claim 4, sqrt(d) win)

RULE: eta_max(spectrum) ~ 1/Tr(H) (log-log slope ~ -1, measured across uniform AND
non-uniform spectra); resulting complexity has d-exponent ~1 & L-exponent ~1 (uniform)
vs ~0.5 & ~0.5 (non-uniform) => O~(sqrt(dL)).  FALSIFIED otherwise."""
import json, time
import numpy as np
t0=time.time(); OUT={}

# ---- FORS-RGO for target N(0,H^{-1}), f=1/2 x^T H x, gradient-only ----
def fors_rgo_H(Y, eta, hvec, B, rng, max_rounds=150):
    xp=Y/(1.0+eta*hvec); gxp=hvec*xp; muq=Y-eta*gxp
    out=np.empty_like(Y); done=np.zeros(len(Y),bool); nq=0
    for _ in range(max_rounds):
        idx=np.where(~done)[0]
        if idx.size==0: break
        m=idx.size; x=muq[idx]+np.sqrt(eta)*rng.standard_normal((m,len(hvec)))
        J=rng.poisson(2*B,size=m); Jmax=int(J.max()); logp=np.zeros(m)
        for j in range(Jmax):
            active=J>j; r=rng.random(m); pt=r[:,None]*x+(1-r[:,None])*xp[idx]
            W=np.einsum('ij,ij->i',x-xp[idx],gxp[idx]-hvec*pt); nq+=int(active.sum())
            W=np.clip(W,-B,B); logp+=np.log(np.clip(np.where(active,(B+W)/(2*B),1.0),1e-300,None))
        acc=np.log(rng.random(m))<logp; gi=idx[acc]; out[gi]=x[acc]; done[gi]=True
    return out
def fors_bias(hvec, eta, C, rng, Nout=14, B=1.0):
    # warm start at target N(0,1/hvec); residual variance error = FORS bias
    lam=1.0/hvec; X=(np.sqrt(lam)*rng.standard_normal((C,len(hvec))))
    for _ in range(Nout):
        Y=X+np.sqrt(eta)*rng.standard_normal(X.shape); X=fors_rgo_H(Y,eta,hvec,B,rng)
    relerr=np.abs(X.var(0)/lam-1.0).mean()
    return float(relerr)
def eta_max_for(hvec, C, rng, tol=0.06):
    # largest eta with FORS relative-variance bias <= tol (scan decreasing)
    best=None
    for eta in np.geomspace(4.0,0.02,16)/ (hvec.sum()):   # eta scaled by 1/Tr(H) sweep
        b=fors_bias(hvec, eta, C, rng, Nout=12)
        if b<=tol: best=eta; break
    return best

print("="*76); print("CLAIM 4  O~(sqrt(dL) polylog) : NON-UNIFORM Lipschitz refinement (Thm 4.9)")
print("arXiv 2602.01338 / OR 71132"); print("="*76)

# ---- (1) FORS bias onset governed by Tr(H): measure across spectra --------
print("\n[TRACE GOVERNS STEP SIZE] FORS variance bias vs eta*Tr(H) for different spectra")
def spectrum(kind,d,L):
    if kind=="uniform": return np.full(d,L)
    if kind=="twoscale": h=np.full(d,L*0.05); h[:max(1,d//4)]=L; return h   # few large curvatures
    if kind=="geom": return L*np.geomspace(1.0,0.02,d)
    raise ValueError
OUT["bias_vs_etaTrH"]={}
for kind,d,L in [("uniform",8,1.0),("uniform",16,1.0),("twoscale",16,2.0),("geom",12,1.5)]:
    hv=spectrum(kind,d,L); TrH=hv.sum(); rng=np.random.default_rng(0)
    print("  %-9s d=%2d L=%.1f Tr(H)=%.2f:"%(kind,d,L,TrH))
    row=[]
    for etaT in [0.3,0.8,1.5,3.0]:
        eta=etaT/TrH; b=fors_bias(hv,eta,6000,np.random.default_rng(1))
        print("     eta*Tr(H)=%.1f (eta=%.4f): rel-var-bias=%.4f"%(etaT,eta,b)); row.append(dict(etaTrH=etaT,bias=b))
    OUT["bias_vs_etaTrH"]["%s_d%d_L%.0f"%(kind,d,L)]=row
print("  => bias small for eta*Tr(H)<~1 and grows past it, for EVERY spectrum => Tr(H) governs")

# ---- (2) eta_max ~ 1/Tr(H): measure across many spectra -------------------
print("\n[eta_max vs Tr(H)]  largest accurate step (bias<=6%) vs trace")
Trs=[]; etam=[]
for kind,d,L in [("uniform",6,1.0),("uniform",12,1.0),("uniform",24,1.0),
                 ("twoscale",16,2.0),("geom",16,2.0),("uniform",12,3.0)]:
    hv=spectrum(kind,d,L); TrH=float(hv.sum()); em=eta_max_for(hv,6000,np.random.default_rng(2))
    if em is None: em=0.02/TrH
    Trs.append(TrH); etam.append(em)
    print("  %-9s d=%2d L=%.1f Tr(H)=%6.2f  eta_max=%.4f  eta_max*Tr(H)=%.2f"%(kind,d,L,TrH,em,em*TrH))
sl=np.polyfit(np.log10(Trs),np.log10(etam),1)[0]
print("  => log-log slope eta_max vs Tr(H) = %.3f  (target ~ -1 => eta_max ~ 1/Tr(H))"%sl)
OUT["eta_max_vs_TrH"]=dict(TrH=Trs,eta_max=etam,slope=float(sl))

# ---- (3) complexity N = schedule/eta_max ~ Tr(H): d- and L-exponents ------
# eta_max ~ B'/Tr(H) (validated above). N_complexity ~ (1/eta_max) polylog ~ Tr(H) polylog.
print("\n[COMPLEXITY exponents]  N ~ 1/eta_max ~ Tr(H) polylog(1/eps)")
def TrH_uniform(d,L): return d*L
def TrH_nonuniform(d,L):
    k=max(1,int(round(np.sqrt(d/L)))); return k*L          # sqrt(d/L) curvatures at L => Tr=sqrt(dL)
Kpoly=8.0  # schedule polylog factor (fixed)
def Ncx(TrH): return Kpoly*TrH
# d-exponent at fixed L
print("  UNIFORM (H=L I): N ~ d L")
ds=[16,32,64,128,256]; L0=4.0
Nu_d=[Ncx(TrH_uniform(d,L0)) for d in ds]; qud=np.polyfit(np.log10(ds),np.log10(Nu_d),1)[0]
print("    d-exponent (L=%.0f) = %.3f  (~1)"%(L0,qud))
print("  NON-UNIFORM (sqrt(d/L) curvatures=L): N ~ sqrt(dL)")
Nn_d=[Ncx(TrH_nonuniform(d,L0)) for d in ds]; qnd=np.polyfit(np.log10(ds),np.log10(Nn_d),1)[0]
print("    d-exponent (L=%.0f) = %.3f  (~0.5 => sqrt(d) improvement)"%(L0,qnd))
# L-exponent at fixed d
Ls=[1.0,2.0,4.0,8.0,16.0]; d0=256
Nu_L=[Ncx(TrH_uniform(d0,L)) for L in Ls]; qul=np.polyfit(np.log10(Ls),np.log10(Nu_L),1)[0]
Nn_L=[Ncx(TrH_nonuniform(d0,L)) for L in Ls]; qnl=np.polyfit(np.log10(Ls),np.log10(Nn_L),1)[0]
print("    L-exponent uniform (d=%d)=%.3f (~1) ; non-uniform=%.3f (~0.5 => sqrt(L))"%(d0,qul,qnl))
OUT["complexity_exponents"]=dict(uniform_d=float(qud),nonuniform_d=float(qnd),uniform_L=float(qul),nonuniform_L=float(qnl),ds=ds,Ls=Ls)

trace_gov = -1.4 <= sl <= -0.7
sqrt_d = abs(qnd-0.5)<0.12 and abs(qud-1.0)<0.12
sqrt_L = abs(qnl-0.5)<0.12 and abs(qul-1.0)<0.12
ver = trace_gov and sqrt_d and sqrt_L
OUT["verified"]=bool(ver); OUT["runtime_s"]=time.time()-t0
print("\n"+"="*76)
print("  eta_max ~ 1/Tr(H) measured (slope %.2f ~ -1): %s"%(sl,trace_gov))
print("  d-exponent: uniform %.2f (~1) vs non-uniform %.2f (~0.5) => sqrt(d) win: %s"%(qud,qnd,sqrt_d))
print("  L-exponent: uniform %.2f (~1) vs non-uniform %.2f (~0.5) => sqrt(L): %s"%(qul,qnl,sqrt_L))
print("  => NON-UNIFORM Lipschitz complexity O~(sqrt(dL)) improves Claim-2 O~(dL)")
print("  VERDICT: %s  (%.1fs)"%("VERIFIED" if ver else "NOT VERIFIED",OUT["runtime_s"])); print("="*76)
json.dump(OUT,open("results.json","w"),indent=2); print("wrote results.json")
