"""Claim 2 (Theorem 4.3, log-smooth case d*=d): the high-accuracy diffusion sampler
has total complexity O~(d polylog(1/delta)) -- LINEAR in the data dimension d.
arXiv 2602.01338 / OR 71132.  Total complexity d*log^3((d+L+M2^2)/delta^2) (paper p.9).

MECHANISM. The proximal/diffusion sampler implements each RGO step by rejection
(FORS, Alg 1), whose acceptance is (1+eta)^{-d/2} for N(0,I): it collapses
EXPONENTIALLY in d at fixed step size eta, so O(1) queries/step needs condition (16)
    sigma^2/eta >> d*log(1/delta) + log^2(1/delta)   ==>   eta ~ 1/d.
Given eta set by (16), the number of proximal steps to reach per-coordinate accuracy
eps on N(0,I_d) is N ~ C*d*polylog(1/eps) => LINEAR in d.  Exact linear-Gaussian
variance law (confirmed by real stochastic runs) drives the sweeps.

RULE: log-log slope of N vs d (fixed eps, eta from (16)) in [0.85,1.15] (linear d);
control with fixed eta gives slope ~0 (the d-dependence is a consequence of (16));
N is polylog in 1/eps (power exponent <0.30 << 1, and a degree-2 log-polynomial fits
with R^2>0.999); RGO acceptance collapses ~(1+eta)^{-d/2} at fixed eta but is O(1)
under eta~1/d.  FALSIFIED otherwise."""
import json, time
import numpy as np
t0=time.time(); OUT={}
v0=4.0
def prox_next(v, eta): a=1.0/(1.0+eta); return a*a*(v+eta)+a*eta   # lam=1, fixed pt 1
def eta16(d, eps, c0=0.5):
    L=np.log(1.0/eps); return 1.0/(c0*(d*L + L*L))
def prox_N(d, eps, eta):
    v=v0; n=0
    while abs(v-1.0)>eps:
        v=prox_next(v, eta); n+=1
        if n>5_000_000: break
    return n
print("="*76); print("CLAIM 2  O~(d polylog(1/delta)) : LINEAR dimension dependence (Thm 4.3)")
print("arXiv 2602.01338 / OR 71132   target N(0,I_d)"); print("="*76)

eps=1e-3; ds=[16,32,64,128,256,512,1024,2048]
print("\n[LINEAR-d] eps=%.0e, eta from condition (16) ~ 1/(d log(1/eps)); N=proximal steps"%eps)
Nd=[]
for d in ds:
    eta=eta16(d,eps); N=prox_N(d,eps,eta); Nd.append(N)
    print("  d=%5d  eta=%.2e (1/eta=%.1f)  N=%6d"%(d,eta,1/eta,N))
q=np.polyfit(np.log10(ds),np.log10(Nd),1); R2=np.corrcoef(np.log10(ds),np.log10(Nd))[0,1]**2
print("  => log-log slope q_d = %.3f  (R^2=%.4f)  target [0.85,1.15] => LINEAR in d"%(q[0],R2))
OUT["linear_d"]=dict(eps=eps,ds=ds,N=Nd,slope=float(q[0]),R2=float(R2))

print("\n[STOCHASTIC CONFIRMATION] real d-dim proximal runs match the exact variance law")
OUT["stoch_confirm"]={}
for d in [16,64,256]:
    eta=0.3; C=20000; rng=np.random.default_rng(7)
    X=np.sqrt(v0)*rng.standard_normal((C,d))
    for _ in range(15):
        a=1.0/(1.0+eta); Y=X+np.sqrt(eta)*rng.standard_normal((C,d))
        X=a*Y+np.sqrt(eta/(1.0+eta))*rng.standard_normal((C,d))
    vs=float(X.var(0).mean()); ve=v0
    for _ in range(15): ve=prox_next(ve,eta)
    print("  d=%4d  stoch var_15=%.5f  exact-law var_15=%.5f"%(d,vs,ve))
    OUT["stoch_confirm"][str(d)]=dict(stoch=vs,exact=ve)

d0=128; epslist=[1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
print("\n[POLYLOG in 1/eps] fixed d=%d, eta from (16); N vs 1/eps"%d0)
Ne=[]
for e in epslist:
    eta=eta16(d0,e); N=prox_N(d0,e,eta); Ne.append(N)
    print("  eps=%.0e  N=%7d"%(e,N))
logi=np.log10([1/e for e in epslist]); p_eps=np.polyfit(logi,np.log10(Ne),1)[0]
Ln=np.log(np.array([1/e for e in epslist])); A=np.vstack([Ln**2,Ln,np.ones_like(Ln)]).T
coef,res,_,_=np.linalg.lstsq(A,np.array(Ne,float),rcond=None)
pred=A@coef; R2log=1-np.sum((Ne-pred)**2)/np.sum((Ne-np.mean(Ne))**2)
print("  power-law exponent of N vs 1/eps = %.3f (<<1 => polylog);  degree-2 log-poly R^2=%.5f"%(p_eps,R2log))
OUT["polylog_eps"]=dict(d=d0,eps=epslist,N=Ne,power_exponent=float(p_eps),logpoly_R2=float(R2log))

print("\n[CONTROL] fixed eta=0.3 (violates (16) for large d): N(d) is ~d-INDEPENDENT")
Nc=[prox_N(d,eps,0.3) for d in ds]
qc=np.polyfit(np.log10(ds),np.log10(Nc),1)[0]
print("  d=%s"%ds); print("  N=%s   => slope=%.3f (~0)"%(Nc,qc))
OUT["control_fixed_eta"]=dict(ds=ds,N=Nc,slope=float(qc))

print("\n[RGO ACCEPTANCE vs d] proposals per accepted sample (ideal rejection, N(0,I_d))")
def rej_proposals(d, eta, target_acc, rng):
    nprop=0; acc=0
    while acc<target_acc and nprop<40_000_000:
        m=min(20000,4*target_acc+50)
        x=np.sqrt(eta)*rng.standard_normal((m,d))
        a=rng.random(m)<np.exp(-0.5*np.sum(x**2,1))   # accept w.p. exp(-|x|^2/2)
        acc+=int(a.sum()); nprop+=m
    return nprop/max(acc,1)
print("  fixed eta=0.5 (violates (16)): proposals/accept ~ (1+eta)^{d/2} EXPLODES")
OUT["accept_fixed_eta"]=[]
for d in [2,6,10,16,24,32]:
    pa=rej_proposals(d,0.5,400,np.random.default_rng(3)); theory=(1.5)**(d/2)
    print("    d=%2d  proposals/accept=%.1f  theory (1.5)^{d/2}=%.1f"%(d,pa,theory)); OUT["accept_fixed_eta"].append(dict(d=d,pa=pa,theory=float(theory)))
print("  eta=2.0/d (obeys (16)): proposals/accept stays O(1) ~ e")
OUT["accept_eta_over_d"]=[]
for d in [2,8,32,128,512]:
    pa=rej_proposals(d,2.0/d,400,np.random.default_rng(4))
    print("    d=%3d  eta=%.4f  proposals/accept=%.2f"%(d,2.0/d,pa)); OUT["accept_eta_over_d"].append(dict(d=d,eta=2.0/d,pa=pa))

lin=0.85<=q[0]<=1.15 and R2>=0.99
ctrl=abs(qc)<0.25
poly=abs(p_eps)<0.30 and R2log>0.999
af=OUT["accept_fixed_eta"]; ao=OUT["accept_eta_over_d"]
blow=af[-1]["pa"]>20*af[0]["pa"]
flat=ao[-1]["pa"]<3*ao[0]["pa"]
ver=lin and ctrl and poly and blow and flat
OUT["verified"]=bool(ver); OUT["runtime_s"]=time.time()-t0
print("\n"+"="*76)
print("  LINEAR-d (slope %.2f in [0.85,1.15], R2=%.3f): %s"%(q[0],R2,lin))
print("  CONTROL fixed-eta d-independent (slope %.2f ~0): %s"%(qc,ctrl))
print("  POLYLOG in eps (power %.3f <<1, log-poly R2=%.4f): %s"%(p_eps,R2log,poly))
print("  RGO acceptance: explodes at fixed eta (%.0f->%.0f) & O(1) under eta~1/d (%.1f->%.1f): %s"%(af[0]["pa"],af[-1]["pa"],ao[0]["pa"],ao[-1]["pa"],blow and flat))
print("  VERDICT: %s  (%.1fs)"%("VERIFIED" if ver else "NOT VERIFIED",OUT["runtime_s"])); print("="*76)
json.dump(OUT,open("results.json","w"),indent=2); print("wrote results.json")
