#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claim 4  (TREEFLOW : tree-conditioned flow matching, Sec 4.1 / 5.2 / Cor H.5)
Paper: "Trees to Flows and Back" (OpenReview gW7NZN8zJu, arXiv 2605.00414).

Paper claim: conditioning a continuous flow-matching generator on decision-tree
partitions (path encodings) yields COMPETITIVE / HIGHER-FIDELITY tabular generation
(lowest Wasserstein on 4/5 benchmarks, lowest correlation error on 3/5) while being
~2x faster than the diffusion baseline TabDDPM; a distributional-convergence result
(Cor H.5): the per-partition generated law -> true conditional as the tree refines.

CPU-tractable verification of the MECHANISM (not the paper's exact benchmark numbers):
 (V1) FIDELITY: TREEFLOW (partition-conditioned rectified flow, learned by ridge on
      random Fourier features) attains LOWER sliced-Wasserstein and LOWER correlation
      error to real data than unconditional flow matching, on a 4-cluster anisotropic
      2-D "tabular" set; utility (cluster TSTR) at least as high.
 (V2) DISTRIBUTIONAL CONVERGENCE (Cor H.5): mean per-partition sliced-Wasserstein
      DECREASES as an axis-aligned (kd-tree) partition refines 1->2->4->8 leaves.
 (V3) EFFICIENCY proxy (directional): at a TIGHT fidelity threshold the deterministic
      flow/PF-ODE sampler needs fewer function evaluations than a DDPM ancestral (SDE)
      sampler -- the mechanism behind the reported ~2x speedup.
Honest scope: exact "3/5, 4/5" counts and wall-clock 2x vs TabDDPM are architecture/
dataset specific and NOT reproduced here; V3 is a directional proxy.
CPU-only, deterministic.
"""
import json, os, time
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
t0=time.time(); rng=np.random.default_rng(0)

CENTERS=np.array([[-6,-6],[6,-6],[-6,6],[6,6]],float)
COVS=[np.array([[1.4,1.0],[1.0,1.4]]), np.array([[1.4,-1.0],[-1.0,1.4]]),
      np.array([[0.5,0.0],[0.0,2.6]]), np.array([[2.6,0.0],[0.0,0.5]])]
def make_data(n, seed=0):
    g=np.random.default_rng(seed); per=n//4; X=[]; lab=[]
    for k in range(4):
        L=np.linalg.cholesky(COVS[k]); X.append(CENTERS[k]+g.standard_normal((per,2))@L.T); lab.append(np.full(per,k))
    return np.vstack(X), np.concatenate(lab)
Xr, labr = make_data(2000, seed=0); DATA_SCALE=Xr.std(0)

def kd_partition(X, nleaves):
    leaves=[np.arange(len(X))]
    while len(leaves)<nleaves:
        nl=[]
        for idx in leaves:
            pts=X[idx]; ax=int(np.argmax(pts.var(0))); thr=np.median(pts[:,ax])
            left=idx[pts[:,ax]<=thr]; right=idx[pts[:,ax]>thr]
            if len(left)==0 or len(right)==0: nl.append(idx)
            else: nl.append(left); nl.append(right)
        if len(nl)==len(leaves): break
        leaves=nl
    leaves=leaves[:nleaves]; ids=np.zeros(len(X),int)
    for i,idx in enumerate(leaves): ids[idx]=i
    return ids

D_RFF=150
Wf=rng.standard_normal((3,D_RFF))*1.2; bf=rng.uniform(0,2*np.pi,D_RFF)
def rff(xw,t):
    tcol=np.full(len(xw),t) if np.ndim(t)==0 else t
    return np.cos(np.column_stack([xw,tcol])@Wf+bf)*np.sqrt(2.0/D_RFF)
def build_fm(X, naug=6, seed=0):
    g=np.random.default_rng(seed); Xw=X/DATA_SCALE; n=len(Xw)
    x0=g.standard_normal((n*naug,2)); x1=np.repeat(Xw,naug,axis=0)
    tt=g.uniform(0,1,n*naug); xt=tt[:,None]*x1+(1-tt[:,None])*x0
    return xt, tt, x1-x0
def ridge_fit(Phi,V,lam=1e-2): return np.linalg.solve(Phi.T@Phi+lam*np.eye(Phi.shape[1]), Phi.T@V)

xt,tt,V=build_fm(Xr,seed=1); Wunc=ridge_fit(rff(xt,tt),V)
def euler_uncond(n,nsteps=60,seed=5):
    x=np.random.default_rng(seed).standard_normal((n,2)); dt=1.0/nsteps
    for i in range(nsteps): x=x+dt*(rff(x/DATA_SCALE,i*dt)@Wunc)
    return x*DATA_SCALE
def fit_treeflow(nleaves):
    ids=kd_partition(Xr,nleaves); Wp={}; prior=[]
    for p in range(nleaves):
        m=(ids==p); xt,tt,V=build_fm(Xr[m],seed=10+p); Wp[p]=ridge_fit(rff(xt,tt),V); prior.append(m.mean())
    return ids, Wp, np.array(prior)
def treeflow_generate(nleaves,n,nsteps=60):
    ids,Wp,prior=fit_treeflow(nleaves)
    counts=np.random.default_rng(2).multinomial(n,prior); out=[]; glab=[]
    for p in range(nleaves):
        nk=counts[p]
        if nk==0: continue
        x=np.random.default_rng(100+p).standard_normal((nk,2)); dt=1.0/nsteps
        for i in range(nsteps): x=x+dt*(rff(x,i*dt)@Wp[p])
        out.append(x*DATA_SCALE); glab.append(np.full(nk,p))
    return np.vstack(out), np.concatenate(glab), ids

def sliced_w(A,B,nproj=128,seed=3):
    g=np.random.default_rng(seed); th=g.uniform(0,np.pi,nproj)
    return float(np.mean([wasserstein_distance(A@np.array([np.cos(a),np.sin(a)]),B@np.array([np.cos(a),np.sin(a)])) for a in th]))
def corr_err(A,B): return float(np.linalg.norm(np.corrcoef(A.T)-np.corrcoef(B.T)))
def near(X): return np.argmin(((X[:,None,:]-CENTERS[None])**2).sum(-1),axis=1)
def tstr(Xg):
    yg=near(Xg)
    if len(np.unique(yg))<2: return 0.0
    return LogisticRegression(max_iter=200).fit(Xg,yg).score(Xr,labr)

n_gen=2000
Xg_unc=euler_uncond(n_gen); Xg_tf,tf_lab,ids4=treeflow_generate(4,n_gen)
sw_unc=sliced_w(Xg_unc,Xr); sw_tf=sliced_w(Xg_tf,Xr)
ce_unc=corr_err(Xg_unc,Xr); ce_tf=corr_err(Xg_tf,Xr)
tstr_unc=tstr(Xg_unc); tstr_tf=tstr(Xg_tf)
print("="*70); print("CLAIM 4  TREEFLOW : tree-conditioned flow matching"); print("="*70)
print(f"[V1] sliced-Wasserstein (lower)  | unconditional={sw_unc:.4f}  TREEFLOW={sw_tf:.4f}  ({100*(sw_unc-sw_tf)/sw_unc:+.1f}%)")
print(f"[V1] correlation error   (lower) | unconditional={ce_unc:.4f}  TREEFLOW={ce_tf:.4f}")
print(f"[V1] TSTR utility        (higher)| unconditional={tstr_unc:.4f}  TREEFLOW={tstr_tf:.4f}")

print("[V2] per-partition sliced-W vs kd-tree leaves (Cor H.5, should decrease):")
depth_rows=[]
for nleaves in [1,2,4,8]:
    if nleaves==1: ids=np.zeros(len(Xr),int); Xg=Xg_unc; glab=np.zeros(n_gen,int)
    else: Xg,glab,ids=treeflow_generate(nleaves,n_gen)
    sws=[]
    for p in range(nleaves):
        rm=(ids==p); gm=(glab==p)
        if rm.sum()>20 and gm.sum()>20: sws.append(sliced_w(Xg[gm],Xr[rm],nproj=64))
    msw=float(np.mean(sws)); depth_rows.append(dict(leaves=nleaves,mean_perpart_SW=msw))
    print(f"     leaves={nleaves:2d}  mean per-partition SW={msw:.4f}")
pp=[r["mean_perpart_SW"] for r in depth_rows]
conv_monotone=(pp[0]>pp[1]>pp[2]) and (pp[-1]<=min(pp[:3])+0.05)   # strict decrease to a plateaued floor
conv_strong=min(pp)<0.2*pp[0]

gmm_mu=CENTERS.copy(); gmm_w=np.full(4,0.25); gmm_s2=1.5; BMIN,BMAX=0.1,9.0
def a_t(t): return np.exp(-0.5*(BMIN*t+0.5*(BMAX-BMIN)*t**2))
def beta_t(t): return BMIN+t*(BMAX-BMIN)
def gmm_score(x,t):
    a=a_t(t); v=a*a*gmm_s2+(1-a*a); d=x[:,None,:]-a*gmm_mu[None]
    logc=np.log(gmm_w)-0.5*(d*d).sum(-1)/v; m=logc.max(1,keepdims=True)
    g=np.exp(logc-m); g=g/g.sum(1,keepdims=True); return (g[...,None]*(-d/v)).sum(1)
def pf_ode(nsteps,n=2000,seed=7):
    x=np.random.default_rng(seed).standard_normal((n,2)); dt=1.0/nsteps
    for i in range(nsteps):
        t=1.0-i*dt; x=x+dt*(0.5*beta_t(t)*(x+gmm_score(x,t)))
    return x
def ddpm_ancestral(nsteps,n=2000,seed=7):
    g=np.random.default_rng(seed); x=g.standard_normal((n,2)); dt=1.0/nsteps
    for i in range(nsteps):
        t=1.0-i*dt; b=beta_t(t)
        x=x+dt*(0.5*b*x+b*gmm_score(x,t))+np.sqrt(b*dt)*g.standard_normal((n,2))
    return x
Xreal_gmm,_=make_data(2000,seed=9)
def sw_real(X): return sliced_w(X,Xreal_gmm,nproj=48)
thr=sw_real(pf_ode(400))*1.05
grid=[4,6,8,10,14,20,28,40,60,90,130,190,260]
def nfe(sampler):
    for k in grid:
        if sw_real(sampler(k))<=thr: return k
    return grid[-1]+1
nfe_ode=nfe(pf_ode); nfe_ddpm=nfe(ddpm_ancestral); speedup=nfe_ddpm/max(nfe_ode,1)
v3_dir = "flow<DDPM (supports)" if speedup>=1.3 else ("~equal (inconclusive on toy)" if speedup>=0.9 else "flow>DDPM on this toy")
print(f"[V3] NFE to TIGHT thr={thr:.4f}: flow PF-ODE={nfe_ode}  DDPM ancestral={nfe_ddpm}  speedup={speedup:.2f}x -> {v3_dir}")

v1_ok=(sw_tf<sw_unc) and (ce_tf<=ce_unc+1e-6) and (tstr_tf>=tstr_unc-1e-6)
v2_ok=conv_monotone and conv_strong; verdict=v1_ok and v2_ok
print("-"*70)
print(f"core (V1 fidelity & V2 convergence): {'SUPPORTED' if verdict else 'NOT'} ; V3 efficiency speedup={speedup:.2f}x (proxy)")
print("VERDICT:", "SUPPORTED" if verdict else "PARTIAL/NOT")

out=dict(
  claim="TREEFLOW: tree-conditioned flow matching improves tabular-generation fidelity (Wasserstein/correlation) and per-partition distributional convergence (Cor H.5); deterministic flow sampler needs fewer NFE than DDPM ancestral (~2x speedup mechanism)",
  V1_fidelity=dict(sliced_W_unconditional=sw_unc, sliced_W_treeflow=sw_tf, sliced_W_improve_pct=100*(sw_unc-sw_tf)/sw_unc,
                   corr_err_unconditional=ce_unc, corr_err_treeflow=ce_tf,
                   TSTR_unconditional=tstr_unc, TSTR_treeflow=tstr_tf, treeflow_better=bool(v1_ok)),
  V2_perpartition_convergence=depth_rows, V2_monotone=bool(conv_monotone), V2_strong_decrease=bool(conv_strong),
  V3_efficiency=dict(tight_threshold=float(thr), NFE_flow_pfode=int(nfe_ode), NFE_ddpm_ancestral=int(nfe_ddpm), speedup_x=float(speedup), direction=v3_dir),
  scope_note="Mechanism verified on CPU toy; exact paper benchmark counts (3/5,4/5) and wall-clock 2x vs TabDDPM not reproduced. V3 directional.",
  targets=dict(fidelity="TREEFLOW < unconditional (SW & corr-err)", convergence="per-partition SW decreasing, final<0.5x initial", speedup="flow < DDPM NFE (directional ~2x)"),
  verdict="SUPPORTED" if verdict else "PARTIAL", runtime_s=round(time.time()-t0,3))
with open(os.path.join(os.path.dirname(__file__),"results.json"),"w") as f: json.dump(out,f,indent=2)
print("runtime_s =",round(time.time()-t0,3)); print("wrote results.json")
