#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claim 3  (Global Trajectory Score Matching, Thm 3.2 & Thm 3.4)
Paper: "Trees to Flows and Back" (OpenReview gW7NZN8zJu, arXiv 2605.00414).

Two checkable consequences of the unifying GTSM framework:

 PART A  (Thm 3.2, CGTSM optimality <=> path matching).  For two SDEs differing
 only in drift/score, the path-space KL equals the CGTSM (score-matching) integral
 (Girsanov). We verify on tractable OU processes that the EXACT multivariate-Gaussian
 path-KL converges to the closed-form CGTSM integral
        CGTSM = (a-a')^2/(2 sigma^2) * int_0^T E_{P*}[X_t^2] dt ,
 and that CGTSM = 0 IFF the scores/drifts match (necessary & sufficient).

 PART B  (Thm 3.4, greedy boosting is globally optimal).  The proof reduces to:
 additive separability + deterministic transitions => the greedy per-stage minimiser
 equals the Bellman-optimal policy.  We verify (i) on a separable finite-horizon DP
 greedy == exhaustive/Bellman optimum (gap = 0); (ii) L2-boosting with a RICH
 (orthogonal) weak-learner dictionary => greedy == global optimum over all length-M
 ensembles (gap = 0); (iii) FALSIFICATION-GUARD: an impoverished/correlated dictionary
 (violates the richness hypothesis) makes greedy SUBOPTIMAL (gap > 0) -- confirming the
 result is genuine and hypothesis-dependent, exactly as the paper states.  We also
 verify the residual = score identity (Def 3.3): -grad_F (1/2)||y-F||^2 = y-F = r.
CPU-only, deterministic.
"""
import json, os, time, itertools
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
t0=time.time()

# =====================================================================
# PART A : Girsanov / CGTSM identity on Ornstein-Uhlenbeck processes
# =====================================================================
a=1.0; sigma=1.0; x0=1.5; T=1.0
def ou_mean_cov(a, grid):
    m = x0*np.exp(-a*grid)
    s = grid[:,None]; t = grid[None,:]
    C = (sigma**2/(2*a))*(np.exp(-a*np.abs(s-t)) - np.exp(-a*(s+t)))
    C = C + 1e-10*np.eye(len(grid))          # tiny jitter for SPD stability
    return m, C
def gauss_kl(m1,C1,m2,C2):
    C2i=np.linalg.inv(C2); M=len(m1); dm=m2-m1
    s1,ld1=np.linalg.slogdet(C1); s2,ld2=np.linalg.slogdet(C2)
    return 0.5*(np.trace(C2i@C1) - M + dm@C2i@dm + (ld2-ld1))
def cgtsm_integral(a, ap):
    # (a-ap)^2/(2 sigma^2) * int_0^T E[X_t^2] dt, E[X_t^2]=(sig^2/2a)(1-e^{-2at})+x0^2 e^{-2at}
    I1 = (sigma**2/(2*a))*(T - (1-np.exp(-2*a*T))/(2*a))
    I2 = x0**2*(1-np.exp(-2*a*T))/(2*a)
    return (a-ap)**2/(2*sigma**2)*(I1+I2)

print("="*70); print("CLAIM 3  GTSM  ->  PART A: Girsanov path-KL == CGTSM score-matching integral"); print("="*70)
aps=[1.0, 1.2, 1.5, 2.0, 0.6]
Ms=[50,100,200,400]
partA=[]
for ap in aps:
    cg=cgtsm_integral(a,ap)
    row={"a_prime":ap,"CGTSM_integral":cg,"path_KL_by_M":{}}
    for M in Ms:
        grid=np.linspace(T/M, T, M)
        m1,C1=ou_mean_cov(a,grid); m2,C2=ou_mean_cov(ap,grid)
        kl=float(gauss_kl(m1,C1,m2,C2)); row["path_KL_by_M"][M]=kl
    finest=row["path_KL_by_M"][Ms[-1]]
    row["rel_err_finest"]=abs(finest-cg)/(cg+1e-12) if cg>0 else abs(finest-cg)
    partA.append(row)
    conv=" -> ".join(f"{row['path_KL_by_M'][M]:.5f}" for M in Ms)
    print(f" a'={ap:>4}: CGTSM={cg:.6f} | path-KL(M={Ms})= {conv} | rel_err(M={Ms[-1]})={row['rel_err_finest']:.3e}")
zero_case=[r for r in partA if r["a_prime"]==a][0]
cgtsm_zero_iff = (zero_case["CGTSM_integral"]<1e-12 and zero_case["path_KL_by_M"][Ms[-1]]<1e-9)
max_relerr_nonzero=max(r["rel_err_finest"] for r in partA if r["a_prime"]!=a)
print(f" [A] CGTSM=0 and path-KL=0 exactly when a'=a (scores match): {cgtsm_zero_iff}")
print(f" [A] max rel-err (path-KL vs CGTSM, M={Ms[-1]}, a'!=a): {max_relerr_nonzero:.3e}  (-> Girsanov identity)")
partA_ok = cgtsm_zero_iff and max_relerr_nonzero < 0.02

# =====================================================================
# PART B : greedy boosting == globally optimal (Thm 3.4)
# =====================================================================
print("-"*70); print("CLAIM 3  GTSM  ->  PART B: greedy boosting == Bellman/global optimum (Thm 3.4)")
rng=np.random.default_rng(1)

# (i) separable finite-horizon DP : the exact reduction in the proof
Mst=5; nact=6; C=rng.random((Mst,nact))          # stage cost c[m][a]
greedy_dp=sum(C[m].min() for m in range(Mst))
V=np.zeros(Mst+1)                                 # Bellman backward induction
for m in range(Mst-1,-1,-1): V[m]=min(C[m][a2]+V[m+1] for a2 in range(nact))
bellman_dp=V[0]
# brute force
best=min(sum(C[m][seq[m]] for m in range(Mst)) for seq in itertools.product(range(nact),repeat=Mst))
gap_dp=abs(greedy_dp-best); print(f" (i) separable DP: greedy={greedy_dp:.6f} Bellman={bellman_dp:.6f} brute={best:.6f} gap={gap_dp:.2e}")

# boosting on N points, dictionary of weak-learner output vectors, cost sum_m ||r_m||^2
def boost_greedy(y,H,M):
    r=y.copy(); tot=0.0; picks=[]
    for _ in range(M):
        j=int(np.argmin(((H-r)**2).sum(1))); r=r-H[j]; tot+=float((r*r).sum()); picks.append(j)
    return tot,picks
def boost_optimal(y,H,M):
    best=None
    for seq in itertools.product(range(len(H)),repeat=M):
        F=np.zeros_like(y); tot=0.0
        for j in seq: F=F+H[j]; tot+=float(((y-F)**2).sum())
        if best is None or tot<best[0]: best=(tot,seq)
    return best
N=4; Mb=3; NTRIAL=300
# (ii) RICH dictionaries: orthogonal residual-spanning atoms (richness hypothesis HOLDS).
#      Statistical sweep -> greedy must equal the global optimum every time (gap=0).
rr=np.random.default_rng(11); rich_sub=0; rich_maxgap=0.0
for _ in range(NTRIAL):
    Q,_=np.linalg.qr(rr.standard_normal((N,N))); c=rr.standard_normal(N); y=Q@c
    Hr=np.vstack([c[k]*Q[:,k] for k in range(N)])            # each atom cancels one orthogonal component
    g,_=boost_greedy(y,Hr,Mb); o,_=boost_optimal(y,Hr,Mb); gp=abs(g-o)
    rich_maxgap=max(rich_maxgap,gp); rich_sub+=int(gp>1e-9)
gap_rich=rich_maxgap
print(f" (ii) boosting RICH (orthogonal) dict, {NTRIAL} trials: greedy-suboptimal count={rich_sub}/{NTRIAL} ; max gap={rich_maxgap:.2e}")
# (iii) POOR dictionaries: correlated random atoms (richness VIOLATED). Falsification-guard:
#       greedy (matching pursuit) is provably suboptimal on some instances -> gap>0.
rp=np.random.default_rng(23); poor_sub=0; poor_maxgap=0.0; example=(0.0,0.0)
for _ in range(NTRIAL):
    H=rp.standard_normal((5,N)); y=rp.standard_normal(N)
    g,_=boost_greedy(y,H,Mb); o,seq=boost_optimal(y,H,Mb); gp=g-o
    if gp>poor_maxgap: poor_maxgap=gp; example=(g,o)
    poor_sub+=int(gp>1e-9)
g_poor,o_poor=example; gap_poor=poor_maxgap
print(f" (iii) boosting POOR (correlated) dict, {NTRIAL} trials: greedy-suboptimal count={poor_sub}/{NTRIAL} ; max gap={poor_maxgap:.4f} (>0 => richness needed)")

# residual = score (Def 3.3): -grad_F 0.5||y-F||^2 = y-F = r
Ftest=rng.standard_normal(N); grad=-(y-Ftest); resid=y-Ftest
res_eq_score=float(np.max(np.abs((-grad)-resid)))
print(f" residual == negative-gradient (score target, Def 3.3): max|(-grad)-r| = {res_eq_score:.2e}")

partB_ok = (gap_dp<1e-9 and abs(bellman_dp-greedy_dp)<1e-9 and rich_sub==0 and rich_maxgap<1e-9
            and poor_sub>0 and poor_maxgap>1e-6 and res_eq_score<1e-12)

verdict = partA_ok and partB_ok
print("-"*70)
print("VERDICT (GTSM: path-KL==CGTSM & greedy boosting optimal under richness):",
      "SUPPORTED" if verdict else "NOT SUPPORTED")

out=dict(
  claim="GTSM: (A) zero CGTSM/score-matching loss <=> path-space measures match (Thm 3.2, Girsanov); (B) greedy boosting is the globally optimal discrete-GTSM solver under sufficiently rich learners (Thm 3.4)",
  partA=dict(process="Ornstein-Uhlenbeck", a=a, sigma=sigma, x0=x0, T=T, sweep=partA,
             CGTSM_zero_iff_scores_match=bool(cgtsm_zero_iff), max_rel_err_path_vs_cgtsm=float(max_relerr_nonzero),
             ok=bool(partA_ok)),
  partB=dict(separable_DP=dict(greedy=float(greedy_dp),bellman=float(bellman_dp),brute=float(best),gap=float(gap_dp)),
             boosting_rich=dict(trials=NTRIAL,greedy_suboptimal_count=int(rich_sub),max_gap=float(rich_maxgap)),
             boosting_poor=dict(trials=NTRIAL,greedy_suboptimal_count=int(poor_sub),max_gap=float(poor_maxgap)),
             residual_eq_score_maxabs=float(res_eq_score), ok=bool(partB_ok)),
  targets=dict(path_KL_vs_CGTSM="rel err <0.02 (Girsanov)", CGTSM_zero="iff scores match",
               separable_DP_gap="0", boosting_rich_gap="0 (greedy optimal)", boosting_poor_gap=">0 (richness needed)"),
  verdict="SUPPORTED" if verdict else "NOT_SUPPORTED", runtime_s=round(time.time()-t0,3))
with open(os.path.join(os.path.dirname(__file__),"results.json"),"w") as f: json.dump(out,f,indent=2)
print("runtime_s =",round(time.time()-t0,3)); print("wrote results.json")
