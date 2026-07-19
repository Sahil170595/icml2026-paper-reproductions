"""
Claim 3 (Sec 4.1, Fig 2e-g): HSR predictive representations are more ROBUST to
task-induced policy changes than standard SR. After adapting the policy from task
G1 to task G2, the standard SR matrix undergoes drastic reorganisation whereas
the HSR matrix is far less variable. Quantitatively (Fig 2g), the relative change
    rho(M) = ||M_1 - M_2||_F^2 / ||M_1||_F^2
is significantly LOWER for HSR than for SR (paper: two-sided t-test p<0.001).
Paper: arXiv 2602.12753 / OpenReview txswvMHt4u (Yu & Lengyel).

Mechanism: HSR is assembled from TASK-AGNOSTIC eigenoption models (B^a, F^a); only
the high-level selection mu_g changes across tasks, so the assembled matrix moves
less than the primitive-policy SR (I-gamma P_{pi*_g})^-1, which is rebuilt entirely.
We measure rho for SR and HSR over 20 random (G1,G2) goal pairs -> two-sample t-test.
"""
import json, time, numpy as np
import hsr_core as H
from math import sqrt, erf

def smdp_optimal_policy(Blist, Flist, r, N, nact, tol=1e-8, max_it=1000):
    Br=np.stack([Blist[a]@r for a in range(nact)],axis=1)
    V=np.zeros(N)
    for _ in range(max_it):
        Q=Br+np.stack([Flist[a]@V for a in range(nact)],axis=1)
        Vn=Q.max(axis=1)
        if np.max(np.abs(Vn-V))<tol: V=Vn; break
        V=Vn
    Q=Br+np.stack([Flist[a]@V for a in range(nact)],axis=1)
    pol=Q.argmax(axis=1)
    mu=np.zeros((N,nact)); mu[np.arange(N),pol]=1.0
    return mu,pol

def welch_t(a,b):
    a=np.array(a); b=np.array(b); na,nb=len(a),len(b)
    va,vb=a.var(ddof=1),b.var(ddof=1)
    t=(a.mean()-b.mean())/sqrt(va/na+vb/nb)
    df=(va/na+vb/nb)**2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
    return t,df

def main():
    t0=time.time(); gamma=0.95
    env=H.four_room(11); N=env['N']
    opts,Mrw=H.eigenoptions(env,gamma,K=8)
    Blist,Flist=H.build_augmented(env,opts,gamma); nact=len(Blist)
    print("="*74)
    print("Claim 3 - HSR representation is robust to policy change (Fig 2e-g)")
    print("arXiv 2602.12753 / OpenReview txswvMHt4u - independent NumPy repro")
    print("="*74)
    print(f"four-room N={N}, gamma={gamma}, augmented actions={nact}")
    rng=np.random.default_rng(0)
    goals=rng.choice(N, size=(20,2), replace=True)
    for i in range(20):
        while goals[i,0]==goals[i,1]: goals[i,1]=rng.integers(N)
    rel_sr=[]; rel_hsr=[]
    for (g1,g2) in goals:
        r1=np.zeros(N); r1[g1]=1.0; r2=np.zeros(N); r2[g2]=1.0
        pol1,_=H.value_iteration(env,r1,gamma); pol2,_=H.value_iteration(env,r2,gamma)
        Msr1=H.sr_matrix(H.policy_transition(env,pol1),gamma)
        Msr2=H.sr_matrix(H.policy_transition(env,pol2),gamma)
        mu1,_=smdp_optimal_policy(Blist,Flist,r1,N,nact)
        mu2,_=smdp_optimal_policy(Blist,Flist,r2,N,nact)
        B1,G1=H.hsr_operator(mu1,Blist,Flist); Mh1=np.linalg.solve(np.eye(N)-G1,B1)
        B2,G2=H.hsr_operator(mu2,Blist,Flist); Mh2=np.linalg.solve(np.eye(N)-G2,B2)
        rel_sr.append(float(np.sum((Msr1-Msr2)**2)/np.sum(Msr1**2)))
        rel_hsr.append(float(np.sum((Mh1-Mh2)**2)/np.sum(Mh1**2)))
    rel_sr=np.array(rel_sr); rel_hsr=np.array(rel_hsr)
    t,df=welch_t(rel_sr,rel_hsr)
    p_two=2*(1-0.5*(1+erf(abs(t)/sqrt(2))))
    ratio=rel_sr.mean()/rel_hsr.mean()
    frac=float(np.mean(rel_hsr<rel_sr))
    print(f"\nRelative representational change rho=||M1-M2||_F^2/||M1||_F^2 over 20 goal pairs:")
    print(f"  SR : mean={rel_sr.mean():.4f}  sem={rel_sr.std(ddof=1)/sqrt(20):.4f}  median={np.median(rel_sr):.4f}")
    print(f"  HSR: mean={rel_hsr.mean():.4f}  sem={rel_hsr.std(ddof=1)/sqrt(20):.4f}  median={np.median(rel_hsr):.4f}")
    print(f"  SR/HSR mean ratio = {ratio:.2f}  (HSR changes {ratio:.2f}x LESS than SR)")
    print(f"  fraction of pairs with rho_HSR < rho_SR = {frac:.2f}")
    print(f"  Welch two-sided t-test: t={t:.3f}, df={df:.1f}, p={p_two:.2e}  (paper: p<0.001)")
    verdict=(rel_hsr.mean()<rel_sr.mean()) and (p_two<0.05) and frac>=0.75
    vtxt='REPRODUCED - HSR significantly more stable' if verdict else 'NOT REPRODUCED - HSR ~ SR (no significant stability gain)'
    print(f"\nOVERALL: {vtxt}")
    print("="*74)
    rt=time.time()-t0
    out=dict(claim="HSR representation is significantly more robust to task-induced policy change than SR (Fig 2g)",
        paper="arXiv 2602.12753 / OpenReview txswvMHt4u", gamma=gamma, N=N, n_pairs=20,
        rho_SR_mean=float(rel_sr.mean()), rho_SR_sem=float(rel_sr.std(ddof=1)/sqrt(20)),
        rho_HSR_mean=float(rel_hsr.mean()), rho_HSR_sem=float(rel_hsr.std(ddof=1)/sqrt(20)),
        SR_over_HSR_stability_factor=float(ratio), frac_pairs_HSR_lower=frac,
        welch_t=float(t), welch_df=float(df), p_two_sided=float(p_two),
        rho_SR=rel_sr.tolist(), rho_HSR=rel_hsr.tolist(),
        overall_reproduced=bool(verdict), verdict=("reproduced" if verdict else "not_reproduced"), runtime_s=rt)
    json.dump(out,open("results.json","w"),indent=2)
    print(f"runtime={rt:.2f}s ; wrote results.json")

if __name__=="__main__": main()
