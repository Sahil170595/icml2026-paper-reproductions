"""
Claim 1 - Theorem 3.1: the HSR Bellman operator T^mu M = B^mu + G^mu M is a
CONTRACTION MAPPING w.r.t. the max-norm, with modulus <= gamma:
        ||T^mu M - T^mu M'||_inf <= gamma ||M - M'||_inf          (Eq 8)
Paper: arXiv 2602.12753 / OpenReview txswvMHt4u (Yu & Lengyel).

T is affine (T M = B + G M) so T M - T M' = G (M - M'); the EXACT one-step
contraction modulus in max-norm is ||G||_inf = max row sum of the nonnegative
continuation kernel G. Proof (App. A.2): this row sum = E_mu[gamma^tau], the
expected discount over option duration tau>=1, hence <= gamma < 1. We verify:
  (A) measured contraction factor c (worst-case probe) == ||G||_inf, is <1 and
      <= gamma (Eq 8 tight); random matrix pairs never exceed it;
  (B) fixed-point iteration M_{k+1}=T M_k converges geometrically to the analytic
      fixed point M*=(I-G)^{-1}B; per-step ratio -> spectral radius rho(G)<=||G||_inf;
  (C) G row sums equal E_mu[gamma^tau] <= gamma (proof core);
  (D) primitive-only HSR fixed point == standard RW-SR (HSR generalises SR).
"""
import json, time, numpy as np
import hsr_core as H

def main():
    t0=time.time(); gamma=0.95
    env=H.four_room(11); N=env['N']
    opts,Mrw=H.eigenoptions(env,gamma,K=8)
    Blist,Flist=H.build_augmented(env,opts,gamma); nact=len(Blist)
    rng=np.random.default_rng(0)
    print("="*74)
    print("Claim 1 - HSR Bellman operator is a max-norm contraction (Theorem 3.1)")
    print("arXiv 2602.12753 / OpenReview txswvMHt4u - independent NumPy repro")
    print("="*74)
    print(f"four-room N={N} states, gamma={gamma}, augmented actions={nact} "
          f"(4 primitives + {len(opts)} eigenoptions)")
    policies={}
    policies['uniform_aug']=np.ones((N,nact))/nact
    mp=np.zeros((N,nact)); mp[:,:4]=0.25; policies['primitive_only']=mp
    mo=np.zeros((N,nact)); mo[:,:4]=0.10; mo[:,4:]=0.60/len(opts); policies['option_heavy']=mo
    results={}
    for name,mu in policies.items():
        B,Gm=H.hsr_operator(mu,Blist,Flist)
        rowsum=Gm.sum(axis=1); Ginf=float(rowsum.max())     # ||G||_inf = contraction modulus
        rho=float(np.max(np.abs(np.linalg.eigvals(Gm))))    # spectral radius (asymptotic rate)
        # (A) worst-case probe D=ones -> ||G D||_inf/||D||_inf = max row sum = ||G||_inf (tight)
        D=np.ones((N,N)); c_worst=float(np.max(np.abs(Gm@D))/np.max(np.abs(D)))
        emp=[]
        for _ in range(200):
            M1=rng.standard_normal((N,N)); M2=rng.standard_normal((N,N))
            emp.append(np.max(np.abs(Gm@(M1-M2)))/np.max(np.abs(M1-M2)))
        emp_max=float(np.max(emp))
        eq8_ok=bool(c_worst<=gamma+1e-9 and emp_max<=gamma+1e-9)
        contraction_ok=bool(Ginf<1.0); le_gamma=bool(Ginf<=gamma+1e-9)
        # (B) fixed-point iteration
        Mstar=np.linalg.solve(np.eye(N)-Gm,B)
        M=rng.standard_normal((N,N)); res=[float(np.max(np.abs(M-Mstar)))]; ratios=[]
        for k in range(600):
            M=B+Gm@M; r=float(np.max(np.abs(M-Mstar)))
            if res[-1]>0: ratios.append(r/res[-1])
            res.append(r)
            if r<1e-12: break
        final_res=res[-1]
        tail=[x for x in ratios[5:] if x>0]; emp_rate=float(np.median(tail)) if tail else float('nan')
        k_check=min(30,len(res)-1); pred_bound=(Ginf**k_check)*res[0]
        bound_ok=bool(res[k_check]<=pred_bound+1e-9)
        results[name]=dict(Ginf=Ginf, rho_G=rho, c_worstcase=c_worst, c_randpairs_max=emp_max,
            eq8_ok=eq8_ok, contraction_ok=contraction_ok, le_gamma=le_gamma,
            emp_rate=emp_rate, final_res=final_res, iters=len(res)-1,
            row_min=float(rowsum.min()), row_max=float(rowsum.max()),
            bound_ok=bound_ok, k_check=k_check)
        print(f"\n[{name}]")
        print(f"  ||G||_inf (contraction modulus)   = {Ginf:.6f}  (<1? {contraction_ok}, <=gamma? {le_gamma})")
        print(f"  measured c (worst-case probe)     = {c_worst:.6f}  == ||G||_inf (Eq8 tight, <=gamma? {c_worst<=gamma+1e-9})")
        print(f"  measured c (max over 200 rand pairs)= {emp_max:.6f}  (never exceeds ||G||_inf)")
        print(f"  spectral radius rho(G)            = {rho:.6f}  (asymptotic convergence rate)")
        print(f"  G row-sum = E_mu[gamma^tau] in [{rowsum.min():.4f},{rowsum.max():.4f}] (<=gamma={gamma})")
        print(f"  fixed-point iter: ||M_k-M*||_inf={final_res:.2e} in {len(res)-1} iters")
        print(f"  empirical per-step ratio (median tail)={emp_rate:.6f} (-> rho(G)={rho:.6f})")
        print(f"  geometric bound res[{k_check}]<=||G||_inf^{k_check}*res0: {res[k_check]:.3e}<={pred_bound:.3e} -> {bound_ok}")
    # (D) HSR reduces to SR
    Bp,Gp=H.hsr_operator(policies['primitive_only'],Blist,Flist)
    Mstar_p=np.linalg.solve(np.eye(N)-Gp,Bp)
    rwsr=H.sr_matrix(H.rw_transition(env),gamma)
    consistency=float(np.max(np.abs(Mstar_p-rwsr)))
    print(f"\n[consistency] ||HSR_primitiveonly_fixedpoint - RW-SR||_inf = {consistency:.2e} "
          f"(HSR generalises SR: {consistency<1e-8})")
    allok=all(results[n]['contraction_ok'] and results[n]['le_gamma'] and results[n]['eq8_ok']
              and results[n]['bound_ok'] and results[n]['final_res']<1e-8 for n in results)
    print("\n"+"="*74)
    print(f"OVERALL: {'REPRODUCED' if (allok and consistency<1e-8) else 'CHECK'} - "
          f"modulus<1 & <=gamma all policies; geometric convergence; HSR->SR consistency")
    print("="*74)
    rt=time.time()-t0
    out=dict(claim="Theorem 3.1 - HSR Bellman operator is a max-norm contraction (modulus ||G||_inf <= gamma)",
             paper="arXiv 2602.12753 / OpenReview txswvMHt4u", gamma=gamma, N=N,
             n_augmented_actions=nact, n_eigenoptions=len(opts), policies=results,
             hsr_reduces_to_sr_residual=consistency,
             overall_reproduced=bool(allok and consistency<1e-8), runtime_s=rt)
    json.dump(out,open("results.json","w"),indent=2)
    print(f"runtime={rt:.2f}s ; wrote results.json")

if __name__=="__main__": main()
