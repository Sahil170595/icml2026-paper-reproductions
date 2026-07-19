"""
Claim 4 - The diffusion distance is a Mahalanobis distance whose matrix is the
Random Walk Diffusion Kernel (RWDK); it generalizes to digraphs via P_(nu).

Paper P-RWDKC (arXiv 2210.00310 / OpenReview 5vI6ApLOg8):
  Prop. 4.1 / Eq.5 (undirected, reversible P, ergodic pi):
        d_t^2(i,j) = || p_t(i,*) - p_t(j,*) ||^2_{1/pi}
                   = (delta_i - delta_j)^T K_t (delta_i - delta_j),
        K_t = P^{2t} D_d^{-1}                  (RWDK, positive definite)
  Def. 4.2 / Eq.6-7 (parametrized, digraph): same identity with the
        parametrized walk P_(nu), reversing measure (nu+xi), and
        K_{t,nu} = P_(nu)^t D_{nu+xi}^{-1}     (P-RWDK, the Alg.1 embedding).
  Sec. 4.2 limit: as t->inf, K_t -> [tr(D_d)]^{-1} 1 1^T  (rank-1).

CHECKABLE CONSEQUENCES (deterministic, CPU, numpy):
  (A) UNDIRECTED: reversing measure = degree d, all pairs (i,j), t in {1,2,4}:
      || p_t(i,*)-p_t(j,*) ||^2_{1/d} == (e_i-e_j)^T [P^{2t}D_d^{-1}] (e_i-e_j),
      and the Gram identity P^t D_d^{-1}(P^t)^T == P^{2t}D_d^{-1}; RWDK symmetric PSD.
  (B) DIGRAPH (parametrized): || p_{t,nu}(i,*)-p_{t,nu}(j,*) ||^2_{1/(nu+xi)} ==
      (e_i-e_j)^T [P_(nu)^{2t} D_{nu+xi}^{-1}] (e_i-e_j); the RWDK P_(nu)^{2t}D^{-1}
      (double power) is symmetric PSD.  Reported nuance: the Eq.7 embedding
      K_{t,nu}=P_(nu)^t D^{-1} is symmetric and PSD for EVEN t but INDEFINITE for
      odd t (min eig < 0), because P_(nu)=D^{-1/2} S D^{1/2} with S symmetric,
      S in [-1,1]; the diffusion-distance (Mahalanobis) matrix is always the
      even power P_(nu)^{2t}D^{-1}, exactly as Prop 4.1 states (K_t=P^{2t}D_d^{-1}).
  (C) LIMIT: as t->inf, K_t -> 1 1^T / tr(D_d) (rank-1), max|.| -> 0.
"""
import json, numpy as np

def undirected(N, p, rng):
    A = (rng.random((N, N)) < p).astype(float); A = np.triu(A, 1); A = A + A.T
    W = A * rng.uniform(0.3, 2.0, size=(N, N)); W = np.triu(W, 1); W = W + W.T
    for i in range(N):
        if W[i].sum() == 0:
            j = (i+1) % N; W[i, j] = W[j, i] = 1.0
    return W

def digraph(N, p, rng):
    A = (rng.random((N, N)) < p).astype(float); np.fill_diagonal(A, 0.0)
    W = A * rng.uniform(0.3, 2.0, size=(N, N))
    for i in range(N):
        if W[i].sum() == 0: W[i, (i+1) % N] = 1.0
    return W

def P_nu_of(W, nu):
    d = W.sum(1); d = np.where(d > 0, d, 1.0); P = W / d[:, None]
    xi = nu @ P; num = (nu[:, None]*P) + (P.T*nu[None, :])
    return num / (nu + xi)[:, None], xi

def maha_check(Pt, m):
    """max over pairs | ||row_i-row_j||^2_{1/m} - (e_i-e_j)^T K (e_i-e_j) |,
       K = Pt D_m^{-1} Pt^T  (= P^{2t} D_m^{-1} under reversibility)."""
    N = Pt.shape[0]; K = Pt @ np.diag(1.0/m) @ Pt.T; diff = 0.0
    for i in range(N):
        for j in range(i+1, N):
            u = Pt[i] - Pt[j]; lhs = float(np.sum(u*u/m))
            e = np.zeros(N); e[i] = 1; e[j] = -1; rhs = float(e @ K @ e)
            diff = max(diff, abs(lhs - rhs))
    return diff, K

def mineig(K): return float(np.linalg.eigvalsh(0.5*(K+K.T)).min())

def main():
    rng = np.random.default_rng(11)
    res = dict(und_maha=0.0, und_K_eq_P2t=0.0, und_Ksym=0.0, und_rwdk_psd=0.0,
               dg_maha=0.0, dg_rwdk_sym=0.0, dg_rwdk_psd=0.0,
               dg_emb_sym=0.0, dg_emb_even_mineig=np.inf, dg_emb_odd_mineig=0.0, limit=0.0)
    for N, p in [(35, 0.2), (50, 0.15)]:
        for s in range(2):
            # (A) undirected
            W = undirected(N, p, rng); d = W.sum(1); P = W / d[:, None]
            for t in (1, 2, 4):
                Pt = np.linalg.matrix_power(P, t)
                dd, K = maha_check(Pt, d)                       # K = P^{2t} D_d^{-1}
                res["und_maha"] = max(res["und_maha"], dd)
                Kpaper = np.linalg.matrix_power(P, 2*t) @ np.diag(1.0/d)
                res["und_K_eq_P2t"] = max(res["und_K_eq_P2t"], float(np.max(np.abs(K - Kpaper))))
                res["und_Ksym"] = max(res["und_Ksym"], float(np.max(np.abs(K - K.T))))
                res["und_rwdk_psd"] = min(res["und_rwdk_psd"], mineig(K))
            # (B) digraph parametrized
            Wd = digraph(N, p, rng)
            for nu in (np.ones(N), rng.uniform(0.3, 1.0, N)):
                Pn, xi = P_nu_of(Wd, np.asarray(nu, float)); m = nu + xi
                for t in (1, 2, 4):
                    Pt = np.linalg.matrix_power(Pn, t)
                    dd, Kgram = maha_check(Pt, m)               # RWDK = P_(nu)^{2t} D_m^{-1}
                    res["dg_maha"] = max(res["dg_maha"], dd)
                    res["dg_rwdk_sym"] = max(res["dg_rwdk_sym"], float(np.max(np.abs(Kgram - Kgram.T))))
                    res["dg_rwdk_psd"] = min(res["dg_rwdk_psd"], mineig(Kgram))
                    Kemb = Pt @ np.diag(1.0/m)                  # Eq.7 single-power embedding
                    res["dg_emb_sym"] = max(res["dg_emb_sym"], float(np.max(np.abs(Kemb - Kemb.T))))
                    if t % 2 == 0:
                        res["dg_emb_even_mineig"] = min(res["dg_emb_even_mineig"], mineig(Kemb))
                    else:
                        res["dg_emb_odd_mineig"] = min(res["dg_emb_odd_mineig"], mineig(Kemb))
    # (C) rank-1 limit
    W = undirected(60, 0.15, rng); d = W.sum(1); P = W / d[:, None]
    Pbig = np.linalg.matrix_power(P, 4096)
    Klim = Pbig @ Pbig @ np.diag(1.0/d)
    res["limit"] = float(np.max(np.abs(Klim - np.ones((60, 60)) / d.sum())))

    ok = (res["und_maha"] < 1e-10 and res["und_K_eq_P2t"] < 1e-10 and res["und_Ksym"] < 1e-12 and
          res["und_rwdk_psd"] > -1e-9 and res["dg_maha"] < 1e-10 and res["dg_rwdk_sym"] < 1e-12 and
          res["dg_rwdk_psd"] > -1e-9 and res["dg_emb_sym"] < 1e-12 and
          res["dg_emb_even_mineig"] > -1e-9 and res["limit"] < 1e-8)
    print("="*74)
    print("Claim 4  Diffusion distance = Mahalanobis distance with RWDK kernel")
    print("arXiv 2210.00310 / OpenReview 5vI6ApLOg8  (numpy %s)" % np.__version__)
    print("="*74)
    print("(A) UNDIRECTED (reversing measure = degree d), t in {1,2,4}:")
    print(f"    max| ||p_t(i)-p_t(j)||^2_(1/d) - (e_i-e_j)^T K (e_i-e_j) |  = {res['und_maha']:.3e}  (=0)")
    print(f"    max| P^t D_d^-1 (P^t)^T - P^(2t) D_d^-1 |                   = {res['und_K_eq_P2t']:.3e}  (=0)")
    print(f"    RWDK K_t symmetry = {res['und_Ksym']:.3e} ; min eig(K_t) = {res['und_rwdk_psd']:.3e}  (>=0 PSD)")
    print("(B) DIGRAPH parametrized (reversing measure nu+xi), t in {1,2,4}:")
    print(f"    max| ||p_(t,nu)(i)-p_(t,nu)(j)||^2_(1/(nu+xi)) - (e_i-e_j)^T P_(nu)^(2t)D^-1 (e_i-e_j) | = {res['dg_maha']:.3e}  (=0)")
    print(f"    RWDK P_(nu)^(2t)D^-1 : sym = {res['dg_rwdk_sym']:.3e} ; min eig = {res['dg_rwdk_psd']:.3e}  (>=0 PSD)")
    print(f"    NUANCE Eq.7 embedding P_(nu)^t D^-1 : symmetric = {res['dg_emb_sym']:.3e} ;")
    print(f"       min eig EVEN t = {res['dg_emb_even_mineig']:.3e} (>=0 PSD) ; ODD t = {res['dg_emb_odd_mineig']:.3e} (<0 indefinite)")
    print(f"       => the Mahalanobis/RWDK matrix is the EVEN power P_(nu)^(2t)D^-1, per Prop 4.1")
    print(f"(C) diffusion limit t->inf : max| K_t - 1 1^T/tr(D_d) | = {res['limit']:.3e}  (-> 0, rank-1)")
    print("-"*74)
    print("VERDICT:", "PASS" if ok else "FAIL")
    print("="*74)
    out = dict(
        claim="diffusion distance == Mahalanobis with RWDK K_t=P^{2t}D_d^{-1} (Prop 4.1); parametrized digraph RWDK P_(nu)^{2t}D^{-1} symmetric PSD; Eq.7 single-power embedding PSD only for even t; rank-1 limit",
        numpy=np.__version__,
        undirected_mahalanobis_maxdiff=res["und_maha"],
        undirected_Kt_eq_P2t_Dd_inv_maxdiff=res["und_K_eq_P2t"],
        undirected_RWDK_symmetry=res["und_Ksym"], undirected_RWDK_min_eig=res["und_rwdk_psd"],
        digraph_parametrized_mahalanobis_maxdiff=res["dg_maha"],
        digraph_RWDK_double_power_symmetry=res["dg_rwdk_sym"],
        digraph_RWDK_double_power_min_eig=res["dg_rwdk_psd"],
        digraph_Eq7_embedding_symmetry=res["dg_emb_sym"],
        digraph_Eq7_embedding_even_t_min_eig=res["dg_emb_even_mineig"],
        digraph_Eq7_embedding_odd_t_min_eig=res["dg_emb_odd_mineig"],
        rank1_diffusion_limit_maxdiff=res["limit"],
        verdict="PASS" if ok else "FAIL")
    json.dump(out, open("results.json", "w"), indent=2)
    print("wrote results.json")

if __name__ == "__main__":
    main()
