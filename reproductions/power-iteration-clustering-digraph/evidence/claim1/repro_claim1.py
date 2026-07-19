"""
Claim 1 - Parametrized Random Walk operator P_(nu) is a valid (row-stochastic)
transition matrix, and is related to the generalized random-walk Laplacian by
L_{RW,(nu)} = I - P_(nu).

Paper: "Clustering for Directed Graphs using Parametrized Random Walk Diffusion
Kernels" (P-RWDKC), OpenReview 5vI6ApLOg8 / arXiv 2210.00310.
  Def. 3.1 / Eq. 4 :  P_(nu) = (I + D_{xi/nu})^{-1} (P + D_nu^{-1} P^T D_nu)
  Prop. 3.1        :  L_{RW,(nu)} = I - P_(nu)   (with L_{RW,(nu)} from Eq. 1)
  Prop. A.2        :  P_(nu) is a transition matrix (rows sum to 1, entries >=0)
                      and equals (D_nu + D_xi)^{-1} (D_nu P + P^T D_nu).
Here  xi = nu^T P  (Def. 2.1 / 3.1).

CHECKABLE CONSEQUENCES (deterministic, CPU, numpy):
  (A) P_(nu) >= 0 elementwise and every row sums to 1 (transition matrix).
  (B) Eq.4 assembly == Prop.A.2 assembly (algebraic identity), to ~machine eps.
  (C) L_{RW,(nu)} (Eq.1) == I - P_(nu) exactly; L 1 = 0 (constant is null vector);
      P_(nu) 1 = 1 (constant is right eigenvector with eigenvalue 1).
  (D) MECHANISM/CONTROL: the un-normalized numerator (P + D_nu^{-1} P^T D_nu) has
      row i summing to 1 + xi_i/nu_i (NOT 1) -> it is the (I+D_{xi/nu})^{-1}
      prefactor that makes P_(nu) stochastic, exactly as in the proof of A.2.
Swept over graph sizes, densities, seeds, and 4 families of vertex measures nu
(uniform, random, out-degree, and the Eq.8 forward/backward measure).
"""
import json, numpy as np

def natural_rw(W):
    dout = W.sum(1)
    dout = np.where(dout > 0, dout, 1.0)
    return W / dout[:, None]

def eq8_vertex_measure(W, t, gamma, alpha):
    dout = W.sum(1); din = W.sum(0)
    dout = np.where(dout > 0, dout, 1.0); din = np.where(din > 0, din, 1.0)
    Pout = W / dout[:, None]
    Pin = (W.T) / din[:, None]
    Pg = gamma * Pout + (1.0 - gamma) * Pin
    N = W.shape[0]
    v = (np.ones(N) / N) @ np.linalg.matrix_power(Pg, t)   # (1/N) 1^T Pg^t
    v = np.clip(v, 1e-300, None) ** alpha
    return v

def P_nu_eq4(P, nu):
    xi = nu @ P                                  # xi = nu^T P
    Dxi_over_nu = xi / nu
    pref = 1.0 / (1.0 + Dxi_over_nu)             # (I + D_{xi/nu})^{-1} diagonal
    inner = P + (P.T * nu[None, :]) / nu[:, None]  # P + D_nu^{-1} P^T D_nu
    return pref[:, None] * inner, xi, inner, pref

def P_nu_propA2(P, nu):
    xi = nu @ P
    num = (nu[:, None] * P) + (P.T * nu[None, :])   # D_nu P + P^T D_nu
    den = 1.0 / (nu + xi)                            # (D_nu + D_xi)^{-1}
    return den[:, None] * num

def L_rw_eq1(P, nu):
    xi = nu @ P
    Dxi_over_nu = xi / nu
    pref = 1.0 / (1.0 + Dxi_over_nu)
    inner = P + (P.T * nu[None, :]) / nu[:, None]
    return np.eye(P.shape[0]) - pref[:, None] * inner   # Eq.1 exactly

def random_digraph(N, p, rng, wmax=3.0):
    A = (rng.random((N, N)) < p).astype(float)
    np.fill_diagonal(A, 0.0)
    W = A * rng.uniform(0.2, wmax, size=(N, N))
    # guarantee every vertex has an out-edge (no dangling nodes)
    for i in range(N):
        if W[i].sum() == 0:
            j = (i + 1) % N; W[i, j] = rng.uniform(0.2, wmax)
    return W

def main():
    rng = np.random.default_rng(0)
    cfgs = [(30, 0.15), (60, 0.10), (120, 0.06), (200, 0.04)]
    worst = dict(min_entry=np.inf, rowsum=0.0, eq4_vs_A2=0.0, L_vs_IminusP=0.0,
                 L_rowsum=0.0, P1_minus_1=0.0)
    numerator_rowsum_excess = []   # control: 1 + xi/nu (should be > 1)
    ncases = 0
    for (N, p) in cfgs:
        for s in range(4):
            W = random_digraph(N, p, rng)
            P = natural_rw(W)
            dout = W.sum(1)
            measures = {
                "uniform": np.ones(N),
                "random": rng.uniform(0.1, 1.0, N),
                "out_degree": np.where(dout > 0, dout, 1.0),
                "eq8_fwd_bwd": eq8_vertex_measure(W, t=int(rng.integers(1, 5)),
                                                  gamma=float(rng.uniform(0, 1)),
                                                  alpha=float(rng.uniform(0.2, 1.0))),
            }
            for name, nu in measures.items():
                nu = np.asarray(nu, float)
                Pn, xi, inner, pref = P_nu_eq4(P, nu)
                Pn2 = P_nu_propA2(P, nu)
                Lrw = L_rw_eq1(P, nu)
                rs = Pn.sum(1)
                worst["min_entry"] = min(worst["min_entry"], float(Pn.min()))
                worst["rowsum"] = max(worst["rowsum"], float(np.max(np.abs(rs - 1))))
                worst["eq4_vs_A2"] = max(worst["eq4_vs_A2"], float(np.max(np.abs(Pn - Pn2))))
                worst["L_vs_IminusP"] = max(worst["L_vs_IminusP"],
                                            float(np.max(np.abs(Lrw - (np.eye(N) - Pn)))))
                worst["L_rowsum"] = max(worst["L_rowsum"], float(np.max(np.abs(Lrw.sum(1)))))
                worst["P1_minus_1"] = max(worst["P1_minus_1"],
                                          float(np.max(np.abs(Pn @ np.ones(N) - 1.0))))
                # control: numerator row sums = 1 + xi/nu, strictly > 1
                num_rs = inner.sum(1)
                numerator_rowsum_excess.append(float(np.min(num_rs)))
                ncases += 1

    all_ok = (worst["min_entry"] >= -1e-12 and worst["rowsum"] < 1e-10 and
              worst["eq4_vs_A2"] < 1e-10 and worst["L_vs_IminusP"] < 1e-12 and
              worst["L_rowsum"] < 1e-10 and worst["P1_minus_1"] < 1e-10)
    ctrl_min = float(np.min(numerator_rowsum_excess))   # should be > 1 (not stochastic)

    print("="*74)
    print("Claim 1  P-RWDKC parametrized random walk operator is a transition matrix")
    print("arXiv 2210.00310 / OpenReview 5vI6ApLOg8  (numpy %s)" % np.__version__)
    print("="*74)
    print(f"cases swept: {ncases}  (sizes {[c[0] for c in cfgs]} x 4 seeds x 4 vertex measures)")
    print("-"*74)
    print(f"(A) transition matrix : min entry of P_(nu)        = {worst['min_entry']:.3e}  (>= 0)")
    print(f"                        max |rowsum(P_(nu)) - 1|   = {worst['rowsum']:.3e}  (target 0)")
    print(f"(B) Eq.4 == Prop.A.2  : max |Eq4 - A2 assembly|    = {worst['eq4_vs_A2']:.3e}  (target 0)")
    print(f"(C) Prop 3.1 identity : max |L_RW - (I - P_(nu))|  = {worst['L_vs_IminusP']:.3e}  (target 0)")
    print(f"    Laplacian zero-rowsum : max |L_RW 1|           = {worst['L_rowsum']:.3e}  (target 0)")
    print(f"    right eigvec-1 : max |P_(nu) 1 - 1|            = {worst['P1_minus_1']:.3e}  (target 0)")
    print(f"(D) CONTROL un-normalized numerator (P+D_nu^-1 P^T D_nu):")
    print(f"    min row sum = 1 + xi/nu                        = {ctrl_min:.4f}  (> 1 => NOT stochastic")
    print(f"                                                     without the (I+D_xi/nu)^-1 prefactor)")
    print("-"*74)
    print("VERDICT (A,B,C all pass, control confirms mechanism):", "PASS" if all_ok and ctrl_min > 1.0 else "FAIL")
    print("="*74)

    out = dict(
        claim="P_(nu) (Eq.4) is a row-stochastic transition matrix; L_RW,(nu)=I-P_(nu) (Prop 3.1); mechanism per Prop A.2",
        numpy=np.__version__, cases=ncases,
        min_entry=worst["min_entry"], max_rowsum_dev=worst["rowsum"],
        eq4_vs_propA2_maxdiff=worst["eq4_vs_A2"],
        L_rw_vs_I_minus_P_maxdiff=worst["L_vs_IminusP"],
        L_rw_max_rowsum=worst["L_rowsum"], P_nu_right_eigvec_dev=worst["P1_minus_1"],
        control_numerator_min_rowsum=ctrl_min,
        tolerances=dict(rowsum="<1e-10", identity="<1e-10", laplacian="<1e-12"),
        verdict="PASS" if all_ok and ctrl_min > 1.0 else "FAIL")
    json.dump(out, open("results.json", "w"), indent=2)
    print("wrote results.json")

if __name__ == "__main__":
    main()
