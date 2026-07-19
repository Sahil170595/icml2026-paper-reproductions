"""
Independent NumPy reproduction of CLAIM 2 (Section 2.1) of
"Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural
Networks" (OpenReview aIH1jyU37z; no arXiv).

CLAIM 2: The sheaf diffusion operator is presented as a DROP-IN GENERALIZATION of
the diffusion operation used in graph convolutional networks (Section 2.1).

Verified to MACHINE PRECISION.  For the TRIVIAL sheaf (all stalks R^1, all
restriction maps = 1) the normalized sheaf diffusion collapses EXACTLY onto the
GCN propagation, so a sheaf-diffusion layer is a strict superset of a GCN layer
(recovered at stalk dimension d = 1).

  Sheaf Laplacian        L_F     = delta^T delta
  Normalized sheaf Lap.  Delta_F = D_F^{-1/2} L_F D_F^{-1/2}      (D_F = block-diag of L_F)
  Sheaf diffusion prop.  P_F     = I - Delta_F

Two GCN diffusion operators are matched:
  plain symmetric norm.  A_sym   = D^{-1/2} A D^{-1/2}            = I - L_sym
  Kipf-Welling renorm.   A_hat   = D~^{-1/2} (A + I) D~^{-1/2}    (Kipf & Welling 2017)

Note: a GCN self-loop is NOT a sheaf edge (a self-loop coboundary F x_i - F x_i = 0
vanishes).  The renormalization trick is realized on the sheaf side EXACTLY by
normalizing the trivial-sheaf Laplacian with the self-loop-augmented degree
D~ = diag(L_F) + I:  since L~ = D~ - (A+I) = D - A = L_F, one has
A_hat = I - D~^{-1/2} L_F D~^{-1/2}.

CHECKS (each over many random graphs: ER, SBM, path, star):
  (D1) trivial sheaf, degree normalization:          P_F == A_sym
  (D2) trivial sheaf, self-loop-augmented normaliz.:  I - D~^{-1/2} L_F D~^{-1/2} == A_hat
  (D3) layer output equality:                        op X W == A_hat X W (random X, W)
  (G)  strict generalization: nontrivial O(d>1) sheaf gives an (n*d)x(n*d) block
       operator with non-scalar off-blocks, reducing to a scalar GCN only at d=1.

COMPARISON RULE: max operator/layer difference < 1e-10 across ALL random graphs.
FALSIFICATION: any difference > 1e-8.
"""
import json
import numpy as np


def sheaf_laplacian(edges, node_dims, edge_dims, restr):
    voff = np.cumsum([0] + list(node_dims))
    eoff = np.cumsum([0] + list(edge_dims))
    N, M = int(voff[-1]), int(eoff[-1])
    delta = np.zeros((M, N))
    for ei, (u, v) in enumerate(edges):
        delta[eoff[ei]:eoff[ei + 1], voff[v]:voff[v + 1]] += restr[(v, ei)]
        delta[eoff[ei]:eoff[ei + 1], voff[u]:voff[u + 1]] -= restr[(u, ei)]
    return delta.T @ delta, voff


def norm_prop(L, add_self=0.0):
    """I - D^{-1/2} L D^{-1/2} with D = diag(L) + add_self (add_self=1 => renorm trick)."""
    d = np.diag(L).copy() + add_self
    d[d <= 0] = 1.0
    Dm = 1.0 / np.sqrt(d)
    return np.eye(L.shape[0]) - (Dm[:, None] * L * Dm[None, :])


def trivial_sheaf(edges):
    r = {}
    for ei, (u, v) in enumerate(edges):
        r[(u, ei)] = np.array([[1.0]])
        r[(v, ei)] = np.array([[1.0]])
    return r


def gcn_prop(A, self_loops=True):
    n = A.shape[0]
    At = A + np.eye(n) if self_loops else A
    d = At.sum(1)
    d[d <= 0] = 1.0
    Dm = 1.0 / np.sqrt(d)
    return Dm[:, None] * At * Dm[None, :]


def edges_of(A):
    n = A.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] > 0]


def random_graph(kind, n, rng):
    if kind == "er":
        A = (rng.random((n, n)) < 0.25).astype(float); A = np.triu(A, 1); A = A + A.T
    elif kind == "sbm":
        y = (np.arange(n) >= n // 2).astype(int)
        P = np.where(y[:, None] == y[None, :], 0.4, 0.1)
        A = (rng.random((n, n)) < P).astype(float); A = np.triu(A, 1); A = A + A.T
    elif kind == "path":
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
    elif kind == "star":
        A = np.zeros((n, n))
        for i in range(1, n):
            A[0, i] = A[i, 0] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def main():
    print("=" * 74)
    print("CLAIM 2  Sheaf diffusion is a DROP-IN generalization of GCN diffusion")
    print("         (Section 2.1)  -  OpenReview aIH1jyU37z, independent NumPy")
    print("=" * 74)
    TOL = 1e-10
    worst = {"D1_sym": 0.0, "D2_kipf_renorm": 0.0, "D3_layer_output": 0.0}
    n_graphs = 0
    for ki, kind in enumerate(["er", "sbm", "path", "star"]):
        for seed in range(30):
            rg = np.random.default_rng(1000 * ki + seed)
            n = int(rg.integers(5, 14))
            A = random_graph(kind, n, rg)
            if edges_of(A) == []:
                continue
            for i in np.where(A.sum(1) == 0)[0]:
                j = (i + 1) % n; A[i, j] = A[j, i] = 1.0
            E = edges_of(A)
            n_graphs += 1
            LF, _ = sheaf_laplacian(E, [1]*n, [1]*len(E), trivial_sheaf(E))

            # (D1) plain symmetric-normalized GCN diffusion
            Pf = norm_prop(LF, add_self=0.0)
            worst["D1_sym"] = max(worst["D1_sym"],
                                  float(np.max(np.abs(Pf - gcn_prop(A, self_loops=False)))))

            # (D2) Kipf-Welling renorm via self-loop-augmented degree D~ = diag(L_F)+I
            Pf_hat = norm_prop(LF, add_self=1.0)
            Ahat = gcn_prop(A, self_loops=True)
            worst["D2_kipf_renorm"] = max(worst["D2_kipf_renorm"],
                                          float(np.max(np.abs(Pf_hat - Ahat))))

            # (D3) layer output equality on random features/weights
            F, Cout = 8, 5
            X = rg.standard_normal((n, F)); W = rg.standard_normal((F, Cout))
            worst["D3_layer_output"] = max(worst["D3_layer_output"],
                                           float(np.max(np.abs(Pf_hat @ X @ W - Ahat @ X @ W))))

    print(f"random graphs tested: {n_graphs} (ER / SBM / path / star, n=5..13)")
    print(f"(D1) trivial sheaf == A_sym = D^-1/2 A D^-1/2:         max|diff| = {worst['D1_sym']:.2e}")
    print(f"(D2) trivial sheaf renorm == Kipf-Welling A_hat:      max|diff| = {worst['D2_kipf_renorm']:.2e}")
    print(f"(D3) sheaf layer  op X W  ==  GCN layer  A_hat X W:   max|diff| = {worst['D3_layer_output']:.2e}")

    # (G) strict generalization: O(3) sheaf
    rgg = np.random.default_rng(7)
    n = 6
    A = random_graph("er", n, rgg)
    for i in np.where(A.sum(1) == 0)[0]:
        j = (i + 1) % n; A[i, j] = A[j, i] = 1.0
    E = edges_of(A)
    d = 3
    def orth(k, r):
        Q, _ = np.linalg.qr(r.standard_normal((k, k))); return Q
    rr = {}
    for ei, (u, v) in enumerate(E):
        rr[(u, ei)] = orth(d, rgg); rr[(v, ei)] = orth(d, rgg)
    Lo, _ = sheaf_laplacian(E, [d]*n, [d]*len(E), rr)
    Po = norm_prop(Lo, add_self=0.0)
    dim_o = Po.shape[0]
    voff = np.cumsum([0] + [d]*n)
    nonscalar = []
    for (u, v) in E:
        B = Po[voff[u]:voff[u]+d, voff[v]:voff[v]+d]
        c = np.trace(B)/d
        nonscalar.append(float(np.linalg.norm(B - c*np.eye(d))))
    frac_nonscalar = float(np.mean([x > 1e-6 for x in nonscalar]))
    print(f"(G)  O(3) sheaf operator dim = {dim_o} = n*d = {n}*{d} (vs GCN {n}x{n}); "
          f"off-blocks non-scalar fraction = {frac_nonscalar:.2f} "
          f"(strictly richer than scalar GCN, reduces to A_hat at d=1)")

    checks = {
        "D1_sym": worst["D1_sym"] < TOL,
        "D2_kipf_renorm": worst["D2_kipf_renorm"] < TOL,
        "D3_layer_output": worst["D3_layer_output"] < TOL,
        "G_strict_generalization": (dim_o == n * d) and (frac_nonscalar > 0.5),
    }
    overall = all(checks.values())
    print("-" * 74)
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}  "
          f"(sheaf diffusion is a drop-in generalization of GCN diffusion)")
    print("=" * 74)
    res = dict(n_graphs=n_graphs, worst=worst, gen_dim=int(dim_o),
               gen_frac_nonscalar=frac_nonscalar,
               checks={k: bool(v) for k, v in checks.items()},
               overall_pass=bool(overall))
    print("JSON_SUMMARY=" + json.dumps(res))


if __name__ == "__main__":
    main()
