"""
Independent NumPy reproduction of CLAIM 1 (Section 3) of
"Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural
Networks" (OpenReview aIH1jyU37z; no arXiv).

CLAIM 1: The paper introduces sheaf neural networks by replacing graph-Laplacian
diffusion with SHEAF-Laplacian diffusion that can encode ASYMMETRIC, SIGNED, and
VARYING-DIMENSIONAL relations (Section 3).

This is a construction/mathematics claim. It is verified to MACHINE PRECISION by
building the sheaf Laplacian L_F = delta^T delta of a cellular sheaf and checking:

  (R) REDUCTION   : the trivial sheaf (stalks R^1, restriction maps = 1) yields
                    exactly the ordinary graph Laplacian L = D - A.
  (S) SIGNED      : a 1-d sheaf with restriction maps in {+1,-1} yields exactly
                    the SIGNED graph Laplacian D_|A| - A_signed (encodes signed
                    relations that the plain graph Laplacian cannot).
  (A) ASYMMETRIC  : an O(2) sheaf with rotation restriction maps yields symmetric
                    PSD L_F whose off-diagonal blocks are genuine rotations
                    (NON-scalar), i.e. orthogonal/asymmetric "transport" that no
                    scalar-weighted graph Laplacian can represent.
  (V) VARYING-DIM : stalks of different dimensions -> rectangular restriction maps
                    -> a valid symmetric PSD L_F on a space of dimension != n.
  (P) PSD/SYMM    : L_F = delta^T delta is symmetric PSD for ALL sheaves (random
                    control, many seeds).

COMPARISON RULE (all must hold):
  - reduction/signed residual  max|L_F - target|   < 1e-10   (exact identity)
  - symmetry  max|L_F - L_F^T|                      < 1e-12
  - PSD  min eigenvalue(L_F)                        > -1e-9
  - rotational off-block distance to nearest cI     > 1e-1   (genuinely non-scalar)
FALSIFICATION: any identity residual > 1e-8, or L_F not PSD, or the signed/trivial
reduction disagrees with the graph Laplacian.
"""
import json
import numpy as np


def build_sheaf_laplacian(edges, node_dims, edge_dims, restr):
    """L_F = delta^T delta for a cellular sheaf.
    restr[(node, edge_index)] = matrix of shape (edge_dim, node_dim).
    Oriented coboundary for edge e=(u,v):  (delta x)_e = F_{v<e} x_v - F_{u<e} x_u.
    """
    voff = np.cumsum([0] + list(node_dims))
    eoff = np.cumsum([0] + list(edge_dims))
    N, M = int(voff[-1]), int(eoff[-1])
    delta = np.zeros((M, N))
    for ei, (u, v) in enumerate(edges):
        delta[eoff[ei]:eoff[ei + 1], voff[v]:voff[v + 1]] += restr[(v, ei)]
        delta[eoff[ei]:eoff[ei + 1], voff[u]:voff[u + 1]] -= restr[(u, ei)]
    return delta.T @ delta, delta, voff


def graph_laplacian(edges, V):
    A = np.zeros((V, V))
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return np.diag(A.sum(1)) - A, A


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def main():
    print("=" * 74)
    print("CLAIM 1  Sheaf-Laplacian diffusion generalizes graph-Laplacian diffusion")
    print("         (encodes ASYMMETRIC / SIGNED / VARYING-DIM relations, Section 3)")
    print("OpenReview aIH1jyU37z  -  independent NumPy implementation")
    print("=" * 74)
    res = {}
    TOL = 1e-10

    # test graph: 4 nodes, cycle + chord
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    V = 4

    # ---------- (R) REDUCTION: trivial sheaf == graph Laplacian ----------
    nd = [1] * V
    ed = [1] * len(edges)
    restr = {}
    for ei, (u, v) in enumerate(edges):
        restr[(u, ei)] = np.array([[1.0]])
        restr[(v, ei)] = np.array([[1.0]])
    Lsheaf, _, _ = build_sheaf_laplacian(edges, nd, ed, restr)
    Lgraph, A = graph_laplacian(edges, V)
    r_red = float(np.max(np.abs(Lsheaf - Lgraph)))
    print(f"(R) trivial-sheaf L_F == D - A         max|diff| = {r_red:.2e}")
    res["reduction_residual"] = r_red

    # ---------- (S) SIGNED: +/-1 restriction maps == signed Laplacian ----------
    esign = [+1, -1, +1, -1, +1]
    restr_s = {}
    for ei, (u, v) in enumerate(edges):
        restr_s[(u, ei)] = np.array([[1.0]])
        restr_s[(v, ei)] = np.array([[float(esign[ei])]])
    Lsigned, _, _ = build_sheaf_laplacian(edges, [1] * V, [1] * len(edges), restr_s)
    Asig = np.zeros((V, V))
    for ei, (u, v) in enumerate(edges):
        Asig[u, v] = Asig[v, u] = esign[ei]
    Lsig_expect = np.diag(np.abs(Asig).sum(1)) - Asig
    r_sig = float(np.max(np.abs(Lsigned - Lsig_expect)))
    sym_sig = float(np.max(np.abs(Lsigned - Lsigned.T)))
    eig_sig = float(np.linalg.eigvalsh(Lsigned).min())
    signed_differs = float(np.max(np.abs(Lsigned - Lgraph)))  # != plain graph Lap
    print(f"(S) signed-sheaf L_F == D-A_signed     max|diff| = {r_sig:.2e}  "
          f"sym={sym_sig:.1e} minEig={eig_sig:+.4f}")
    print(f"    signed L_F differs from plain graph Laplacian by max {signed_differs:.3f} "
          f"(sign info the graph Laplacian cannot carry)")
    res["signed_residual"] = r_sig
    res["signed_symmetry"] = sym_sig
    res["signed_min_eig"] = eig_sig
    res["signed_vs_graph_gap"] = signed_differs

    # ---------- (A) ASYMMETRIC: O(2) rotational sheaf ----------
    nd_o = [2] * V
    ed_o = [2] * len(edges)
    restr_o = {}
    for ei, (u, v) in enumerate(edges):
        restr_o[(u, ei)] = rot(0.0)
        restr_o[(v, ei)] = rot(0.7 * ei + 0.3)      # different map each end -> asymmetric
    Lo, _, voff = build_sheaf_laplacian(edges, nd_o, ed_o, restr_o)
    sym_o = float(np.max(np.abs(Lo - Lo.T)))
    eig_o = float(np.linalg.eigvalsh(Lo).min())
    u0, v0 = edges[0]
    blk = Lo[voff[u0]:voff[u0] + 2, voff[v0]:voff[v0] + 2]
    exp_blk = -restr_o[(u0, 0)].T @ restr_o[(v0, 0)]
    r_blk = float(np.max(np.abs(blk - exp_blk)))
    c_star = float(np.trace(blk) / 2.0)             # nearest scalar cI (Frobenius)
    nonscalar = float(np.linalg.norm(blk - c_star * np.eye(2)))
    print(f"(A) O(2)-sheaf L_F symmetric={sym_o:.1e}  minEig={eig_o:+.4f}  "
          f"off-block==-Fu^T Fv max|diff|={r_blk:.2e}")
    print(f"    off-block distance to nearest scalar cI = {nonscalar:.4f} "
          f"(non-scalar rotation; not representable by weighted graph Laplacian)")
    res["rot_symmetry"] = sym_o
    res["rot_min_eig"] = eig_o
    res["rot_block_residual"] = r_blk
    res["rot_nonscalar_dist"] = nonscalar

    # ---------- (V) VARYING-DIMENSIONAL stalks ----------
    rng = np.random.default_rng(0)
    nd_v = [2, 3, 1, 2]
    ed_v = [2, 1, 1, 2, 1]
    restr_v = {}
    for ei, (u, v) in enumerate(edges):
        restr_v[(u, ei)] = rng.standard_normal((ed_v[ei], nd_v[u]))
        restr_v[(v, ei)] = rng.standard_normal((ed_v[ei], nd_v[v]))
    Lv, _, _ = build_sheaf_laplacian(edges, nd_v, ed_v, restr_v)
    sym_v = float(np.max(np.abs(Lv - Lv.T)))
    eig_v = float(np.linalg.eigvalsh(Lv).min())
    print(f"(V) varying-dim stalks {nd_v}: L_F shape={Lv.shape} (total dim "
          f"{Lv.shape[0]} != n={V})  sym={sym_v:.1e}  minEig={eig_v:+.2e}")
    res["vardim_total"] = int(Lv.shape[0])
    res["vardim_symmetry"] = sym_v
    res["vardim_min_eig"] = eig_v

    # ---------- (P) PSD/SYMMETRY control: random sheaves ----------
    worst_sym = 0.0
    worst_eig = np.inf
    worst_red = 0.0
    for seed in range(200):
        rg = np.random.default_rng(1000 + seed)
        n = rg.integers(4, 9)
        p = 0.5
        A2 = (rg.random((n, n)) < p).astype(float)
        A2 = np.triu(A2, 1)
        E2 = [(i, j) for i in range(n) for j in range(i + 1, n) if A2[i, j] > 0]
        if not E2:
            continue
        d = rg.integers(1, 4)   # common stalk dim
        rr = {}
        for ei, (u, v) in enumerate(E2):
            rr[(u, ei)] = rg.standard_normal((d, d))
            rr[(v, ei)] = rg.standard_normal((d, d))
        Lr, _, _ = build_sheaf_laplacian(E2, [d] * n, [d] * len(E2), rr)
        worst_sym = max(worst_sym, float(np.max(np.abs(Lr - Lr.T))))
        worst_eig = min(worst_eig, float(np.linalg.eigvalsh(Lr).min()))
        # trivial-sheaf reduction control on the same random graph
        rt = {}
        for ei, (u, v) in enumerate(E2):
            rt[(u, ei)] = np.array([[1.0]])
            rt[(v, ei)] = np.array([[1.0]])
        Lt, _, _ = build_sheaf_laplacian(E2, [1] * n, [1] * len(E2), rt)
        Lg2, _ = graph_laplacian(E2, n)
        worst_red = max(worst_red, float(np.max(np.abs(Lt - Lg2))))
    print(f"(P) random control (200 sheaves): worst |L-L^T|={worst_sym:.2e}  "
          f"worst minEig={worst_eig:+.2e}  worst trivial-reduction residual={worst_red:.2e}")
    res["ctrl_worst_symmetry"] = worst_sym
    res["ctrl_worst_min_eig"] = worst_eig
    res["ctrl_worst_reduction"] = worst_red

    # ---------- VERDICT ----------
    checks = {
        "reduction_exact": r_red < TOL,
        "signed_exact": r_sig < TOL,
        "signed_symm": sym_sig < 1e-12,
        "signed_psd": eig_sig > -1e-9,
        "rot_symm": sym_o < 1e-12,
        "rot_psd": eig_o > -1e-9,
        "rot_block_exact": r_blk < TOL,
        "rot_nonscalar": nonscalar > 1e-1,
        "vardim_symm": sym_v < 1e-12,
        "vardim_psd": eig_v > -1e-9,
        "ctrl_symm": worst_sym < 1e-9,
        "ctrl_psd": worst_eig > -1e-8,
        "ctrl_reduction": worst_red < TOL,
    }
    res["checks"] = {k: bool(v) for k, v in checks.items()}
    overall = all(checks.values())
    res["overall_pass"] = bool(overall)
    print("-" * 74)
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}  "
          f"(sheaf-Laplacian is a strict generalization encoding "
          f"signed/asymmetric/varying-dim relations)")
    print("=" * 74)
    print("JSON_SUMMARY=" + json.dumps(res))


if __name__ == "__main__":
    main()
