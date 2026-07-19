"""
CLAIM 3 (space scoreboard) / paper Sec. 3.3 - Tucker Attention is "fully
compatible with flash-attention" (and, combined with repro_claim5.py, RoPE).
Paper: arXiv 2603.30033 / ErcPPRZaiq, Sec. 3.3 "Flash Attention".

The paper's compatibility argument (Sec. 3.3, verified against the arXiv HTML
source) is NOT a throughput claim -- it is an ALGORITHMIC claim: (1) Tucker's
key/value projections K=X U3, V=X~U3 are SHARED across all n_H heads (like
MQA / n_KV=1), so only a single KV chunk pair needs to be resident in SRAM per
attention computation; (2) those chunks live in the small latent rank r3
(=d_c^K), not the head dimension d_H, so the online-softmax recurrence used by
flash-attention (Dao, 2023) applies to Tucker Attention with NO custom kernel:
"we ... invoke the production-level flash-attention kernel in PyTorch, which
natively supports GQA-style grouped heads. Since Tucker Attention uses a
single shared KV pair for all query heads, this corresponds to GQA with
n_KV=1 (i.e. MQA) from the kernel's perspective."

FlashAttention is an EXACT algorithm (Dao, 2022/2023): tiling KV into blocks
and accumulating the output with running max/normalizer (the online-softmax
recurrence) reproduces full-softmax attention exactly (up to floating-point
summation order) -- it never approximates. We therefore verify Tucker's flash
compatibility on CPU as an EXACTNESS statement:

  (a) implement the FlashAttention-1 tiled/online-softmax recurrence in numpy
      from scratch (blocked over KV tiles, running max m_i + running
      normalizer l_i, rescaled accumulator O_i -- Dao 2022 Alg. 1);
  (b) run Tucker Attention's factored forward (Sec. 3, Def. 2.1: per-head
      query Q_i = C x1 U1 x2 (X U2) restricted to head i, SHARED key/value
      K=X U3, V=X Ut3) through this tiled kernel and show it reproduces the
      naive full-softmax Tucker forward (already verified == MHA/GQA/MLA in
      claims 1-3) to machine precision, across shapes / tile sizes / seeds,
      including non-divisor block sizes that stress the tiling remainder path;
  (c) do the same for the MHA / GQA(n_KV=2) / MLA Tucker constructions from
      Theorems B.2/B.3/B.4 (claims 1/2/3), each checked against its own
      reference layer, not just against the naive Tucker forward;
  (d) compose tiled + latent-RoPE (Def. 3.1, already verified in
      repro_claim5.py) + Tucker together and show the combination is still
      exact to machine precision;
  (e) measure the memory-profile argument directly: track the size (in
      elements) of the largest intermediate array actually allocated by the
      naive path (the N x N logits matrix) vs the tiled path (the block_q x
      block_k tile) as N grows -- the tiled path's peak is O(block^2),
      constant in N, vs naive's O(N^2).

This verifies ALGORITHMIC flash-attention compatibility (exactness of Tucker
Attention under the flash tiled/online-softmax recurrence, and that its
shared-KV/latent-rank structure is what lets it reuse an unmodified flash
kernel). It does NOT measure GPU kernel throughput/wall-clock, which needs a
real CUDA flash-attention kernel and is out of CPU scope -- consistent with
Claim 6's GPU-only validation-accuracy split.
"""
import json, time, hashlib, numpy as np
from tucker_common import (mode1, mode2, mode3, tucker_reconstruct, softmax_rows,
                            matn_rank, tucker_attention_forward, mha_reference,
                            build_mha_tensors)

t0 = time.time()

# =============================================================================
# (a) FlashAttention-1 tiled / online-softmax recurrence, single head, numpy.
#     Dao (2022) "FlashAttention", Algorithm 1: block over KV, keep a running
#     row-max m_i and running normalizer l_i, rescale the output accumulator
#     O_i whenever the running max is updated.  Never materializes the full
#     (N,N) score matrix -- peak intermediate is (block_q, block_k).
# =============================================================================
def flash_tile_1head(Q, K, V, scale, bq, bk):
    N, r = Q.shape
    M = K.shape[0]
    dv = V.shape[1]
    O = np.zeros((N, dv))
    peak = 0
    for qs in range(0, N, bq):
        qe = min(qs + bq, N)
        Qb = Q[qs:qe]
        nb = qe - qs
        Ob = np.zeros((nb, dv))
        lb = np.zeros(nb)
        mb = np.full(nb, -np.inf)
        for ks in range(0, M, bk):
            ke = min(ks + bk, M)
            Kb = K[ks:ke]; Vb = V[ks:ke]
            Sij = (Qb @ Kb.T) * scale                     # (nb, nkb) -- NEVER (N,N)
            peak = max(peak, Sij.size)
            mij = Sij.max(axis=1)
            mnew = np.maximum(mb, mij)
            Pij = np.exp(Sij - mnew[:, None])
            peak = max(peak, Pij.size)
            alpha = np.exp(mb - mnew)
            lb = alpha * lb + Pij.sum(axis=1)
            Ob = alpha[:, None] * Ob + Pij @ Vb
            mb = mnew
        O[qs:qe] = Ob / lb[:, None]
    return O, peak

def naive_1head(Q, K, V, scale):
    S = (Q @ K.T) * scale
    peak = S.size
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=1, keepdims=True)
    return P @ V, peak

# sanity: flash-tiled == naive on a single random head, several tile sizes.
rng0 = np.random.default_rng(100)
_Q = rng0.standard_normal((37, 6)); _K = rng0.standard_normal((41, 6)); _V = rng0.standard_normal((41, 5))
sanity_errs = []
for bq, bk in [(1, 1), (5, 7), (16, 16), (37, 41), (9, 3)]:
    o_f, _ = flash_tile_1head(_Q, _K, _V, 1/np.sqrt(6), bq, bk)
    o_n, _ = naive_1head(_Q, _K, _V, 1/np.sqrt(6))
    sanity_errs.append(float(np.max(np.abs(o_f - o_n))))
err_sanity = max(sanity_errs)

# =============================================================================
# (b) Multi-head Tucker Attention through the tiled kernel.  Per Sec. 3.3, the
#     key/value projections K=X U3, V=X~U3 are shared across ALL heads (single
#     KV pair, like MQA); only the query Q_i = (X U2) contracted with core
#     C x1 U1 restricted to head i differs per head.  We decompose the
#     post-softmax combination (already verified in tucker_attention_forward,
#     tucker_common.py) into a per-head effective value matrix V_i_eff so each
#     head reduces to a single-head flash/naive attention with SHARED K.
# =============================================================================
def tucker_forward_naive_peak(C, U1, U2, U3, Ct, U1t, U2t, U3t, X, dH):
    out = tucker_attention_forward(C, U1, U2, U3, Ct, U1t, U2t, U3t, X, dH)
    nH = C.shape[0]; N = X.shape[0]
    return out, nH * N * N     # peak = size of the (nH,N,N) logits array

def tucker_forward_tiled(C, U1, U2, U3, Ct, U1t, U2t, U3t, X, dH, bq, bk):
    nH = C.shape[0]; N = X.shape[0]; d = U2t.shape[0]
    P2 = X @ U2                              # (N, r2)
    K  = X @ U3                              # (N, r3)  SHARED across heads (MQA-style)
    T  = mode1(C, U1)                        # (nH, r2, r3)
    Qh = np.einsum('nb,ibc->inc', P2, T)     # (nH, N, r3)  per-head query
    V  = X @ U3t                             # (N, r3t) SHARED across heads
    Mi = mode1(Ct, U1t)                      # (nH, r2t, r3t)
    scale = 1.0 / np.sqrt(dH)
    out = np.zeros((N, d))
    peak = 0
    for i in range(nH):
        Vi_pre = V @ Mi[i].T                 # (N, r2t)
        Vi_eff = Vi_pre @ U2t.T              # (N, d)  per-head effective value
        Oi, p = flash_tile_1head(Qh[i], K, Vi_eff, scale, bq, bk)
        peak = max(peak, p)
        out += Oi
    return out, peak

print("=" * 78)
print("CLAIM 3 (flash-attention half)  Tucker Attention is flash-attention compatible")
print("Paper Sec. 3.3: shared KV (single pair, all heads) + latent-rank chunks")
print("=" * 78)
print(f"(a) FlashAttention-1 tiled recurrence vs naive softmax, 1 head, 5 tile sizes")
print(f"    max abs error = {err_sanity:.3e}")

# --- generic (non-special-case) random Tucker configurations ---------------
generic_cases = []
shapes = [(3, 22, 7, 5, 47, [(7, 11), (16, 16), (47, 47), (5, 6)]),
          (4, 24, 6, 8, 63, [(9, 13), (32, 32), (63, 20)]),
          (2, 18, 9, 4, 30, [(4, 4), (30, 30), (7, 12)])]
for seed, (nH, d, dH, r2, N, tiles) in enumerate(shapes):
    r3 = r2 - 1 if r2 > 2 else r2           # asymmetric r2 != r3 to stress general case
    rng = np.random.default_rng(200 + seed)
    X  = rng.standard_normal((N, d))
    U1 = np.eye(nH)
    U2 = rng.standard_normal((d, r2)); U3 = rng.standard_normal((d, r3))
    C  = rng.standard_normal((nH, r2, r3))
    U2t = rng.standard_normal((d, d)); U3t = rng.standard_normal((d, r3))
    Ct  = rng.standard_normal((nH, d, r3))
    ref_out, peak_naive = tucker_forward_naive_peak(C, U1, U2, U3, Ct, U1, U2t, U3t, X, dH)
    for (bq, bk) in tiles:
        t_out, peak_tiled = tucker_forward_tiled(C, U1, U2, U3, Ct, U1, U2t, U3t, X, dH, bq, bk)
        err = float(np.max(np.abs(t_out - ref_out)))
        generic_cases.append(dict(seed=200 + seed, nH=nH, d=d, dH=dH, r2=r2, r3=r3, N=N,
                                   bq=bq, bk=bk, err=err, peak_naive=int(peak_naive),
                                   peak_tiled=int(peak_tiled)))
err_generic = max(c["err"] for c in generic_cases)
print(f"\n(b) generic random Tucker Attention: tiled vs naive-Tucker forward,"
      f" {len(generic_cases)} (shape,tile-size) combos, 3 seeds")
print(f"    max abs error = {err_generic:.3e}")
for c in generic_cases[:4]:
    print(f"      nH={c['nH']} N={c['N']} r2={c['r2']} r3={c['r3']} bq={c['bq']} bk={c['bk']}"
          f"  err={c['err']:.2e}  peak_naive={c['peak_naive']}  peak_tiled={c['peak_tiled']}")
print(f"      ... ({len(generic_cases)-4} more rows in results.json)")

# =============================================================================
# (c) recovered special cases (Thm B.2/B.3/B.4): build the SAME Tucker core /
#     factor tensors used in claims 1-3, verify tiled == naive-Tucker AND
#     tiled == the original reference layer (MHA / GQA / MLA), not merely a
#     transitive argument through the other claims.
# =============================================================================
def gqa_reference(X, WQ_heads, WK_kv, WV_kv, WO_heads, group_of, dH):
    N, d = X.shape; nH = len(WQ_heads)
    out = np.zeros((N, d))
    for i in range(nH):
        g = group_of[i]
        Qi = X @ WQ_heads[i]; Ki = X @ WK_kv[g]; Vi = X @ WV_kv[g]; Oi = WO_heads[i]
        A = softmax_rows(Qi @ Ki.T / np.sqrt(dH))
        out += A @ Vi @ Oi
    return out

def mla_reference(X, WDQ, WUQ, WDKV, WUK, WUV, WO_heads, nH, dH):
    N, d = X.shape
    out = np.zeros((N, d))
    LQ = X @ WDQ; LKV = X @ WDKV
    for i in range(nH):
        WiUQ = WUQ[:, i*dH:(i+1)*dH]; WiUK = WUK[:, i*dH:(i+1)*dH]; WiUV = WUV[:, i*dH:(i+1)*dH]
        Oi = WO_heads[i]
        Qi = LQ @ WiUQ; Ki = LKV @ WiUK; Vi = LKV @ WiUV
        A = softmax_rows(Qi @ Ki.T / np.sqrt(dH))
        out += A @ Vi @ Oi
    return out

special = {}

# --- MHA (Thm B.2): U3 = full concat of per-head keys, core = block selector.
rng = np.random.default_rng(11)
nH, dH, N = 4, 5, 53
d = nH * dH
X = rng.standard_normal((N, d))
WQ = rng.standard_normal((d, d)); WK = rng.standard_normal((d, d)); WV = rng.standard_normal((d, d))
WO = rng.standard_normal((d, d))
ref_mha = mha_reference(X, WQ, WK, WV, WO, nH, dH)
W, Wt = build_mha_tensors(WQ, WK, WV, WO, nH, dH)
U1 = np.eye(nH); U2 = WQ.copy(); U3 = WK.copy()
# MHA core: C_i selects head-i's own query/key block from the full
# concatenated (d-dim) query/key mode -- the Thm B.2 construction.
C = np.zeros((nH, d, d))
for i in range(nH):
    for l in range(dH):
        C[i, i*dH + l, i*dH + l] = 1.0
Wcheck = tucker_reconstruct(C, U1, U2, U3)
errW = float(np.max(np.abs(Wcheck - W)))
# Post-softmax factors: U2t = I_d (output mode kept full), U3t = WV (value
# projection factor, mirrors U3=WK for the key/query side), core Ct_i places
# WO_i^T into the columns of head i's own value block so that
# Ct_i @ WV^T == WO_i^T @ WiV^T == (WiV @ WO_i)^T == Wt_i (matches
# build_mha_tensors' Wt[i] = (WiV @ WiO).T exactly).
U1t = np.eye(nH)
U2t = np.eye(d)
U3t = WV.copy()
Ctt = np.zeros((nH, d, d))
for i in range(nH):
    Ctt[i][:, i*dH:(i+1)*dH] = WO[i*dH:(i+1)*dH, :].T
Wtcheck = tucker_reconstruct(Ctt, U1t, U2t, U3t)
errWt = float(np.max(np.abs(Wtcheck - Wt)))
naive_mha, peak_naive_mha = tucker_forward_naive_peak(C, U1, U2, U3, Ctt, U1t, U2t, U3t, X, dH)
tiled_mha, peak_tiled_mha = tucker_forward_tiled(C, U1, U2, U3, Ctt, U1t, U2t, U3t, X, dH, 7, 11)
special["MHA"] = dict(nH=nH, dH=dH, d=d, N=N,
                       err_tucker_construction=errW,
                       err_tucker_construction_post=errWt,
                       err_naive_vs_reference=float(np.max(np.abs(naive_mha - ref_mha))),
                       err_tiled_vs_reference=float(np.max(np.abs(tiled_mha - ref_mha))),
                       err_tiled_vs_naive=float(np.max(np.abs(tiled_mha - naive_mha))),
                       peak_naive=int(peak_naive_mha), peak_tiled=int(peak_tiled_mha))

# --- GQA (Thm B.3, n_KV=2): U3 = concat of n_KV distinct KV heads (shared
#     within a group), core routes head i to its group.
rng = np.random.default_rng(12)
nH, nKV, dH, N = 8, 2, 4, 61
d = nH * dH
X = rng.standard_normal((N, d))
WQ_heads = [rng.standard_normal((d, dH)) for _ in range(nH)]
WO_heads = [rng.standard_normal((dH, d)) for _ in range(nH)]
WK_kv = [rng.standard_normal((d, dH)) for _ in range(nKV)]
WV_kv = [rng.standard_normal((d, dH)) for _ in range(nKV)]
per = nH // nKV
group_of = [i // per for i in range(nH)]
ref_gqa = gqa_reference(X, WQ_heads, WK_kv, WV_kv, WO_heads, group_of, dH)
U1 = np.eye(nH); U2 = np.concatenate(WQ_heads, axis=1); U3 = np.concatenate(WK_kv, axis=1)
r3 = nKV * dH
C = np.zeros((nH, d, r3))
for i in range(nH):
    g = group_of[i]
    for l in range(dH):
        C[i, i*dH + l, g*dH + l] = 1.0
U2t = np.concatenate([O.T for O in WO_heads], axis=1); U3t = np.concatenate(WV_kv, axis=1)
Ct = np.zeros((nH, d, r3))
for i in range(nH):
    g = group_of[i]
    for l in range(dH):
        Ct[i, i*dH + l, g*dH + l] = 1.0
naive_gqa, peak_naive_gqa = tucker_forward_naive_peak(C, U1, U2, U3, Ct, U1, U2t, U3t, X, dH)
tiled_gqa, peak_tiled_gqa = tucker_forward_tiled(C, U1, U2, U3, Ct, U1, U2t, U3t, X, dH, 9, 5)
special["GQA_nKV2"] = dict(nH=nH, nKV=nKV, dH=dH, d=d, N=N,
                            err_naive_vs_reference=float(np.max(np.abs(naive_gqa - ref_gqa))),
                            err_tiled_vs_reference=float(np.max(np.abs(tiled_gqa - ref_gqa))),
                            err_tiled_vs_naive=float(np.max(np.abs(tiled_gqa - naive_gqa))),
                            peak_naive=int(peak_naive_gqa), peak_tiled=int(peak_tiled_gqa))

# --- MLA (Thm B.4): shared low-rank KV down-projection W^DKV.
rng = np.random.default_rng(13)
nH, dH, dcQ, dcK, N = 6, 6, 8, 6, 59
d = nH * dH
X = rng.standard_normal((N, d))
WDQ = rng.standard_normal((d, dcQ)); WDKV = rng.standard_normal((d, dcK))
WUQ = rng.standard_normal((dcQ, d)); WUK = rng.standard_normal((dcK, d)); WUV = rng.standard_normal((dcK, d))
WO_heads = [rng.standard_normal((dH, d)) for _ in range(nH)]
ref_mla = mla_reference(X, WDQ, WUQ, WDKV, WUK, WUV, WO_heads, nH, dH)
U1 = np.eye(nH); U2 = WDQ.copy(); U3 = WDKV.copy()
C = np.zeros((nH, dcQ, dcK))
for i in range(nH):
    C[i] = WUQ[:, i*dH:(i+1)*dH] @ WUK[:, i*dH:(i+1)*dH].T
U1t = np.eye(nH); U2t = np.concatenate([O.T for O in WO_heads], axis=1); U3t = WDKV.copy()
Ct = np.zeros((nH, d, dcK))
for i in range(nH):
    Ct[i][i*dH:(i+1)*dH, :] = WUV[:, i*dH:(i+1)*dH].T
naive_mla, peak_naive_mla = tucker_forward_naive_peak(C, U1, U2, U3, Ct, U1t, U2t, U3t, X, dH)
tiled_mla, peak_tiled_mla = tucker_forward_tiled(C, U1, U2, U3, Ct, U1t, U2t, U3t, X, dH, 8, 8)
special["MLA"] = dict(nH=nH, dH=dH, d=d, dcQ=dcQ, dcK=dcK, N=N,
                       err_naive_vs_reference=float(np.max(np.abs(naive_mla - ref_mla))),
                       err_tiled_vs_reference=float(np.max(np.abs(tiled_mla - ref_mla))),
                       err_tiled_vs_naive=float(np.max(np.abs(tiled_mla - naive_mla))),
                       peak_naive=int(peak_naive_mla), peak_tiled=int(peak_tiled_mla))

err_special = max(v["err_tiled_vs_reference"] for v in special.values())
print(f"\n(c) recovered special cases through the tiled kernel (vs their OWN reference layer)")
for k, v in special.items():
    print(f"      {k:10s} tiled_vs_ref={v['err_tiled_vs_reference']:.2e}"
          f"  tiled_vs_naiveTucker={v['err_tiled_vs_naive']:.2e}"
          f"  peak_naive={v['peak_naive']}  peak_tiled={v['peak_tiled']}")

# =============================================================================
# (d) tiled + latent-RoPE (Def 3.1 / repro_claim5.py) + Tucker, combined.
#     Rotate the per-head query Qh[i] by R(m,r3) and the shared key K by
#     R(n,r3) BEFORE the tiled/naive kernel (Eq 12: Qhat_m x3 (K_n R(n)R(m)^T)
#     == dot(Qhat_m R(m), K_n R(n)); v@M == M.T@v for row vectors, see
#     repro_claim5.py's identical logit derivation). Then compare
#     tiled-rotated vs naive-rotated -- both must match to machine precision,
#     i.e. RoPE composes cleanly with the flash tiling.
# =============================================================================
def rope_matrix(pos, dim, base=10000.0):
    R = np.zeros((dim, dim))
    for j in range(dim // 2):
        theta = base ** (-2.0 * j / dim)
        a = pos * theta; c, s = np.cos(a), np.sin(a)
        R[2*j, 2*j] = c;   R[2*j, 2*j+1] = -s
        R[2*j+1, 2*j] = s; R[2*j+1, 2*j+1] = c
    return R

def rotate_rows(Xrows, dim, base=10000.0):
    N = Xrows.shape[0]
    out = np.zeros_like(Xrows)
    for m in range(N):
        out[m] = Xrows[m] @ rope_matrix(m, dim, base)
    return out

rng = np.random.default_rng(14)
nH, dH, dcQ, dcK, N = 4, 6, 12, 16, 50
d = nH * dH
X = rng.standard_normal((N, d))
WDQ = rng.standard_normal((d, dcQ)); WDKV = rng.standard_normal((d, dcK))
WUQ = rng.standard_normal((dcQ, d)); WUK = rng.standard_normal((dcK, d)); WUV = rng.standard_normal((dcK, d))
WO_heads = [rng.standard_normal((dH, d)) for _ in range(nH)]
U1 = np.eye(nH); U2 = WDQ.copy(); U3 = WDKV.copy()
C = np.zeros((nH, dcQ, dcK))
for i in range(nH):
    C[i] = WUQ[:, i*dH:(i+1)*dH] @ WUK[:, i*dH:(i+1)*dH].T
U1t = np.eye(nH); U2t = np.concatenate([O.T for O in WO_heads], axis=1); U3t = WDKV.copy()
Ct = np.zeros((nH, d, dcK))
for i in range(nH):
    Ct[i][i*dH:(i+1)*dH, :] = WUV[:, i*dH:(i+1)*dH].T

P2 = X @ U2; K_shared = X @ U3
T = mode1(C, U1)
Qh = np.einsum('nb,ibc->inc', P2, T)         # (nH, N, dcK)
K_rot = rotate_rows(K_shared, dcK)
V = X @ U3t
Mi = mode1(Ct, U1t)
scale = 1.0 / np.sqrt(dH)

naive_rope_out = np.zeros((N, d)); tiled_rope_out = np.zeros((N, d))
peak_naive_rope = 0; peak_tiled_rope = 0
for i in range(nH):
    Qi_rot = rotate_rows(Qh[i], dcK)
    Vi_pre = V @ Mi[i].T
    Vi_eff = Vi_pre @ U2t.T
    o_n, p_n = naive_1head(Qi_rot, K_rot, Vi_eff, scale)
    o_t, p_t = flash_tile_1head(Qi_rot, K_rot, Vi_eff, scale, 7, 11)
    naive_rope_out += o_n; tiled_rope_out += o_t
    peak_naive_rope = max(peak_naive_rope, p_n * nH)   # nH heads materialized as (nH,N,N) in naive path
    peak_tiled_rope = max(peak_tiled_rope, p_t)
err_rope_tiled = float(np.max(np.abs(tiled_rope_out - naive_rope_out)))
print(f"\n(d) tiled + latent-RoPE + Tucker (MLA-shaped, nH={nH} N={N} dcK={dcK}) combined")
print(f"    max abs error tiled-vs-naive (both RoPE-rotated) = {err_rope_tiled:.3e}")
print(f"    (repro_claim5.py already verifies latent RoPE itself: A=1.61e-15 B=2.27e-13 C=2.56e-13)")

# =============================================================================
# (e) memory profile: peak intermediate ARRAY SIZE (elements) actually
#     allocated by naive (N,N logits) vs tiled (block_q,block_k tile) as N
#     grows, single representative head, fixed 64x64 tile.
# =============================================================================
mem_profile = []
rngm = np.random.default_rng(999)
for N in [128, 512, 2048, 4096]:
    Qm = rngm.standard_normal((N, 8)); Km = rngm.standard_normal((N, 8)); Vm = rngm.standard_normal((N, 8))
    scale = 1.0 / np.sqrt(8)
    _, pk_naive = naive_1head(Qm, Km, Vm, scale)
    _, pk_tiled = flash_tile_1head(Qm, Km, Vm, scale, 64, 64)
    mem_profile.append(dict(N=N, peak_naive_elems=int(pk_naive), peak_tiled_elems=int(pk_tiled),
                             ratio=round(pk_naive / pk_tiled, 2)))
print(f"\n(e) memory profile: peak intermediate array size (elements), naive O(N^2) vs tiled O(block^2)")
for r in mem_profile:
    print(f"      N={r['N']:5d}  peak_naive={r['peak_naive_elems']:>9d}"
          f"  peak_tiled={r['peak_tiled_elems']:>6d}  ratio={r['ratio']:>8.1f}x")

# =============================================================================
# verdict
# =============================================================================
tol = 1e-9
ok_sanity  = err_sanity < tol
ok_generic = err_generic < tol
ok_special = err_special < tol
ok_rope    = err_rope_tiled < tol
ok_mem     = all(r["peak_tiled_elems"] < r["peak_naive_elems"] for r in mem_profile) and \
             (mem_profile[-1]["peak_tiled_elems"] == mem_profile[0]["peak_tiled_elems"])
verified = ok_sanity and ok_generic and ok_special and ok_rope and ok_mem

print("\n" + "-" * 78)
print("MEASURED vs TARGET (tol < 1e-9 for all exactness checks)")
print(f"  (a) flash-tile kernel sanity (1 head, 5 tile sizes)     {err_sanity:.2e}  {'PASS' if ok_sanity else 'FAIL'}")
print(f"  (b) generic Tucker tiled vs naive ({len(generic_cases)} combos)     {err_generic:.2e}  {'PASS' if ok_generic else 'FAIL'}")
print(f"  (c) MHA/GQA/MLA tiled vs reference layer                {err_special:.2e}  {'PASS' if ok_special else 'FAIL'}")
print(f"  (d) tiled+RoPE+Tucker vs naive+RoPE+Tucker               {err_rope_tiled:.2e}  {'PASS' if ok_rope else 'FAIL'}")
print(f"  (e) tiled peak intermediate size constant in N, naive grows O(N^2)  {'PASS' if ok_mem else 'FAIL'}")
print(f"\nVERDICT: Tucker Attention is exact under the FlashAttention tiled/online-softmax")
print(f"recurrence (shared-KV, latent-rank chunking, per Sec 3.3) -> {'VERIFIED' if verified else 'NOT VERIFIED'}")
print("=" * 78)

res = dict(
    claim="Tucker Attention is fully compatible with flash-attention (Sec. 3.3) -- "
          "algorithmic exactness under the FlashAttention-1 tiled online-softmax recurrence",
    tol=tol,
    a_flash_kernel_sanity=dict(err=err_sanity, tile_sizes=[[1, 1], [5, 7], [16, 16], [37, 41], [9, 3]]),
    b_generic_tucker_cases=generic_cases,
    b_generic_max_err=err_generic,
    c_special_cases=special,
    c_special_max_err=err_special,
    d_rope_plus_tiled=dict(config=dict(nH=nH, dH=dH, dcQ=dcQ, dcK=dcK, N=N),
                            err_tiled_vs_naive=err_rope_tiled,
                            peak_naive_elems=int(peak_naive_rope), peak_tiled_elems=int(peak_tiled_rope)),
    e_memory_profile=mem_profile,
    checks=dict(sanity=bool(ok_sanity), generic=bool(ok_generic), special=bool(ok_special),
                rope=bool(ok_rope), memory=bool(ok_mem)),
    verified=bool(verified),
    runtime_s=round(time.time() - t0, 3),
    numpy=np.__version__,
)
res["script_sha256"] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
json.dump(res, open("results_flash.json", "w"), indent=2)
print("[wrote results_flash.json]  runtime=%.3fs" % (time.time() - t0))
