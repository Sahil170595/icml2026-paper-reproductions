"""
CLAIM 1 - MHA is an EXACT special case of Tucker Attention.
Paper: Tucker Attention (arXiv 2603.30033 / OpenReview ErcPPRZaiq), Sec. 2, Thm B.2.

Two things are verified to machine precision on random inputs (CPU, float64):
  (a) The tensor reformulation of MHA (Eqs. 2-6) reproduces the standard MHA layer.
  (b) The Tucker factorization  W = C x2 (W^Q Pi) x3 (W^K Pi)  (Thm B.2, delta core
      of Lemma B.1) reconstructs the pre-softmax attention tensor, and feeding those
      Tucker factors (analogously for the post-softmax tensor) into the *factored*
      Tucker Attention forward recovers the MHA output.  Tucker ranks = (nH, d, d).
"""
import json, time, hashlib, numpy as np
from tucker_common import (mode1, tucker_reconstruct, softmax_rows, matn_rank,
                           attention_from_tensors, tucker_attention_forward,
                           mha_reference, build_mha_tensors)

np.set_printoptions(precision=3)
t0 = time.time()
rng = np.random.default_rng(0)

# -- config (small; exactness is dimension-independent) --
N, nH, dH = 7, 4, 5
d = nH * dH                        # d_model = 20
X  = rng.standard_normal((N, d))
WQ = rng.standard_normal((d, d)); WK = rng.standard_normal((d, d))
WV = rng.standard_normal((d, d)); WO = rng.standard_normal((d, d))

print("="*74)
print("CLAIM 1  MHA is an exact special case of Tucker Attention  (Thm B.2)")
print("arXiv 2603.30033 / ErcPPRZaiq   independent numpy float64, CPU single-thread")
print("="*74)
print(f"N={N}  n_H={nH}  d_H={dH}  d_model={d}")

# -- reference MHA --
ref = mha_reference(X, WQ, WK, WV, WO, nH, dH)

# -- (a) tensor reformulation (Eqs 2-6) --
W, Wt = build_mha_tensors(WQ, WK, WV, WO, nH, dH)     # (nH,d,d),(nH,d,d)
out_tensor, _ = attention_from_tensors(W, Wt, X, dH)
err_reform = float(np.max(np.abs(out_tensor - ref)))

# -- (b) Thm B.2 delta-core Tucker factorization of the pre-softmax tensor --
# C[i,n,m] = sum_{l} delta_{n,i*dH+l} delta_{m,i*dH+l};  U2=W^Q, U3=W^K (Pi = I here).
C = np.zeros((nH, d, d))
for i in range(nH):
    for l in range(dH):
        C[i, i*dH + l, i*dH + l] = 1.0
U1 = np.eye(nH); U2 = WQ.copy(); U3 = WK.copy()
W_rec = tucker_reconstruct(C, U1, U2, U3)
err_pre = float(np.max(np.abs(W_rec - W)))

# post-softmax tensor: Wtilde_i^T = W_i^{O,T} W_i^{V,T}  -> same delta core,
# U2t = (W^O)^T (column-blocked W_i^{O,T}), U3t = W^V.
Ct = C.copy(); U1t = np.eye(nH); U2t = WO.T.copy(); U3t = WV.copy()
Wt_rec = tucker_reconstruct(Ct, U1t, U2t, U3t)
err_post = float(np.max(np.abs(Wt_rec - Wt)))

# -- (b) factored Tucker Attention forward with the constructed factors --
out_tucker = tucker_attention_forward(C, U1, U2, U3, Ct, U1t, U2t, U3t, X, dH)
err_tucker = float(np.max(np.abs(out_tucker - ref)))

# -- measured Tucker ranks of the attention tensors --
ranks_W  = (matn_rank(W, 0), matn_rank(W, 1), matn_rank(W, 2))
ranks_Wt = (matn_rank(Wt, 0), matn_rank(Wt, 1), matn_rank(Wt, 2))
target_rank = (nH, d, d)

print(f"\n(a) tensor reformulation (Eq 6) vs standard MHA : max|.| = {err_reform:.3e}")
print(f"(b) Thm B.2 pre-softmax  reconstruction  max|W_rec - W|   = {err_pre:.3e}")
print(f"(b) Thm B.2 post-softmax reconstruction  max|Wt_rec - Wt| = {err_post:.3e}")
print(f"(b) factored Tucker forward vs MHA output max|.|          = {err_tucker:.3e}")
print(f"measured Tucker ranks  W  (head,query,key)   = {ranks_W}   target {target_rank}")
print(f"measured Tucker ranks  Wt (head,out,value)   = {ranks_Wt}   target {target_rank}")

tol = 1e-9
verified = (err_reform < tol and err_pre < tol and err_post < tol and
            err_tucker < tol and ranks_W == target_rank and ranks_Wt == target_rank)
print("\n" + "-"*74)
print("MEASURED vs TARGET")
print(f"  reformulation error {err_reform:.2e}  (<1e-9)   {'PASS' if err_reform<tol else 'FAIL'}")
print(f"  Tucker->MHA output  {err_tucker:.2e}  (<1e-9)   {'PASS' if err_tucker<tol else 'FAIL'}")
print(f"  Tucker ranks (nH,d,d)=({nH},{d},{d})           {'PASS' if ranks_W==target_rank else 'FAIL'}")
print(f"\nVERDICT: MHA is an EXACT special case of Tucker Attention -> "
      f"{'VERIFIED' if verified else 'NOT VERIFIED'}")
print("="*74)

res = dict(claim="MHA is an exact special case of Tucker Attention (Thm B.2)",
           config=dict(N=N, nH=nH, dH=dH, d_model=d, dtype="float64", seed=0),
           err_reformulation_eq6=err_reform, err_pre_reconstruction=err_pre,
           err_post_reconstruction=err_post, err_tucker_forward_vs_mha=err_tucker,
           measured_ranks_W=list(ranks_W), measured_ranks_Wt=list(ranks_Wt),
           target_ranks=list(target_rank), tol=tol, verified=bool(verified),
           runtime_s=round(time.time()-t0, 3), numpy=np.__version__)
res["script_sha256"] = hashlib.sha256(open(__file__,'rb').read()).hexdigest()
json.dump(res, open("results.json","w"), indent=2)
print("[wrote results.json]  runtime=%.3fs" % (time.time()-t0))
