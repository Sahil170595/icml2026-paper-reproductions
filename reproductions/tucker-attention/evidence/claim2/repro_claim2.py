"""
CLAIM 2 - GQA (Grouped-Query Attention, incl. MQA as n_KV=1) is an EXACT special
case of Tucker Attention with maximal Tucker rank (nH, d_model, n_KV*d_H).
Paper: arXiv 2603.30033 / ErcPPRZaiq, Sec. 2.2.1, Theorem B.3.

CPU float64.  We build a reference GQA layer (n_KV distinct KV heads broadcast to
nH query heads), construct Tucker core+factors whose key/value mode has exactly
n_KV*d_H columns, verify the factored Tucker forward reproduces GQA to machine
precision, and measure that the numerical rank of the key/value mode is n_KV*d_H.
"""
import json, time, hashlib, numpy as np
from tucker_common import (tucker_reconstruct, softmax_rows, matn_rank,
                           attention_from_tensors, tucker_attention_forward)

t0 = time.time()
rng = np.random.default_rng(1)

def gqa_reference(X, WQ_heads, WK_kv, WV_kv, WO_heads, group_of, dH):
    """GQA: query head i uses KV head group_of[i]. Returns output + pre/post tensors."""
    N, d = X.shape; nH = len(WQ_heads)
    out = np.zeros((N, d)); W = np.zeros((nH, d, d)); Wt = np.zeros((nH, d, d))
    for i in range(nH):
        g = group_of[i]
        Qi = X @ WQ_heads[i]; Ki = X @ WK_kv[g]; Vi = X @ WV_kv[g]; Oi = WO_heads[i]
        A = softmax_rows(Qi @ Ki.T / np.sqrt(dH))
        out += A @ Vi @ Oi
        W[i]  = WQ_heads[i] @ WK_kv[g].T
        Wt[i] = (WV_kv[g] @ Oi).T
    return out, W, Wt

def run(nH, nKV, dH, N, tag):
    d = nH * dH
    X = rng.standard_normal((N, d))
    WQ_heads = [rng.standard_normal((d, dH)) for _ in range(nH)]
    WO_heads = [rng.standard_normal((dH, d)) for _ in range(nH)]
    WK_kv    = [rng.standard_normal((d, dH)) for _ in range(nKV)]
    WV_kv    = [rng.standard_normal((d, dH)) for _ in range(nKV)]
    per = nH // nKV
    group_of = [i // per for i in range(nH)]              # contiguous grouping

    ref, W, Wt = gqa_reference(X, WQ_heads, WK_kv, WV_kv, WO_heads, group_of, dH)

    # -- Tucker factors: query mode U2 = [W_1^Q..W_nH^Q] (d x d); key mode
    #    U3 = Kstack = [W_1^{K,GQA}..W_nKV^{K,GQA}] (d x nKV*dH). Core routes head i
    #    to its group g(i).  U2t = [W_i^{O,T}] (d x d); U3t = Vstack (d x nKV*dH).
    U1 = np.eye(nH)
    U2 = np.concatenate(WQ_heads, axis=1)                 # (d, d)
    U3 = np.concatenate(WK_kv,   axis=1)                  # (d, nKV*dH)
    r3 = nKV * dH
    C = np.zeros((nH, d, r3))
    for i in range(nH):
        g = group_of[i]
        for l in range(dH):
            C[i, i*dH + l, g*dH + l] = 1.0
    W_rec = tucker_reconstruct(C, U1, U2, U3)
    err_pre = float(np.max(np.abs(W_rec - W)))

    U1t = np.eye(nH)
    U2t = np.concatenate([O.T for O in WO_heads], axis=1) # (d, d)
    U3t = np.concatenate(WV_kv, axis=1)                   # (d, nKV*dH)
    Ct = np.zeros((nH, d, r3))
    for i in range(nH):
        g = group_of[i]
        for l in range(dH):
            Ct[i, i*dH + l, g*dH + l] = 1.0
    Wt_rec = tucker_reconstruct(Ct, U1t, U2t, U3t)
    err_post = float(np.max(np.abs(Wt_rec - Wt)))

    out_t = tucker_attention_forward(C, U1, U2, U3, Ct, U1t, U2t, U3t, X, dH)
    err_out = float(np.max(np.abs(out_t - ref)))

    ranks_W  = (matn_rank(W, 0), matn_rank(W, 1), matn_rank(W, 2))
    target   = (nH, d, r3)
    print(f"[{tag}] nH={nH} nKV={nKV} dH={dH} d={d} N={N}")
    print(f"    Tucker pre/post reconstruction err = {err_pre:.3e} / {err_post:.3e}")
    print(f"    factored Tucker forward vs GQA out  = {err_out:.3e}")
    print(f"    measured ranks W (head,query,key)   = {ranks_W}  target {target}")
    ok = err_pre<1e-9 and err_post<1e-9 and err_out<1e-9 and ranks_W==target
    print(f"    -> {'EXACT special case' if ok else 'MISMATCH'}\n")
    return dict(tag=tag, nH=nH, nKV=nKV, dH=dH, d_model=d, N=N,
                err_pre=err_pre, err_post=err_post, err_out=err_out,
                measured_ranks_W=list(ranks_W), target_ranks=list(target), ok=bool(ok))

print("="*74)
print("CLAIM 2  GQA / MQA are exact special cases of Tucker Attention (Thm B.3)")
print("rank (n_H, d_model, n_KV*d_H)   independent numpy float64, CPU single-thread")
print("="*74)
cases = [run(8, 2, 4, 9, "GQA n_KV=2"),
         run(8, 4, 4, 9, "GQA n_KV=4"),
         run(8, 1, 4, 9, "MQA (n_KV=1)")]
verified = all(c["ok"] for c in cases)
print("-"*74)
print("MEASURED vs TARGET (max abs output error GQA vs Tucker, and key-mode rank)")
for c in cases:
    print(f"  {c['tag']:14s}  out_err={c['err_out']:.2e}  key-rank={c['measured_ranks_W'][2]}"
          f" (target n_KV*d_H={c['target_ranks'][2]})  {'PASS' if c['ok'] else 'FAIL'}")
print(f"\nVERDICT: GQA/MQA are EXACT special cases (rank n_KV*d_H on the key/value mode)"
      f" -> {'VERIFIED' if verified else 'NOT VERIFIED'}")
print("="*74)

res = dict(claim="GQA/MQA are exact special cases of Tucker Attention, rank (nH,d,nKV*dH) (Thm B.3)",
           cases=cases, verified=bool(verified), tol=1e-9,
           runtime_s=round(time.time()-t0,3), numpy=np.__version__)
res["script_sha256"] = hashlib.sha256(open(__file__,'rb').read()).hexdigest()
json.dump(res, open("results.json","w"), indent=2)
print("[wrote results.json]  runtime=%.3fs" % (time.time()-t0))
