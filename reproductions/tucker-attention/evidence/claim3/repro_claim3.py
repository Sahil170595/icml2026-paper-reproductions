"""
CLAIM 3 - MLA (Multi-head Latent Attention) is an EXACT special case of Tucker
Attention.  Pre-softmax tensor has maximal Tucker rank (nH, d_c^Q, d_c^K); the
post-softmax tensor has rank (nH, d_model, d_c^K).
Paper: arXiv 2603.30033 / ErcPPRZaiq, Sec. 2.2.2, Theorem B.4.

CPU float64.  Reference MLA uses low-rank query/key/value: W_i^Q = W^DQ W_i^UQ,
W_i^K = W^DKV W_i^UK, W_i^V = W^DKV W_i^UV (shared KV down-proj) with full-rank
per-head output W_i^O.  We build the Tucker core+factors (U2=W^DQ, U3=W^DKV),
verify the factored Tucker forward reproduces MLA to machine precision, and
measure the attention-tensor ranks equal (nH, d_c^Q, d_c^K)/(nH, d, d_c^K).
"""
import json, time, hashlib, numpy as np
from tucker_common import (tucker_reconstruct, softmax_rows, matn_rank,
                           tucker_attention_forward)

t0 = time.time()
rng = np.random.default_rng(2)

def mla_reference(X, WDQ, WUQ, WDKV, WUK, WUV, WO_heads, nH, dH, dcQ, dcK):
    """Shared-KV MLA. Returns output + pre/post-softmax attention tensors."""
    N, d = X.shape
    out = np.zeros((N, d)); W = np.zeros((nH, d, d)); Wt = np.zeros((nH, d, d))
    LQ = X @ WDQ                      # (N, dcQ) latent query
    LKV = X @ WDKV                    # (N, dcK) latent key/value
    for i in range(nH):
        WiUQ = WUQ[:, i*dH:(i+1)*dH]  # (dcQ, dH)
        WiUK = WUK[:, i*dH:(i+1)*dH]  # (dcK, dH)
        WiUV = WUV[:, i*dH:(i+1)*dH]  # (dcK, dH)
        Oi   = WO_heads[i]            # (dH, d)
        Qi = LQ  @ WiUQ               # (N, dH)
        Ki = LKV @ WiUK              # (N, dH)
        Vi = LKV @ WiUV              # (N, dH)
        A  = softmax_rows(Qi @ Ki.T / np.sqrt(dH))
        out += A @ Vi @ Oi
        WiQ = WDQ  @ WiUQ            # (d, dH)
        WiK = WDKV @ WiUK           # (d, dH)
        WiV = WDKV @ WiUV           # (d, dH)
        W[i]  = WiQ @ WiK.T
        Wt[i] = (WiV @ Oi).T
    return out, W, Wt

def run(nH, dH, dcQ, dcK, N, tag):
    d = nH * dH
    X    = rng.standard_normal((N, d))
    WDQ  = rng.standard_normal((d, dcQ))
    WDKV = rng.standard_normal((d, dcK))
    WUQ  = rng.standard_normal((dcQ, d))     # stacked per-head up-proj (query)
    WUK  = rng.standard_normal((dcK, d))     # key
    WUV  = rng.standard_normal((dcK, d))     # value
    WO_heads = [rng.standard_normal((dH, d)) for _ in range(nH)]

    ref, W, Wt = mla_reference(X, WDQ, WUQ, WDKV, WUK, WUV, WO_heads, nH, dH, dcQ, dcK)

    # -- pre-softmax Tucker: U2 = W^DQ (d x dcQ), U3 = W^DKV (d x dcK),
    #    core C_i = W_i^UQ (W_i^UK)^T  in R^{dcQ x dcK}. --
    U1 = np.eye(nH); U2 = WDQ.copy(); U3 = WDKV.copy()
    C = np.zeros((nH, dcQ, dcK))
    for i in range(nH):
        C[i] = WUQ[:, i*dH:(i+1)*dH] @ WUK[:, i*dH:(i+1)*dH].T
    W_rec = tucker_reconstruct(C, U1, U2, U3)
    err_pre = float(np.max(np.abs(W_rec - W)))

    # -- post-softmax Tucker: Wt_i = (W_i^V W_i^O)^T = W_i^{O,T} W_i^{UV,T} W^{DKV,T}
    #    output mode full (U2t = [W_i^{O,T}] d x d), value mode U3t = W^DKV (d x dcK),
    #    core Ct_i in R^{d x dcK} routes head i's output block. --
    U1t = np.eye(nH); U2t = np.concatenate([O.T for O in WO_heads], axis=1); U3t = WDKV.copy()
    Ct = np.zeros((nH, d, dcK))
    for i in range(nH):
        Ct[i] = np.zeros((d, dcK))
        Ct[i][i*dH:(i+1)*dH, :] = WUV[:, i*dH:(i+1)*dH].T   # (dH, dcK) block
    Wt_rec = tucker_reconstruct(Ct, U1t, U2t, U3t)
    err_post = float(np.max(np.abs(Wt_rec - Wt)))

    out_t = tucker_attention_forward(C, U1, U2, U3, Ct, U1t, U2t, U3t, X, dH)
    err_out = float(np.max(np.abs(out_t - ref)))

    ranks_W  = (matn_rank(W, 0),  matn_rank(W, 1),  matn_rank(W, 2))
    ranks_Wt = (matn_rank(Wt, 0), matn_rank(Wt, 1), matn_rank(Wt, 2))
    # Thm B.4 gives *maximal* ranks; the realized post-softmax output-mode rank is
    # min(d_model, nH*dcK) (equals the paper's maximal d_model when nH*dcK >= d_model).
    out_mode = min(d, nH * dcK)
    tW  = (nH, dcQ, dcK); tWt = (nH, out_mode, dcK); maximal_Wt = (nH, d, dcK)
    print(f"[{tag}] nH={nH} dH={dH} d={d} dcQ={dcQ} dcK={dcK} N={N}")
    print(f"    Tucker pre/post reconstruction err = {err_pre:.3e} / {err_post:.3e}")
    print(f"    factored Tucker forward vs MLA out  = {err_out:.3e}")
    print(f"    measured ranks W  (head,query,key)  = {ranks_W}  target {tW}")
    print(f"    measured ranks Wt (head,out,value)  = {ranks_Wt}  target {tWt}")
    ok = (err_pre<1e-9 and err_post<1e-9 and err_out<1e-9 and ranks_W==tW and ranks_Wt==tWt)
    print(f"    -> {'EXACT special case' if ok else 'MISMATCH'}\n")
    return dict(tag=tag, nH=nH, dH=dH, d_model=d, dcQ=dcQ, dcK=dcK, N=N,
                err_pre=err_pre, err_post=err_post, err_out=err_out,
                measured_ranks_W=list(ranks_W), measured_ranks_Wt=list(ranks_Wt),
                target_ranks_W=list(tW), target_ranks_Wt=list(tWt),
                maximal_ranks_Wt=list(maximal_Wt), out_mode_eq_dmodel=bool(out_mode==d),
                ok=bool(ok))

print("="*74)
print("CLAIM 3  MLA is an exact special case of Tucker Attention (Thm B.4)")
print("pre rank (n_H, d_c^Q, d_c^K); post rank (n_H, d, d_c^K)  numpy float64 CPU")
print("="*74)
cases = [run(6, 6, 8,  6,  9, "MLA dcQ=8 dcK=6"),
         run(6, 6, 12, 8,  9, "MLA dcQ=12 dcK=8"),
         run(4, 8, 16, 16, 9, "MLA dcQ=dcK=16")]
verified = all(c["ok"] for c in cases)
print("-"*74)
print("MEASURED vs TARGET")
for c in cases:
    print(f"  {c['tag']:18s} out_err={c['err_out']:.2e}  ranks_W={tuple(c['measured_ranks_W'])}"
          f"=target{tuple(c['target_ranks_W'])}  {'PASS' if c['ok'] else 'FAIL'}")
print(f"\nVERDICT: MLA is an EXACT special case of Tucker Attention -> "
      f"{'VERIFIED' if verified else 'NOT VERIFIED'}")
print("="*74)

res = dict(claim="MLA is an exact special case of Tucker Attention (Thm B.4)",
           cases=cases, verified=bool(verified), tol=1e-9,
           runtime_s=round(time.time()-t0,3), numpy=np.__version__)
res["script_sha256"] = hashlib.sha256(open(__file__,'rb').read()).hexdigest()
json.dump(res, open("results.json","w"), indent=2)
print("[wrote results.json]  runtime=%.3fs" % (time.time()-t0))
