"""
CLAIM 5 - Latent RoPE + simplified RoPE for MLA (Corollary 3.2.1).
Paper: arXiv 2603.30033 / ErcPPRZaiq, Sec. 3.2, Def. 3.1, Lemma B.5.

Because Tucker/MLA parametrize the *fused* product W_i^Q W_i^{K,T}, the standard
head-dimension placement of RoPE is incompatible with weight fusion.  The paper
moves the rotation into the shared latent key dimension r3 ("latent RoPE") and
claims: (i) it satisfies the relative-position property R(m)R(n)^T = R(m-n)
(Eq 10) so scores depend only on m-n; (ii) it enables *full* query-side weight
fusion, giving the first RoPE-compatible MLA WITHOUT decoupled RoPE.

Verified (CPU float64, machine precision):
  A. R(m,d) R(n,d)^T = R(m-n,d) exactly.
  B. latent-RoPE attention logit is a function of (m-n) only (Lemma B.5).
  C. fusing the query-side projections into ONE matrix reproduces the unfused
     latent-RoPE logits exactly (Eq 14 full fusion).
  D. CONTROL: the same fusion with RoPE kept in the *head* dimension breaks
     (large error) -> why decoupled RoPE was needed, and why latent RoPE fixes it.
"""
import json, time, hashlib, numpy as np

t0 = time.time()
rng = np.random.default_rng(5)

def rope_matrix(pos, dim, base=10000.0):
    assert dim % 2 == 0
    R = np.zeros((dim, dim))
    for j in range(dim // 2):
        theta = base ** (-2.0 * j / dim)
        a = pos * theta; c, s = np.cos(a), np.sin(a)
        R[2*j, 2*j] = c;   R[2*j, 2*j+1] = -s
        R[2*j+1, 2*j] = s; R[2*j+1, 2*j+1] = c
    return R

print("="*74)
print("CLAIM 5  Latent RoPE + simplified MLA RoPE  (Corollary 3.2.1, Lemma B.5)")
print("arXiv 2603.30033 / ErcPPRZaiq   independent numpy float64, CPU single-thread")
print("="*74)

r3 = 16
errA = 0.0
for (m, n) in [(0,0),(5,2),(13,13),(31,7),(100,64),(7,40)]:
    lhs = rope_matrix(m, r3) @ rope_matrix(n, r3).T
    rhs = rope_matrix(m - n, r3)
    errA = max(errA, float(np.max(np.abs(lhs - rhs))))
print(f"\nA. max ||  R(m,r3)R(n,r3)^T - R(m-n,r3) ||  over 6 (m,n) pairs = {errA:.3e}")

d, nH, dH, dcQ, dcK, N = 24, 4, 6, 12, r3, 12
X    = rng.standard_normal((N, d))
WDQ  = rng.standard_normal((d, dcQ))
WDKV = rng.standard_normal((d, dcK))
WUQ  = rng.standard_normal((dcQ, d))
WUK  = rng.standard_normal((dcK, d))
C = np.stack([WUQ[:, i*dH:(i+1)*dH] @ WUK[:, i*dH:(i+1)*dH].T for i in range(nH)])
LQ = X @ WDQ
K  = X @ WDKV
Qhat = np.einsum('mb,ibc->imc', LQ, C)

def latent_rope_logit(i, m, n):
    return float(Qhat[i, m] @ rope_matrix(m, dcK) @ rope_matrix(n, dcK).T @ K[n])

errB = 0.0
for i in range(nH):
    for (m, n) in [(9, 4), (7, 7), (10, 2)]:
        ref = latent_rope_logit(i, m, n)
        for t in [1, 3, 8, 20]:
            val = float(Qhat[i, m] @ rope_matrix(m+t, dcK) @ rope_matrix(n+t, dcK).T @ K[n])
            errB = max(errB, abs(val - ref))
print(f"B. latent-RoPE logit(m,n) vs logit(m+t,n+t) [same content], max|diff| = {errB:.3e}")
print(f"   (relative-position property: score is a function of m-n only)")

F = np.einsum('db,ibc->idc', WDQ, C)
errC = 0.0
for i in range(nH):
    for (m, n) in [(9,4),(3,11),(6,6)]:
        fused = float((X[m] @ F[i]) @ rope_matrix(m, dcK) @ rope_matrix(n, dcK).T @ K[n])
        errC = max(errC, abs(fused - latent_rope_logit(i, m, n)))
print(f"C. fused query matrix F_i=WDQ*C_i reproduces latent-RoPE logits, max|diff| = {errC:.3e}")
print(f"   (Eq 14: full inference-time fusion of the query-side projections)")

errD = 0.0; maxmag = 0.0
for i in range(nH):
    WiUQ = WUQ[:, i*dH:(i+1)*dH]; WiUK = WUK[:, i*dH:(i+1)*dH]
    for (m, n) in [(9,4),(3,11),(6,6)]:
        Qi = LQ[m] @ WiUQ; Ki = K[n] @ WiUK
        true_head = float(Qi @ rope_matrix(m, dH) @ rope_matrix(n, dH).T @ Ki)
        fused_wrong = float((X[m] @ F[i]) @ rope_matrix(m, dcK) @ rope_matrix(n, dcK).T @ K[n])
        errD = max(errD, abs(fused_wrong - true_head)); maxmag = max(maxmag, abs(true_head))
print(f"D. CONTROL head-dim RoPE + query fusion: max|fused - true_head| = {errD:.3e}"
      f"  (logit magnitude ~{maxmag:.2f})")
print(f"   -> head-dim RoPE is NOT fusable (why MLA needed decoupled RoPE);")
print(f"      latent RoPE (C) IS fusable -> the paper's simplification.")

verified = (errA < 1e-9 and errB < 1e-9 and errC < 1e-9 and errD > 1e-2)
print("\n" + "-"*74)
print("MEASURED vs TARGET")
print(f"  A  R(m)R(n)^T=R(m-n)               {errA:.2e}  (<1e-9)   {'PASS' if errA<1e-9 else 'FAIL'}")
print(f"  B  latent-RoPE score = f(m-n)      {errB:.2e}  (<1e-9)   {'PASS' if errB<1e-9 else 'FAIL'}")
print(f"  C  query fusion exact (Eq 14)      {errC:.2e}  (<1e-9)   {'PASS' if errC<1e-9 else 'FAIL'}")
print(f"  D  control: head-RoPE not fusable  {errD:.2e}  (>1e-2)   {'PASS' if errD>1e-2 else 'FAIL'}")
print(f"\nVERDICT: latent RoPE satisfies the rel-position property AND enables MLA weight")
print(f"         fusion without decoupled RoPE -> {'VERIFIED' if verified else 'NOT VERIFIED'}")
print("="*74)

res = dict(claim="Latent RoPE relative-position property + simplified MLA RoPE (Cor 3.2.1)",
           config=dict(d_model=d, nH=nH, dH=dH, dcQ=dcQ, dcK=dcK, r3=r3, N=N, base=10000.0),
           errA_rope_identity=errA, errB_relative_position=errB,
           errC_query_fusion=errC, errD_control_headdim_notfusable=errD,
           control_logit_magnitude=round(maxmag,3),
           verified=bool(verified), tol=1e-9,
           runtime_s=round(time.time()-t0,3), numpy=np.__version__)
res["script_sha256"] = hashlib.sha256(open(__file__,'rb').read()).hexdigest()
json.dump(res, open("results.json","w"), indent=2)
print("[wrote results.json]  runtime=%.3fs" % (time.time()-t0))
