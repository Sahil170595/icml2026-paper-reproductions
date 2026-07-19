"""
CLAIM 6 - ViT experiment: Tucker Attention matches GQA/MLA validation performance
with almost an order of magnitude fewer parameters (Paper Sec 4.1, Figure 3).

The *validation-accuracy* result requires training a real ViT on real images and
is NOT reproducible on CPU without fabrication.  This script does the part that IS
decisively checkable on CPU, and defers the full-scale accuracy result to the
accompanying GPU kit (gpu_job/).  On CPU we verify:
  (S1) the PyTorch Tucker Attention *layer* computes the paper's factored
       attention (Eqs 4-6) correctly -> matches an independent numpy reference to
       float64 machine precision;
  (S2) the layer is correctly differentiable -> torch.autograd.gradcheck passes
       (input Jacobian) AND analytic vs central-difference parameter gradients agree;
  (S3) the layer's trainable-parameter count matches the analytic Tucker formula
       and is ~an order of magnitude below GQA/MLA at the ViT-B.16 config (the
       parameter axis of Figure 3).
NO validation numbers are reported here; see gpu_job/RUN_GPU.md.
"""
import json, time, hashlib, numpy as np, torch
from tucker_common import attention_from_tensors, tucker_reconstruct

torch.manual_seed(0); np.random.seed(0)
torch.set_default_dtype(torch.float64)
t0 = time.time()


class TuckerAttention(torch.nn.Module):
    """Factored Tucker Attention layer (Def 2.1 + Sec 3 implementation)."""
    def __init__(self, d, nH, dH, r1, r2, r3):
        super().__init__()
        self.d, self.nH, self.dH = d, nH, dH
        g = lambda *s: torch.nn.Parameter(torch.randn(*s) * 0.05)
        self.C  = g(r1, r2, r3); self.U1  = g(nH, r1); self.U2  = g(d, r2); self.U3  = g(d, r3)
        self.Ct = g(r1, r2, r3); self.U1t = g(nH, r1); self.U2t = g(d, r2); self.U3t = g(d, r3)

    def forward(self, X):                          # X: (N, d) -> (N, d)
        dH = self.dH
        P2 = X @ self.U2;  P3 = X @ self.U3
        T  = torch.einsum('abc,ia->ibc', self.C, self.U1)
        logits = torch.einsum('nb,ibc,mc->inm', P2, T, P3) / (dH ** 0.5)
        H1 = torch.softmax(logits, dim=-1)
        V  = X @ self.U3t
        St = torch.einsum('abc,ia->ibc', self.Ct, self.U1t)
        S  = torch.einsum('ibc,mc->ibm', St, V)
        H2 = torch.einsum('ibm,jb->ijm', S, self.U2t)
        return torch.einsum('ijl,ikl->jk', H1, H2)


print("="*74)
print("CLAIM 6  ViT experiment  (CPU layer sanity; full accuracy -> GPU kit)")
print("arXiv 2603.30033 / ErcPPRZaiq   PyTorch %s float64, CPU single-thread" % torch.__version__)
print("="*74)

d, nH, dH, r1, r2, r3, N = 32, 4, 8, 4, 12, 12, 6
layer = TuckerAttention(d, nH, dH, r1, r2, r3)
X = torch.randn(N, d, requires_grad=True)

# (S1) forward vs independent numpy Eq-6 reference
with torch.no_grad():
    out_torch = layer(X).numpy()
    W  = tucker_reconstruct(layer.C.numpy(),  layer.U1.numpy(),  layer.U2.numpy(),  layer.U3.numpy())
    Wt = tucker_reconstruct(layer.Ct.numpy(), layer.U1t.numpy(), layer.U2t.numpy(), layer.U3t.numpy())
    out_ref, _ = attention_from_tensors(W, Wt, X.detach().numpy(), dH)
err_fwd = float(np.max(np.abs(out_torch - out_ref)))
print(f"\n(S1) layer forward vs numpy Eq-6 reference : max|.| = {err_fwd:.3e}")

# (S2) gradcheck (input Jacobian) + analytic-vs-FD parameter gradients
gc = bool(torch.autograd.gradcheck(lambda x: layer(x), (X,), eps=1e-6, atol=1e-6, rtol=1e-4))
print(f"(S2) torch.autograd.gradcheck (input Jacobian) : {gc}")

loss = (layer(X) ** 2).sum(); loss.backward()
gU2 = layer.U2.grad.detach().clone()
h = 1e-6; rel_errs = []
with torch.no_grad():
    for (i, j) in [(0, 0), (5, 3), (17, 7), (30, 11)]:
        base = layer.U2[i, j].item()
        layer.U2[i, j] = base + h; lp = (layer(X) ** 2).sum().item()
        layer.U2[i, j] = base - h; lm = (layer(X) ** 2).sum().item()
        layer.U2[i, j] = base
        fd = (lp - lm) / (2 * h)
        rel_errs.append(abs(fd - gU2[i, j].item()) / (abs(fd) + 1e-12))
max_grad_rel = float(max(rel_errs))
print(f"(S2) analytic vs central-difference dL/dU2 (4 entries): max rel err = {max_grad_rel:.3e}")

# (S3) parameter count vs analytic formula + reduction at ViT-B.16
n_params = sum(p.numel() for p in layer.parameters())
analytic = 2 * (r1*r2*r3 + nH*r1 + d*r2 + d*r3)
print(f"(S3) layer trainable params = {n_params}  analytic 2*(r1r2r3+nH r1+d r2+d r3) = {analytic}"
      f"  match={n_params == analytic}")

D, NH, DH = 768, 12, 64                              # ViT-B.16 (Table 5)
def mha(D):            return 4*D*D
def gqa(D, DH, nKV):   return 2*D*D + 2*nKV*DH*D
def mla_sh(D, dc):     return D*D + 5*D*dc
def tuck(D, NH, a, b): return 2*(a*b*b + NH*a + 2*D*b)
vit = dict(MHA=mha(D), GQA_nKV1=gqa(D,DH,1), MLA_dc16=mla_sh(D,16),
           Tucker_8_16_16=tuck(D,NH,8,16), Tucker_8_32_32=tuck(D,NH,8,32))
red_vs_gqa = vit['GQA_nKV1'] / vit['Tucker_8_16_16']
red_vs_mla = vit['MLA_dc16'] / vit['Tucker_8_16_16']
print(f"(S3) ViT-B.16 attn params/layer: MHA={vit['MHA']}  GQA(nKV=1)={vit['GQA_nKV1']}"
      f"  MLA(dc=16)={vit['MLA_dc16']}  Tucker[8,16,16]={vit['Tucker_8_16_16']}")
print(f"     Tucker[8,16,16] reduction vs GQA(nKV=1)={red_vs_gqa:.1f}x  vs MLA(dc=16)={red_vs_mla:.1f}x")

sane = (err_fwd < 1e-9 and gc and max_grad_rel < 1e-5 and n_params == analytic and red_vs_gqa > 5)
print("\n" + "-"*74)
print("MEASURED vs TARGET  (CPU layer correctness; accuracy result is in the GPU kit)")
print(f"  S1 forward == numpy Eq-6        {err_fwd:.2e} (<1e-9)   {'PASS' if err_fwd<1e-9 else 'FAIL'}")
print(f"  S2 gradcheck + param-grad FD     {gc} / {max_grad_rel:.1e} (<1e-5)   "
      f"{'PASS' if gc and max_grad_rel<1e-5 else 'FAIL'}")
print(f"  S3 params==formula; {red_vs_gqa:.1f}x<GQA, {red_vs_mla:.1f}x<MLA at ViT-B.16   "
      f"{'PASS' if n_params==analytic and red_vs_gqa>5 else 'FAIL'}")
print(f"\nLAYER SANITY: {'PASS' if sane else 'FAIL'}  (Tucker layer is correct & differentiable,")
print(f"  parameters ~10x below GQA/MLA at the ViT config).")
print(f"NOTE: the paper's validation-accuracy match (Fig 3) is NOT claimed here; it")
print(f"  requires a real ViT trained on ImageNet -> see gpu_job/RUN_GPU.md (hf jobs run).")
print("="*74)

res = dict(claim="ViT: Tucker matches GQA/MLA val perf with ~10x fewer params (Fig 3)",
           status="CPU layer sanity PASS; full-scale accuracy deferred to GPU kit (not fabricated)",
           cpu_layer_config=dict(d=d, nH=nH, dH=dH, r1=r1, r2=r2, r3=r3, N=N, dtype="float64"),
           err_forward_vs_numpy=err_fwd, gradcheck_input_jacobian=bool(gc),
           param_grad_max_rel_err=max_grad_rel,
           layer_params=int(n_params), analytic_params=int(analytic),
           param_count_match=bool(n_params == analytic),
           vit_b16_params_per_layer=vit,
           reduction_tucker_vs_gqa_x=round(red_vs_gqa, 2),
           reduction_tucker_vs_mla_x=round(red_vs_mla, 2),
           cpu_layer_sanity_pass=bool(sane),
           accuracy_result="deferred_to_gpu_kit",
           runtime_s=round(time.time()-t0, 3), torch=torch.__version__, numpy=np.__version__)
res["script_sha256"] = hashlib.sha256(open(__file__,'rb').read()).hexdigest()
json.dump(res, open("results.json","w"), indent=2)
print("[wrote results.json]  runtime=%.3fs" % (time.time()-t0))
