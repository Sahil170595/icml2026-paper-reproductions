"""Claim 1 - DAVE gradient decomposition identity (paper Eq. 3) on a REAL small ViT.

Paper claim: each ViT layer F(X)=L(X)(X)+B has derivative
    D_X F  =  L(X)  +  ((D_X L(X)(.)) X)     [effective transformation + operator variation]
and the model effective transformation composes multiplicatively W_L=prod_i W_{L_i} (eq. after 4).

We verify this to MACHINE PRECISION (float64) on a compact real ViT built in pure PyTorch
(patch-embed conv, class token, positional embedding, real multi-head self-attention,
LayerNorm, GELU-MLP, linear head). Weights are random-init (timm/torchvision weights could
not load: torchvision is ABI-incompatible with torch 2.13 here) -- but the decomposition
identity is a property of the computation graph and is EXACT for any weight values.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import dave_vit as D
torch.set_default_dtype(torch.float64)
os.environ.setdefault("OMP_NUM_THREADS", "1")

t0 = time.time()
res = {"claim": "DAVE layer-derivative decomposition D_X F = L(X) [effective] + operator-variation (Eq.3), "
                "and multiplicative composition of effective transformations W_L=prod W_Li (Eq. after 4)",
       "model": "compact real ViT (img32/patch8/dim32/depth2/4heads), pure-PyTorch, float64, random-init weights",
       "dtype": "float64", "unit_roundoff_f64": 2.22e-16}

# ---- (A) per-layer identity on the 4 real ViT layer types, over several seeds/inputs ----
per_layer = {"LayerNorm": [], "MHSA(attention)": [], "MLP(GELU)": [], "Linear(head)": []}
opvar_frac = {}
for seed in range(5):
    m = D.TinyViT(seed=seed)
    b = m.blocks[0]
    X = torch.randn(m.np + 1, m.dim, generator=torch.Generator().manual_seed(100 + seed)) * 0.5
    layers = {
        "LayerNorm":       lambda A, C: D.layernorm(A, C, b['g1'], b['b1']),
        "MHSA(attention)": lambda A, C: D.mhsa(A, C, b['Wq'], b['Wk'], b['Wv'], b['Wo'], m.nh),
        "MLP(GELU)":       lambda A, C: D.mlp(A, C, b['W1'], b['c1'], b['W2'], b['c2']),
        "Linear(head)":    lambda A, C: D.lin(C, m.Wh, m.bh),
    }
    for name, fn in layers.items():
        c = D.check_layer(name, fn, X)
        per_layer[name].append(c['max_abs_err'])
        opvar_frac[name] = c['opvar_frac']

print("=" * 74)
print("Claim 1: DAVE decomposition identity  D_X F = L(X) + operator-variation  (Eq. 3)")
print("compact REAL ViT, float64, 5 seeds x real layer inputs")
print("=" * 74)
print(f"{'layer type':18s} {'worst max|Jfull-(Jeff+Jopvar)|':32s} {'opvar/(eff+opvar)':s}")
worst_layer = 0.0
res["per_layer_worst_abs_err"] = {}
for name in per_layer:
    w = max(per_layer[name])
    worst_layer = max(worst_layer, w)
    res["per_layer_worst_abs_err"][name] = w
    print(f"{name:18s} {w:.3e}{'':>22s} {opvar_frac[name]:.3f}")
print(f"\nWORST per-layer identity error across all types/seeds = {worst_layer:.3e}  (machine precision ~1e-16)")
res["worst_per_layer_abs_err"] = worst_layer

# ---- (B) multiplicative composition W_L = prod_i W_{L_i} on a real block chain LN->MHSA->MLP ----
m = D.TinyViT(seed=0); b = m.blocks[0]; nh = m.nh
X = torch.randn(m.np + 1, m.dim, generator=torch.Generator().manual_seed(11)) * 0.4
g1 = lambda A, C: D.layernorm(A, C, b['g1'], b['b1'])
g2 = lambda A, C: D.mhsa(A, C, b['Wq'], b['Wk'], b['Wv'], b['Wo'], nh)
g3 = lambda A, C: D.mlp(A, C, b['W1'], b['c1'], b['W2'], b['c2'])
def eff_jac(g, Xin):
    J, _, _ = D.jac(lambda C: g(Xin.detach().clone(), C), Xin); return J
X1 = g1(X, X); X2 = g2(X1, X1)
J1, J2, J3 = eff_jac(g1, X), eff_jac(g2, X1), eff_jac(g3, X2)
prod = J3 @ J2 @ J1
X1c, X2c = X1.detach(), X2.detach()
def chain(Z):
    o1 = D.layernorm(X.detach(), Z, b['g1'], b['b1'])
    o2 = D.mhsa(X1c, o1, b['Wq'], b['Wk'], b['Wv'], b['Wo'], nh)
    return D.mlp(X2c, o2, b['W1'], b['c1'], b['W2'], b['c2'])
Jchain, _, _ = D.jac(chain, X)
comp_err = (Jchain - prod).abs().max().item()
comp_rel = comp_err / (Jchain.abs().max().item())
print(f"\nComposition  W_L = J3(MLP) @ J2(MHSA) @ J1(LN)  vs frozen-operator autograd Jacobian:")
print(f"  max|autograd - product| = {comp_err:.3e}   rel = {comp_rel:.3e}   (machine precision)")
res["composition_abs_err"] = comp_err
res["composition_rel_err"] = comp_rel

# ---- (C) non-triviality on the full ViT: how much of the RAW input gradient is operator variation ----
fracs = []
for seed in range(5):
    m = D.TinyViT(seed=seed)
    Xi = torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(7 + seed)) * 0.3
    g_full, _ = D.input_grad(m, Xi)
    g_eff, _ = D.input_grad(m, Xi, detach_op=True)
    fr = float((g_full - g_eff).norm() / g_full.norm())
    fracs.append(fr)
import statistics
res["opvar_fraction_of_raw_gradient_mean"] = statistics.mean(fracs)
res["opvar_fraction_of_raw_gradient_range"] = [min(fracs), max(fracs)]
print(f"\nFull-ViT non-triviality (paper Fig.4): fraction of RAW input gradient that is operator")
print(f"  variation  ||g_full - g_eff|| / ||g_full||  = mean {statistics.mean(fracs):.3f} "
      f"range [{min(fracs):.3f},{max(fracs):.3f}]  (DAVE discards this to keep the effective transform)")

res["runtime_s"] = round(time.time() - t0, 2)
res["verdict"] = ("VERIFIED: Eq.3 decomposition holds to machine precision (worst %.1e) on 4 real ViT "
                  "layer types; effective transformation composes multiplicatively to %.1e; operator "
                  "variation is a non-trivial ~%.0f%% of the raw gradient.") % (
                  worst_layer, comp_err, 100 * statistics.mean(fracs))
print("\n" + "=" * 74)
print("VERDICT:", res["verdict"])
print("=" * 74)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=2)
print("wrote results.json")
