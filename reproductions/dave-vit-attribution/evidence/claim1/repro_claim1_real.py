"""Claim 1 on REAL pretrained weights: DAVE decomposition identity (Eq. 3) on
timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k with real ImageNet photos.

Stages (argv[1]), each cached to _cache/c1_<stage>.json:
  identity  - model-level identity  D_X F = D_C F(A,C)|_{A=C=X} + D_A F(A,C)|_{A=C=X}
              in float64 on real images, + equivalence of the C-path gradient with the
              frozen-operator (effective transformation) gradient, + operator-variation
              fraction of the raw gradient (paper Fig. 4).
  layers    - per-sublayer identity (LayerNorm / MHSA / GELU-MLP / Linear head) using the
              REAL block-0 and block-11 pretrained weights and REAL hidden states, via
              vector-Jacobian rows u^T J in float64.
  classcons - class-consistency of DAVE attributions: same image, different target
              classes -> attribution maps must differ; same class, same image -> exact.
Run all stages, then 'report' merges into results_real.json.
"""
import sys, os, json, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.dirname(HERE)
sys.path.insert(0, EP)
CACHE = os.path.join(EP, "_cache")
import vit_pretrained as V  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(1)


def model64():
    return V.ViT(dtype=torch.float64)


def grads_dual(m, X, target):
    """Returns (g_full, g_C, g_A, g_eff) for d logit[target] / dX, float64."""
    Xf = X.clone().requires_grad_(True)
    y = m.forward(Xf[None])[0, target]
    g_full, = torch.autograd.grad(y, Xf)
    A = X.clone().requires_grad_(True)
    C = X.clone().requires_grad_(True)
    y2 = m.forward(C[None], dual_leaf=A[None])[0, target]
    g_A, g_C = torch.autograd.grad(y2, [A, C])
    Xe = X.clone().requires_grad_(True)
    ye = m.forward(Xe[None], detach_op=True)[0, target]
    g_eff, = torch.autograd.grad(ye, Xe)
    return g_full, g_C, g_A, g_eff, float(y.detach())


def stage_identity():
    m = model64()
    paths = V.image_paths()[::12][:5]  # 5 images, one per class stride
    out = []
    for p in paths:
        X = V.preprocess(p, torch.float64)
        w = os.path.basename(p).split("_")[0]
        t = V.WNID_IDX[w]
        g_full, g_C, g_A, g_eff, y = grads_dual(m, X, t)
        rec = g_C + g_A
        err = (g_full - rec).abs().max().item()
        rel = err / g_full.abs().max().item()
        eff_match = (g_C - g_eff).abs().max().item()
        opfrac = (g_full - g_eff).norm().item() / g_full.norm().item()
        out.append(dict(img=os.path.basename(p), target=t, logit=y,
                        max_abs_err=err, rel_err=rel,
                        effpath_vs_detach_err=eff_match, opvar_frac=opfrac))
        print(p.split("/")[-1], "err %.3e rel %.3e eff-match %.3e opvar %.3f" % (err, rel, eff_match, opfrac))
    json.dump(out, open(os.path.join(CACHE, "c1_identity.json"), "w"), indent=1)


def stage_layers():
    m = model64()
    X = V.preprocess(V.image_paths()[7], torch.float64)
    # real hidden state entering block 0 and block 6
    with torch.no_grad():
        h0 = torch.cat([m.cls[None].to(torch.float64).expand(1, 1, -1),
                        (m.patchify(X[None]) @ m.Wpe + m.bpe)], 1) + m.pos
    def ln(A, C, g, b, eps=1e-6):
        sig = torch.sqrt(A.var(-1, unbiased=False, keepdim=True) + eps)
        return (C - C.mean(-1, keepdim=True)) / sig * g + b
    def mhsa(A, C, blk):
        B, t, d = C.shape; nh = 3; dh = d // nh
        qA = A @ blk["Wqkv"] + blk["bqkv"]; qC = C @ blk["Wqkv"] + blk["bqkv"]
        Q = qA[..., :d].reshape(B, t, nh, dh).transpose(1, 2)
        K = qA[..., d:2*d].reshape(B, t, nh, dh).transpose(1, 2)
        Vv = qC[..., 2*d:].reshape(B, t, nh, dh).transpose(1, 2)
        P = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(dh), -1)
        return (P @ Vv).transpose(1, 2).reshape(B, t, d) @ blk["Wo"] + blk["bo"]
    def mlp(A, C, blk):
        pA = A @ blk["W1"] + blk["c1"]; pC = C @ blk["W1"] + blk["c1"]
        gate = 0.5 * (1.0 + torch.erf(pA / math.sqrt(2.0)))
        return (gate * pC) @ blk["W2"] + blk["c2"]
    def head(A, C):
        return C[:, 0] @ m.Wh + m.bh  # static linear: no operator
    gen = torch.Generator().manual_seed(1)
    res = []
    for bi in (0, 11):
        blk = m.blocks[bi]
        for name, fn in [("layernorm", lambda A, C: ln(A, C, blk["g1"], blk["b1"])),
                         ("mhsa", lambda A, C: mhsa(A, C, blk)),
                         ("mlp_gelu", lambda A, C: mlp(A, C, blk)),
                         ("linear_head", head)]:
            worst = 0.0
            for k in range(3):  # 3 random Jacobian rows
                u = torch.randn(fn(h0, h0).shape, generator=gen, dtype=torch.float64)
                Xf = h0.clone().requires_grad_(True)
                gf, = torch.autograd.grad((fn(Xf, Xf) * u).sum(), Xf)
                A = h0.clone().requires_grad_(True); C = h0.clone().requires_grad_(True)
                s = (fn(A, C) * u).sum()
                gA, gC = torch.autograd.grad(s, [A, C], allow_unused=True)
                gA = torch.zeros_like(h0) if gA is None else gA
                rel = ((gf - (gA + gC)).abs().max() / gf.abs().max()).item()
                worst = max(worst, rel)
            res.append(dict(block=bi, layer=name, worst_rel_err=worst))
            print("block", bi, name, "worst rel err %.3e" % worst)
    json.dump(res, open(os.path.join(CACHE, "c1_layers.json"), "w"), indent=1)


def stage_classcons():
    m = V.ViT()  # float32 fine here
    paths = V.image_paths()[2::12][:4]
    out = []
    for p in paths:
        X = V.preprocess(p)
        w = os.path.basename(p).split("_")[0]
        t = V.WNID_IDX[w]
        alts = [i for i in sorted(V.WNID_IDX.values()) if i != t]
        t2 = alts[hash(w) % len(alts)]
        def dave_map(target):
            Xf = X.clone().requires_grad_(True)
            y = m.forward(Xf[None], detach_op=True)[0, target]
            g, = torch.autograd.grad(y, Xf)
            return (g * X).sum(0)
        a1, a1b, a2 = dave_map(t), dave_map(t), dave_map(t2)
        c_same = torch.corrcoef(torch.stack([a1.flatten(), a1b.flatten()]))[0, 1].item()
        c_diff = torch.corrcoef(torch.stack([a1.flatten(), a2.flatten()]))[0, 1].item()
        out.append(dict(img=os.path.basename(p), target=t, alt=t2,
                        corr_same_class=c_same, corr_diff_class=c_diff))
        print(os.path.basename(p), "same %.6f diff %.4f" % (c_same, c_diff))
    json.dump(out, open(os.path.join(CACHE, "c1_classcons.json"), "w"), indent=1)


def stage_report():
    ident = json.load(open(os.path.join(CACHE, "c1_identity.json")))
    layers = json.load(open(os.path.join(CACHE, "c1_layers.json")))
    cc = json.load(open(os.path.join(CACHE, "c1_classcons.json")))
    cls = json.load(open(os.path.join(CACHE, "classify_sanity.json")))
    res = dict(
        model="timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k (5.7M params, ImageNet-1k, loaded from HF-hub safetensors)",
        images="Imagenette v2 (fast.ai) real ImageNet validation photos, 224x224, timm eval transform",
        classify_sanity=dict(n=cls["n"], top1_acc=cls["acc"],
                             mean_p_true=round(sum(r["p_true"] for r in cls["records"]) / cls["n"], 4)),
        model_identity=dict(
            n_images=len(ident),
            worst_max_abs_err=max(r["max_abs_err"] for r in ident),
            worst_rel_err=max(r["rel_err"] for r in ident),
            worst_effpath_vs_detach=max(r["effpath_vs_detach_err"] for r in ident),
            opvar_frac_mean=sum(r["opvar_frac"] for r in ident) / len(ident),
            opvar_frac_min=min(r["opvar_frac"] for r in ident),
            opvar_frac_max=max(r["opvar_frac"] for r in ident),
            per_image=ident),
        per_layer_identity=dict(worst_rel_err=max(r["worst_rel_err"] for r in layers), rows=layers),
        class_consistency=dict(
            corr_same_class_min=min(r["corr_same_class"] for r in cc),
            corr_diff_class_mean=sum(r["corr_diff_class"] for r in cc) / len(cc),
            corr_diff_class_max=max(r["corr_diff_class"] for r in cc),
            rows=cc))
    json.dump(res, open(os.path.join(HERE, "results_real.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("classify_sanity", "model_identity")}, indent=1)[:900])


if __name__ == "__main__":
    dict(identity=stage_identity, layers=stage_layers,
         classcons=stage_classcons, report=stage_report)[sys.argv[1]]()
