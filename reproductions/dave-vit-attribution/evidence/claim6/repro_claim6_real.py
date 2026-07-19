"""Claim 6 (judge C2) on REAL pretrained weights: DAVE vs baselines on
timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k with real ImageNet photos.

Methods (patch-level attribution, 196 patches, target = model top-1):
  dave    - effective-transformation gradient (operators frozen, Eq. 3/4) x input,
            Reynolds average over integer shifts {-3,0,3}^2 (Eq. 6), Gaussian
            low-pass sigma=4px (Eq. 7), summed per 16x16 patch.
  ixg     - Input x full Gradient (patch-summed).
  ig      - Integrated Gradients, 16 steps, gray baseline (batched).
  rollout - attention rollout (Abnar & Zuidema), cls row.
Metrics per image (cached _cache/c6_img_<i>.json; 'report' aggregates):
  deletion AUC  - replace top-k patches (k=0,14,...,196) with gray, mean top-1 prob
                  (lower = more faithful);  insertion AUC - reverse (higher better).
  random-order control for both.
  stability     - Pearson corr of patch attribution before/after Gaussian input
                  noise sigma=0.05 (higher = more stable).
Usage: python repro_claim6_real.py img <i> | report
"""
import sys, os, json
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.dirname(HERE)
sys.path.insert(0, EP)
CACHE = os.path.join(EP, "_cache")
import vit_pretrained as V  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(1)
SHIFTS = [(dy, dx) for dy in (-3, 0, 3) for dx in (-3, 0, 3)]
STEPS = list(range(0, 197, 14))  # 15 occlusion levels
METHODS = ("dave", "ixg", "ig", "rollout")


def patch_sum(a):  # 224,224 -> 196
    return a.reshape(14, 16, 14, 16).sum((1, 3)).flatten()


def gauss_blur(a, sigma=4.0):
    r = int(3 * sigma)
    x = torch.arange(-r, r + 1, dtype=a.dtype)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2)); k /= k.sum()
    k2 = torch.outer(k, k)
    pad = torch.nn.functional.pad(a[None, None], (r, r, r, r), mode="reflect")
    return torch.nn.functional.conv2d(pad, k2[None, None])[0, 0]


def attr_dave(m, X, t):
    acc = torch.zeros(224, 224)
    for dy, dx in SHIFTS:
        Xs = torch.roll(X, (dy, dx), dims=(1, 2))
        Xf = Xs.clone().requires_grad_(True)
        y = m.forward(Xf[None], detach_op=True)[0, t]
        g, = torch.autograd.grad(y, Xf)
        acc += torch.roll((g * Xs).sum(0), (-dy, -dx), dims=(0, 1))
    return patch_sum(gauss_blur((acc / len(SHIFTS)).double()).float())


def attr_ixg(m, X, t):
    Xf = X.clone().requires_grad_(True)
    y = m.forward(Xf[None])[0, t]
    g, = torch.autograd.grad(y, Xf)
    return patch_sum((g * X).sum(0))


def attr_ig(m, X, t, steps=16):
    al = torch.linspace(1.0 / steps, 1.0, steps)[:, None, None, None]
    Xb = (al * X[None]).clone().requires_grad_(True)
    y = m.forward(Xb)[:, t].sum()
    g, = torch.autograd.grad(y, Xb)
    return patch_sum((g.mean(0) * X).sum(0))


def attr_rollout(m, X, t):
    with torch.no_grad():
        _, attns = m.forward(X[None], want_attn=True)
    R = torch.eye(197)
    for P in attns:
        A = P[0].mean(0)  # average heads, 197x197
        A = 0.5 * A + 0.5 * torch.eye(197)
        A = A / A.sum(-1, keepdim=True)
        R = A @ R
    return R[0, 1:].clone()


def attribute(m, X, t, method):
    return dict(dave=attr_dave, ixg=attr_ixg, ig=attr_ig, rollout=attr_rollout)[method](m, X, t)


def occlusion_curve(m, X, t, order, insertion=False):
    """order: patch indices most-important first. Returns mean prob over STEPS."""
    Xp = m.patchify(X[None])[0]           # 196,768
    base = torch.zeros_like(Xp)           # gray (normalized 0)
    states = []
    for k in STEPS:
        keep = Xp.clone() if not insertion else base.clone()
        idx = order[:k]
        if insertion:
            keep[idx] = Xp[idx]
        else:
            keep[idx] = base[idx]
        states.append(keep)
    S = torch.stack(states)               # 15,196,768
    with torch.no_grad():
        tok = S @ m.Wpe + m.bpe
        h = torch.cat([m.cls.expand(S.shape[0], 1, -1), tok], 1) + m.pos
        # reuse forward internals: emulate by direct block loop
        for blk in m.blocks:
            B, tt, d = h.shape; nh, dh = m.nh, 192 // m.nh
            mu = h.mean(-1, keepdim=True)
            sig = torch.sqrt(h.var(-1, unbiased=False, keepdim=True) + 1e-6)
            a = (h - mu) / sig * blk["g1"] + blk["b1"]
            qkv = a @ blk["Wqkv"] + blk["bqkv"]
            Q = qkv[..., :d].reshape(B, tt, nh, dh).transpose(1, 2)
            K = qkv[..., d:2*d].reshape(B, tt, nh, dh).transpose(1, 2)
            Vv = qkv[..., 2*d:].reshape(B, tt, nh, dh).transpose(1, 2)
            P = torch.softmax(Q @ K.transpose(-1, -2) / dh ** 0.5, -1)
            h = h + (P @ Vv).transpose(1, 2).reshape(B, tt, d) @ blk["Wo"] + blk["bo"]
            mu = h.mean(-1, keepdim=True)
            sig = torch.sqrt(h.var(-1, unbiased=False, keepdim=True) + 1e-6)
            a2 = (h - mu) / sig * blk["g2"] + blk["b2"]
            pre = a2 @ blk["W1"] + blk["c1"]
            h = h + (0.5 * pre * (1 + torch.erf(pre / 2 ** 0.5))) @ blk["W2"] + blk["c2"]
        mu = h.mean(-1, keepdim=True)
        sig = torch.sqrt(h.var(-1, unbiased=False, keepdim=True) + 1e-6)
        hN = (h - mu) / sig * m.gN + m.bN
        pr = torch.softmax(hN[:, 0] @ m.Wh + m.bh, -1)[:, t]
    return pr.mean().item(), [round(v, 5) for v in pr.tolist()]


def bench_paths():
    return V.image_paths()[::2][:30]


def stage_img(i):
    m = V.ViT()
    p = bench_paths()[i]
    X = V.preprocess(p)
    with torch.no_grad():
        t = int(m.forward(X[None])[0].argmax())
    g = torch.Generator().manual_seed(1000 + i)
    noise = torch.randn(X.shape, generator=g) * 0.05
    rec = dict(img=os.path.basename(p), target=t, methods={})
    for meth in METHODS:
        a = attribute(m, X, t, meth)
        an = attribute(m, X + noise, t, meth)
        stab = torch.corrcoef(torch.stack([a, an]))[0, 1].item()
        order = a.argsort(descending=True)
        del_auc, del_curve = occlusion_curve(m, X, t, order, insertion=False)
        ins_auc, ins_curve = occlusion_curve(m, X, t, order, insertion=True)
        rec["methods"][meth] = dict(stability=stab, del_auc=del_auc, ins_auc=ins_auc,
                                    del_curve=del_curve, ins_curve=ins_curve)
        print(meth, "del %.4f ins %.4f stab %.4f" % (del_auc, ins_auc, stab))
    rnd = torch.randperm(196, generator=torch.Generator().manual_seed(i))
    dr, _ = occlusion_curve(m, X, t, rnd, insertion=False)
    ir, _ = occlusion_curve(m, X, t, rnd, insertion=True)
    rec["random"] = dict(del_auc=dr, ins_auc=ir)
    json.dump(rec, open(os.path.join(CACHE, "c6_img_%02d.json" % i), "w"), indent=1)


def stage_report():
    import glob as _g
    recs = [json.load(open(f)) for f in sorted(_g.glob(os.path.join(CACHE, "c6_img_*.json")))]
    n = len(recs)
    agg = {}
    for meth in METHODS:
        d = [r["methods"][meth]["del_auc"] for r in recs]
        s = [r["methods"][meth]["ins_auc"] for r in recs]
        st = [r["methods"][meth]["stability"] for r in recs]
        def ms(v):
            mu = sum(v) / n
            sd = (sum((x - mu) ** 2 for x in v) / (n - 1)) ** 0.5
            return dict(mean=mu, sd=sd, sem=sd / n ** 0.5)
        agg[meth] = dict(del_auc=ms(d), ins_auc=ms(s), stability=ms(st))
    agg["random"] = dict(del_auc=dict(mean=sum(r["random"]["del_auc"] for r in recs) / n),
                         ins_auc=dict(mean=sum(r["random"]["ins_auc"] for r in recs) / n))
    wins = {}
    for b in ("ixg", "ig", "rollout"):
        wins[b] = dict(
            deletion=sum(r["methods"]["dave"]["del_auc"] < r["methods"][b]["del_auc"] for r in recs),
            insertion=sum(r["methods"]["dave"]["ins_auc"] > r["methods"][b]["ins_auc"] for r in recs),
            stability=sum(r["methods"]["dave"]["stability"] > r["methods"][b]["stability"] for r in recs))
    res = dict(model="timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
               n_images=n, target="model top-1", occlusion="16x16 patches, gray baseline, k=0..196 step 14",
               aggregate=agg, dave_wins_of_30=wins, per_image=recs)
    json.dump(res, open(os.path.join(HERE, "results_real.json"), "w"), indent=1)
    for meth in METHODS + ("random",):
        a = agg[meth]
        print("%-8s del %.4f  ins %.4f  stab %s" % (
            meth, a["del_auc"]["mean"], a["ins_auc"]["mean"],
            "%.4f" % a["stability"]["mean"] if "stability" in a else "-"))
    print("dave wins", json.dumps(wins))


if __name__ == "__main__":
    if sys.argv[1] == "img":
        stage_img(int(sys.argv[2]))
    else:
        stage_report()
