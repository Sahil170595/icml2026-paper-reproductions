#!/usr/bin/env python3
"""
GPU kit for CLAIM 6 - ViT: Tucker Attention matches GQA/MLA validation performance
with ~an order of magnitude fewer attention parameters (arXiv 2603.30033, Fig 3).

Trains a REAL Vision Transformer on a REAL ImageNet subset (Imagenette: 10 native
ImageNet classes, full-resolution photographs) and compares MHA vs GQA vs MLA vs
Tucker Attention.  Each low-rank variant is initialized from the pretrained MHA
weights exactly as in Appendix C.1.4 (SVD for GQA/MLA, HOSVD for Tucker), then the
model is fine-tuned; we report top-1/top-5 validation accuracy against the number
of trainable attention parameters (the Figure-3 axes).

Faithful to the paper:
  * pretrained ViT init (timm ImageNet weights)   [Sec 4.1]
  * low-rank init by SVD / HOSVD of MHA weights    [Appendix C.1.4]
  * Tucker pre-/post-softmax tensor factorization  [Def 2.1, Eqs 4-6]
  * AdamW + cosine schedule with warmup            [Table 5]

One method per job; sweep to build the Pareto frontier (see RUN_GPU.md):
  python run.py --method tucker --r1 8 --r2 32 --r3 32 --epochs 10 --out out_tucker_8_32_32.json
  python run.py --method gqa --n_kv 1 --epochs 10 --out out_gqa1.json
  python run.py --method mla --dc 32 --epochs 10 --out out_mla32.json
  python run.py --method mha --epochs 10 --out out_mha.json
"""
# /// script
# dependencies = [
#   "torch>=2.1",
#   "timm>=1.0.0",
#   "datasets>=2.18",
#   "torchvision>=0.16",
#   "pillow",
# ]
# ///
import argparse, json, math, time, os
import torch, torch.nn as nn, torch.nn.functional as F


# --------------------------------------------------------------------------- #
def split_heads_cols(W, nH):
    d = W.shape[1]; dH = d // nH
    return [W[:, i*dH:(i+1)*dH] for i in range(nH)]


class MHAWrap(nn.Module):
    def __init__(self, d, nH, WQ, WK, WV, WO, qb, pb):
        super().__init__(); self.d, self.nH, self.dH = d, nH, d // nH
        self.WQ = nn.Parameter(WQ); self.WK = nn.Parameter(WK)
        self.WV = nn.Parameter(WV); self.WO = nn.Parameter(WO)
        self.qb = nn.Parameter(qb) if qb is not None else None
        self.pb = nn.Parameter(pb) if pb is not None else None
    def attn_param_count(self):
        return sum(p.numel() for p in [self.WQ, self.WK, self.WV, self.WO])
    def forward(self, x):
        B, N, d = x.shape; dH = self.dH
        q = x @ self.WQ; k = x @ self.WK; v = x @ self.WV
        if self.qb is not None:
            qb, kb, vb = self.qb.chunk(3); q = q + qb; k = k + kb; v = v + vb
        q = q.view(B, N, self.nH, dH).transpose(1, 2)
        k = k.view(B, N, self.nH, dH).transpose(1, 2)
        v = v.view(B, N, self.nH, dH).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, N, d) @ self.WO
        return o + self.pb if self.pb is not None else o


class GQAWrap(nn.Module):
    def __init__(self, d, nH, n_kv, WQ, WK, WV, WO, qb, pb):
        super().__init__(); self.d, self.nH, self.dH, self.n_kv = d, nH, d // nH, n_kv
        per = nH // n_kv
        self.WQ = nn.Parameter(WQ); self.WO = nn.Parameter(WO)
        Kh = split_heads_cols(WK, nH); Vh = split_heads_cols(WV, nH)
        WKg = torch.cat([torch.stack(Kh[g*per:(g+1)*per]).mean(0) for g in range(n_kv)], 1)
        WVg = torch.cat([torch.stack(Vh[g*per:(g+1)*per]).mean(0) for g in range(n_kv)], 1)
        self.WKg = nn.Parameter(WKg); self.WVg = nn.Parameter(WVg)
        self.pb = nn.Parameter(pb) if pb is not None else None
    def attn_param_count(self):
        return sum(p.numel() for p in [self.WQ, self.WKg, self.WVg, self.WO])
    def forward(self, x):
        B, N, d = x.shape; dH = self.dH; per = self.nH // self.n_kv
        q = (x @ self.WQ).view(B, N, self.nH, dH).transpose(1, 2)
        k = (x @ self.WKg).view(B, N, self.n_kv, dH).transpose(1, 2)
        v = (x @ self.WVg).view(B, N, self.n_kv, dH).transpose(1, 2)
        k = k.repeat_interleave(per, dim=1); v = v.repeat_interleave(per, dim=1)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, N, d) @ self.WO
        return o + self.pb if self.pb is not None else o


class MLAWrap(nn.Module):
    def __init__(self, d, nH, dc, WQ, WK, WV, WO, qb, pb):
        super().__init__(); self.d, self.nH, self.dH, self.dc = d, nH, d // nH, dc
        def lowrank(W, r):
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            return (U[:, :r] * S[:r].sqrt()).contiguous(), (S[:r].sqrt()[:, None] * Vh[:r]).contiguous()
        self.WDQ, self.WUQ  = map(nn.Parameter, lowrank(WQ, dc))
        self.WDKV, self.WUK = map(nn.Parameter, lowrank(WK, dc))
        _, WUV = lowrank(WV, dc); self.WUV = nn.Parameter(WUV)   # shared KV down-proj
        self.WO = nn.Parameter(WO)
        self.pb = nn.Parameter(pb) if pb is not None else None
    def attn_param_count(self):
        return sum(p.numel() for p in [self.WDQ, self.WUQ, self.WDKV, self.WUK, self.WUV, self.WO])
    def forward(self, x):
        B, N, d = x.shape; dH = self.dH
        q = (x @ self.WDQ @ self.WUQ).view(B, N, self.nH, dH).transpose(1, 2)
        lkv = x @ self.WDKV
        k = (lkv @ self.WUK).view(B, N, self.nH, dH).transpose(1, 2)
        v = (lkv @ self.WUV).view(B, N, self.nH, dH).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, N, d) @ self.WO
        return o + self.pb if self.pb is not None else o


def hosvd3(T, ranks):
    """Higher-order SVD of a 3-tensor T -> (core, U0, U1, U2), truncated to `ranks`."""
    Us = []
    for mode, r in enumerate(ranks):
        M = torch.movedim(T, mode, 0).reshape(T.shape[mode], -1)
        U, _, _ = torch.linalg.svd(M, full_matrices=False)
        Us.append(U[:, :r].contiguous())
    core = T
    for mode in range(3):
        core = torch.tensordot(core, Us[mode].T, dims=([mode], [1]))
        core = torch.movedim(core, -1, mode)
    return core.contiguous(), Us[0], Us[1], Us[2]


class TuckerWrap(nn.Module):
    """Tucker Attention (HOSVD of pre-/post-softmax attention tensors, App C.1.4)."""
    def __init__(self, d, nH, r1, r2, r3, WQ, WK, WV, WO, qb, pb):
        super().__init__(); self.d, self.nH, self.dH = d, nH, d // nH
        dH = self.dH
        Qh = split_heads_cols(WQ, nH); Kh = split_heads_cols(WK, nH); Vh = split_heads_cols(WV, nH)
        Oh = [WO[i*dH:(i+1)*dH, :] for i in range(nH)]                 # per-head W_i^O (dH,d)
        Wpre  = torch.stack([Qh[i] @ Kh[i].T for i in range(nH)])      # (nH,d,d)
        Wpost = torch.stack([(Vh[i] @ Oh[i]).T for i in range(nH)])    # (nH,d,d) = Wt_i^T
        C, U1, U2, U3     = hosvd3(Wpre,  (min(r1, nH), r2, r3))
        Ct, U1t, U2t, U3t = hosvd3(Wpost, (min(r1, nH), r2, r3))
        for n, p in dict(C=C, U1=U1, U2=U2, U3=U3, Ct=Ct, U1t=U1t, U2t=U2t, U3t=U3t).items():
            setattr(self, n, nn.Parameter(p))
        self.pb = nn.Parameter(pb) if pb is not None else None
    def attn_param_count(self):
        return sum(p.numel() for p in [self.C, self.U1, self.U2, self.U3,
                                       self.Ct, self.U1t, self.U2t, self.U3t])
    def forward(self, x):
        B, N, d = x.shape; dH = self.dH
        P2 = x @ self.U2; P3 = x @ self.U3                            # (B,N,r2),(B,N,r3)
        T  = torch.einsum('abc,ia->ibc', self.C, self.U1)            # (nH,r2,r3)
        logits = torch.einsum('bnp,ipc,bmc->binm', P2, T, P3) / math.sqrt(dH)
        H1 = torch.softmax(logits, dim=-1)                           # (B,nH,N,N)
        V  = x @ self.U3t                                            # (B,N,r3)
        St = torch.einsum('abc,ia->ibc', self.Ct, self.U1t)         # (nH,r2,r3)
        S  = torch.einsum('ipc,bmc->bipm', St, V)                   # (B,nH,r2,N)
        H2 = torch.einsum('bipm,jp->bijm', S, self.U2t)            # (B,nH,d,N)
        o  = torch.einsum('bijm,bikm->bjk', H1, H2)                # (B,N,d)
        return o + self.pb if self.pb is not None else o


# --------------------------------------------------------------------------- #
def build_variant(attn, method, args):
    """Read a timm Attention (qkv, proj) and return the chosen low-rank variant."""
    d = attn.qkv.in_features; nH = attn.num_heads
    Wqkv = attn.qkv.weight.data                       # (3d, d)
    WQ = Wqkv[:d].T.contiguous(); WK = Wqkv[d:2*d].T.contiguous(); WV = Wqkv[2*d:].T.contiguous()
    WO = attn.proj.weight.data.T.contiguous()         # (d, d)
    qb = attn.qkv.bias.data.clone() if attn.qkv.bias is not None else None
    pb = attn.proj.bias.data.clone() if attn.proj.bias is not None else None
    if method == "mha":    return MHAWrap(d, nH, WQ, WK, WV, WO, qb, pb)
    if method == "gqa":    return GQAWrap(d, nH, args.n_kv, WQ, WK, WV, WO, qb, pb)
    if method == "mla":    return MLAWrap(d, nH, args.dc, WQ, WK, WV, WO, qb, pb)
    if method == "tucker": return TuckerWrap(d, nH, args.r1, args.r2, args.r3, WQ, WK, WV, WO, qb, pb)
    raise ValueError(method)


def replace_attention(model, method, args):
    total = 0
    for blk in model.blocks:
        v = build_variant(blk.attn, method, args)
        total += v.attn_param_count()
        blk.attn = v
    return total


def make_loaders(args):
    import torchvision.transforms as T
    from datasets import load_dataset
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)       # paper ImageNet normalization
    tf_tr = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize(mean, std)])
    tf_va = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])
    ds = load_dataset(args.dataset, args.dataset_config)
    tr_split = "train"; va_split = "validation" if "validation" in ds else "test"
    lbl_key = "label" if "label" in ds[tr_split].column_names else "labels"
    def collate(tf):
        def f(batch):
            xs = torch.stack([tf(b["image"].convert("RGB")) for b in batch])
            ys = torch.tensor([b[lbl_key] for b in batch])
            return xs, ys
        return f
    n_classes = ds[tr_split].features[lbl_key].num_classes
    tl = torch.utils.data.DataLoader(ds[tr_split], batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, collate_fn=collate(tf_tr), drop_last=True)
    vl = torch.utils.data.DataLoader(ds[va_split], batch_size=args.batch, shuffle=False,
                                     num_workers=args.workers, collate_fn=collate(tf_va))
    return tl, vl, n_classes


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); top1 = top5 = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        _, p5 = logits.topk(5, dim=1)
        correct = p5.eq(y[:, None])
        top1 += correct[:, 0].sum().item(); top5 += correct.any(1).sum().item(); n += y.numel()
    return 100.0 * top1 / n, 100.0 * top5 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["mha", "gqa", "mla", "tucker"])
    ap.add_argument("--model", default="vit_small_patch16_224")
    ap.add_argument("--dataset", default="frgfm/imagenette")
    ap.add_argument("--dataset_config", default="320px")
    ap.add_argument("--n_kv", type=int, default=1)
    ap.add_argument("--dc", type=int, default=32)
    ap.add_argument("--r1", type=int, default=8)
    ap.add_argument("--r2", type=int, default=32)
    ap.add_argument("--r3", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.001)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    import timm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tl, vl, n_classes = make_loaders(args)
    model = timm.create_model(args.model, pretrained=True, num_classes=n_classes)
    attn_params = replace_attention(model, args.method, args)
    model.to(device)

    steps = args.epochs * len(tl)
    warmup = max(1, int(args.warmup_frac * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    def lr_at(s):
        if s < warmup: return s / warmup
        prog = (s - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    t0 = time.time(); step = 0
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); step += 1
        top1, top5 = evaluate(model, vl, device)
        print(f"[{args.method}] epoch {ep+1}/{args.epochs}  val top1={top1:.2f}  top5={top5:.2f}  "
              f"loss={loss.item():.3f}  ({time.time()-t0:.0f}s)")

    top1, top5 = evaluate(model, vl, device)
    total_params = sum(p.numel() for p in model.parameters())
    res = dict(method=args.method, model=args.model, dataset=args.dataset,
               ranks=dict(n_kv=args.n_kv, dc=args.dc, r1=args.r1, r2=args.r2, r3=args.r3),
               epochs=args.epochs, val_top1=round(top1, 3), val_top5=round(top5, 3),
               attn_params_total=int(attn_params), model_params_total=int(total_params),
               attn_params_MB_bf16=round(attn_params * 2 / 1e6, 3),
               n_classes=n_classes, device=device, runtime_s=round(time.time() - t0, 1),
               torch=torch.__version__)
    json.dump(res, open(args.out, "w"), indent=2)
    print("RESULT " + json.dumps(res))
    print("[wrote %s]" % args.out)


if __name__ == "__main__":
    main()
