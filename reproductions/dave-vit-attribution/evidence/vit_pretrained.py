"""Pretrained ViT-Tiny/16 (timm vit_tiny_patch16_224.augreg_in21k_ft_in1k) in pure PyTorch.

Loads the HF-hub safetensors checkpoint directly (no timm/torchvision import needed)
into an explicit pre-norm ViT forward that supports DAVE's operator freezing:
with detach_op=True every input-dependent *operator* (LayerNorm statistics, the
softmax attention matrix P, the GELU gate) is detached, so autograd flows only
through the *effective transformation* W_L(X) (paper Eq. 3/4). detach_op=False is
the ordinary full gradient. Also exposes per-block attention matrices for rollout.
"""
import math, os, glob
import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache")
CKPT_REPO = "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k"

# Imagenette wnid -> ImageNet-1k class index / name
WNID_IDX = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
            "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
            "n03445777": 574, "n03888257": 701}
WNID_NAME = {"n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
             "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
             "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
             "n03888257": "parachute"}


def load_state():
    hits = glob.glob(os.path.join(CACHE, "hf", "models--timm--*", "snapshots", "*", "model.safetensors"))
    if not hits:
        from huggingface_hub import hf_hub_download
        hits = [hf_hub_download(CKPT_REPO, "model.safetensors", cache_dir=os.path.join(CACHE, "hf"))]
    return load_file(hits[0])


class ViT:
    """vit_tiny_patch16_224: patch 16, dim 192, depth 12, heads 3, MLP 768, 1000 classes."""

    def __init__(self, dtype=torch.float32):
        sd = {k: v.to(dtype) for k, v in load_state().items()}
        self.dtype = dtype
        self.cls = sd["cls_token"][0]                       # 1,192
        self.pos = sd["pos_embed"][0]                       # 197,192
        w = sd["patch_embed.proj.weight"]                   # 192,3,16,16
        self.Wpe = w.reshape(192, -1).t().contiguous()      # 768,192 (C-order c,h,w per patch)
        self.bpe = sd["patch_embed.proj.bias"]
        self.blocks = []
        for i in range(12):
            p = f"blocks.{i}."
            self.blocks.append(dict(
                g1=sd[p + "norm1.weight"], b1=sd[p + "norm1.bias"],
                Wqkv=sd[p + "attn.qkv.weight"].t().contiguous(), bqkv=sd[p + "attn.qkv.bias"],
                Wo=sd[p + "attn.proj.weight"].t().contiguous(), bo=sd[p + "attn.proj.bias"],
                g2=sd[p + "norm2.weight"], b2=sd[p + "norm2.bias"],
                W1=sd[p + "mlp.fc1.weight"].t().contiguous(), c1=sd[p + "mlp.fc1.bias"],
                W2=sd[p + "mlp.fc2.weight"].t().contiguous(), c2=sd[p + "mlp.fc2.bias"]))
        self.gN, self.bN = sd["norm.weight"], sd["norm.bias"]
        self.Wh = sd["head.weight"].t().contiguous()        # 192,1000
        self.bh = sd["head.bias"]
        self.nh, self.dim, self.depth = 3, 192, 12

    def patchify(self, X):
        # X: B,3,224,224 -> B,196,768 (channel-major within patch, matches conv weight reshape)
        B = X.shape[0]
        P = X.reshape(B, 3, 14, 16, 14, 16).permute(0, 2, 4, 1, 3, 5).reshape(B, 196, 768)
        return P

    def forward(self, X, detach_op=False, want_attn=False, dual_leaf=None):
        """X: B,3,224,224 -> logits B,1000.
        detach_op: freeze operators (LN stats / attention P / GELU gate) => effective path.
        dual_leaf: optional tensor like X routed ONLY into the operators (A-argument);
                   then X feeds only the acted-upon C-argument (Eq. 3 dual-input form)."""
        op = (lambda h: h.detach()) if detach_op else (lambda h: h)
        attns = []

        def ln(A, C, g, b, eps=1e-6):
            mu = A.mean(-1, keepdim=True)
            sig = torch.sqrt(A.var(-1, unbiased=False, keepdim=True) + eps)
            return (C - C.mean(-1, keepdim=True)) / sig * g + b

        # single-path version with op() (used unless dual_leaf given)
        if dual_leaf is None:
            h = torch.cat([self.cls.expand(X.shape[0], 1, -1),
                           self.patchify(X) @ self.Wpe + self.bpe], 1) + self.pos
            for blk in self.blocks:
                B, t, d = h.shape
                dh = d // self.nh
                a = ln(op(h), h, blk["g1"], blk["b1"])
                qkvA = op(a) @ blk["Wqkv"] + blk["bqkv"]
                qkvC = a @ blk["Wqkv"] + blk["bqkv"]
                Q = qkvA[..., :d].reshape(B, t, self.nh, dh).transpose(1, 2)
                K = qkvA[..., d:2 * d].reshape(B, t, self.nh, dh).transpose(1, 2)
                V = qkvC[..., 2 * d:].reshape(B, t, self.nh, dh).transpose(1, 2)
                P = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(dh), dim=-1)
                if want_attn:
                    attns.append(P.detach())
                if detach_op:
                    P = P.detach()
                O = (P @ V).transpose(1, 2).reshape(B, t, d) @ blk["Wo"] + blk["bo"]
                h = h + O
                a2 = ln(op(h), h, blk["g2"], blk["b2"])
                pre = a2 @ blk["W1"] + blk["c1"]
                preA = op(a2) @ blk["W1"] + blk["c1"] if detach_op else pre
                gate = 0.5 * (1.0 + torch.erf(preA / math.sqrt(2.0)))  # GELU gate from A
                act = (gate.detach() if detach_op else gate) * pre
                h = h + act @ blk["W2"] + blk["c2"]
            hN = ln(op(h), h, self.gN, self.bN)
            logits = hN[:, 0] @ self.Wh + self.bh
            return (logits, attns) if want_attn else logits

        # explicit dual-input form: operators read A=dual_leaf, values read C=X
        A_in, C_in = dual_leaf, X
        hA = torch.cat([self.cls.expand(A_in.shape[0], 1, -1),
                        self.patchify(A_in) @ self.Wpe + self.bpe], 1) + self.pos
        hC = torch.cat([self.cls.expand(C_in.shape[0], 1, -1),
                        self.patchify(C_in) @ self.Wpe + self.bpe], 1) + self.pos
        for blk in self.blocks:
            B, t, d = hC.shape
            dh = d // self.nh
            aA = ln(hA, hA, blk["g1"], blk["b1"])
            aC = ln(hA, hC, blk["g1"], blk["b1"])
            qA = aA @ blk["Wqkv"] + blk["bqkv"]
            qC = aC @ blk["Wqkv"] + blk["bqkv"]
            Q = qA[..., :d].reshape(B, t, self.nh, dh).transpose(1, 2)
            K = qA[..., d:2 * d].reshape(B, t, self.nh, dh).transpose(1, 2)
            VA = qA[..., 2 * d:].reshape(B, t, self.nh, dh).transpose(1, 2)
            VC = qC[..., 2 * d:].reshape(B, t, self.nh, dh).transpose(1, 2)
            P = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(dh), dim=-1)
            OA = (P @ VA).transpose(1, 2).reshape(B, t, d) @ blk["Wo"] + blk["bo"]
            OC = (P @ VC).transpose(1, 2).reshape(B, t, d) @ blk["Wo"] + blk["bo"]
            hA, hC = hA + OA, hC + OC
            a2A = ln(hA, hA, blk["g2"], blk["b2"])
            a2C = ln(hA, hC, blk["g2"], blk["b2"])
            preA = a2A @ blk["W1"] + blk["c1"]
            preC = a2C @ blk["W1"] + blk["c1"]
            gate = 0.5 * (1.0 + torch.erf(preA / math.sqrt(2.0)))
            hA = hA + (gate * preA) @ blk["W2"] + blk["c2"]
            hC = hC + (gate * preC) @ blk["W2"] + blk["c2"]
        hN = ln(hA, hC, self.gN, self.bN)
        return hN[:, 0] @ self.Wh + self.bh


def preprocess(path, dtype=torch.float32):
    """timm eval transform: resize shorter->floor(224/0.9)=248 bicubic, center-crop 224, mean/std 0.5."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 248 / min(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    im = im.crop((l, t, l + 224, t + 224))
    x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    return ((x - 0.5) / 0.5).to(dtype)


def image_paths():
    return sorted(glob.glob(os.path.join(CACHE, "images", "*.JPEG")))
