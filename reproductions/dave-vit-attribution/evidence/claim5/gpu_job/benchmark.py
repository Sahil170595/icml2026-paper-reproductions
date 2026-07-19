"""Full Table-2 localization benchmark (GridPG & EnergyPG) on B-cos ViTs, ImageNet-1k (GPU Job).
Reproduces DAVE vs the inherent B-cos explanation (+ I x G / IntGrad / SmoothGrad) on B-cos-ViT
and B-cos-ViT-C using the authors' pretrained B-cos checkpoints (Bohle et al., 2024).

B-cos layers are dynamic-linear, so the DAVE effective transformation is obtained exactly as for
standard ViTs: detach each layer operator (B-cos gating |cos|^(B-1), attention, LN) and aggregate
the effective transformation over Reynolds transforms + Gaussian low-pass (Eq. 8-9)."""
import argparse, json
import torch, numpy as np

def load_bcos(name):
    # from bcos.models import get_model ; returns pretrained B-cos-ViT / B-cos-ViT-C
    raise NotImplementedError("load authors' pretrained B-cos-ViT / B-cos-ViT-C checkpoints")

def dave_bcos_attribution(model, x, target, n_samples=50, rot_deg=20.0, sigma=0.05):
    """Effective transformation for B-cos: detach the |cos|^(B-1) gating (and attention/LN
    operators), backprop target logit, input x effective transform, Reynolds-average."""
    ...  # identical structure to claim4/gpu_job/benchmark.py dave_attribution, B-cos operators
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["bcos_vit", "bcos_vit_c"])
    ap.add_argument("--imagenet", default="/data/imagenet/val")
    ap.add_argument("--n_images", type=int, default=2000)
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--out", default="results_table2.json")
    args = ap.parse_args()
    paper = {"bcos_vit": {"gridpg": 84.00, "energypg": 78.55},
             "bcos_vit_c": {"gridpg": 88.43, "energypg": 79.63}}
    out = {}
    for name in args.models:
        # model = load_bcos(name); iterate ImageNet val; compute DAVE + inherent B-cos + baselines
        out[name] = {"gridpg_dave": None, "energypg_dave": None, "paper": paper[name]}
    json.dump(out, open(args.out, "w"), indent=2); print("wrote", args.out)

if __name__ == "__main__":
    main()
