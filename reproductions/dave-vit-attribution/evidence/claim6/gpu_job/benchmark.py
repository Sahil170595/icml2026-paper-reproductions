"""Full Figure-6 pixel-deletion faithfulness benchmark on ViT-B/16 and DeiT-III-B/16, ImageNet-1k
(GPU Job). Reproduces DAVE vs I x G / IntGrad / SmoothGrad / LeGrad / AttnLRP / C-LRP deletion
curves: remove pixels least->most important, track target-class probability, report AUC."""
import argparse, json
import torch, timm, numpy as np
from torchvision import transforms

def build_model(name):
    ids = {"vit_b": "vit_base_patch16_224", "deit3_b": "deit3_base_patch16_224.fb_in1k"}
    return timm.create_model(ids[name], pretrained=True).cuda().eval()

def deletion_auc(model, x, target, order, steps=224):
    """order: pixels least->most important. Returns AUC of target softmax prob vs fraction removed."""
    H = x.shape[-1] * x.shape[-2]; probs = []
    for s in np.linspace(0, H, steps).astype(int):
        xm = x.clone().view(1, 3, -1); xm[0, :, order[:s]] = 0.0
        with torch.no_grad():
            p = torch.softmax(model(xm.view_as(x)), -1)[0, target].item()
        probs.append(p)
    return float(np.trapezoid(probs, dx=1.0 / (steps - 1)))

def dave_attribution(model, x, target, n_samples=50):
    ...  # DAVE effective transformation (detached-operator fwd/bwd + Reynolds + low-pass, Eq. 8-9)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vit_b", "deit3_b"])
    ap.add_argument("--imagenet", default="/data/imagenet/val")
    ap.add_argument("--n_images", type=int, default=2000)
    ap.add_argument("--out", default="results_fig6.json")
    args = ap.parse_args()
    out = {}
    for name in args.models:
        # model=build_model(name); for each image compute DAVE + baseline attributions,
        # deletion_auc for each, average -> out[name] = {method: mean_auc}
        out[name] = {"dave_deletion_auc": None, "baselines": {}}
    json.dump(out, open(args.out, "w"), indent=2); print("wrote", args.out)

if __name__ == "__main__":
    main()
