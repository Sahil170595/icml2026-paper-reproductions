"""Claim 4 - Localization on conventional ViTs: GridPG & EnergyPG (paper Table 1).

Paper claim: DAVE improves GridPG across ViT-B/16 (60.19%), DeiT-B/16 (63.52%), DeiT-III-B/16
(65.76%), DINO-B/16 (51.33%) and is best/competitive on EnergyPG (DeiT-B 82.23%, DeiT-III-B
82.43%, DINO-B 83.38%) vs I x G, IntGrad, SmoothGrad, LeGrad, AttnLRP, Chefer-LRP on ImageNet-1k.

Large-scale attribution-quality benchmark. Per pilot rules:
  * CPU MECHANISM CHECK: implement GridPG (fraction of positive attribution mass in the target
    grid cell) and EnergyPG (fraction of attribution energy in the ground-truth box) and run
    DAVE (effective transform) vs Input x Gradient on the compact ViT. Random-init weights =>
    scores are near chance and are NOT the ImageNet claim; this validates the metric machinery.
  * emit an `hf jobs run` GPU KIT (gpu_job/) for the full Table-1 benchmark on the real models.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import dave_vit as D
torch.set_default_dtype(torch.float64)
os.environ.setdefault("OMP_NUM_THREADS", "1")
t0 = time.time()
K = 10
print("=" * 74)
print("Claim 4: GridPG & EnergyPG localization (Table 1) - CPU mechanism check")
print("=" * 74)

def energy_pg(a, box):                              # fraction of |attribution| energy inside box
    y0, y1, x0, x1 = box
    e = np.abs(a); tot = e.sum() + 1e-30
    return float(e[y0:y1, x0:x1].sum() / tot)

def grid_pg(a, cell, nside=2):                      # fraction of positive attribution mass in target cell
    H = a.shape[0]; c = H // nside
    r, col = cell // nside, cell % nside
    pos = np.clip(a, 0, None); tot = pos.sum() + 1e-30
    return float(pos[r*c:(r+1)*c, col*c:(col+1)*c].sum() / tot)

rng = np.random.default_rng(1)
gpg = {"DAVE": [], "IxG": []}
epg = {"DAVE": [], "IxG": []}
for seed in range(6):
    m = D.TinyViT(seed=seed)
    Whk = torch.randn(m.dim, K, generator=torch.Generator().manual_seed(700 + seed)) * 0.3
    bhk = torch.zeros(K)
    # 2x2 montage: a distinct blob per cell; the "object" cell holds the strongest blob (its box is GT)
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
    X = torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(seed)) * 0.12
    centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
    tgt = int(rng.integers(0, 4))
    for ci, (cy, cx) in enumerate(centers):
        amp = 1.6 if ci == tgt else 0.5
        X = X + (amp * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 16.0)).unsqueeze(0)
    with torch.no_grad():
        target = int(D.forward_head(m, X, Whk, bhk).argmax())
    box = [(tgt // 2) * 16, (tgt // 2) * 16 + 16, (tgt % 2) * 16, (tgt % 2) * 16 + 16]
    for meth in ["DAVE", "IxG"]:
        a = D.attribution_map(m, X, Whk, bhk, target, "dave" if meth == "DAVE" else "ixg").numpy()
        gpg[meth].append(grid_pg(a, tgt))
        epg[meth].append(energy_pg(a, box))

mg = {k: 100*float(np.mean(v)) for k, v in gpg.items()}
me = {k: 100*float(np.mean(v)) for k, v in epg.items()}
print("localization scores (mean over 6 montages), 2x2 GridPG, chance = 25%:")
print(f"    {'method':8s} {'GridPG %':>9s} {'EnergyPG %':>11s}")
for k in ["DAVE", "IxG"]:
    print(f"    {k:8s} {mg[k]:9.2f} {me[k]:11.2f}")
print("    (mechanism check only: random-init weights => near-chance; NOT the ImageNet Table-1 claim)")

res = {
    "claim": "DAVE improves GridPG on all conventional ViTs (ViT-B 60.19, DeiT-B 63.52, DeiT-III-B 65.76, "
             "DINO-B 51.33) and is best/competitive on EnergyPG vs 6 baselines on ImageNet-1k (Table 1).",
    "scale": "CPU MECHANISM CHECK on compact random-init ViT (NOT the ImageNet Table-1 claim)",
    "gridpg_pct_mean": mg, "energypg_pct_mean": me, "gridpg_chance_pct": 25.0,
    "paper_table1_gridpg": {"ViT-B": 60.19, "DeiT-B": 63.52, "DeiT-III-B": 65.76, "DINO-B": 51.33},
    "paper_table1_energypg": {"DeiT-B": 82.23, "DeiT-III-B": 82.43, "DINO-B": 83.38},
    "gpu_kit": "gpu_job/ (hf jobs run) runs the full Table-1 GridPG/EnergyPG benchmark on real models",
    "verdict": "MECHANISM-CHECK (not verified at scale): GridPG & EnergyPG metrics implemented and run on the "
               "compact ViT. Paper Table-1 targets (DAVE GridPG 60-66%, EnergyPG 82-83%) need real pretrained "
               "ViT-B/DeiT/DINO weights + ImageNet bboxes and are emitted as a GPU kit.",
    "runtime_s": round(time.time() - t0, 2),
}
print("\nVERDICT:", res["verdict"])
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=2)
print("wrote results.json")
