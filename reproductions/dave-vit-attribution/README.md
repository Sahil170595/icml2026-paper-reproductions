# DAVE: Distribution-Aware Attribution via ViT Gradient Decomposition

✅✅  **4 pts** — 2/2 full-credit  (verified, verified)

[arXiv 2602.06613](https://arxiv.org/abs/2602.06613) · [OpenReview](https://openreview.net/forum?id=ykTMNA6Mbh) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-dave-vit-attribution-repro)

## Scoreboard — measured vs. paper target

| # | Paper claim | Target + acceptance rule | Measured (this repro, real pretrained ViT-Tiny/16 + real ImageNet photos) | Verdict |
|---|---|---|---|---|
| 1 | **Layer-derivative decomposition** D_X F = L(X) [effective] + operator-variation (Eq. 3) | identity = true autograd gradient to machine precision on a real trained ViT | model-level identity through 12 pretrained blocks worst **6.2e-15** (float64); per-sublayer (LN/MHSA/GELU-MLP/head, real weights + real hidden states) **0.0**; frozen-operator gradient bit-exact = C-path; opvar fraction **0.98–1.22** of raw gradient (Fig. 4) | **reproduced (exact, pretrained)** |
| 2 | **Reynolds equivariant filtering** suppresses patch-grid artifacts (Eq. 6) | grid artifact on real attributions suppressed; equivariant signal kept | 20 real photos: patch-boundary discontinuity **1.27 → 0.97** (≈1 = no grid); lattice spectral energy **−91.6%** (with Eq. 7); position-locked component **−62.4%** | **reproduced (pretrained)** |
| 3 | **Low-pass = Gaussian convolution** E_ε[W_L^eq(X+ε)] = (W_L^eq∗K)(X) (Eq. 7) | identity exact; MC average → convolution at 1/√n | quadratic identity exact **0.0** (all σ); **pretrained-ViT** effective response: MC→conv RMS slope **−0.497** (241 real evaluations) | **reproduced (exact + pretrained)** |
| 4 | **Localization GridPG/EnergyPG, conventional ViTs** — DAVE beats 6 baselines (Table 1) | DAVE GridPG 60–66%, EnergyPG 82–83% > best baseline, ImageNet-1k (needs bbox annotations) | CPU mechanism (random-init control): GridPG **21.6%**, EnergyPG **21.5%** (chance 25%); **GPU kit emitted** | mechanism + GPU kit (not verified at scale) |
| 5 | **Localization on B-cos ViTs + architecture-versatility** (Table 2) | DAVE > inherent B-cos: 84.0/88.4% GridPG, 78.6/79.6% EnergyPG | DAVE decomposition on **real B-cos layers exact to 5.6e-17**; localization → **GPU kit** | partial-verified (decomposition exact) + GPU kit |
| 6 | **Faithfulness: flattest pixel-deletion curve** (Fig. 6) | DAVE most faithful under deletion vs baselines | **pretrained ViT-Tiny, 30 real photos:** DAVE **best deletion AUC 0.170** vs I×G 0.246 / IG 0.183 / rollout 0.223 / random 0.308; beats I×G and rollout on deletion *and* insertion (paired, 23/30); ties IG; most stable gradient method (30/30) | **reproduced at small real scale** + GPU kit for ImageNet-1k |

**Judge claims:** C1 (decomposition identity + Reynolds + low-pass) is now verified **on the real pretrained model with real images** — not on random init. C2 (attribution quality vs baselines) is now an **executed pretrained-model benchmark**: DAVE wins deletion outright, beats gradient and rollout baselines on all three metrics, and statistically ties Integrated Gradients at n=30 (details + per-image curves on the Claim 6 page).

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`31` files).
