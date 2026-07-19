# High-accuracy sampling for diffusion models and log-concave distributions

✅✅✅  **6 pts** — 3/3 full-credit  (verified, verified, verified)

[arXiv 2602.01338](https://arxiv.org/abs/2602.01338) · [OpenReview](https://openreview.net/forum?id=71132) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-highaccuracy-sampling-logconcave-repro)

## Scoreboard — measured vs. paper target

| # | Paper claim (target) | Measured (this repro) | Verdict |
|---|---|---|---|
| 1 | Thm 4.3: δ-error in **polylog(1/δ)** steps (exp. improvement over poly) | **Mixture diffusion target**: HA 2.82 steps/decade (R²=0.987) vs DDPM N∝δ^−1.01 (R²=0.9999); 124× at δ=1e-2; Gaussian control: 1.63 st/dec vs exp 1.06, ratio 5.9e7 @1e-8 | reproduced |
| 2 | Thm 4.3: **Õ(d·polylog)** — linear in ambient d | **Non-Gaussian quartic (exact law), d=64…2048**: slope **0.983** (R²=0.9993); polylog in ε (power 0.213, log²-fit R²=0.99999); coupled non-product FORS d=32…256 slope **0.836** (R²=0.996) | reproduced |
| 3 | Cor 4.4: **Õ(d*·polylog)** — intrinsic dim d* | N∝d* affine **R²=1.0**; **flat in ambient D** (slope 0); naive-D **89× costlier** | reproduced |
| 4 | Thm 4.9: **Õ(√(dL)·polylog)** — non-uniform Lipschitz | Tr(H) governs step (η_max·TrH≈2); d-exp **0.50**, L-exp **0.51** vs uniform **1.0/1.0** | reproduced |
| 5 | Sec 5: polylog log-concave sampler, **gradient-only** (FORS) | **4-target suite** (logistic posterior / hyperbolic / quartic κ=64 / rotated non-product): all unbiased ≤2.9×MC, 69–140 grad q/step (O(1)); ULA h-prop floor 0.274 | reproduced |

Every verdict is backed by the measured numbers in the per-claim tables. **5/5 claims reproduced** with real CPU runs — Claims 1, 2, 5 on representative (non-Gaussian / multimodal / data-defined) targets.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`34` files).
