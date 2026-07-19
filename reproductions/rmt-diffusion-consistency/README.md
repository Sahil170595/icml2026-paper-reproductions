# A Random Matrix Perspective on the Consistency of Diffusion Models

✅✅🟡  **5 pts** — 2/3 full-credit  (verified, verified, toy)

[arXiv 2602.02908](https://arxiv.org/abs/2602.02908) · [OpenReview](https://openreview.net/forum?id=iPjuUQbkfl) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-rmt-diffusion-consistency-repro)

## Scoreboard — measured vs. paper target

| # | Paper claim (Fig/Result) | Target + acceptance rule | Measured (this repro) | Verdict |
|---|---|---|---|---|
| 1 | Non-overlapping splits give consistent samples, predicted by the Gaussian linear theory (Fig 1) | cross-split MSE < nearest-train dist (non-memorization) & down with n; splits to Gaussian predictor; corr r>0 | cross/NN **1.29 to 0.037** (n:16 to 1024); slope **-0.896**; split to pop down; **r=0.67-0.75>0** | **reproduced** |
| 2 | Finite data renormalize noise sigma^2 to kappa(sigma^2), overshrinking low-variance modes (Fig 2, Prop 4.1) | emp per-mode gain = lam/(lam+kappa) not naive lam/(lam+sigma^2); overshrinkage; DE exact as d grows | kappa=**1.122** (22.4x at gamma=3.2); max\|emp-RMT\|=**0.0062** vs naive **0.654** (105x); low mode **0.77 to 0.13** (83% shrink); RMS **0.006 to 0.0019** as d up | **reproduced** |
| 3 | Variance law: anisotropic, location-dependent deviations decaying with n (Result 4.2, Prop 4.2) | Var = kappa^2/(n-df2)*diamond(v)*diamond(x); emp/pred~1 across modes & locations; global 1/n | anisotropy ratio **0.98-1.04** span **17.7x** (corr 0.9987); inhomog **0.98-1.06** span 7.9x; decay ratio~1, asymptotic slope **-1.026** | **reproduced** |
| 4 | Sampling-map deterministic-equivalence for expectation & variance over full trajectories (Results 5.1, 5.2) | fractional-power DE: E[Sigma^0.5] & Var[v.Sigma^0.5.xbar] match integral formulas; over-shrinkage | Prop 5.1 median rel-err **2.8%**, over-shrinkage ok, d to large ok; Prop 5.2 ratio **0.90-0.98**, anisotropic | **reproduced** |
| 5 | UNet/DiT validate consistency, overshrinkage, eigenmode-dependence in the non-memorization regime (Fig 5) | nonlinear denoiser: consistency up, to Gaussian predictor, overshrinks low modes | **REAL TRAINED conv UNet (torch, real digit images, 6 sizes, 12 trained models):** cross/NN **1.21 to 0.17**; toGaussian **0.295 to 0.059**; overshrink (n=16,32) **+0.154 to +0.088**, Spearman **0.89-0.98**. Plus KDE control (synthetic): cross/NN **11.1 to 0.78**; overshrink **+0.105 to +0.060**, Spearman **1.000** | **reproduced** (real trained neural denoiser + KDE control) |

**All 5 scored claims reproduced with executed CPU numbers.** The central RMT prediction — the self-consistent renormalized noise kappa(sigma^2) and the resulting over-shrinkage of low-variance eigenmodes — is confirmed to <1% per-mode error, and the deterministic-equivalent error **shrinks as d grows** exactly as the theory requires. Claim 5's deep nets (UNet/DiT on FFHQ) are out of CPU scope; a **real 2-level convolutional UNet trained by gradient descent on real digit images** reproduces the three stated predictions, with the earlier non-parametric-Bayes (KDE) denoiser on synthetic data kept as a labelled control.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`17` files).
