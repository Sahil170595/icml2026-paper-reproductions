# Hierarchical Successor Representation for Robust Transfer

✅🔴⚪🔴🟡  **7 pts** — 3/5 full-credit  (verified, falsified, inconclusive, falsified, toy)

[arXiv 2602.12753](https://arxiv.org/abs/2602.12753) · [OpenReview](https://openreview.net/forum?id=txswvMHt4u) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-hierarchical-successor-representation-repro)

## Scoreboard — measured vs. paper target

| # | Paper claim | Target / acceptance rule | Measured (this repro) | Verdict |
|---|---|---|---|---|
| 1 | **HSR Bellman operator is a max-norm contraction** (Thm 3.1): ‖T^μM−T^μM′‖∞ ≤ γ‖M−M′‖∞ | contraction modulus ‖G‖∞ = E_μ[γ^τ] < 1 and ≤ γ; iteration → unique fixed point at that rate | ‖G‖∞ = **0.9253 / 0.9500 / 0.9278** (3 policies) all < 1 and ≤ γ=0.95; worst-case factor **= ‖G‖∞** (tight); iterate → fixed point, resid **9e-13**; HSR→SR resid **0.0** | **reproduced** |
| 2 | **Four-Room transfer: HSR row-features transfer significantly faster than SR** (Fig 2d) | HSR episodes-to-optimal on new goal G2 << SR (both << one-hot Raw) | Raw G2 = **146.8** (fails, cap 150); SR G2 = **54.5**; HSR G2 = **52.0**; HSR vs SR **p=0.854** | **partial** (SR & HSR both beat Raw; HSR **not** sig. faster than SR) |
| 3 | **HSR representation is robust to policy change** (Fig 2g): rel. change ‖M₁−M₂‖²/‖M₁‖² lower for HSR (p<0.001) | ρ_HSR ≪ ρ_SR across goal pairs | ρ_SR = **1.965**, ρ_HSR = **2.031** (SR/HSR ratio **0.97**), p=**0.74**, HSR-lower in **55%** of pairs | **not reproduced** |
| 4 | **HSR-NMF sparse basis** (Fig 4): NMF on HSR is sparse, elevated at bottlenecks, matches SR-SVD; NMF on SR collapses | HSR-NMF sparser + bottleneck-ratio ≫1; SR-NMF MSE ≫ SR-SVD | NMF Gini eHSR **0.624** < eSR **0.699**; bottleneck ratio **1.01** (no elevation); HSR-NMF MSE 0.0117 ≈ SR-SVD 0.0108; SR-NMF 0.0116 (**no collapse**) | **not reproduced** (only "HSR-NMF ≈ SR-SVD recon" holds) |
| 5 | **Scalable exploration with HSR** (Fig 5): HSR coverage > SR, gap **grows** with maze size | mean coverage gap > 0 and increasing in N | mean gap **+0.062** (helps at N=40,68,104; reverses at N=148,200); gap-vs-N slope **−1.1e-3** (should be > 0) | **partial** (helps small/mid mazes; gap does **not** scale) |

**Headline.** The paper's **theoretical backbone reproduces exactly**: the HSR Bellman operator `T^μM = B^μ + G^μM` is a max-norm contraction with modulus `‖G‖∞ = E_μ[γ^τ] ≤ γ` (Thm 3.1), value-iteration converges geometrically to the analytic fixed point `(I−G^μ)^{−1}B^μ`, and HSR **exactly** reduces to the standard SR when only primitive actions are used (residual 0.0). The four **empirical HSR-over-SR advantages do not reproduce** in this independent bounded tabular build: the faithfully-constructed expected-HSR is **numerically almost identical to the expected-SR** (Claims 3–4: room-block structure, reconstruction, and stability all ≈ equal), so the reported gains in transfer speed, NMF sparsity/bottleneck structure, and exploration scaling largely vanish. Where a shared mechanism is genuinely at work — temporally-extended (option) actions — we *do* see a benefit (Raw one-hot fails transfer while SR/HSR succeed; options raise coverage on small/mid mazes), but the **HSR-specific** advantage over the flat SR is absent.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`21` files).
