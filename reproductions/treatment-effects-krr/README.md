# Estimating Continuous Treatment Effects with Two-Stage Kernel Ridge Regression

✅✅  **4 pts** — 2/2 full-credit  (verified, verified)

[arXiv 2604.13410](https://arxiv.org/abs/2604.13410) · [OpenReview](https://openreview.net/forum?id=ziqS4yXFQX) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-treatment-effects-krr-repro)

## Scoreboard — measured vs paper target

| # | Claim | Paper target / acceptance rule | Measured (executed) | Verdict |
|---|---|---|---|---|
| 1 | Two-stage KRR adapts to the simpler induced effect from covariate averaging (rate set by 1-D target RKHS `H`, not the (d+1)-dim nuisance `F`) | two-stage log–log MISE slope within ±0.15 of −1 (smooth kernel) **and** ≥0.2 steeper than direct-`f*`, d∈{3,5}, n≥1000 | slope_h = **−1.017** (d=3), **−0.874** (d=5); steeper than direct by **+0.244 / +0.315**; MISE_h grows ×1.6 vs MISE_f ×3.6 from d=3→5 | **verified** |
| 2 | Fully data-driven model selection (Algorithm 2) is provably adaptive to overlap γ and kernel regularity | selected-λ MISE within a constant factor of the oracle-λ across all γ×kernel cells; selected rate ≈ oracle rate; no fixed λ uniformly good | worst-cell ρ = MISE_sel/MISE_oracle = **1.344** (≤1.5) over 6 cells; selected slope within **0.06 / 0.11** of oracle (RBF/Laplace); λ\* shifts, theory-floor fixed λ worse (**1.516**) | **verified** |

Total fresh compute ≈ 48 s across 2 commands. Per-claim detail, controls, and limitations are on the claim pages; exact commands, runtimes and sha256 are on the Evidence-and-rerun page.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`4` files).
