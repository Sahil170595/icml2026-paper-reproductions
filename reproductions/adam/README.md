# Towards Understanding Adam Convergence on Highly Degenerate Polynomials

✅✅  **4 pts** — 2/2 full-credit  (verified, verified)

[arXiv 2603.09581](https://arxiv.org/abs/2603.09581) · [OpenReview](https://openreview.net/forum?id=uYWVGk1Qt0) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-adam-repro)

## Scoreboard — measured vs target (both claims executed)

| # | Claim | Key measured quantity | Measured | Target | Verdict |
|---|---|---|--:|--:|:--|
| 1 | Adam converges locally linearly; GD/Momentum sub-linear | Adam ratio xₜ₊₁/xₜ (k=4) | 0.98202 | β₂^{1/(2(k−2))}=0.98202 | REPRODUCED |
| 1 | " | Adam loss-log slope (k=6) | −0.05443 | −0.05443 | REPRODUCED |
| 1 | " | GD log-log slope (k=4) | −0.4965 | −1/(k−2)=−0.5000 | REPRODUCED |
| 1 | " | final \|x\| Adam vs GD (k=4) | 1.8e−82 vs 7.1e−2 | Adam→0 | REPRODUCED |
| 2 | Acceleration from vₜ ⁄ gₜ² **decoupling** | rate-vs-β₂ regression slope (decoupled sweep) | 0.99998 (R²=1.0000) | 1.0 | REPRODUCED |
| 2 | " | vₜ/vₜ₋₁ at β₂=0.99 (Lemma 5.4) | 0.990000 | β₂=0.990 | REPRODUCED |
| 2 | " | max coupling ratio Rₜ=vₜ/gₜ² | 4.9e+97 | →∞ (decoupled) | REPRODUCED |
| 2 | **control**: force vₜ:=gₜ² (no decoupling) | rate-vs-β₂ regression slope | 0.000 (R²=0) | 0 (β₂ inert) | MECHANISM CONFIRMED |
| 2 | **control**: force vₜ:=gₜ² | converged? / tail max \|x\| | never / 1.0e−3 | linear conv. lost | MECHANISM CONFIRMED |

Claim 1: Adam is geometric at β₂^{1/(2(k−2))} to 5 decimals and hits machine zero; GD/Momentum follow t^{−1/(k−2)}. Claim 2: the geometric rate moves with β₂ exactly as β₂^{1/(2(k−2))} (regression slope 1.0, R²=1.0, k∈{4,6}); the decoupling fingerprints (vₜ/vₜ₋₁→β₂, Rₜ→∞) hold; and the coupled control that removes the decoupling destroys linear convergence and makes β₂ inert — isolating decoupling as the cause. Total experiments run: 2 (Claim 1 `artifacts/adam_repro.py`, Claim 2 `.trackio/logbook/evidence-package/claim2/repro_claim2.py`); ~2.6 s combined.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`3` files).
