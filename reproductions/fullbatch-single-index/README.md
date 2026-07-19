# Full-Batch Gradient Descent Outperforms One-Pass SGD: Sample Complexity Separation in Single-Index Learning

✅✅  **4 pts** — 2/2 full-credit  (verified, verified)

[arXiv 2602.02431](https://arxiv.org/abs/2602.02431) · [OpenReview](https://openreview.net/forum?id=QItZDBVCT0) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-fullbatch-single-index-repro)

## Scoreboard — measured vs paper target

| # | Claim | Key measured numbers (real run) | Paper target | Status |
|---|---|---|---|---|
| 1 | Full-batch GD outperforms one-pass SGD (sample-complexity separation) | weak-recovery δ\*=n/d: **trunc 3.13–4.20 < plain 5.21–6.47 < one-pass SGD 10.6–11.7**; plain δ\*∝log d **slope 0.58, R²0.954**; at fixed δ=4 recovery **trunc 75–100 % vs plain 0–21 % vs SGD 0 %** | trunc Θ(d) ≪ plain,SGD Θ(d log d); δ\*∝log d (Fig 1c) | Separation confirmed (ratio-growth directional) |
| 2 | Strong recovery needs n≳d and T≳log d (squared loss, truncated quadratic) | T_recover∝log d **slope 7.25, R²0.977** (beats √d 0.891); recover @ n=40d (all d, min overlap 0.951), fail @ n=0.25d (all d, 0/4) | T≳C·log d/η and n≳d (Thm 4.1) | Reproduced |

**Claim 1** (uncovered before this pass) is now backed by a deterministic separation experiment: the truncated full-batch flow recovers at the smallest samples/dim, the plain-quadratic flow's threshold grows exactly like log d (Thm 3.1 / Fig 1c, computed via the exact gradient-flow limit), and a genuine one-pass online SGD needs ~3× more samples and collapses at fixed budget. **Claim 2** is the pre-existing, unchanged T∝log d / Θ(d)-threshold reproduction.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`3` files).
