# Keep Everyone Happy: Online Fair Division of Numerous Items with Few Copies

✅✅  **4 pts** — 2/2 full-credit  (verified, verified)

[arXiv 2408.12845](https://arxiv.org/abs/2408.12845) · [OpenReview](https://openreview.net/forum?id=2XMLJj67yY) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-fair-division-repro)

## Scoreboard — measured vs paper target

| Claim | Metric | Paper target / rule | Measured (5 seeds) | Verdict |
|---|---|---|---|---|
| 1 — sub-linear regret, Thm 1 R_T=O(sqrt(dT logT)) | log-log regret exponent b | UCB b<1; sqrt-band [0.45,0.65] on tight instance | UCB b=**0.496** (tight) / **0.343** (OFD) | **VERIFIED** |
| 1 — average regret R_T/T -> 0 | R_T/T vs no-learning | UCB >=10x below Uniform | **68.7x** (3.49e-3 vs 0.240); Uniform b=0.997/1.000 | **VERIFIED** |
| 2 — learns linear utility theta* from limited feedback | recovery ‖theta_hat_T - theta*‖ | -> 0, decay exponent p>0 | 0.250 -> **0.066**, p=**0.307** | **VERIFIED** |
| 2 — predicts utilities of unseen item-agent pairs | held-out utility RMSE | -> noise floor sigma, << controls | **0.008** vs 0.301 (no-learn) / 0.309 (shuffled) | **VERIFIED** |
| 2 — learning beats no-learning allocation | OFD-UCB vs OFD-Uniform regret | UCB << Uniform | **61 vs 1088 (18x)** | **VERIFIED** |

Only **1 of N=10** item-agent utilities is observed per arriving item (the paper's "numerous items, few copies" regime), yet OFD-UCB's ridge model recovers theta* and predicts all pairs near-exactly. Per-claim pages give the full setup, controls, falsification conditions, and rerun commands; the evidence page lists environment, runtimes and checksums.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`6` files).
