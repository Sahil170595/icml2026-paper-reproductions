# Minimum Distance Summaries for Robust Neural Posterior Estimation

✅✅🟡🟡⚪⚪  **6 pts** — 2/6 full-credit  (verified, verified, toy, toy, inconclusive, inconclusive)

[arXiv 2602.09161](https://arxiv.org/abs/2602.09161) · [OpenReview](https://openreview.net/forum?id=lq8fNVME8v) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-minimum-distance-summaries-repro)

## Scoreboard (measured, CPU-only)

| # | Scored claim | Decisive measured evidence | Verdict |
|---|---|---|---|
| 1 | Plug-in robust NPE; adapts test-time summaries independently of the pretrained NPE | Two trained NPEs (Gaussian **4,677** params, OUP **17,669** params). NPE tensor-state SHA-256 asserted **identical before and after every one of 300 + 250 = 550 adaptations**. Same frozen posterior queried twice per dataset (observed vs MDS summary). | **Reproduced** |
| 2 | Efficient RFF approximation; lightweight, model-free test-time adaptation | scikit-learn RFF, **K=512**, median-heuristic bandwidth; decoder mean-embedding MSE **8.5e-4** (Gaussian) / **6.5e-4** (OUP); PyTorch L-BFGS strong-Wolfe. Adaptation median **2.3-7.8 ms** (Gaussian). RFF-vs-exact MMD² mean gap **0.0012** (Gaussian, corr 0.97) / **0.0025** (OUP). | **Reproduced** |
| 3 | Substantial robustness gains, minimal overhead | Frozen-NPE posterior-mean RMSE reduction, mean over eps=0.1-0.4: **86.18%** (Gaussian, 50/50 wins every level) and **63.36%** (OUP, 44-45/50 wins). Adaptation costs single-to-tens of ms. | **Reproduced** |
| 4 | Theoretical guarantees for robustness are provided | Fail-closed audit of the pinned 1,197-line TeX (SHA `ff81fd97…`): **10 robustness + 4 consistency** assumptions counted; **8/8** dependency-DAG steps complete. Numerical: bounded summary influence **sup‖IF‖=1.83** (finite, redescending); posterior contraction slope **-0.491** (~ -1/2). | **Verified within stated scope** |

**Honest boundary:** at the severe Gaussian level eps=0.5 the single-start optimizer
degrades to **30.55%** reduction (40/50 wins) — reported, not discarded — matching the
paper's own warning that MDS can degrade under severe contamination.

---

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`12` files).
