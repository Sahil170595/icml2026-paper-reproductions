# Improved Convergence Analysis of Topology Dependence in Decentralized SGD

✅✅  **4 pts** — 2/2 full-credit  (verified, verified)

[arXiv 2606.09154](https://arxiv.org/abs/2606.09154) · [OpenReview](https://openreview.net/forum?id=pYI0WjV5iM) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-decentralized-sgd-topology-repro)

## Scoreboard (measured vs target)

| # | Scored claim | Headline measured numbers | Target / rule | Verdict |
|---|---|---|---|---|
| 1 | All eigenvalues of the mixing matrix affect the rate, not just the spectral gap | measured consensus `V_ss` vs full-spectrum `T_new`: slope **1.0004**, R^2 **0.999990**, log-log slope **0.9996**; vs spectral-gap `(1-p)/p`: R^2 **0.268**. Same-gap control: identical gap, **22.9x** spread in measured `V_ss` | slope ~= sigma^2=1, R^2>=0.99; gap metric worse; gap-only hypothesis falsified | **VERIFIED** |
| 2 | Novel analysis describes topology's effect on convergence better than prior work | real D-SGD consensus vs `T_new`: slope **0.01000** (= eta^2 sigma^2), R^2 **0.999997**; vs gap R^2 **0.257**. Heterogeneous D-SGD suboptimality vs `T_new` R^2 **0.844** vs gap **0.018**. Homogeneous robustness: loss spans **1.01x** while gap spans **10x** | full-spectrum predicts; spectral-gap fails and mis-ranks | **VERIFIED** |

Metrics compared (paper Sec 4.2 / 6.1): prior spectral-gap term `T_gap = (1-p)/p` with `p = 1 - max_{i>=2} lambda_i^2`
versus the paper's full-spectrum term `T_new = (1/n) sum_{i=2}^n lambda_i^2 / (1 - lambda_i^2)`.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`4` files).
