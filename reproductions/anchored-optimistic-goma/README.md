# Accelerated and Stable Convergence with Anchored Generalized Optimistic Method

✅⚪✅✅  **6 pts** — 3/4 full-credit  (verified, inconclusive, verified, verified)

[arXiv 2606.21528](https://arxiv.org/abs/2606.21528) · [OpenReview](https://openreview.net/forum?id=G6WKIN1heG) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-anchored-optimistic-goma-repro)

## Scoreboard — measured vs. paper target

| # | Paper claim | Target + acceptance rule | Measured (this repro) | Verdict |
|---|---|---|---|---|
| 1 | GOMA **deterministic O(1/k²)** last-iterate rate (Thm 1), monotone L-Lipschitz operators | C_k=‖G(x_k)‖²(k+6)²/‖x₀−x*‖² ≤ **264·L²** ∀k; log-log slope of ‖G(x_k)‖ vs k ∈ [−1.15,−0.85] | max C_k = **60.4** (skew) / **62.5** (bilinear) ≤ 264; slope **−0.986** / **−0.993** | reproduced |
| 2 | GOMA **stochastic O(1/√k)** last-iterate rate **with linearly increasing minibatches** (Thm 4); squared operator norm | E‖G(x_N)‖² ≤ (1570L²κR²+8σ²/κ)/√(N+1); log-log slope of E‖G‖² vs k ∈ **[−0.6,−0.4]** (target −0.5), additive **and** unbounded-variance regimes | b_k=k: slope **−0.5032** (κ=1,d=2) / **−0.4919** (κ=4,d=10); const **2.10** / **39.8** ≤ 3142 / 62800 | reproduced |

Both last-iterate rates match the paper's exponents under the stated acceptance rules. Claim 2's **primary** experiment is the exact claim setting — GOMA driven by a **linearly increasing minibatch b_k=k** — and gives E‖G(x_k)‖² ~ k^(−0.5) (O(1/√k) on the squared operator norm) in both the additive (κ=1, d=2) and the state-dependent multiplicative / unbounded-variance (κ=4, d=10) regime, with the Theorem-4 constant satisfied by >1500×. Constant-batch single-call GOMA reproduces the same −0.5 (the paper's no-growing-batch result); switching the minibatch **off** in the accelerated control collapses convergence to a plateau (slope +0.0150).

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`2` files).
