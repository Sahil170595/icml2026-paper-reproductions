# Maximum Likelihood Reinforcement Learning

🟡✅  **3 pts** — 1/2 full-credit  (toy, verified)

[arXiv 2602.02710](https://arxiv.org/abs/2602.02710) · [OpenReview](https://openreview.net/forum?id=EeuLO2BjFN) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-maxlik-rl-repro)

## Scoreboard — measured vs. paper target

| # | Paper claim (Abstract) | Target + acceptance rule | Measured (this repro) | Verdict |
|---|---|---|---|---|
| 1 | Compute-indexed family of sample-based objectives **interpolates** standard RL → exact maximum likelihood as sampling compute N grows | grad identity exact; family monotone in N; endpoints = standard RL (N=1) and exact ML (N→∞) | grad_RL = p·grad_ML to **1.4e-17**; per-problem MaxRL weight **p^(1/N)** rises monotonically p→1; Box-Cox value **p−1 → log p**; sample-based J_N ↗ log p | reproduced |
| 2 | Objectives admit a **simple, unbiased** policy-gradient estimator (non-differentiable sampling) | MC mean of estimator = exact ∇J_N within 95% CI (∀ K,N); a mis-specified control is biased | all 4 (K,N) cases: max\|z\|<2, every component in 95% CI, cos≈1.000; **1/N-biased control** rejected (max\|z\| 132–1434) | reproduced |
| 3 | Objectives **converge to maximum-likelihood** optimization in the infinite-compute limit | J_N → log E[r]; gap decays **O(1/N)** (log-log slope ≈ −1) | J_N → log p; slope **−1.007** ∈ [−1.2,−0.8]; N·Δ_N → **0.151** vs delta-method **0.147** (3%); value/grad gaps slope −0.985/−0.977 | reproduced |
| 4 | MaxRL **Pareto-dominates** existing methods in all models/tasks tested | mechanism: MaxRL improves coverage/pass@k without losing on the frontier vs standard RL/GRPO | matched-compute REINFORCE: MaxRL **dominates GRPO on 100%** of problems, pass@1 **0.842 vs 0.401**, hard-problems **66×**; allocation optimum: RL abandons **21/40**, ML covers all | mechanism reproduced (toy) |
| 5 | Up to **20× test-time scaling efficiency** vs GRPO counterpart | mechanism: k_RL/k_MaxRL to reach target success rate > 1 (direction) | measured efficiency **7× / 26× / 71× / 166×** at τ=0.5/0.6/0.7/0.8 (REINFORCE); 20× bracketed; standard RL cannot reach τ≥0.6 at the allocation optimum | mechanism reproduced (toy) |

**Definitions used throughout.** Softmax policy π=softmax(θ) over K discrete rollouts; per-rollout correctness r_a∈[r_min,1]; p=E_{a~π}[r_a] is the model's "likelihood of a correct rollout." **Standard RL** maximizes J_RL=E[r]=p; **exact maximum likelihood** maximizes J_ML=log E[r]=log p; the **MaxRL family** is J_N=E_{a_1..a_N~π}[log((1/N)Σ_i r_{a_i})] (the log of the sample-mean reward over a group of N rollouts, the canonical Monte-Carlo estimator of the log-marginal).

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`12` files).
