# Beyond Softmax: A Natural Parameterization for Categorical Random Variables

✅✅⚪🟡⚪  **5 pts** — 2/5 full-credit  (verified, verified, inconclusive, toy, inconclusive)

[arXiv 2509.24728](https://arxiv.org/abs/2509.24728) · [OpenReview](https://openreview.net/forum?id=ClBpWdkPZd) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-catnat-beyond-softmax-repro)

## Scoreboard — both claims (measured vs paper)

| # | Claim (paper) | Decisive measured numbers (this repro) | Paper target | Evidence |
|---|---|---|---|---|
| 1 | catnat FIM is **diagonal** vs softmax **dense**; better conditioning aids gradient descent | catnat off-diag/diag = **6.6e-17** exact, **5.6e-4** Monte-Carlo (N=2e6) vs softmax **0.354**; Corollary 4.3 Gᵢᵢ/P(aᵢ) = **0.2500** (max dev 5.7e-15); plain-GD to tol **105** iters (catnat ν) vs **182** (softmax) | (π/A)²=0.25; softmax dense O(1) | executed CPU, exact + MC |
| 2 | **consistently higher test performance** across GSL, VAEs, RL | VAE proxy, exact held-out **test NLL** (↓): softmax **25.63 ± 2.97** vs catnat σ **23.79 ± 1.48** vs catnat ν **23.27 ± 1.78**; both catnat beat softmax at **all 4** learning rates; ν also ~2× lower variance | Table 3: both catnat < softmax everywhere, ν best in majority | toy CPU proxy executed (~23 s) + full GPU job prepared |

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`6` files).
