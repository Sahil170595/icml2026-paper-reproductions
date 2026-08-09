# Design-Based Anytime-Valid Inference for Randomized Experiments with Delayed Outcomes and Staggered Entry

✅✅⚪⚪🟡  **5 pts** — 2/5 full-credit  (verified, verified, inconclusive, inconclusive, toy)

[arXiv 2603.25971](https://arxiv.org/abs/2603.25971) · [OpenReview](https://openreview.net/forum?id=FXWnvznHMW) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-anytime-valid-delayed-repro)

## Scoreboard (measured vs target)

| Claim | Test (paper anchor) | Measured | Target / rule | Verdict |
|---|---|---|---|---|
| 1 | Per-arm IPW error increment mean-zero, E[dM_k given past] (Thm 4.4) | max abs(z)=3.34; terminal E[M_T] z=+0.17 | ~0, abs(z)<=4 | PASS |
| 1 | Lemma 4.7 identity Cov(M_s,M_t)=Var(M_s), single arm | max abs(z)=2.89 (abs(ratio-1)=0.021) | holds, abs(z)<=4 | PASS |
| 1 | Difference error (Delta_hat - Delta) NOT a martingale (Prop 4.8) | max abs(z)=37.0 (Cov/Var ratio=1.35) | violated, abs(z)>=6 | PASS |
| 1 | Anytime-valid single-arm CS uniform miscoverage (Thm 4.6) | 0.44% | <= alpha=5% | PASS |
| 1 | Naive fixed-n CI uniform miscoverage under monitoring (single-look 95%) | 70.4% | > alpha (over-rejects) | PASS |
| 2 | Width ratio symmetric limit 2*sqrt(pi(1-pi)), pi=.25/.5/.75 | 0.8635 / 1.0000 / 0.8635 | 0.8660 / 1.0000 / 0.8660 (<=2%) | PASS |
| 2 | Width ratio asymmetric limit sqrt(1-pi), pi=.25/.5/.75 | 0.8760 / 0.7122 / 0.5000 | 0.8660 / 0.7071 / 0.5000 (<=2%) | PASS |
| 2 | Union bound strictly tighter, R_pi < 1 (asymmetric) | < 1 for all pi | R_pi < 1 | PASS |

**Both claims: real_verified.** Claim 1 - the Horvitz-Thompson (IPW) per-arm estimation error is a martingale wrt the single-arm event-time filtration F_t(w) (Def 4.3 / Thm 4.4); the treatment-effect error is not a martingale under any filtration (Prop 4.8); and the martingale structure yields anytime-valid confidence sequences (Thm 4.6), while a naive fixed-n CI over-rejects under optional stopping. Claim 2 - the union-bound confidence sequence is strictly tighter than the design-based variance upper bound when arms have asymmetric variance clocks (Prop 4.12), matching the closed-form limits 2*sqrt(pi(1-pi)) and sqrt(1-pi).

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`3` files).
