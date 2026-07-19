# Row-Stochastic Matrices Can Provably Outperform Doubly Stochastic Matrices in Decentralized Learning

✅✅✅  **6 pts** — 3/3 full-credit  (verified, verified, verified)

[arXiv 2511.19513](https://arxiv.org/abs/2511.19513) · [OpenReview](https://openreview.net/forum?id=GAQE4Wr53f) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-row-stochastic-decentralized-repro)

## Scoreboard (measured vs. paper target)

| # | Scored claim | Paper target | Measured | Verdict |
|---|---|---|--:|:--:|
| 1 | Row-stoch `W` self-adjoint, DS `W^ds` not → penalty amplifies consensus error | `W` DB residual `≈0`; `W^ds` `O(1)`; DS prefactor `>1` | `W` DB **1.1e-16**, self-adj **5.5e-16**; `W^ds` DB **≥2.59e-2**; DS prefactor up to **1.491** | **reproduced** |
| 2 | Row-stoch converges strictly faster (weighted Hilbert-space analysis) | `W` prefactor `=1`; `W^ds` inflated; `ρ_Λ<ρ_J` | `W` prefactor **1+9e-14**; `W^ds` **12/15** infl (real run **1.000/1.456**); `ρ_Λ<ρ_J` **6/6** | **reproduced** |
| 3 | Sufficient conditions + topology guidelines for the advantage | Thm 7.1 holds on degree-matched; conditional on random | degree-matched **6/6** (cond ⇔ faster); random **2/9** (conditional) | **reproduced** |

**All three scored claims are reproduced.** The row-stochastic mixing matrix satisfies exact weighted detailed balance (self-adjoint under `D_λ` at machine precision), pinning its weighted transient prefactor to 1; the doubly-stochastic matrix breaks detailed balance by `O(10⁻²–10⁻¹)` and pays a genuine `κ_λ` penalty (up to 1.491); and Theorem 7.1's sufficient condition holds — and predicts strictly-faster contraction — on the paper's degree-matched design (6/6) while only conditionally on adversarial weights (2/9), matching the paper's conditional framing.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`4` files).
