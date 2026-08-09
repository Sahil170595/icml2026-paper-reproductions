# Characterization of Gaussian Universality Breakdown in High-Dimensional Empirical Risk Minimization

🟡⚪🟡⚪⚪  **2 pts** — 0/5 full-credit  (toy, inconclusive, toy, inconclusive, inconclusive)

[arXiv 2604.03146](https://arxiv.org/abs/2604.03146) · [OpenReview](https://openreview.net/forum?id=UHQDfvZBFi) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-gaussian-universality-repro)

## Scoreboard — measured vs target (real executed numbers)

| # | Paper claim | Measured quantity | Target / rule | Measured | Reproduced |
|---|---|---|---|---|---|
| 1 | Min–max characterization approximates the **mean** `mu_thetahat` and **covariance** `C_thetahat` of the ERM estimator under non-Gaussian data (Thm 4.3 / Sec 5) | rel. gap of (mean `m*`, cov `alpha*^2=tr(C_th)`, risk) vs deterministic-equivalent theory across Gaussian / Rademacher / Exponential designs | all gaps < 4% | **max gap 1.15%** (m 0.50%, cov 0.62%, risk 1.15%) | **yes** |
| 1b | …performance universality holds **even when score universality fails** (Sec 5) | ridge risk: true bimodal mixture vs matched-moment Gaussian; score non-Gaussianity | perf gap < 5% AND score non-Gaussian | **perf gap 0.28%**; score KS-p=3.2e-186, exkurt −0.49 | **yes** |
| 2 | Projection `x^T thetahat` ≈ convolution of `x^T mu*` with an independent **centered Gaussian of variance tr(C_th Cx)** (Thm 6.1) | KS(empirical score, `x^T mu_hat` ⊛ `N(0,alpha*^2)`); non-Gaussianity of projection | KS D < 0.02 AND projection non-Gaussian | **KS D=0.0030** (p=1.00); proj exkurt −0.52, KS-p=4.3e-102 | **yes** |

Deterministic-equivalent theory is independently validated: analytic `g=0.82843` vs direct-resolvent `g=0.82880` (0.05% agreement).

### Claim 1 detail — mean / covariance / performance universality
`p=200, n=400, gamma=0.5, lambda=0.5, sigma^2=0.25, M=2000`. Theory: `m*=0.58579`, `alpha*^2=tr(C_th)=0.087311`, `risk*=0.25888`.

| design law | mean m (meas / gap) | cov alpha^2 (meas / gap) | risk (meas / gap) |
|---|---|---|---|
| Gaussian | 0.58517 / 0.10% | 0.08747 / 0.19% | 0.25955 / 0.26% |
| Rademacher (exkurt −2) | 0.58560 / 0.03% | 0.08757 / 0.30% | 0.25930 / 0.16% |
| Exponential (skew 2, exkurt 6) | 0.58287 / 0.50% | 0.08785 / 0.62% | 0.26186 / 1.15% |

### Claim 2 detail — Theorem 6.1 convolution structure
`p=250, n=500, lambda=0.1, sigma^2=0.01, weights (0.3,0.7), M=4000, K=20000`.

| quantity | target | measured |
|---|---|---|
| KS(score, `x^T mu_hat` ⊛ `N(0,alpha*^2)`) | D < 0.02 | **D=0.0030** (p=1.00) |
| projection `x^T mu_hat` non-Gaussian | exkurt<0, KS-p≪1e-3 | exkurt=−0.520, KS-p=4.3e-102 |
| `alpha*^2 = tr(C_th Cx)` | (derived) | 0.011372 |
| conditional score Gaussian (5 fixed `x0`) | mean within 2 SE, var ratio ~1, KS-p>0.05 | all 5 pass (var ratio 1.000) |

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`5` files).
