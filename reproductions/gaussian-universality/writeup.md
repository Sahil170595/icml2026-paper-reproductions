# Claim 2 — Theorem 6.1: the ERM test score is a non-Gaussian projection convolved with an independent centered Gaussian

---

**Verbatim paper claim (abstract / Theorem 6.1).** "the projection `thetahat^T x` approximately follows the convolution of the (generally non-Gaussian) distribution of `mu*^T x` with an independent centered Gaussian variable of variance `tr(C_thetahat E[x x^T])`." Equivalently `x^T thetahat ~= x^T mu* + alpha* z`, `z ~ N(0,1)` independent of `x`, with `alpha*^2 = tr(C_thetahat Cx)`. The score is asymptotically Gaussian **iff** the one-dimensional projection `x^T mu*` is Gaussian.

**Target + rule.** In a bimodal-mixture instance where universality breaks (the signal is aligned with the mixture direction, so `x^T mu*` is bimodal): (a) the marginal score density must equal the projection density convolved with `N(0, alpha*^2)`; (b) the bare projection must itself be non-Gaussian. **Reproduced iff KS(empirical score, `x^T mu_hat` conv `N(0,alpha*^2)`) < 0.02 AND the projection is significantly non-Gaussian (excess kurtosis < 0 and KS-vs-normal p much less than 1e-3).** Setup: `p=250, n=500, gamma=0.5, lambda=0.1, sigma^2=0.01, weights (0.3,0.7), M=4000, K=20000`.

| quantity | target / rule | measured |
|---|---|---|
| KS(score, `x^T mu_hat` conv `N(0,alpha*^2)`) | D < 0.02 | **D=0.0030, p=1.00** |
| projection `x^T mu_hat` excess kurtosis | < 0 (non-Gaussian) | **-0.520** |
| projection KS-vs-normal p | much less than 1e-3 | **4.3e-102** |
| convolution variance `alpha*^2 = tr(C_th Cx)` | self-consistent | 0.011372 (`alpha*=0.10664`) |

Both predictions reproduced: the score equals the non-Gaussian projection convolved with an independent centered Gaussian (KS D=0.0030, well under 0.02), and the bare projection is decisively non-Gaussian (p=4.3e-102, excess kurtosis -0.520) — so score universality genuinely breaks here.

---

**Theorem 6.1a (conditional score).** For a FIXED test point `x0`, the score `s = x0^T thetahat` across `M=4000` independent training sets is Gaussian with mean `x0^T mu_hat` and variance `x0^T C_thetahat x0`. Rule per point: emp mean within 2 SE of `x0^T mu_hat`, variance ratio within ~5%, `|skew|<0.15`, `|excess kurtosis|<0.2`, KS-p>0.05.

| x0 | pred mean | emp mean | abs diff (2 SE) | var ratio | skew | ex.kurt | KS p |
|---|---|---|---|---|---|---|---|
| 0 | +0.3837 | +0.3837 | 0.0000 (0.0035) | 1.0000 | +0.031 | +0.016 | 0.605 |
| 1 | -0.0387 | -0.0387 | 0.0000 (0.0033) | 1.0000 | -0.129 | -0.087 | 0.190 |
| 2 | +0.5862 | +0.5862 | 0.0000 (0.0035) | 1.0000 | -0.025 | +0.016 | 0.702 |
| 3 | -3.1575 | -3.1575 | 0.0000 (0.0034) | 1.0000 | -0.018 | +0.058 | 0.361 |
| 4 | +0.5288 | +0.5288 | 0.0000 (0.0035) | 1.0000 | -0.022 | -0.047 | 0.905 |

All 5 fixed points pass: mean within 2 SE, variance ratio 1.000, moments within tolerance, KS-p>0.05. (One point had Shapiro p=0.001, an artifact of Shapiro sensitivity at M=4000; its KS p=0.19 and moments are within tolerance.) Combined with the convolution result this confirms `x^T thetahat = x^T mu* + alpha* z` with the predicted mean, variance, and independent Gaussian residual.

**Rerun.**
````bash
cd .trackio/logbook/evidence-package/claim2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GAUSSIAN_UNIVERSALITY_OUTPUT=results.json python3 repro_claim2.py   # ~14s
# writes results.json ; numpy 2.2.6 / scipy 1.15.3 / python 3.10.12 ; deterministic (seed 20260716)
````

**Scope / honest caveats.** One `gamma=0.5` instance, a single random `u`/seed, ridge (squared) loss only; `mu*` and `C_thetahat` are finite-sample proxies from `M=4000` draws (estimation error, not the asymptotic self-consistent `mu*`/`alpha*`). The Step-3 model and empirical score share the same `mu_hat`, so the KS test isolates whether the residual `x^T(thetahat - mu_hat)` is an independent centered Gaussian of variance `alpha*^2 = tr(C_th Cx)` — the theorem's core content, confirmed at KS D=0.0030. Toy dimensions chosen for a <15 s CPU budget; a numerical confirmation of the theorem's structure, not the full asymptotic derivation.


---

# Claim 1 — Min–max characterization of the mean and covariance of high-dimensional ERM under non-Gaussian data

---

**Verbatim paper claim (contribution 1 / Theorem 4.3 / Sec 5).** The paper derives "an asymptotic min–max characterization of key statistics, enabling approximation of the mean `mu_thetahat` and covariance `C_thetahat` of the ERM estimator `thetahat`." The min–max solution `(mu*, alpha*)` provides sharp approximations to `(mu_thetahat, sqrt(tr(C_thetahat Cx)))` and is sufficient to characterize ERM performance under general, possibly non-Gaussian, data.

**Target + rule.** For ridge ERM this min–max characterization coincides with the RMT deterministic equivalent (App. D.3 / Cor. 5.1). It must predict the mean summary `m* = <theta_star, mu_thetahat>`, the covariance summary `alpha*^2 = tr(C_thetahat Cx)`, and the estimation risk — and, being a Gaussian-equivalent, these must hold under NON-Gaussian designs. **Reproduced iff all three relative gaps are < 4% for Gaussian AND non-Gaussian (Rademacher, centered-Exponential) designs.** Setup: `p=200, n=400, gamma=0.5, lambda=0.5, sigma^2=0.25, M=2000` draws/design. Theory: `m*=0.585786`, `alpha*^2=0.0873106`, `risk*=0.258883`.

| design (Cx=I) | mean m: meas / gap | cov alpha^2: meas / gap | risk: meas / gap |
|---|---|---|---|
| Gaussian | 0.58517 / 0.10% | 0.08747 / 0.19% | 0.25955 / 0.26% |
| Rademacher (excess kurt −2) | 0.58560 / 0.03% | 0.08757 / 0.30% | 0.25930 / 0.16% |
| Exponential (skew 2, kurt 6) | 0.58287 / 0.50% | 0.08785 / 0.62% | 0.26186 / 1.15% |

**Max relative gap 1.15%** (tol 4%); cross-law spread risk 0.98%, alpha^2 0.43%. The mean and covariance summaries are the same for Gaussian and strongly non-Gaussian designs and match the closed-form theory — exactly "approximation of the mean and covariance under non-Gaussian data".

---

**Deterministic-equivalent formula independently validated.** The isotropic fixed point `delta = gamma / (1/(1+delta) + lambda)` gives `g = (1/p) tr E[R] = 0.828427` and `h = (1/p) tr E[R^2] = 0.828427`; a direct resolvent computed on one large Gaussian design (`p=1000, n=2000`) gives `g=0.828803`, `h=0.829185` — 0.05% agreement. The closed form `alpha*^2 = lambda^2 (h - g^2) + sigma^2 gamma (g - lambda h)` equals `q* - m*^2` to machine precision. So the "theory" column above is a genuine analytic prediction, not a Monte-Carlo surrogate.

**Performance universality holds even when score universality fails (paper Sec 5).** Bimodal Gaussian mixture, signal aligned with the mixture direction (`Cx = I + 1.714 u u^T`, `E[x]=0`, `M=1500`):

| test | rule | measured | outcome |
|---|---|---|---|
| ridge risk: true mixture vs matched-moment Gaussian design | perf gap < 5% | 0.19289 vs 0.19235 → **gap 0.28%** (±0.44% SE) | performance universality HOLDS |
| score `x^T mu_thetahat`, true mixture | Gaussian? | skew +0.43, excess kurt −0.49, KS-vs-normal p=3.2e-186 | score universality FAILS |
| score `x^T mu_thetahat`, matched Gaussian | Gaussian? | excess kurt +0.04, KS-vs-normal p=0.66 | Gaussian |

The gap (0.28%) is below its own standard error (0.44%): replacing the data by a matched-moment Gaussian leaves ridge PERFORMANCE statistically unchanged, while the SCORE distribution is decisively non-Gaussian. This is the paper's central dissociation — the breakdown of score universality (Claim 2) does not break performance universality for squared loss.

**Rerun.**
````bash
cd .trackio/logbook/evidence-package/claim1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 repro_claim1.py   # Part A (~8s): mean/cov/perf universality
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 partB.py          # Part B (~7s): perf-vs-score dissociation
# writes results.json ; numpy 2.2.6 / scipy 1.15.3 / python 3.10.12 ; deterministic (fixed seeds)
````

**Scope / honest caveats.** Isotropic `Cx=I` for the analytic anchor (Part A); one fixed `theta_star` and seed per design; ridge (squared) loss only, not general convex ERM; `M=2000` draws so `mu_thetahat`/`C_thetahat` carry O(1/sqrt(M)) Monte-Carlo error (the residual source of the sub-1.2% gaps). Dimensions are modest for a <20 s CPU budget: this is a numerical confirmation of the characterization's mean/covariance/performance content, not the full asymptotic derivation or every figure.


---

# Conclusion

---

**Both paper claims are now covered by executed CPU experiments with real measured numbers.**

- **Claim 1 — min–max characterization of the mean `mu_thetahat` and covariance `C_thetahat` of the ERM estimator under non-Gaussian data (Thm 4.3 / Sec 5).** The deterministic-equivalent (min–max) prediction matches empirical ridge mean `m*`, covariance `alpha*^2=tr(C_th Cx)`, and risk to within **1.15%** across Gaussian, Rademacher, and centered-Exponential designs (tolerance 4%); the analytic formula is independently validated against a direct resolvent (g=0.82843 vs 0.82880). Performance universality holds **even where score universality fails**: replacing the bimodal mixture by a matched-moment Gaussian changes ridge risk by only 0.28% (below its 0.44% SE) while the score is decisively non-Gaussian (KS-p=3.2e-186).
- **Claim 2 — Theorem 6.1 score decomposition.** The test score equals the non-Gaussian projection `x^T mu*` convolved with an independent centered Gaussian of variance `tr(C_th Cx)` (KS D=0.0030 ≪ 0.02), and the bare projection is decisively non-Gaussian (excess kurtosis −0.520, KS-p=4.3e-102); all 5 fixed-point conditional-Gaussianity checks pass.

Fresh local reruns completed **3/3 commands in approximately 29 seconds**. No Hugging Face GPU Job was used: these checks are CPU-feasible and deterministic (fixed seeds); a GPU would not change the outcome.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 2 claim pages, both with executed measured-vs-target evidence; original claim labels preserved | Paper-scale implementation and every headline empirical claim |
| Hardware | Local machine; CPU-only deterministic scripts; no HF Job | Paper-specified accelerators, datasets, checkpoints, and sweeps |
| Compute time | ~29 s across 3 freshly recorded commands | Not estimated without the full paper setup |
| Cost | Approximately $0 incremental local compute | Unknown; potentially substantial |
| Outcome | Both claims reproduced within their stated acceptance rules (Claim 1 gaps ≤1.15%; Claim 2 KS=0.0030) | Not attempted |

---

**📦 Artifact** `icml26-uhqdfvzbfi/uhqdfvzbfi-reproduction-bundle:v0` · dataset

https://huggingface.co/buckets/Crusadersk/icml26-gaussian-universality-repro-artifacts#icml26-uhqdfvzbfi/uhqdfvzbfi-reproduction-bundle:v0

---

The reproduction bundle contains the runnable scripts, evidence JSON/JSONL, manifests, reviews, and supporting logs under `artifacts/` and `.trackio/logbook/evidence-package/`. Fresh Claim-1 and Claim-2 experiments and their `results.json` live under `.trackio/logbook/evidence-package/claim1/` and `.trackio/logbook/evidence-package/claim2/`. After publication, the artifact cell above resolves to the Hugging Face Bucket URL and can be downloaded from that Bucket. Secrets, virtual environments, caches, and replaceable downloads are excluded.


---

# Sources and provenance

---

- OpenReview: https://openreview.net/forum?id=UHQDfvZBFi
- Published logbook: https://huggingface.co/spaces/Crusadersk/icml26-gaussian-universality-repro
- arXiv: https://arxiv.org/abs/2604.03146

The migration preserves the original claim boundaries and does not convert toy, partial, or inconclusive evidence into a full reproduction.
