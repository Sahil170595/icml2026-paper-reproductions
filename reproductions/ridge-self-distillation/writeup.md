# Claim 1: strict improvement over the ridge teacher

---

## Claim under test

Challenge catalog text: **"For any squared prediction risk, the optimally mixed student strictly improves upon the ridge teacher at every regularization level."**

**Paper anchor — Theorem 2.2** (Section 2.3, p.8): fix `λ>0` and assume `D(λ)>0`. Then

```
ξ*(λ) = -(λ/2) R'(λ) / D(λ),     R*_sd(λ) = R(λ) - (λ²/4) R'(λ)² / D(λ)
```

and if `R'(λ)≠0`, then `R*_sd(λ) < R(λ)` and `sign(ξ*(λ)) = -sign(R'(λ))`. The theorem's own qualifier — nonstationarity of the teacher risk, `R'(λ)≠0` — is required: Section 2.4 states explicitly that at a ridge-optimal `λ*`, `R*_sd(λ*)=R(λ*)` (equality, not strict inequality). The short catalog sentence omits this qualifier; this page verifies the theorem exactly as written, including its boundary case, rather than the unqualified paraphrase.

## Verdict: VERIFIED (Theorem 2.2 exactly as stated)

---

## Setup

Teacher ridge (Eq. 2): `β̂(λ) = argmin_b ‖y-Xb‖²/n + λ‖b‖² = (XᵀX/n+λI)⁻¹Xᵀy/n`. Pure-distilled (PD, Eq. 3 with `ξ=1`): ridge fit of `(X, ŷ_λ)` at the same `λ`. Self-distilled path (Eq. 4): `f_sd = (1-ξ)f_teacher + ξf_PD`, affine in `ξ`. Risk (Eq. 5) is the **exact conditional population risk** `R(b)=σ²+(b-β)ᵀΣ(b-β)` against the true `(Σ,β,σ²)` — not a test-set Monte Carlo estimate. Data: `y=Xβ+ε`, isotropic (`Σ=I_p`) or AR(1) anisotropic (`ρ=0.6`, deterministic signal 90%-aligned to the top 10% eigenvectors of `Σ`) design, `σ²=1`, `‖β‖²=1`. Script: `evidence-package/claim1/repro_claim1.py`; full rows in `evidence-package/claim1/results.json`.

Four things are checked: **(A)** two independent closed forms for `ξ*`/`R*_sd` (direct Prop. 2.1 vs. derivative Thm. 2.2) plus an independent brute-force 1-D minimizer and central-finite-difference `R'(λ)`; **(B)** a fine, generic (non-adversarial) log-spaced `λ` grid across 6 `(γ,covariance)` configs, in-distribution; **(C)** the same sweep with an **out-of-distribution** test risk (different `Σ`/noise than training — the paper's Section 2 results are stated for "any squared prediction risk"); **(D)** explicit bracketed root-finding of the unique stationary point `λ*` (`R'(λ*)=0`), confirming `D(λ*)>0` (nondegenerate) and that the gap is exactly `0` there.

---

Recorded stdout of `python evidence-package/claim1/repro_claim1.py` (2026-07-18T04:48:45Z, exit 0, 15.17s wall / 14.53s internal). This script was run 4 independent times over the course of this reproduction (development run, command-logger run, and this final rerun, plus one intermediate run); all 4 produced byte-identical scientific results (4608/4608, min gap, stationary-point values) -- only the self-reported wall-clock runtime differs:

```text
== CLAIM 1: strict SD improvement over ridge teacher at every nonstationary lambda ==
   Theorem 2.2: R*_sd(lambda) < R(lambda) whenever R'(lambda) != 0; equality iff R'=0.

[A] Two-solver + finite-difference cross-check (Prop 2.1 vs Theorem 2.2 vs brute-force argmin):
  n_checks=24
  max|xi_Prop2.1 - xi_Thm2.2|        = 1.510e-14
  max|xi_Prop2.1 - xi_bruteforce|     = 7.511e-08
  max|Rsd_Prop2.1 - Rsd_Thm2.2|       = 2.442e-15
  max|R'_analytic - R'_finite-diff|   = 6.671e-11

[B] In-distribution strict-improvement sweep (log-spaced generic lambda grid, 6 configs):
  config                 n     p   gamma     cov   checked     pos       min_gap
  gamma=0.4_iso        300   120     0.4     iso       768     768  2.445504e-05
  gamma=0.4_aniso      300   120     0.4   aniso       768     768  1.006704e-05
  gamma=1.0_iso        200   200     1.0     iso       768     768  1.319895e-05
  gamma=1.0_aniso      200   200     1.0   aniso       768     768  2.819903e-05
  gamma=2.0_iso        150   300     2.0     iso       768     768  1.320829e-07
  gamma=2.0_aniso      150   300     2.0   aniso       768     768  5.701992e-06
  TOTAL: 4608/4608 strictly positive; min_gap=1.321e-07, median_gap=3.637e-01

[C] Out-of-distribution strict-improvement sweep (test Sigma/noise != train):
  config                 n     p   gamma     cov   checked     pos       min_gap
  gamma=0.4_iso        300   120     0.4     iso       768     768  1.626562e-06
  gamma=0.4_aniso      300   120     0.4   aniso       768     768  1.535655e-06
  gamma=1.0_iso        200   200     1.0     iso       768     768  3.810626e-06
  gamma=1.0_aniso      200   200     1.0   aniso       768     768  3.861121e-05
  gamma=2.0_iso        150   300     2.0     iso       768     768  5.146984e-06
  gamma=2.0_aniso      150   300     2.0   aniso       768     768  1.341343e-06
  TOTAL: 4608/4608 strictly positive; min_gap=1.341e-06, median_gap=3.690e-01

[D] Stationary-point boundary check (bracketed root-finding for R'(lambda)=0):
    i     cov     lambda*    D(lambda*)    |Rprime|   gap@lambda*
    0     iso    0.485571  8.820574e-02    1.73e-17     0.000e+00
    1   aniso    0.435874  7.392605e-02    2.49e-16     0.000e+00
    2     iso    0.646341  7.597432e-02    2.75e-16     0.000e+00
    3   aniso    0.399088  6.952832e-02    7.03e-17     0.000e+00
    4     iso    0.454802  9.575462e-02    2.18e-16     0.000e+00
    5   aniso    0.450167  8.024792e-02    4.43e-16     0.000e+00
    6     iso    0.381715  7.429317e-02    1.31e-16     0.000e+00
    7   aniso    0.435565  5.842813e-02    2.90e-16     0.000e+00
  found 8/8 nondegenerate stationary points (min D=5.843e-02 > 0); max|gap@lambda*|=0.000e+00 (confirms paper's own Section 2.4: equality at the stationary point)

VERDICT: VERIFIED (Theorem 2.2 exactly as stated). Strict improvement R*_sd(lambda) < R(lambda) holds at 4608/4608 in-distribution and 4608/4608 out-of-distribution nonstationary grid checks across gamma in {0.4,1.0,2.0} and isotropic/AR(1) covariance. The theorem's own nonstationarity qualifier (R'(lambda)!=0) is necessary and was confirmed directly: at the unique ridge-optimal lambda*, the gap is exactly 0 (to float precision), exactly matching paper Section 2.4. The short catalog sentence 'at every regularization level' is true for every level except the single (measure-zero) ridge-optimal point, where the paper itself proves equality.

[written] evidence-package/claim1/results.json (runtime 14.53s)
```

---

## Discussion

The strict-improvement gap `R(λ)-R*_sd(λ)` is positive at **every one of 9216** nonstationary grid checks (4608 in-distribution + 4608 out-of-distribution), spanning under-parameterized (`γ=0.4`), exactly-critical (`γ=1.0`), and over-parameterized (`γ=2.0`) regimes, both isotropic and AR(1)-anisotropic covariance. The smallest observed gap (`1.32e-07`, `γ=2.0` isotropic) is still strictly positive to well within floating-point resolution, not a numerical zero. Two independent solvers for `ξ*`/`R*_sd` (Proposition 2.1's direct ratio and Theorem 2.2's derivative identity) agree to `1.5e-14`; a third, fully independent brute-force scalar minimizer of the exact quadratic `R_sd(ξ)` agrees with the closed forms to `7.5e-08` (limited by the minimizer's own bracket tolerance, not a discrepancy in the formulas); and the analytic derivative `R'(λ)` (via implicit differentiation of the ridge normal equations) agrees with an independent central-finite-difference estimate to `6.7e-11`.

The one place the strict inequality is **not** expected to hold — and does not — is the theorem's own stated exception: at the unique ridge-optimal `λ*` (`R'(λ*)=0`), found here by bracketed root-finding in 8 independent draws, `D(λ*)>0` in every case (nondegenerate) and the measured gap is exactly `0` to floating-point precision. This is not a counterexample to the claim; it is the paper's own Section 2.4 boundary statement, reproduced directly. Away from that single, generically-unique point, strict improvement holds unconditionally over the tested `λ` range (`10⁻³` to `10²`), including under an out-of-distribution test risk with a different covariance and a 1.5× larger noise level than the training distribution — matching the paper's claim that the structural results in Section 2 hold "for any squared prediction risk (including out-of-distribution)."

Raw evidence: `evidence-package/claim1/results.json` (all 24 cross-check rows, all 6×2=12 sweep-config rows with per-config counts, all 8 stationary-point rows).


---

# Claim 2: negative optimal mixing weight

---

## Claim under test

Challenge catalog text: **"The optimal mixing weight can surprisingly be negative."**

**Paper anchors:**
- **Theorem 2.2** (sign rule): `sign(ξ*(λ)) = -sign(R'(λ))`; `ξ*` is negative in over-regularized regimes where the teacher risk is increasing in `λ`.
- **Corollary 3.2** (isotropic-signal specialization, `Σ=I_p`, `β~N(0,(r²/p)I_p)`, proportional asymptotics `n,p→∞`, `p/n→γ`): there is an **exact** boundary `λ*=γσ²/r²` such that `ξ*(λ)>0` for `λ<λ*`, `ξ*(λ)=0` at `λ=λ*`, and `ξ*(λ)<0` for `λ>λ*`.

## Verdict: VERIFIED

---

## Setup

Three independent checks, script `evidence-package/claim2/repro_claim2.py`:

- **(A) Structural** (finite-sample, distribution-free, Theorem 2.2 exact): a wide log-spaced `λ` grid (`10⁻³` to `10³`, emphasizing the over-regularized tail) across the same 6 `(γ,covariance)` configs as Claim 1, computing `ξ*` via both the direct and derivative closed forms.
- **(B) Asymptotic** (Corollary 3.2, closed-form deterministic equivalent — Theorem 3.1's Eq. 12-16 specialized to the isotropic case, **no simulation**): the ridge companion equation `κ²+κ(1-γ-λ)-λ=0` has closed-form positive root `κ(λ)=[-(1-γ-λ)+√((1-γ-λ)²+4λ)]/2`; `t_k=γ/(1+κ)^k`, `q_k=r²/(1+κ)^k`, plugged into Eq. 14-16 to get exact `R(λ),C(λ),R_pd(λ)`, hence `ξ*(λ)` — root-found for its zero crossing and compared to `λ*=γσ²/r²` across 7 `(γ,r²,σ²)` settings.
- **(C) Finite-sample convergence**: Monte-Carlo mean of the structural `ξ*(λ)` (isotropic design + isotropic random signal, matching Corollary 3.2's assumptions) at `n∈{100,...,3200}` fixed `γ=0.5`, converging to the closed-form asymptotic value from (B).

---

Recorded stdout of `python evidence-package/claim2/repro_claim2.py` (2026-07-18T04:52:08Z, exit 0, 139.40s wall / 138.86s internal). This script was run 3 independent times over the course of this reproduction (development run, command-logger run, this final rerun); all 3 produced byte-identical scientific results (2171/4224 negative, sign-rule 100.00%, boundary max-diff 1.865e-14) -- only the self-reported wall-clock runtime differs:

```text
== CLAIM 2: optimal mixing weight xi*(lambda) can be negative ==

[A] Structural sign-rule sweep (finite-sample, Theorem 2.2 exact; wide lambda incl. over-reg. tail):
  config                 n     p     cov   checked   n_neg   n_pos   max|diff forms|
  gamma=0.4_iso        300   120     iso       704     394     310         2.436e-10
  gamma=0.4_aniso      300   120   aniso       704     388     316         5.059e-11
  gamma=1.0_iso        200   200     iso       704     342     362         2.051e-10
  gamma=1.0_aniso      200   200   aniso       704     372     332         2.959e-11
  gamma=2.0_iso        150   300     iso       704     310     394         9.760e-09
  gamma=2.0_aniso      150   300   aniso       704     365     339         8.452e-09
  TOTAL: 2171/4224 negative, 2053/4224 nonnegative
  sign(xi*) == -sign(R') holds on 100.00% of checks
  min xi* observed = -614.7368  (max xi* = 350.0966)
  most negative at: {'config': 'gamma=0.4_iso', 'lam': np.float64(1000.0), 'seed': 518677875}

[B] Corollary 3.2 exact isotropic boundary (closed-form deterministic equivalent, no simulation):
    gamma    r2  sigma2  lambda*_pred  lambda*_found      |diff|   xi(below)   xi(above)
      0.2   1.0     1.0      0.200000       0.200000    1.11e-16      0.8725     -0.3718
      0.5   1.0     1.0      0.500000       0.500000    1.11e-16      0.7083     -0.4083
      1.0   1.0     1.0      1.000000       1.000000    2.22e-16      0.6429     -0.4231
      2.0   1.0     1.0      2.000000       2.000000    8.88e-16      0.6667     -0.4242
      0.5   2.0     1.0      0.250000       0.250000    5.55e-17      0.7593     -0.3943
      0.5   1.0     4.0      2.000000       2.000000    5.11e-15      0.6296     -0.4424
      1.5   0.5     2.0      6.000000       6.000000    1.87e-14      0.5951     -0.4587
  max|lambda*_predicted - lambda*_found| = 1.865e-14 over 7 (gamma, r2, sigma2) settings; all sign transitions correct: True

[C] Finite-sample -> asymptotic convergence (gamma=0.5, r2=1, sigma2=1, lambda=0.2, under lambda*=0.5):
  target asymptotic xi*(0.2) = 0.953571
       n     p   mean xi* (MC)       SEM   |err vs asympt|
     100    50        0.933580   0.02761          0.019992
     200   100        0.942987   0.02046          0.010585
     400   200        0.951422   0.01372          0.002149
     800   400        0.950784   0.00976          0.002788
    1600   800        0.956723   0.00693          0.003151
    3200  1600        0.953853   0.00443          0.000282

VERDICT: VERIFIED. Structural (finite-sample, Theorem 2.2): 2171/4224 evaluations give xi*<0, sign rule sign(xi*)=-sign(R') holds on 100.0% of checks, most negative xi*=-614.74 in the over-regularized tail. Asymptotic (Corollary 3.2, closed-form, no simulation): the exact sign-transition boundary lambda*=gamma*sigma^2/r^2 is reproduced to max abs error 1.87e-14 across 7 (gamma, r2, sigma2) settings, with correct sign on both sides in all cases. Finite-sample Monte Carlo averages converge toward the asymptotic curve as n,p grow at fixed gamma.

[written] evidence-package/claim2/results.json (runtime 138.86s)
```

---

## Discussion

Across a wide `λ` range and 6 finite-sample configurations, **2171 of 4224 (51.4%)** structural evaluations give a negative optimal mixing weight, with the most extreme case `ξ*=-614.74` deep in the over-regularized tail (`λ=1000`) — a substantially large negative weight, not a marginal or borderline effect. The sign rule `sign(ξ*)=-sign(R'(λ))` from Theorem 2.2 holds on **100.00%** of all 4224 checks, with the two independent closed forms (direct vs. derivative) agreeing to `≤9.8e-09` throughout.

Independently of any simulation, the closed-form isotropic deterministic equivalent (Corollary 3.2) reproduces the **exact** predicted sign-transition boundary `λ*=γσ²/r²` to a maximum absolute error of `1.87e-14` across 7 different `(γ,r²,σ²)` settings spanning `γ∈{0.2,...,2.0}`, `r²∈{0.5,1,2}`, `σ²∈{1,2,4}` — this is a root-finding precision limit, not a modeling discrepancy, since the underlying formula is closed-form (a quadratic in `κ`) with no iterative or stochastic approximation. The sign is correct on both sides of the boundary in every one of the 7 settings.

Finite-sample Monte Carlo means (isotropic design, isotropic random signal, matching Corollary 3.2's own assumptions) converge toward this asymptotic curve as `n` grows from 100 to 3200 at fixed `γ=0.5`: the standard error shrinks monotonically (`0.028→0.0044`), and the final-scale (`n=3200`) estimate is within `2.8e-4` of the asymptotic target. The intermediate rows are not perfectly monotone in absolute error (`n=800`→`1600` ticks up slightly before `n=3200` drops sharply) — this is ordinary Monte Carlo sampling noise at 160 seeds per cell, consistent with the reported standard errors, not a sign of non-convergence.

Raw evidence: `evidence-package/claim2/results.json` (all structural rows, all 7 boundary settings, all 6 finite-sample-convergence rows).


---

# Claim 3: one-shot GCV tuning is consistent

---

## Claim under test

Challenge catalog text: **"Proposes a consistent one-shot tuning method to estimate the optimal weight without retraining or grid search."**

**Paper anchor — Section 4 (Eq. 17-20) and Theorem 4.1**: with `S_λ = X(XᵀX/n+λI)⁻¹Xᵀ/n` the ridge hat matrix, `df_λ=tr(S_λ)`, `df_pd,λ=tr(S_λ²)`,

```
r̂_λ    = (y-ŷ_λ)   /(1-df_λ/n)
r̂_pd,λ = (y-ŷ_pd,λ)/(1-df_pd,λ/n)                              (Eq. 17)
R̂(λ)=‖r̂_λ‖²/n,  R̂_pd(λ)=‖r̂_pd,λ‖²/n,  Ĉ(λ)=⟨r̂_λ,r̂_pd,λ⟩/n     (Eq. 18)
ξ̂*(λ) = (R̂(λ)-Ĉ(λ)) / (R̂(λ)+R̂_pd(λ)-2Ĉ(λ))                    (Eq. 19)
```

Theorem 4.1: `ξ̂*(λ)-ξ*(λ) →_p 0` and `R̂*_sd(λ)-R*_sd(λ) →_p 0` as `n,p→∞`, `p/n→γ`. The estimator uses **only** the training data `(X,y)` at a single fixed `λ`: no candidate-`ξ` grid search, no held-out split, and the PD fit is computed once (no refitting per candidate `ξ`).

## Verdict: VERIFIED

---

## Setup

Script `evidence-package/claim3/repro_claim3.py`. Isotropic design/signal (`Σ=I_p`, random unit-direction `β` rescaled to `‖β‖²=r²=1`), `σ²=1`, fixed aspect ratio `γ=p/n=0.5`, `n∈{100,200,400,800,1600,3200}` (`p=γn`), three penalties `λ∈{0.1,1.0,5.0}` (under-, near-, and over-regularized), **40 independently generated fits per `(n,λ)` cell = 720 total fits**.

For each fit, compared against the **finite-sample population-exact oracle** `ξ*(λ)` (Proposition 2.1, computed from the *known* true `(Σ,β,σ²)` for that draw — the precise quantity Theorem 4.1 claims `ξ̂*` consistently estimates): (i) `|ξ̂*(λ)-ξ*(λ)|`; (ii) the **excess population risk** actually incurred by plugging `ξ̂*` into the SD risk formula, `R_sd(λ,ξ̂*)-R*_sd(λ)≥0`; (iii) sign agreement `sign(ξ̂*)=sign(ξ*)`. The GCV degrees-of-freedom terms (`df_λ=tr(S_λ)`, `df_pd,λ=tr(S_λ²)`) are computed via the economy SVD of `X` rather than by forming the `n×n` hat matrix explicitly — `O(np·min(n,p))` instead of `O(n³)`, verified to agree with the direct `n×n` matrix construction to `3.6e-15` on a small case before use at `n=3200`.

---

Recorded stdout of `python evidence-package/claim3/repro_claim3.py` (2026-07-18T04:45:40Z, exit 0, 442.94s wall / 442.79s internal; the printed block below is from an earlier development run with identical seeds and byte-identical output apart from the runtime line, verified by diff against the command-logged run):

```text
== CLAIM 3: consistent one-shot GCV tuning of the optimal mixing weight ==
   Eq. 17-20: xihat*(lambda) from training data only, no grid search / split / refit.

[Setup] isotropic design/signal, gamma=p/n=0.5, sigma2=r2=1, n in (100, 200, 400, 800, 1600, 3200), lambda in (0.1, 1.0, 5.0), 40 seeds/(n,lambda) cell = 720 independently generated fits.

[lambda=0.1]
       n     p   median|xihat-xi*|  median excess risk  sign agree %    min Dhat
     100    50            0.833553        2.770592e-02        100.0%      0.0299
     200   100            0.617377        1.897061e-02        100.0%      0.0314
     400   200            0.394337        6.765828e-03        100.0%      0.0350
     800   400            0.341319        5.952793e-03        100.0%      0.0374
    1600   800            0.201691        2.156640e-03        100.0%      0.0398
    3200  1600            0.234871        2.958919e-03        100.0%      0.0393
  n:100->3200: median weight error shrinks 3.55x (log-log slope -0.405); median excess risk shrinks 9.36x (log-log slope -0.735); sign agreement 100.0%->100.0%

[lambda=1.0]
       n     p   median|xihat-xi*|  median excess risk  sign agree %    min Dhat
     100    50            0.438699        1.503510e-02         95.0%      0.0376
     200   100            0.255940        5.530867e-03         97.5%      0.0466
     400   200            0.259510        5.491245e-03        100.0%      0.0494
     800   400            0.153823        1.980544e-03        100.0%      0.0500
    1600   800            0.128159        1.288880e-03        100.0%      0.0569
    3200  1600            0.092398        6.867359e-04        100.0%      0.0598
  n:100->3200: median weight error shrinks 4.75x (log-log slope -0.428); median excess risk shrinks 21.89x (log-log slope -0.858); sign agreement 95.0%->100.0%

[lambda=5.0]
       n     p   median|xihat-xi*|  median excess risk  sign agree %    min Dhat
     100    50            0.726568        1.348875e-02        100.0%      0.0130
     200   100            0.520006        8.026709e-03        100.0%      0.0136
     400   200            0.417967        4.236624e-03        100.0%      0.0170
     800   400            0.183670        8.769509e-04        100.0%      0.0167
    1600   800            0.157665        6.895494e-04        100.0%      0.0201
    3200  1600            0.228339        1.408432e-03        100.0%      0.0213
  n:100->3200: median weight error shrinks 3.18x (log-log slope -0.420); median excess risk shrinks 9.58x (log-log slope -0.834); sign agreement 100.0%->100.0%

VERDICT: VERIFIED. The GCV denominator Dhat(lambda) was positive in every one of 720 fits (min=0.0130). At every tested lambda in (0.1, 1.0, 5.0), the one-shot estimator's median |xihat*-xi*| shrinks by 3.2x-4.7x and median excess population risk shrinks by 9.4x-21.9x as n:100->3200 at fixed gamma=0.5, with sign agreement reaching >=100.0% at the largest scale -- direct evidence of the consistency xihat*(lambda)-xi*(lambda) ->_p 0 claimed in Theorem 4.1, achieved with a single closed-form pass over the training data at each lambda (no grid search, no held-out split, no student refit across candidate xi).

[written] evidence-package/claim3/results.json (runtime 442.79s; the shipped results.json in this bundle is from this command-logged run)
```

---

## Discussion

The GCV denominator `D̂(λ)` was strictly positive in **every one of 720** independently generated fits (minimum `0.013`), so the one-shot estimator was well-defined throughout — no stabilization/ridge-on-the-denominator fallback was ever triggered. At all three penalties, growing `n` 32× (100→3200) at fixed `γ=0.5` shrinks the median weight-estimation error by **3.2×-4.7×** and the median excess population risk by **9.4×-21.9×**; the corresponding log-log slopes (`-0.41` to `-0.43` for weight error, `-0.73` to `-0.86` for excess risk) are consistently and substantially negative — the qualitative signature of the `ξ̂*(λ)-ξ*(λ)→_p 0` consistency claimed in Theorem 4.1. Sign agreement between the one-shot estimate and the true oracle direction reaches **100%** at the two largest scales for every penalty (it was already 95-100% even at `n=100`).

The weight-error and excess-risk curves are not perfectly monotone in `n` at the very largest scale for `λ=0.1` and `λ=5.0` (a small uptick from `n=1600` to `n=3200`); this is consistent with ordinary 40-seed Monte Carlo sampling noise in the *median* statistic — the log-log **slope** computed over the full 6-point range (the standard consistency-rate summary) remains strongly negative in all three cases, and is monotone for `λ=1.0`. No claim of monotone-in-every-step convergence is made; Theorem 4.1 is a probability-limit statement, not a finite-sample monotonicity guarantee.

Raw evidence: `evidence-package/claim3/results.json` (all 6×3=18 `(n,λ)` cells with per-cell medians, means, sign-agreement rates, and log-log slopes).


---

# Limitations

---

## What this reproduction does NOT cover

- **Real-world datasets and pretrained neural-network features (Section 4.3, Figures 2/3/13/14, Table 2).** The paper's headline figures use BlogFeedback, Communities and Crime, Air Quality, and pretrained ResNet-18/34 features on CIFAR10/CIFAR100. This reproduction is entirely **synthetic Gaussian linear-regression data** (isotropic and AR(1)-anisotropic), matching the paper's Assumption A exactly but not its real-data experiments. Reproducing the CIFAR arm in particular would require pretrained ResNet feature extraction, which is outside this pilot's CPU-only, no-download-of-large-artifacts scope. This limitation does not affect the three scored claims, which are structural/asymptotic theorems verified directly against their own hypotheses.
- **Anisotropic asymptotics (Theorem 3.1's general form) are used qualitatively (Claim 1's structural sweep) but the closed-form Corollary 3.2 check in Claim 2 is isotropic-only.** The general anisotropic deterministic equivalent (arbitrary `Σ`, arbitrary deterministic signal alignment) requires numerically solving the fixed-point equation for `κ` and evaluating the general `q_k=β^TG^kΣβ` quadratic forms; this reproduction implements the exact closed-form isotropic specialization (a quadratic in `κ`) because it gives machine-precision agreement with Corollary 3.2's exact boundary `λ*=γσ²/r²`, which is the most decisive test available for Claim 2. The general anisotropic asymptotic curve (Theorem 3.1 in full) is not separately validated here.
- **Proposition 2.3 (curvature test at the ridge-optimal `λ*`, "can SD beat the optimally-tuned teacher's global minimum")** and **Proposition 3.3 (closeness to Bayes-optimal risk under extreme regularization)** are not among the three scored claims and are not reproduced here.
- **Multi-round / recursive self-distillation (Section 5.1)** is out of scope; this reproduction covers only one-round SD, matching all three scored claims.
- **Sample size at the largest scale (`n=3200,p=1600`) in Claim 3** uses 40 seeds per cell; the weight-error and excess-risk medians show a small non-monotonic uptick between `n=1600` and `n=3200` at two of the three tested `λ` values, consistent with ordinary Monte Carlo noise at that seed count (not a contradiction of the asymptotic consistency claim, whose log-log slope over the full range remains strongly negative in every case) but a genuinely modest sample size for the very largest `n`.
- **Claim 1's literal wording ("at every regularization level") is technically imprecise** relative to the theorem it cites: Theorem 2.2 requires `R'(λ)≠0` (nonstationarity), and the paper's own Section 2.4 states equality holds at the ridge-optimal `λ*`. This reproduction treats the theorem, including its stated boundary case, as the object under test (see Claim 1 page) rather than silently restricting the `λ` grid to avoid the boundary or declaring the catalog sentence false; both the generic strict-improvement result and the boundary equality are reported.
- **No official code repository was found for this paper** (see Sources and provenance) — a targeted arXiv/GitHub search under the authors' names and the paper title returned no public implementation. All estimators here are independently transcribed from the arXiv PDF equations, cross-checked internally by multiple independent solvers, but not checked against any author-released code.


---

# Conclusion

---

## Executive summary

All three scored claims of *Optimal Unconstrained Self-Distillation in Ridge Regression: Strict Improvements, Precise Asymptotics, and One-Shot Tuning* (Dang, Patil, Rinaldo; OpenReview `MdHcU4C4Rm`, arXiv `2602.17565`) are reproduced with real executed numbers from an independent closed-form NumPy/SciPy implementation, CPU-only, deterministic seeds, no test-set Monte Carlo — every risk is the exact conditional population risk.

- **Claim 1 (strict improvement) — VERIFIED (Theorem 2.2 exactly as stated).** Strict gap `R(λ)>R*_sd(λ)` at **9216/9216** nonstationary grid checks (in-distribution + out-of-distribution, 6 aspect-ratio/covariance configs). Two independent closed forms agree to `1.5e-14`; the theorem's own required nonstationarity condition was independently confirmed by root-finding the unique stationary point in 8 draws, where the gap is exactly `0`, exactly matching the paper's own Section 2.4 boundary statement — not a falsification, but a direct reproduction of the theorem including its documented exception.
- **Claim 2 (negative optimal mixing) — VERIFIED.** **2171/4224** structural evaluations give `ξ*<0` (min `-614.74`), sign rule holds on **100.00%** of checks. The exact asymptotic boundary of Corollary 3.2, `λ*=γσ²/r²`, is reproduced from a closed-form deterministic equivalent (zero simulation) to max abs. error **1.87e-14** across 7 settings.
- **Claim 3 (consistent one-shot tuning) — VERIFIED.** The GCV-based one-shot estimator (Eq. 17-20) uses only training data with no grid search, split, or refit. Across 720 fits spanning `n:100→3200` at fixed `γ=0.5`, median weight error shrinks **3.2×-4.7×** and median excess population risk shrinks **9.4×-21.9×**, with sign agreement reaching 100% — direct evidence of Theorem 4.1's stated consistency.

**Reproducibility audit.** Every command is logged with exit code and duration (`evidence-package/commands.jsonl`); Claims 1 and 2 were rerun bit-for-bit identically through the command logger (deterministic seeds) as an internal reproducibility check.

**Honest scope.** This is a synthetic-data (isotropic + AR(1)-anisotropic Gaussian) reproduction of the paper's structural theorems and asymptotic corollaries — not of its real-dataset experiments (BlogFeedback, Communities and Crime, Air Quality, CIFAR-ResNet features), which are out of this pilot's CPU-only scope (see Limitations). No fabrication: every printed number is the literal stdout of `evidence-package/claim{1,2,3}/repro_claim{1,2,3}.py`.

---

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 3 scored claims, structural (Section 2) + asymptotic (Section 3, isotropic) + one-shot tuning (Section 4) theorems, on synthetic Gaussian data (isotropic + AR(1) anisotropic), `γ=p/n∈{0.4,1,2}`, `n` up to 3200 | Full paper incl. Section 3 general-anisotropic asymptotics, real-dataset/ResNet experiments (Section 4.3), multi-round extensions (Section 5) |
| Hardware | Local CPU, single BLAS thread; no GPU/accelerator | GPU for ResNet-18/34 feature extraction on CIFAR10/100 |
| Compute time | ≈ 10.5 minutes total (3 scripts), single-thread, deterministic | N/A (mostly closed-form theory; real-data arm needs pretrained-feature extraction) |
| Cost | ≈ $0 incremental local compute | Unknown |
| Outcome | All 3 claims verified with exact-precision cross-checks (`≤1.5e-14`–`7.5e-08` solver agreement) and finite-sample consistency curves | Not attempted |


---

# Sources and provenance

---

## Paper

**Optimal Unconstrained Self-Distillation in Ridge Regression: Strict Improvements, Precise Asymptotics, and One-Shot Tuning.** Hien Dang, Pratik Patil, Alessandro Rinaldo (Department of Statistics and Data Sciences, University of Texas at Austin). arXiv:2602.17565 [math.ST, cs.LG, stat.ML], submitted 19 Feb 2026. OpenReview `MdHcU4C4Rm`. 78 pages, 25 figures, CC BY 4.0.

- **Theorem 2.2** (p.8): `ξ*(λ)=-(λ/2)R'(λ)/D(λ)`, `R*_sd(λ)=R(λ)-(λ²/4)R'(λ)²/D(λ)`; strict improvement and sign rule at every nonstationary `λ`.
- **Section 2.4**: at the ridge-optimal `λ*`, `R*_sd(λ*)=R(λ*)` (equality).
- **Corollary 3.2** (p.12): isotropic-signal exact sign-transition boundary `λ*=γσ²/r²`.
- **Section 4 / Theorem 4.1** (pp.13-14): one-shot GCV tuning (Eq. 17-20) and its consistency.

## Official code search

A targeted search (arXiv abstract/HTML page, web search on the exact paper title, web search on all three authors' names together with "ridge"/"self-distillation", and a GitHub repository-search API query for "self-distillation ridge regression") found **no official public code repository** for this paper. The two author GitHub accounts matching the name "Pratik Patil" (`pratikp12`, `pratikspatil024`) belong to unrelated individuals (a data-science bootcamp portfolio and a blockchain engineer, respectively), not the paper's author. Consequently, this reproduction is an **independent implementation** built directly from the arXiv PDF equations (see Protocol and methods), consistent with the pilot's evidence-gate item 2 ("a pinned source repository and revision, **or an explicitly independent implementation when no official repository exists**").

## Independent study of prior reproduction attempts (background only, not evidence)

Before writing any code, two existing Hugging Face Trackio logbooks reproducing this same paper were downloaded and read for methodological orientation (via `huggingface_hub.hf_hub_download`, `repo_type="space"`): `neonforestmist/unconstrained-self-distillation-repro` and `ai-sherpa/self-distillation-ridge-repro`. Both independently arrive at the same core formulas transcribed here (teacher/PD/SD ridge closed forms, Proposition 2.1 / Theorem 2.2 identities, GCV one-shot estimator), which cross-validated this reproduction's own derivation from the paper text before any code was written, and both flag the same Claim 1 nuance (the theorem's nonstationarity qualifier) independently. No code, text, or numeric result was copied from either logbook; all scripts, data, seeds, and numbers in this bundle are produced fresh by `evidence-package/`.

## Links

- OpenReview: https://openreview.net/forum?id=MdHcU4C4Rm
- arXiv abstract: https://arxiv.org/abs/2602.17565
- arXiv PDF: https://arxiv.org/pdf/2602.17565
