# Claim 1: Claim 2 / Proposition 4.12. With boundary function (Theorem 4.6 form)

---

**Claim 2 (Proposition 4.12).** The union-bound confidence-sequence width ratio `R_pi(V0,V1) = [b(V0;a/2)+b(V1;a/2)] / b(V1/(1-pi)+V0/pi; a)` reproduces the two closed-form asymptotic limits, and the union bound is strictly tighter (`R_pi<1`) in the asymmetric regime. Executed by `artifacts/repro.py` (a=0.05, eta=1.0; limits taken at V=1e18). Every number is printed by a real run.

| pi | symmetric R_pi (V=1e18) | target 2*sqrt(pi(1-pi)) | rel err | asymmetric R_pi (V1=1e18) | target sqrt(1-pi) | rel err | R_pi<1 |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.863485 | 0.866025 | 0.29% | 0.875957 | 0.866025 | 1.15% | yes (V1>=1e4) |
| 0.50 | 1.000000 | 1.000000 | 0.00% | 0.712197 | 0.707107 | 0.72% | yes (V1>=1e2) |
| 0.75 | 0.863485 | 0.866025 | 0.29% | 0.500012 | 0.500000 | 0.00% | yes (V1>=1e2) |

**Verdict: real_verified.** Both asymptotic limits match within ~2% for pi in {0.25, 0.5, 0.75}, and the union bound is strictly tighter (`R_pi < 1`) in the asymmetric-clock regime. Comparison rule, robustness sweep over (a, eta), and full stdout are on the Evidence and rerun page. Rerun: `cd artifacts && python3 repro.py`.

---

**Paper claim.** Claim 2 / Proposition 4.12. With boundary function (Theorem 4.6 form)

**Paper anchor.** See the original experiment report

**Reproduction status.** `bounded local check`

**Evidence contract.** See Evidence and rerun page

This status is copied from the existing reproduction bundle and is not strengthened by the Trackio migration. The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.


---

# Claim 1: IPW estimation error is a martingale wrt a single-arm filtration

---

**Claim 1 (Section 4, arXiv 2603.25971v1 / FXWnvznHMW).** The Horvitz-Thompson (IPW) per-arm estimation error is a martingale wrt a specific single-arm event-time filtration. Executed design-based reproduction below (only randomness = treatment assignment; N=400 fixed potential outcomes; R=20000 Monte-Carlo draws; fixed seeds; reran identical). Every number is printed by a real run of `repro_claim1.py`.

| # | Test (paper anchor) | Measured | Target / rule | Verdict |
|---|---|---|---|---|
| A1 | Per-arm increment mean-zero, E[dM_k given past] (Thm 4.4) | max abs(z)=3.34; terminal E[M_T] z=+0.17 | ~0, abs(z)<=4 | PASS |
| A2 | Increment unpredictable from past (slope 0) | worst abs(z)=2.06 (slope -9.8e-4) | ~0, abs(z)<=4 | PASS |
| A3 | Lemma 4.7: Cov(M_s,M_t)=Var(M_s), single arm | max abs(z)=2.89 (max abs(ratio-1)=0.021) | holds, abs(z)<=4 | PASS |
| B | Difference error (Delta_hat - Delta) NOT a martingale (Prop 4.8) | max abs(z)=37.0 (Cov/Var ratio=1.35) | violated, abs(z)>=6 | PASS |
| C | Single-arm CS uniform miscoverage (Thm 4.6) | 0.44% | <= alpha=5% | PASS |
| C | Naive fixed-n CI uniform miscoverage (continuous monitoring) | 70.4% | > alpha (over-rejects) | PASS |
| C | Naive fixed-n CI single-look coverage (control) | 94.7% / 95.2% | ~ 1-alpha=95% | PASS |

**Verbatim claim (Section 4):** "The IPW treatment-effect estimation error is *not* a martingale with respect to any single filtration, but *each arm's* IPW estimation error is shown to be a martingale with respect to an arm-specific event-time filtration."

**Verdict: VERIFIED (real).** The per-arm IPW error is a martingale wrt the single-arm event-time filtration F_t(w) (A1-A3 sit at the Monte-Carlo noise floor); the treatment-effect (difference) error is not a martingale under any filtration (B, a 37-sigma violation reproducing Prop 4.8); and the martingale structure yields anytime-valid coverage while the naive fixed-n CI over-rejects under optional stopping (C).

---

## Target, comparison rule, falsification

**Target (Section 4).** (i) each arm's IPW error `M_t(w) = r_hat_t(w) - r_t(w)` is a martingale wrt the single-arm filtration `F_t(w)` (Def 4.3: `w_i` revealed at the potential event time `t_i(w)`); Thm 4.4 gives this for the IPW special case (augmentation `m=0`). By Lemma 4.7 a mean-zero martingale must satisfy `Cov(M_s,M_t)=Var(M_s)` for `s<=t`, with conditionally mean-zero increments unpredictable from the past. (ii) the treatment-effect error `Delta_hat_t - Delta_t` is NOT a martingale under any filtration (Prop 4.8: `Cov(D_s,D_t) != Var(D_s)`, from cross-arm covariance + differential delay). (iii) the martingale yields an anytime-valid confidence sequence `r_hat_t(w) +/- b(V_hat_t(w); alpha)` (Thm 4.6) with uniform-over-time coverage `>= 1-alpha`.

**Comparison rule (PASS).** A1, A2: `abs(z) <= 4` (increments consistent with conditional mean 0 and no past-predictability). A3: max standardized `abs(z) <= 4` for the single arm (Lemma 4.7 identity holds within MC error). B: max standardized `abs(z) >= 6` for the difference (identity decisively violated). C: single-arm CS uniform miscoverage `<= alpha` AND naive fixed-n CI uniform miscoverage `> alpha` AND naive single-look coverage `~ 1-alpha`.

**Falsification (honest failure modes).** The martingale claim would be FALSIFIED if the single-arm increments showed a significant nonzero conditional mean or were predictable from the past (`abs(z) >> 4`), or if the Lemma 4.7 identity failed for the single arm, or if the CS uniform miscoverage substantially exceeded `alpha`. The negative result (B) would be falsified if the difference process satisfied `Cov(D_s,D_t)=Var(D_s)`. None occurred: single-arm statistics sit at the MC noise floor (2-3 sigma over hundreds of tests) while the difference violation is 37 sigma.

## Setup (design-based; only randomness is assignment)

- N=400 units with FIXED potential outcomes (seed 0): staggered entry `E_i ~ Uniform(0,10)`; internal delays `s_i(w)` lognormal with treatment accelerated (median arm-1 event time 6.75 vs arm-0 9.81, i.e. asymmetric arrival clocks / pull-forward); bounded magnitudes, treatment lower (`beta1=0.6 < beta0=1.0`), matching Dataset 5.1's structure.
- Treatment `w_i ~ Bernoulli(pi=0.5)`, `R=20000` independent MC assignment vectors (seed 12345) -- the sole source of randomness (design-based framework: potential outcomes are constants).
- Estimator: Horvitz-Thompson IPW (`m=0`, the special case of Thm 4.4). Single-arm event-time ordering by `t_i(w)`. Boundary `b(V;alpha)=sqrt((V*eta^2+1)/eta^2 * log((V*eta^2+1)/alpha^2))`, `alpha=0.05`, `eta=0.1`.
- Deterministic: fixed seeds; a second run reproduced every number identically.

## Controls

- **eta is a free boundary constant** (Thm 4.6 holds for any `eta^2>0`), chosen to match the information scale, NOT tuned to data; the anytime-valid guarantee is independent of it.
- **Same statistic, both processes:** A3 (single arm) and B (difference) use the identical `Cov/Var` identity test -- apples-to-apples -- so the 2.9-sigma vs 37-sigma contrast is a genuine property difference, not a metric artifact.
- **Single-look control:** the naive CI's single-look coverage is ~95%, confirming the CI is valid pointwise; only continuous monitoring (400 looks) breaks it, isolating the anytime-valid property.
- **Extreme-value floor:** A1 max abs(z)=3.34 over 400 event steps and A3 max abs(z)=2.89 over 66 time-pairs are the expected maxima of that many standard normals under a true martingale null.

## Limitations

- Faithful **mechanism** reproduction (martingale structure + anytime-valid coverage), not a bit-for-bit rerun of Dataset 5.1 or Figures 12-15; absolute magnitudes depend on the DGP, but the qualitative facts are decisive and DGP-robust.
- Uses IPW (`m=0`); the event-time AIPW augmentation class (Thm 4.4 general form / Section 5) is not separately exercised here.
- The CS is the asymptotic mixture confidence sequence, hence (as expected) conservative: 0.44% uniform miscoverage < 5%, consistent with the paper's Table 2 oracle coverage of 98.7-100%.

## Rerun

```
cd .trackio/logbook/evidence-package/claim1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 repro_claim1.py
```

---

````bash
$ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 repro_claim1.py
````

exit 0 * 1.15s

````output
==========================================================================
CLAIM 1 - per-arm IPW error martingale wrt single-arm filtration
Anytime-Valid Inference Under Outcome Delay (arXiv 2603.25971v1, FXWnvznHMW)
==========================================================================
design: N=400 fixed potential outcomes; only randomness = assignment
        pi=0.5, R=20000 MC draws, alpha=0.05, eta(boundary)=0.1
        arm-1 median event time 6.75 (pull-forward) < arm-0 median 9.81  -> asymmetric clocks
        mean |y0|=1.246  mean |y1|=0.672 (treatment lower magnitude), bounded

--------------------------------------------------------------------------
(A) PER-ARM IPW ERROR IS A MARTINGALE wrt single-arm filtration (arm 1)
--------------------------------------------------------------------------
  A1 increment E[dM_k|past]~0 : max|z_k| over 400 event steps = 3.34 (frac |z|<=2 = 0.960)
     terminal error mean E[M_T]  = +0.0160  (z = +0.17; expect 0)
  A2 increment unpredictable  : worst Cov(M_(k-1),dM_k) z = -2.06 (slope = -9.83e-04; expect 0)
  A3 Lemma4.7 Cov(M_s,M_t)=Var(M_s): max|ratio-1| = 0.0206, max|z| = -2.89
     example (s=9.26, t=10.65): ratio=0.9904  z=-2.89
  => A1 PASS | A2 PASS | A3 PASS  (martingale property holds)

--------------------------------------------------------------------------
(B) NEGATIVE RESULT: treatment-effect (difference) error is NOT a martingale
    under ANY filtration  (Prop 4.8: Cov(D_s,D_t) != Var(D_s))
--------------------------------------------------------------------------
  max|ratio-1| = 0.5177, max standardized |z| = +37.04
     example (s=3.99, t=6.88): Cov/Var ratio=1.3453  z=+37.04  (!= 1 => violation)
  => PASS (violation is decisive; martingale fails for the difference)

--------------------------------------------------------------------------
(C) ANYTIME-VALID coverage from the martingale CS  vs  naive pointwise CI
    single-arm CS (Thm 4.6) checked at all 400 event times (continuous monitoring)
--------------------------------------------------------------------------
  confidence SEQUENCE  uniform coverage = 99.56%  (miscover 0.44% <= 5%?  yes)
  naive pointwise CI   uniform coverage = 29.59%  (miscover 70.42% -> over-rejects under optional stopping)
  naive pointwise CI   single-look coverage: mid=94.73%  late=95.17%  (~95% => pointwise valid, only sequential use breaks)
  => PASS (martingale CS is anytime-valid; naive CI is not)

==========================================================================
OVERALL CLAIM 1: VERIFIED
  per-arm IPW error IS a martingale wrt the single-arm filtration (A);
  the treatment-effect error is NOT a martingale under any filtration (B);
  the martingale CS is anytime-valid while the naive CI over-rejects (C).
==========================================================================
wrote results.json  (runtime 1.08s)
````


---

# Conclusion

---

Both scored claims are reproduced with executed evidence. Claim 1: an independent design-based Monte-Carlo simulation (fixed potential outcomes; only randomness is treatment assignment) confirms the Horvitz-Thompson (IPW) per-arm estimation error is a martingale wrt the single-arm event-time filtration (increment conditional mean-zero and Lemma 4.7 identity both at the Monte-Carlo noise floor), that the treatment-effect (difference) error is NOT a martingale under any filtration (a decisive 37-sigma violation of the same identity, reproducing Prop 4.8), and that the martingale structure yields anytime-valid confidence-sequence coverage (uniform miscoverage 0.44% <= alpha=5%) while a naive fixed-n CI over-rejects under continuous monitoring (uniform miscoverage 70.4%). Claim 2: the union-bound width-ratio scaling law of Prop 4.12 is reproduced within ~1.15% of its closed-form limits 2*sqrt(pi(1-pi)) and sqrt(1-pi), with the union bound strictly tighter (R_pi<1) in the asymmetric-clock regime. This Trackio-native record covers 2 claim page(s) and preserves the reports, scripts, evidence, and rerun output. Fresh local reruns completed 2/2 command(s) in approximately 1.3 seconds. No Hugging Face GPU Job was used: both checks are CPU-feasible (deterministic NumPy/SciPy, <2 s each), not GPU-limited.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 2 claim page(s) with executed evidence; original claim labels preserved | Paper-scale implementation and every headline empirical claim |
| Hardware | Local machine; CPU-oriented scripts unless a recorded command says otherwise; no HF Job | Paper-specified accelerators, datasets, checkpoints, and sweeps |
| Compute time | ~1.3 s across 2 freshly recorded command(s) (CPU, single-threaded) | Not estimated without the full paper setup |
| Cost | Approximately $0 incremental local compute | Unknown; potentially substantial |
| Outcome | Both scored claims verified with executed numbers: per-arm IPW martingale + anytime-valid coverage (Claim 1) and the union-bound width-ratio law (Claim 2). | Not attempted |

---

**📦 Artifact** `icml26-fxwnvznhmw/fxwnvznhmw-reproduction-bundle:v0` · dataset

https://huggingface.co/buckets/Crusadersk/icml26-anytime-valid-delayed-repro-artifacts#icml26-fxwnvznhmw/fxwnvznhmw-reproduction-bundle:v0

---

The reproduction bundle contains the runnable scripts, evidence JSON/JSONL, manifests, reviews, and supporting logs under `artifacts/`. After publication, the artifact cell above resolves to the Hugging Face Bucket URL and can be downloaded from that Bucket. Secrets, virtual environments, caches, and replaceable downloads are excluded.


---

# Sources and provenance

---

- OpenReview: https://openreview.net/forum?id=FXWnvznHMW
- Published logbook: https://huggingface.co/spaces/Crusadersk/icml26-anytime-valid-delayed-repro
- arXiv: https://arxiv.org/abs/2603.25971

The migration preserves the original claim boundaries and does not convert toy, partial, or inconclusive evidence into a full reproduction.
