# Claim 1: Claim 1 — Theorem 3.1. In a two-armed Gaussian-prior bandit, GP-TS incu…

---

**Paper claim.** Claim 1 — Theorem 3.1. In a two-armed Gaussian-prior bandit, GP-TS incurs a polynomial (not exponential) high-probability regret tail

**Paper anchor.** See the original experiment report

**Reproduction status.** `bounded local check`

**Evidence contract.** See Evidence and rerun page

This status is copied from the existing reproduction bundle and is not strengthened by the Trackio migration. The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.

---

**Target (Theorem 3.1).** With `X={x1,x2}`, `eps_t~N(0,1)`, `(f(x1),f(x2))~N(0, [[1,1/2],[1/2,1]])`, and `T>e`: `Pr(R_T >= T/2) >= c1*T^(-c2)` with `c1,c2>0` — a **polynomial** (not exponential) high-probability tail, giving polynomial dependence on `1/delta`.

**Pass rule.** log-log fit of `Pr(R_T>=T/2)` vs `T` is linear with finite slope `-c2` (order 1) and `R2 > 0.99`, and beats the exponential fit; successive ratios `P(2T)/P(T)` roughly constant.

Exact Gaussian conjugate 2-armed GP-TS, **M=300000** runs, seed 20260716 (real stdout):

| T | 200 | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|
| hits / 300000 | 13807 | 3833 | 878 | 221 | 73 |
| `Pr(R_T>=T/2)` | 0.046023 | 0.012777 | 0.002927 | 0.000737 | 0.000243 |

| model | fit | measured | target |
|---|---|---|---|
| polynomial `log P = -c2*log T + log c1` | `c2 = 1.924`, `c1 ~ 1214` | **R2 = 0.99800** | linear log-log, slope order 1, R2>0.99 ok |
| exponential `log P = -a*T + b` | `a = 1.59e-3` | R2 = 0.84611 | must fit worse ok |

Successive ratios `P(2T)/P(T)` = 0.278, 0.229, 0.252, 0.330 ~ constant near `2^(-c2)=0.263` (polynomial signature). **Status: VERIFIED** — polynomial (not exponential) tail reproduced in the theorem's exact two-armed Gaussian-prior setting. Full script/output on the Evidence and rerun page.


---

# Claim 2 — Second moment ⟹ improved delta bound (Thm 3.2) + expected lenient regret (Thm 3.3)

---

**Paper claim (verbatim).** "Second moment of cumulative regret upper bound yields improved regret upper bound on δ, with expected lenient regret bounds also established."

**Paper anchors.** Contribution (ii) = **Theorem 3.2**: `E[R_T^2] = O(T*gamma_T*log T)`, hence by Markov's inequality `Pr( R_T <= sqrt(E[R_T^2]/delta) ) >= 1-delta` — the dependence on the failure probability delta is tightened from the previously established `1/delta` (first-moment Markov; Russo & Van Roy 2014; Takeno et al. 2024) to `1/sqrt(delta)`. Contribution (iii) = **Theorem 3.3**: the expected **lenient** regret is polylogarithmic in T, and the count of Delta-suboptimal rounds `|T_Delta|` is bounded (polylog).

**Pass rule.** (2A) the `1/sqrt(delta)` bound is valid at every delta (empirical exceedance <= delta) AND is tighter than the old `1/delta` bound with improvement factor scaling as `delta^(-1/2)` (fitted exponent ~ 0.5). (2B) lenient regret and Delta-bad-pull counts grow polylogarithmically (fitted exponent ~ 0, constant increment per horizon-doubling), not like the growing cumulative regret.

### 2A — measured vs target (improved delta dependence)

Faithful GP-TS on a finite GP bandit (SE kernel, K=8 grid points on [0,1], Bayesian `f~prior`, noise 0.05, exact conjugate posterior), horizon T=300, **M=40000** independent trials, seed 20260717. Measured `E[R_T]=7.496`, `E[R_T^2]=69.827`, sd=3.694.

| delta | empirical `(1-delta)` quantile of `R_T` | old bound `E[R_T]/delta` (~ 1/delta) | new bound `sqrt(E[R_T^2]/delta)` (~ 1/sqrt(delta)) | exceedance of new bound (must be <= delta) | improvement factor old/new |
|---|---|---|---|---|---|
| 0.50000 | 6.781 | 14.99 | 11.818 | 0.07090 | 1.269 |
| 0.25000 | 8.464 | 29.98 | 16.713 | 0.01883 | 1.794 |
| 0.12500 | 10.201 | 59.97 | 23.635 | 0.00573 | 2.537 |
| 0.06250 | 12.238 | 119.93 | 33.425 | 0.00165 | 3.588 |
| 0.03125 | 14.643 | 239.86 | 47.270 | 0.00060 | 5.074 |
| 0.01563 | 17.564 | 479.72 | 66.850 | 0.00028 | 7.176 |

- **improvement-factor exponent vs `1/delta` = 0.5000** (target 0.5) — the advantage of the new bound over the old grows exactly as `delta^(-1/2)`, i.e. the paper's `1/sqrt(delta)` tightening.
- new `delta^(-1/2)` bound **valid at every delta** (exceedance <= delta everywhere): True.
- `E[R_T^2]` is finite and sub-quadratic in T (T-sweep exponent 0.339 <= 1 ⟹ `O(T*polylog)` upper bound holds): `E[R_T^2]` = 45.61, 56.28, 72.78, 91.72 at T = 75, 150, 300, 600.
- empirical `(1-delta)`-quantile exponent q = 0.271 (<= 0.5): GP-TS's true tail is no heavier than the second-moment worst case, consistent with the bound.

### 2B — measured vs target (lenient regret is polylog, Thm 3.3)

GP-TS, finite GP bandit (SE kernel, K=12 on [0,1], noise 0.05), **M=1500** trials, seed 20260718, horizon up to 1024. `LR_T = sum_{t: gap>=Delta} gap`, `|T_Delta| = #{t: gap >= Delta}`.

| T | cumulative `E[R_T]` | bad pulls `E|T_Delta|` (Delta=0.3) | lenient `E[LR_T]` (Delta=0.3) | bad pulls `E|T_Delta|` (Delta=0.6) | lenient `E[LR_T]` (Delta=0.6) |
|---|---|---|---|---|---|
| 64 | 7.372 | 6.161 | 5.937 | 3.591 | 4.855 |
| 128 | 8.275 | 6.592 | 6.258 | 3.745 | 5.066 |
| 256 | 9.212 | 6.991 | 6.562 | 3.895 | 5.268 |
| 512 | 10.222 | 7.417 | 6.851 | 4.038 | 5.445 |
| 1024 | 11.358 | 7.841 | 7.152 | 4.170 | 5.630 |

- growth exponents (Delta=0.3): cumulative `R_T` = 0.155, bad-pulls = 0.087, **lenient `LR_T` = 0.067**; (Delta=0.6): bad-pulls = 0.054, **lenient = 0.053**.
- lenient regret grows by a near-**constant increment per horizon-doubling** (Delta=0.3: +0.32, +0.30, +0.29, +0.30), the signature of **polylogarithmic** (`~log T`) growth — Theorem 3.3 — while the number of Delta-bad pulls is essentially saturated (6.16 -> 7.84 over a 16x horizon). Cumulative regret keeps growing over the same range.

**Status: VERIFIED** — both the `1/sqrt(delta)` improvement (Thm 3.2) and the polylog lenient regret (Thm 3.3) are reproduced with the measured numbers above.

---

**Why this setup is faithful.** Theorems 3.2 and 3.3 are proved for GP-TS in the Bayesian setting (objective `f` a GP sample path) and explicitly cover the finite-domain `|X|<inf` case. The experiments run exactly that: `f` drawn from the SE-kernel GP prior over a finite grid, GP-TS with an exact Gaussian conjugate posterior (full posterior sample per round via batched Cholesky, argmax queried), Bayesian expectations estimated by Monte Carlo over independent draws of `f`.

**Scope / honesty.**
- 2A's `delta^(-1/2)` improvement factor `(E[R_T]/delta)/sqrt(E[R_T^2]/delta) = (E[R_T]/sqrt(E[R_T^2]))*delta^(-1/2)` is structural (it follows from the two Markov inequalities); what the experiment establishes empirically is that GP-TS's cumulative regret has a **finite, well-behaved second moment** (`E[R_T^2]=69.8`, sub-quadratic in T) so that the new bound is both valid and materially tighter (7.2x at delta=0.0156). The specific constants are instance-dependent.
- On a fixed finite domain the realized tail is lighter than the worst case (quantile exponent 0.27 < 0.5) and cumulative regret is itself only mildly growing; the contrast lenient-vs-cumulative is therefore modest in absolute exponent but clearly present and in the predicted direction.
- The `gamma_T` (maximum information gain) factor in `O(T*gamma_T*log T)` is polylog for SE and is not separately measured; the reproduction checks the resulting `E[R_T^2]=O(T*polylog)` upper-bound behaviour directly.

**Rerun.**
```bash
cd .trackio/logbook/evidence-package/claim2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 repro_claim2.py all
```
CPU-only, deterministic. Writes `results.json`; all numbers on this page are copied verbatim from stdout / `results.json`. Runtimes and sha256 are on the Evidence and rerun page.


---

# Claim 3 — Improved sqrt(T) cumulative regret upper bound (Thm 3.5)

---

**Paper claim (verbatim).** "Improved cumulative regret upper bound on time horizon T provided along with well-posedness of associated ODE limit." (The "ODE / well-posedness" fragment appears nowhere in the paper and is a mis-extraction — see the scope cell; the reproducible core is the sqrt(T) upper bound.)

**Paper anchor / target.** Contribution (iv) = **Theorem 3.5**: with high probability `R_T = O(sqrt(T)*log T)` (SE kernel) / `~O(sqrt(T))` (Matern nu>2), the sqrt(T)-type sublinear rate matching GP-UCB.

**Pass rule.** Upper bound ⟹ verified iff over the horizon grid: (a) sublinear `R_T/T -> 0`; (b) `rho(T)=R_T/(sqrt(T)*log T)` bounded / non-increasing (so `R_T <= C*sqrt(T)*log T` for fixed C at every T); (c) linear regret rejected (it would make `rho ~ sqrt(T)/log T` grow, and fits far worse).

### Measured vs target

Continuous-domain GP-TS, GP sample path (SE kernel) on a K=200 grid of [0,1] finer than the horizon, exact Matheron pathwise sampling, ell=0.02, noise 0.25, Bayesian E[R_T] over **M=40** GP draws, seed0=4000, wall 32.9 s.

| T | `E[R_T]` | `R_T/T` (-> 0) | `rho=R_T/(sqrt(T)*log T)` (<= C) | `rho` if regret were linear (grows) |
|---|---|---|---|---|
| 32 | 50.094 | 1.5654 | 2.5551 | 1.6322 |
| 64 | 67.487 | 1.0545 | 2.0284 | 1.9236 |
| 128 | 81.704 | 0.6383 | 1.4884 | 2.3317 |
| 256 | 95.341 | 0.3724 | 1.0746 | 2.8854 |
| 512 | 110.629 | 0.2161 | 0.7837 | 3.6272 |
| 1024 | 126.396 | 0.1234 | 0.5698 | 4.6166 |

- **Sublinear:** `R_T/T` falls monotonically 1.565 -> 0.123 (`R_T = o(T)`). `sublinear = True`.
- **Upper bound holds:** `rho(T)` non-increasing 2.555 -> 0.570, so `R_T <= C*sqrt(T)*log T` with `C = 2.555` at every horizon. `upper_bound_holds = True`.
- **Linear rejected:** under linear regret `rho` would instead rise 1.63 -> 4.62; single-param fits give linear `a*T` R^2 = -2.61 (rejected) vs log-form `a+b*log T` R^2 = 0.9989. `linear_rejected = True`.
- Per-doubling increments `E[R_2T]-E[R_T]` = 17.39, 14.22, 13.64, 15.29, 15.77 (near-constant ⟹ realized rate `~ log T`, within the `sqrt(T)*log T` bound); power exponent p = 0.258.

**Status: VERIFIED** (upper bound) — sublinear regret staying below `2.555*sqrt(T)*log T` at every horizon while rejecting linear regret: the sqrt(T)-type improved bound of Theorem 3.5.

---

**ODE fragment — reported honestly, not faked.** The auto-extracted claim mentions "well-posedness of associated ODE limit". The paper (arXiv 2603.09276) contains **no** ordinary differential equation, ODE, flow, or well-posedness statement anywhere — every occurrence of the substring "ode" in the source is inside the word "m-ode-l". That fragment corresponds to no result in this paper; it is a mis-extraction, is not simulatable, and is flagged rather than fabricated. The paper-supported, reproducible core of Claim 3 is the improved sqrt(T) upper bound tested above.

**Why this setup is faithful and why the realized rate is below sqrt(T).** Theorem 3.5's sqrt(T) bound is a *continuous-domain, worst-case* upper bound; its `sqrt(T*gamma_T)` mechanism needs the domain complexity to keep growing, so the reproduction uses a grid finer than the horizon and exact pathwise posterior sampling (Matheron), the standard continuous-GP-TS emulation. For a *typical* GP sample path on a compact domain, GP-TS localises the optimum and realised regret grows only `~ log T` — which is `<= sqrt(T)*log T`, satisfying (not violating) the upper bound with margin. The verification therefore targets the upper-bound inequality (`rho` bounded/non-increasing) and rejection of the linear-regret null — the correct empirical content of an `O(sqrt(T)*log T)` claim. Forcing the realised rate to sit exactly at sqrt(T) needs a domain whose resolution scales with T (`K ~ T`), outside the CPU/40-s-per-run budget; the upper-bound check is invariant to that.

**Scope / honesty.**
- SE-kernel case of Theorem 3.5 (the cleaner of the two); the Matern `nu>2` case is not separately run.
- Constant `C = 2.555` is instance-dependent; the claim's content is the sqrt(T)-type *rate*, which the `rho`-table and linear-rejection establish.

**Rerun.**
```bash
cd .trackio/logbook/evidence-package/claim3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 repro_claim3.py
```
CPU-only, deterministic (numpy.random.default_rng, fixed seeds). Writes `results.json`; all numbers on this page are copied verbatim from stdout / `results.json`.


---

# Conclusion

---

All three paper claims are covered by independent, deterministic, CPU-only reproductions, each reproducing the tested mechanism within its stated acceptance rule. This Trackio-native record covers 3 claim page(s) (Thm 3.1 polynomial lower-bound tail; Thm 3.2 improved delta bound + Thm 3.3 lenient regret; Thm 3.5 improved sqrt(T) upper bound) and preserves the original report, scripts, evidence, and rerun output. Fresh local reruns completed 4/4 command(s) in approximately 163 seconds. No Hugging Face GPU Job was used: these are theory/regret-scaling checks that are fully CPU-feasible (exact Gaussian-conjugate GP-TS, NumPy/SciPy), not GPU-limited. The auto-extracted "ODE / well-posedness" fragment of Claim 3 does not correspond to any statement in the paper and is flagged honestly rather than fabricated.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 3 claim page(s), all covered by executed evidence; original claim labels preserved | Paper-scale implementation and every headline empirical claim |
| Hardware | Local machine; CPU-only NumPy/SciPy scripts; no HF Job | Paper-specified accelerators, datasets, checkpoints, and sweeps |
| Compute time | ~163 s across 4 freshly recorded command(s) | Not estimated without the full paper setup |
| Cost | Approximately $0 incremental local compute | Unknown; potentially substantial |
| Outcome | All three claims reproduced within their stated acceptance rules (Thm 3.1 / 3.2+3.3 / 3.5). | Not attempted |

---

**📦 Artifact** `icml26-yzob5onbco/yzob5onbco-reproduction-bundle:v0` · dataset

https://huggingface.co/buckets/Crusadersk/icml26-ts-bayesopt-repro-artifacts#icml26-yzob5onbco/yzob5onbco-reproduction-bundle:v0

---

The reproduction bundle contains the runnable scripts, evidence JSON/JSONL, manifests, reviews, and supporting logs under `artifacts/` and `.trackio/logbook/evidence-package/`. After publication, the artifact cell above resolves to the Hugging Face Bucket URL and can be downloaded from that Bucket. Secrets, virtual environments, caches, and replaceable downloads are excluded.


---

# Sources and provenance

---

- OpenReview: https://openreview.net/forum?id=yZoB5oNBco
- Published logbook: https://huggingface.co/spaces/Crusadersk/icml26-ts-bayesopt-repro
- arXiv: https://arxiv.org/abs/2603.09276

The migration preserves the original claim boundaries and does not convert toy, partial, or inconclusive evidence into a full reproduction.
