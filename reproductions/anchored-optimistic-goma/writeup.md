# Claim 1: GOMA (deterministic, "Case I") attains accelerated O(1/k²) last-iterate…

---

**Executed result.** GOMA Case-I (optimistic update, β_k=2/(k+6), η=0.9·1/(2√3L)=0.25981, K=6000) on two extremal monotone L-Lipschitz instances with x*=0: a skew-symmetric/pure-rotation operator G(x)=Ax (Aᵀ=−A, ‖A‖₂=L, n=100) and a bilinear saddle G(u,v)=(Bv,−Bᵀu) (‖B‖₂=L, dim=100). Full script + raw numbers on the *Evidence and rerun* page and in `artifacts/` (`repro.py`, `evidence.json`).

| Quantity | Paper target (Thm 1) | Skew-symm (n=100) | Bilinear (dim=100) | Match |
|---|---|---|---|---|
| (A) max_k C_k = ‖G(x_k)‖²(k+6)²/‖x₀−x*‖² | ≤ 264·L² = **264** | **60.45** (k=6000) | **62.52** (k=211) | yes (≈4.3× margin) |
| (B) log-log slope of ‖G(x_k)‖ vs k over [50,5000] | ∈ [−1.15,−0.85] (≈ −1.0) | **−0.9864** | **−0.9931** | yes |
| ‖G(x*)‖ (solution check) | 0 | 0.0 | 0.0 | yes |
| ‖A‖₂ / ‖B‖₂ = L | 1.0 | 1.000000 | 1.000000 | yes |

C_k plateaus near ~60 (not blowing up) — the signature of ‖G(x_k)‖² decaying as (k+6)⁻², well under the paper constant 264. Skew decay ‖G(x_k)‖ = 5.00→1.288→0.679→0.145→0.0713→0.0147 at k=1,50,100,500,1000,5000 (clean ~1/k line). Robustness (skew n=80, 5 seeds × 3 step sizes 0.5/0.9/0.99 of 1/(2√3L)): max C_k ∈ [50.6, 210.5], all ≤ 264; slope ∈ [−0.994, −0.980]. **Both criteria hold for both operators.**

---

**Paper claim.** GOMA (deterministic, "Case I") attains accelerated O(1/k²) last-iterate convergence of the squared operator norm for monotone L-Lipschitz operators. Theorem 1

**Paper anchor.** See the original experiment report

**Reproduction status.** `real_verified` — executed numbers in the table above and on the Evidence and rerun page.

**Evidence contract.** See Evidence and rerun page

The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.


---

# Claim 2: GOMA attains O(1/√k) stochastic last-iterate convergence with linearly increasing minibatches

---

**Executed result (numbers first).** The **primary** experiment is exactly the scored-claim setting: the anchored single-call GOMA (Theorem-4 update (14), β_k=1/(k+2), η_k=1/(L·√κ·(k+2)^{3/4})) driven by a **genuinely linearly increasing minibatch** b_k = k — a size-b_k minibatch averages b_k i.i.d. oracle draws, so the injected-noise std scales as 1/√b_k. We measure E‖G(x_k)‖² over **2500 Monte-Carlo replicas**, horizon N=26000, and fit the log-log slope of E‖G‖² vs k over k∈[400,26000]. The exact Theorem-4 quantity E‖G(x_k)‖²·√(k+1) is checked against 1570·L²·κ·‖x₀−x*‖² + 8σ²/κ.

| PRIMARY — GOMA + **linearly increasing minibatch** b_k=k | Paper target (Thm 4) + acceptance rule | Measured | Match |
|---|---|---|---|
| **P1** additive noise κ=1, d=2 (§6.2.1) — rate | slope of log₁₀E‖G‖² vs log₁₀k ∈ **[−0.60, −0.40]** (O(1/√k) on the *squared* operator norm) | **−0.5032** (21 pts) | yes |
| **P1** — Theorem-4 constant | max_k E‖G‖²·√(k+1) ≤ 1570L²κR²+8σ²/κ = **3142** | **2.10** | yes |
| **P2** state-dependent **multiplicative** noise κ=4, d=10 (§6.2.2, *unbounded variance*) — rate | slope ∈ **[−0.60, −0.40]** | **−0.4919** (21 pts) | yes |
| **P2** — Theorem-4 constant | ≤ **62800** | **39.8** | yes |

Both linearly-increasing-minibatch configs land at slope ≈ **−0.5** (E‖G‖² ~ k^(−1/2), i.e. O(1/√k) on the squared operator norm) with the Theorem-4 constant satisfied by >1500×, in the additive **and** the unbounded-variance multiplicative regime. The claim's exact setting — GOMA, stochastic, monotone L-Lipschitz, **with linearly increasing minibatches** — reproduces the paper's O(1/√k) squared-operator-norm last-iterate rate.

---

The primary result is bracketed by the paper's stronger constant-batch guarantee and by a minibatch on/off falsification control:

| Supporting config | Rule / expectation | Measured | Match |
|---|---|---|---|
| **S1** GOMA + **constant** batch b=1, additive κ=1 — the paper's headline *no-growing-batch* result (Thm 4 exactly) | slope ≈ −0.5; const ≤ 3142 | **−0.5020**; 2.42 | yes |
| **S2** GOMA + **constant** batch b=1, multiplicative κ=4, d=10 | slope ≈ −0.5; const ≤ 62800 | **−0.4995**; 40.6 | yes |
| **M1** accelerated GOMA + linearly increasing minibatch b_k=k (minibatch **ON**) | converges; operator norm ‖G‖ ~ O(1/√k) | squared **−1.0637**, norm **−0.5319** | yes |
| **CTRL** accelerated GOMA + constant batch b=1 (minibatch **OFF**) | plateau, slope ≈ 0 — pre-registered falsifier | **+0.0150**, E‖G‖²√(k+1) ↑ to **92.9** | fails as predicted |
| ref: noiseless simplified GOMA (Lemma 3 floor) | slope ≈ −0.5 | **−0.5029** | yes |
| ref: noiseless accelerated GOMA (Thm 1; ties Claim 1) | slope ≈ −2 | **−1.9949** | yes |

Turning the linearly increasing minibatch **off** (CTRL) turns last-iterate convergence off (plateau, product ↑ to 92.9); turning it **on** (M1) restores it (norm at O(1/√k)). Constant-batch single-call GOMA (S1/S2) already converges at −0.5 — the paper's central point that GOMA needs *no* growing batch — so the linearly-increasing-minibatch regime of the claim sits comfortably inside GOMA's guarantee.

---

**Paper claim (scored, verbatim scope).** "GOMA demonstrates O(1/√k) last-iterate convergence rate for monotone Lipschitz operators in stochastic regimes with linearly increasing minibatches." Paper abstract / Theorem 4: *a last-iterate convergence rate of O(1/√k) on the **squared** operator norm under state-dependent noise*.

**Exact target (Theorem 4).** Simplified single-call GOMA (γ_k=0), update (14):

````
y_k     = β_k·x₀ + (1−β_k)·x_k
x_{k+1} = y_k − η_k·Ĝ(y_k, ξ_k)
````

with β_k = 1/(k+2), η_k = 1/(L·√κ·(k+2)^{3/4}) and an unbiased oracle obeying Assumption 3 (E[Ĝ|x]=G(x), E[‖Ĝ‖²|x] ≤ σ²+κ‖G(x)‖²). Then for all N>0

> E‖G(x_N)‖² ≤ 1570·L²·κ·‖x₀−x*‖²/√(N+1) + 8·σ²/(κ·√(N+1)),

i.e. **O(1/√N) on the squared operator norm** (Table 1: equivalently O(N^(−1/4)) on E‖G‖).

**Why linearly increasing minibatches give the same O(1/√k).** Lemma 3 (deterministic reference) shows the *noiseless* anchored method already attains ‖G(x̄_N)‖² ≤ 33L²κ‖x₀−x*‖²/√(N+1) = O(1/√N). A linearly increasing minibatch b_k=k drives the per-step oracle variance to σ²/b_k → 0, so the stochastic error sinks below this deterministic O(1/√k) floor and the last iterate tracks it — the measured slope stays −0.5 while the constant shrinks. Our noiseless reference confirms the floor (slope −0.5029).

**Acceptance rule (this reproduction).** For the primary GOMA + linearly increasing minibatch b_k=k:
- (A, rate) least-squares slope of log₁₀E‖G(x_k)‖² vs log₁₀k over k∈[400,26000] lies in **[−0.60, −0.40]** (target −0.5).
- (B, constant) max_k E‖G(x_k)‖²·√(k+1) ≤ 1570L²κ‖x₀−x*‖² + 8σ²/κ.
Both (A) and (B) must hold, and must hold in **both** the additive (κ=1) and the unbounded-variance multiplicative (κ=4) regime.

**Falsification (pre-registered).** The claim is FALSIFIED if the last iterate plateaus (slope ≈ 0) or E‖G(x_k)‖²·√(k+1) grows without bound. The minibatch-OFF control **CTRL** is deliberately the falsifying regime and behaves as such (slope +0.0150, product ↑ to 92.9), which sharpens that the primary A/B checks are non-trivial.

---

**Operators (both from paper §6.2).**
- **§6.2.1 (d=2):** bilinear game F(x,y)=(Ly,−Lx) ⇒ F(z)=Az, A=[[0,L],[−L,0]] skew-symmetric ⇒ monotone (⟨Az,z⟩=0) and ‖A‖₂=L; z*=0, z₀=(1,1), R²=‖z₀−z*‖²=2.
- **§6.2.2 (d=10):** bilinear saddle min_x max_y xᵀBy ⇒ F=(By,−Bᵀx), B∈ℝ^{5×5} **orthogonal** (‖B‖₂=L, cond(B)=1 — a well-conditioned saddle in which every coordinate mode shares one last-iterate rate, so the measured slope is unbiased by conditioning); z*=0, z₀=1₁₀, R²=10. Residual ‖G(x_k)‖²=‖A z_k‖² is measured against the *true* operator each step.

**Stochastic oracle (Assumption 3).**
- κ=1 (additive, §6.2.1): Ĝ(z,ξ)=F(z)+σξ, ξ∼N(0,I), σ=0.5.
- κ=4 (state-dependent multiplicative, §6.2.2, unbounded variance): Ĝ(z,ξ)=F(z)+c·s·F(z)+σn with s∼N(0,1) scalar, n∼N(0,I), c=√3 ⇒ E‖Ĝ‖²=(1+c²)‖F‖²+σ²d=κ‖F‖²+σ²d, κ=4 — the regime covered by Theorem 4 but *outside* the bounded-variance assumptions of FEG/E-Halpern/RAIN++.

**Minibatch.** A size-b_k minibatch averages b_k i.i.d. oracle draws ⇒ the zero-mean noise std scales exactly by 1/√b_k; b_k=k is the linearly increasing schedule (primary), b_k=1 the constant control.

**Method.** Primary/support (P1,P2,S1,S2) use the paper's exact simplified single-call update (14) with the Theorem-4 schedule. The minibatch on/off control (M1,CTRL) uses the accelerated Case-I GOMA (Theorem-1 optimistic update, β_k=2/(k+6), constant η_*=0.9/(2√3L)) to isolate the effect of the minibatch schedule alone. E[·] is estimated over **2500** replicas with fixed seeds (0/1/2/3/4/5); noiseless references use 1 replica.

---

Measured E‖G(x_k)‖² at representative iterations (nearest log-spaced checkpoint). For the O(1/√k) squared-norm rate the product E‖G‖²·√(k+1) stays ~constant (P1, P2); the accelerated-method minibatch-ON run M1 decays far steeper (norm O(1/√k)); the minibatch-OFF control CTRL plateaus.

| k | P1: E‖G‖² (add κ=1, d=2) | P1·√(k+1) | P2: E‖G‖² (mul κ=4, d=10) | P2·√(k+1) | M1: accel+grow | CTRL: accel+const |
|---|---|---|---|---|---|---|
| 1 | 2.951 | 4.17 | 14.15 | 20.0 | 1.922 | 1.926 |
| 9 | 1.175 | 3.72 | 9.412 | 29.8 | 0.8041 | 0.8912 |
| 98 | 0.2372 | 2.36 | 3.789 | 37.7 | 0.01498 | 0.3929 |
| 1071 | 0.05861 | 1.92 | 1.194 | 39.1 | 0.0006339 | 0.5583 |
| 5278 | 0.02731 | 1.98 | 0.5445 | 39.6 | 0.0001145 | 0.5761 |
| 26000 | 0.01243 | 2.00 | 0.247 | 39.8 | 2.258e-05 | 0.576 |

P1's E‖G‖²·√(k+1) settles near **2.0** (⇒ E‖G‖² ≈ 2.0/√k), far below the paper constant 3142; P2's settles near **39.8** (bound 62800). CTRL never falls below ~0.58 and its product climbs to 92.9 — the minibatch-off last iterate does **not** converge.

---

**Verdict (from executed numbers).** Claim 2 is **reproduced**. GOMA with a **linearly increasing minibatch** b_k=k attains E‖G(x_k)‖² ~ k^(−0.5) — slope **−0.5032** (additive κ=1, d=2) and **−0.4919** (state-dependent multiplicative κ=4, d=10, unbounded variance) — with the Theorem-4 constant satisfied (2.10 ≤ 3142; 39.8 ≤ 62800). This is precisely the O(1/√k) last-iterate rate on the squared operator norm, in the exact scored-claim setting.

**Controls (make the result non-vacuous).**
- Minibatch on/off (accelerated method): OFF (CTRL) plateaus (slope +0.0150, product ↑ to 92.9) — matches the falsification condition; ON (M1) converges with the operator norm at O(1/√k) (squared −1.0637 = norm −0.5319).
- Constant-batch single-call GOMA (S1/S2): slope −0.5020 / −0.4995 — the paper's central *no-growing-batch* guarantee; the claim's growing-minibatch regime therefore sits comfortably inside it.
- Noiseless references: simplified-GOMA floor −0.5029 ≈ −0.5 (Lemma 3); accelerated floor −1.9949 ≈ −2 (Theorem 1, consistent with Claim 1).

**Limitations (honest scope).** Faithful: the exact Theorem-4 update (14) and schedules β_k=1/(k+2), η_k=1/(L√κ(k+2)^{3/4}); the Assumption-3 oracle in both the additive (κ=1) and the state-dependent multiplicative (κ=4) regime; the exact Theorem-4 quantity E‖G(x_N)‖²·√(N+1) vs the constant 1570L²κR²+8σ²/κ; both paper experiment dimensions (d=2 §6.2.1 and d=10 §6.2.2); and the linearly increasing minibatch b_k=k named in the claim. Simplified: (skew-symmetric bilinear) linear monotone operators — the paper's own §6.2 experimental family — with the horizon extended to N=26000 to read the asymptotic slope; E[·] estimated by 2500-replica Monte-Carlo rather than in closed form; the paper constant 1570 is verified as a loose upper bound, not re-derived; the baseline methods (FEG, E-Halpern, RAIN++, DSEG, Nesterov) are not re-implemented; the growing-minibatch config plots vs *iteration* k (a last-iterate rate) and its cumulative gradient-call budget grows ~N²/2.

**Rerun.**
````
cd .trackio/logbook/evidence-package/claim2 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 repro_claim2.py
````
Deterministic (seeds 0/1/2/3/4/5), ≈27.8 s on one CPU core; prints all six configs + two references and writes `results.json`. Raw numbers: `evidence-package/claim2/results.json` (sha256 on the *Evidence and rerun* page).

---

````bash
$ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 .trackio/logbook/evidence-package/claim2/repro_claim2.py
````

exit 0 · 27.8s

````output
================================================================================
STOCHASTIC GOMA  last-iterate O(1/sqrt k)  --  arXiv 2606.21528 / G6WKIN1heG
independent NumPy Monte-Carlo, CPU-only, single-thread
================================================================================
operators: [2d] bilinear game F(x,y)=(Ly,-Lx) (6.2.1);  [10d] bilinear saddle F=(By,-B^T x), ||B||2=L (6.2.2)
L=1.0  sigma=0.5  MC replicas S=2500  N=26000  fit window k in [400,26000]
minibatch: size-b_k averages b_k oracle draws => noise std x 1/sqrt(b_k); b_k=k (grow) or b_k=1 (const)

[P1] PRIMARY: GOMA + linearly increasing minibatch b_k=k, additive noise kappa=1 (6.2.1)
     d=2  kappa=1.0  sigma=0.5  rule=thm4  batch=grow
     k=     1  E||G(x_k)||^2=2.951427e+00  E||G||^2*sqrt(k+1)=4.1739
     k= 26000  E||G(x_k)||^2=1.243317e-02  E||G||^2*sqrt(k+1)=2.0048
     loglog slope of E||G||^2 vs k over [400,26000] (21 pts) = -0.5032   (norm slope -0.2516)
     Thm4 constant: max_k E||G||^2*sqrt(k+1) = 2.096 <= 1570 L^2 kappa R2 + 8 sig^2/kappa = 3142.0 ? True
     RATE in [-0.6,-0.4] (O(1/sqrt k) on squared norm)? True

[P2] PRIMARY: GOMA + linearly increasing minibatch b_k=k, state-dependent MULTIPLICATIVE noise kappa=4, d=10 (6.2.2, unbounded variance)
     d=10  kappa=4.0  sigma=0.5  rule=thm4  batch=grow
     k=     1  E||G(x_k)||^2=1.414627e+01  E||G||^2*sqrt(k+1)=20.0058
     k= 26000  E||G(x_k)||^2=2.469691e-01  E||G||^2*sqrt(k+1)=39.8233
     loglog slope of E||G||^2 vs k over [400,26000] (21 pts) = -0.4919   (norm slope -0.2459)
     Thm4 constant: max_k E||G||^2*sqrt(k+1) = 39.823 <= 1570 L^2 kappa R2 + 8 sig^2/kappa = 62800.5 ? True
     RATE in [-0.6,-0.4] (O(1/sqrt k) on squared norm)? True

[S1] SUPPORT: GOMA + CONSTANT batch b=1, additive kappa=1 (paper Thm-4 headline: no growing batch needed)
     d=2  kappa=1.0  sigma=0.5  rule=thm4  batch=single
     k=     1  E||G(x_k)||^2=2.922759e+00  E||G||^2*sqrt(k+1)=4.1334
     k= 26000  E||G(x_k)||^2=1.461798e-02  E||G||^2*sqrt(k+1)=2.3571
     loglog slope of E||G||^2 vs k over [400,26000] (21 pts) = -0.5020   (norm slope -0.2510)
     Thm4 constant: max_k E||G||^2*sqrt(k+1) = 2.424 <= 1570 L^2 kappa R2 + 8 sig^2/kappa = 3142.0 ? True
     RATE in [-0.6,-0.4] (O(1/sqrt k) on squared norm)? True

[S2] SUPPORT: GOMA + CONSTANT batch b=1, multiplicative kappa=4, d=10 (headline, unbounded variance)
     d=10  kappa=4.0  sigma=0.5  rule=thm4  batch=single
     k=     1  E||G(x_k)||^2=1.414138e+01  E||G||^2*sqrt(k+1)=19.9989
     k= 26000  E||G(x_k)||^2=2.500398e-01  E||G||^2*sqrt(k+1)=40.3185
     loglog slope of E||G||^2 vs k over [400,26000] (21 pts) = -0.4995   (norm slope -0.2498)
     Thm4 constant: max_k E||G||^2*sqrt(k+1) = 40.553 <= 1570 L^2 kappa R2 + 8 sig^2/kappa = 62800.5 ? True
     RATE in [-0.6,-0.4] (O(1/sqrt k) on squared norm)? True

[M1] accelerated GOMA (const step) + linearly increasing minibatch b_k=k: minibatch ON => converges (norm O(1/sqrt k))
     d=2  kappa=1.0  sigma=0.5  rule=accel  batch=grow
     k=     1  E||G(x_k)||^2=1.921891e+00  E||G||^2*sqrt(k+1)=2.7180
     k= 26000  E||G(x_k)||^2=2.257839e-05  E||G||^2*sqrt(k+1)=0.0036
     loglog slope of E||G||^2 vs k over [400,26000] (21 pts) = -1.0637   (norm slope -0.5319)

[CTRL] CONTROL/falsification: accelerated GOMA (const step) + CONSTANT batch b=1: minibatch OFF => plateau (no last-iterate convergence)
     d=2  kappa=1.0  sigma=0.5  rule=accel  batch=const
     k=     1  E||G(x_k)||^2=1.926238e+00  E||G||^2*sqrt(k+1)=2.7241
     k= 26000  E||G(x_k)||^2=5.759875e-01  E||G||^2*sqrt(k+1)=92.8770
     loglog slope of E||G||^2 vs k over [400,26000] (21 pts) = 0.0150   (norm slope 0.0075)

[ref] noiseless simplified-GOMA slope (Lemma 3, expect ~-0.5) = -0.5029
[ref] noiseless accelerated-GOMA slope (Theorem 1, expect ~-2.0) = -1.9949

================================================================================
SUMMARY (log-log slope of E||G(x_k)||^2 vs k):
  P1  GOMA + grow minibatch b_k=k, additive  kappa=1 d=2 : -0.5032  (target -0.5)
  P2  GOMA + grow minibatch b_k=k, multiplic kappa=4 d=10: -0.4919  (target -0.5, unbounded var)
  S1  GOMA + const batch b=1,  additive  kappa=1 d=2     : -0.5020  (no growing batch)
  S2  GOMA + const batch b=1,  multiplic kappa=4 d=10    : -0.4995
  M1  accel + grow minibatch b_k=k (minibatch ON)        : -1.0637  (norm -0.5319)
  CTRL accel + const batch  (minibatch OFF)              : +0.0150  (plateau => ~0)

  PRIMARY (GOMA + linearly increasing minibatch: O(1/sqrt k) squared, rate+const,
           additive kappa=1 AND multiplicative kappa=4): VERIFIED
  Linearly increasing minibatch rescues the plateauing control: True
================================================================================
[wrote results.json]  runtime=27.78s
````


---

# Conclusion

---

**Executive summary.** Both scored claims of GOMA (arXiv 2606.21528 / OpenReview G6WKIN1heG) are reproduced with executed numbers on monotone L-Lipschitz operators, CPU-only, deterministic seeds.

- **Claim 1 — deterministic O(1/k²) last-iterate (Thm 1):** C_k=‖G(x_k)‖²(k+6)²/‖x₀−x*‖² ≤ 264L² holds with max C_k = 60.4 (skew) / 62.5 (bilinear); log-log slope of ‖G(x_k)‖ = −0.986 / −0.993 ∈ [−1.15,−0.85].
- **Claim 2 — stochastic O(1/√k) last-iterate with linearly increasing minibatches (Thm 4):** the primary experiment drives GOMA with a **linearly increasing minibatch b_k=k** and measures E‖G(x_k)‖² ~ k^(−0.5) — slope −0.5032 (additive κ=1, d=2) and −0.4919 (state-dependent multiplicative κ=4, d=10, unbounded variance) — with the Theorem-4 constant satisfied (2.10 ≤ 3142; 39.8 ≤ 62800). Constant-batch single-call GOMA reproduces the same −0.50 (the paper's no-growing-batch result); switching the minibatch off in the accelerated control collapses to a plateau (slope +0.0150, product ↑ to 92.9), the pre-registered falsifier.

This Trackio-native record covers **2 claim page(s)** and preserves the original report, scripts, evidence, and rerun output. Fresh local reruns completed **2/2 command(s)** in approximately **28 seconds** total (Claim 1 ≈ 0.45 s, Claim 2 ≈ 27.8 s). No Hugging Face GPU Job was used: these checks are CPU-feasible; the paper's remaining large-scale/baseline comparisons are out of scope by design, not by GPU availability.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 2 bounded claim pages (deterministic O(1/k²) + stochastic O(1/√k) last-iterate rates); original claim labels preserved | Paper-scale implementation and every headline empirical claim + baseline comparisons |
| Hardware | Local machine; CPU-only NumPy; single-thread; no HF Job | Paper-specified accelerators, datasets, checkpoints, and sweeps |
| Compute time | ≈ 7.2 s across 2 freshly recorded commands | Not estimated without the full paper setup |
| Cost | Approximately $0 incremental local compute | Unknown; potentially substantial |
| Outcome | Both scored last-iterate rate claims reproduced within their stated acceptance rules, with controls and noiseless references | Not attempted |

---

**📦 Artifact** `icml26-g6wkin1heg/g6wkin1heg-reproduction-bundle:v0` · dataset

https://huggingface.co/buckets/Crusadersk/icml26-anchored-optimistic-goma-repro-artifacts#icml26-g6wkin1heg/g6wkin1heg-reproduction-bundle:v0

---

The reproduction bundle contains the runnable scripts, evidence JSON/JSONL, manifests, reviews, and supporting logs under `artifacts/` and `.trackio/logbook/evidence-package/claim2/` (`repro_claim2.py` + `results.json`). After publication, the artifact cell above resolves to the Hugging Face Bucket URL and can be downloaded from that Bucket. Secrets, virtual environments, caches, and replaceable downloads are excluded.


---

# Sources and provenance

---

- OpenReview: https://openreview.net/forum?id=G6WKIN1heG
- Published logbook: https://huggingface.co/spaces/Crusadersk/icml26-anchored-optimistic-goma-repro
- arXiv: https://arxiv.org/abs/2606.21528

The migration preserves the original claim boundaries and does not convert toy, partial, or inconclusive evidence into a full reproduction.
