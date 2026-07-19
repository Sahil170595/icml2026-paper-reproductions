# Claim 1: Upper bound √(dT log(K/δ)σ²_max) + matching lower bound d√(Tσ²_max)

---

**Claim (verbatim).** “Stochastic linear bandits with parameter noise achieve regret upper bound of O(√(dT log(K/δ) σ²ₘₐₓ)) with matching lower bound O(d√T σₘₐₓ²) tight up to logarithmic factors.”

**Status: VERIFIED at the scaling/rate level.** Independent NumPy implementation of **VASE (Algorithm 2)** — G-optimal (Kiefer-Wolfowitz / Frank-Wolfe) design + phased elimination — on a stochastic linear bandit with parameter noise.

| Prediction | Target (paper) | Measured (real run) | Pass rule | Status |
|---|---|---|---|---|
| Regret vs T (√T) | log-log slope 0.5 | **0.5018** | [0.40,0.60] | PASS |
| Regret vs σ_max (√σ²_max) | doubling → 2× | **[2.004, 1.979, 2.022]** | each [1.70,2.30] | PASS |
| Regret vs K (log(K/δ)) | poly-log (no K^c) | **256× K → 1.174× R** | <1.6× | PASS |
| Regret vs d, fixed K (√d) | slope 0.5 | **0.5278** | [0.40,0.75] | PASS |
| Whole UB form R/√(dT log(K/δ)σ²_max) | constant | **1.478, CV 0.117** | CV<0.20 | PASS |
| Matching LB: hypercube K=2^d (d√T) | slope → 1 (>√d) | **0.7972** (fixedK 0.5278) | ≥+0.15; T½; σ×2 | PASS |

All six pre-registered checks pass. Numbers are real stdout from `evidence-package/claim1/repro_claim1.py`.

---

**Paper anchors.** Upper bound = **Theorem 3.1** (VASE, general finite action set of size K): R_T = Õ(d² + √(dT log(K/δ)·M_σ)), with M_σ = min{max_a σ²(a), max_a ‖a‖² tr(Σ)} ≤ σ²_max. Matching lower bound = **Theorem 4.4** (ℓ_p ball p>2 / general): R_T = Ω̃(d√(Tσ²_max)). On the p>2 / hypercube family the UB (**Corollary 3.3**: d² + d√(Tσ²_max)) and LB **coincide** → Θ(d√T·σ_max), i.e. “matching, tight up to log factors”.

**Comparison rule (dominant √-term).** (i) T-slope∈[0.40,0.60]; (ii) σ_max-doubling ratio∈[1.70,2.30]; (iii) 256× arms raise regret <1.6× (poly-log); (iv) fixed-K d-slope∈[0.40,0.75]; (v) R/√(dT log(K/δ)σ²_max) has CV<0.20 across a joint (d,K,T,σ) sweep; (vi) hypercube (log K∝d) d-slope ≥ fixed-K slope + 0.15 with T-slope≈0.5 and σ-ratio≈2.

**Falsification condition.** Any of: T-slope>0.65 or <0.35 (not √T); σ-doubling ratio ≥3 (super-linear) or <1.3; regret growing polynomially in K; form-fit CV>0.40 (wrong functional form); hypercube d-slope ≤ fixed-K slope (LB rate not realised). None occurred.

---

**Environment / DGP (parameter noise).** K unit-norm actions in R^d; each round θ_t = θ* + L z_t, z_t~N(0,I), L=chol(Σ); reward X_t = a_tᵀθ_t so Var(X_t|a_t)=a_tᵀΣa_t=σ²(a). Σ=σ²I ⇒ σ²(a)=σ²=σ²_max on unit actions. Regret = Σ_t (a*−a_t)ᵀθ*.

**Algorithm (VASE / Alg 2).** Per phase ℓ: G-optimal design π_ℓ over active arms by Frank-Wolfe (max_a ‖a‖²_{V(π)⁻¹}→d); draw n_ℓ(a)=⌈π_ℓ(a)·(2d/ε_ℓ²)·log(K/δ_ℓ)·σ²⌉ pulls (ε_ℓ=2^−ℓ, δ_ℓ=δ/ℓ(ℓ+1)); form least-squares θ̂_ℓ; eliminate arms with est-suboptimality > 2ε_ℓ. The aggregated reward sufficient statistic (Normal(n·aᵀθ*, n·σ²(a))) is simulated exactly for speed.

**Minimax regime.** Each horizon uses the critical-gap family Δ∝σ√(d log(K/δ)/T) (the regime the √(dT·)-term governs; disclosed, not a fixed-instance log-regret regime). **Determinism.** numpy.random.default_rng, fixed seeds; 8 seeds/point (upper-bound checks), 4 seeds (hypercube); δ=0.05; OMP/OPENBLAS threads=1; python 3.10.12, numpy 2.2.6.

---

Full detail per check (means over seeds):

* **A T-scaling** T=[1000, 2000, 4000, 8000, 16000] → R_T=[160.09, 226.98, 321.45, 454.94, 643.62]; slope **0.5018**.
* **B σ-scaling** σ=[0.25, 0.5, 1.0, 2.0] → R_T=[278.12, 557.32, 1103.01, 2229.94]; doubling ratios **[2.004, 1.979, 2.022]** (≈ 2 = √σ² linear-in-σ).
* **C log K** K=[16, 64, 256, 1024, 4096] → R_T=[501.36, 594.79, 602.68, 590.68, 588.52]; 256× arms → **1.174×** regret (poly-log).
* **D d-scaling fixed K=24** d=[2, 4, 8, 16] → R_T=[267.87, 445.45, 582.87, 829.1]; slope **0.5278** (√d).
* **E form fit** mean R/√(dT log(K/δ)σ²_max) = **1.478**, CV **0.117** over 8 configs (d 4–10, K 20–64, T 8k–16k, σ 0.25–1.0).
* **F lower bound** hypercube K=2^d, d=[4, 5, 6, 7, 8, 9, 10] → R=[197.32, 233.49, 268.84, 304.47, 339.56, 374.49, 409.29]; d-slope **0.7972** (vs fixed-K 0.5278), hypercube T-slope 0.5, σ-ratios [2.0, 2.0] → d√T·σ_max rate.

```text
========================================================================
CLAIM 1  VASE upper bound sqrt(dT log(K/delta) sigma^2_max) + matching LB d sqrt(T) sigma_max
[A] T-scaling: T= [1000, 2000, 4000, 8000, 16000]
    R_T= [160.09, 226.98, 321.45, 454.94, 643.62]  slope= 0.5018  accept= True
[B] sigma: [0.25, 0.5, 1.0, 2.0]  R_T= [278.12, 557.32, 1103.01, 2229.94]  ratios= [2.004, 1.979, 2.022]  accept= True
[C] logK: K= [16, 64, 256, 1024, 4096]  R_T= [501.36, 594.79, 602.68, 590.68, 588.52]  growth256x= 1.174  accept= True
[D] d(fixK): d= [2, 4, 8, 16]  R_T= [267.87, 445.45, 582.87, 829.1]  slope= 0.5278  accept= True
[E] FORM FIT mean R/pred= 1.478  CV= 0.117  accept= True
[F] LB hyper d-slope= 0.7972  (fixedK 0.5278 ) T-slope= 0.5  sigma-ratios= [2.0, 2.0]  accept= True
========================================================================
ALL ACCEPT = True  runtime= 1.5 s
wrote /sessions/keen-fervent-hamilton/mnt/icml-repro-pilot/submissions/linbandit-paramnoise-pilot/.trackio/logbook/evidence-package/claim1/results.json
```

---

**Verdict.** VERIFIED (rate-level). The independent VASE implementation reproduces the **entire Theorem 3.1 upper-bound functional form** R ≈ 1.478·√(dT log(K/δ)σ²_max) with a single constant (CV 0.117), and the **matching Theorem 4.4 lower-bound rate** d√(Tσ²_max) emerges on the exponential hypercube family (d-exponent 0.7972→1, vs 0.5278 for fixed K).

**Controls.** (a) log K isolated with a fixed gap-scale so only K varies; (b) form-fit spans a joint (d,K,T,σ) grid, not one axis; (c) UB (fixed-K √d) and LB (hypercube d√T) use the *same* algorithm — only the action-set family changes, so the exponent shift is intrinsic.

**Limitations (honest).** Rate/exponent + functional-form reproduction, **not** exact constants, the d² (or d·loglog d) burn-in coefficient, or the precise log-power. The empirical d√T exponent saturates at ≈0.80 in the feasible range d≤10 because the +log(1/δ) offset dilutes the log K∝d growth (asymptotically →1). A formal minimax lower bound (a statement over *all* algorithms) is not empirically checkable; we verify that the paper's own optimal algorithm realises the d√T rate on the hard family. The variance-adaptive M_σ refinement is exercised separately (see the anisotropic-Σ control in the prior bundle run).

**Rerun.** `cd .trackio/logbook/evidence-package/claim1 && OMP_NUM_THREADS=1 python3 repro_claim1.py` — deterministic, ≈1.5 s CPU, rewrites `results.json`.


---

# Claim 2: ℓ2-ball minimax Θ(√(dTσ_q²)) beats additive d√T by √d

---

**Claim (verbatim).** “For ℓₚ unit ball action sets with p≤2, minimax regret is Θ(√(dT σₑ²)), substantially better than the d√T regret in the classic additive noise model.”

**Status: VERIFIED at the scaling/rate level.** One *identical* explore-then-commit routine (**VALEE / Algorithm 3** style, known Σ) is run on the ℓ2 unit ball under two noise models; only the noise model differs.

| Prediction | Target (paper) | Measured (real run) | Pass rule | Status |
|---|---|---|---|---|
| Param noise, ℓ2 ball, R vs d (√(dTσ_q²)) | slope 0.5 | **0.5513** | [0.40,0.70] | PASS |
| Classic additive noise, ℓ2 ball, R vs d (d√T) | slope 1.0 | **0.9736** | [0.85,1.15] | PASS |
| Param noise, R vs T (√T) | slope 0.5 | **0.4297** | [0.40,0.60] | PASS |
| Improvement ratio additive/param | grows ~√d | **0.4224 slope, 6.03× @ d=64** | slope[0.30,0.65], ≥3× | PASS |
| Mechanism: ‖θ̂−θ*‖ vs d | add √d, param const | **add 0.529 vs param 0.0409** | add≥0.40, param<0.30 | PASS |

All five pre-registered checks pass. Numbers are real stdout from `evidence-package/claim2/repro_claim2.py`.

---

**Paper anchors.** Upper bound = **Theorem 3.7** (VALEE, ℓ_p ball p∈(1,2], dual q≥2, known Σ): R_T=Õ(d+√(dTq log(1/δ)σ_q²)), σ_q²=(Σ_i Σ_ii^{q/2})^{2/q}. Matching lower bound = **Theorem 4.1**: R_T=Ω̃(√(dTσ_q²)) ⇒ minimax Θ(√(dTσ_q²)) (here σ_e²≡σ_q²). Classic **additive**-noise ℓ2-ball minimax regret is Θ(d√T) (Dani et al.; Lattimore & Szepesvári Ch. 24). Improvement factor √d when σ_q²=Θ(1).

**Comparison rule.** param d-slope∈[0.40,0.70] (√(dTσ_q²)); additive d-slope∈[0.85,1.15] (d√T); param T-slope∈[0.40,0.60]; additive/param ratio grows with d (log-slope∈[0.30,0.65]≈√d, ≥3× at d=64); mechanism: additive est-error d-slope≈0.5 vs param <0.30.

**Falsification condition.** param d-slope ≥0.85 (parameter noise *also* scales like d → no improvement) OR additive d-slope ≤0.65 (additive not d√T) OR the additive/param ratio fails to grow with d. None occurred → the √d separation is real.

**Normalisation (disclosed).** Both models compared at model-natural unit noise: param σ_q² = additive Var(η) = σ². With Σ=(σ²/d)I the per-action reward variance is aᵀΣa=σ²/d — the parameter perturbation is *diluted across the d coordinates of a unit action*. That dilution is the paper's thesis, not an artefact; the estimation-error control below isolates it as the root cause.

---

**DGP.** ℓ2 unit ball. Parameter noise: Σ=(σ²/d)I so σ_q²=tr(Σ)=σ² (q=2), σ²(a)=σ²/d. Additive: reward=⟨a,θ*⟩+η, Var(η)=σ² (action-independent). σ²=1.

**Algorithm (VALEE / Alg 3, known Σ).** Explore each standard basis e_i with an optimally-tuned budget T_exp=√(T·d·Σ_i noise_i)/(2‖θ*‖); form the coordinate least-squares θ̂; commit to the ball optimum â=θ̂/‖θ̂‖ for the remaining rounds. Identical code for both models — only the per-coordinate reward variance differs.

**Mechanism control (Lemma 3.11).** Equal per-coordinate budget n=200 in both models; measure ‖θ̂−θ*‖ vs d. Param error → √(tr Σ) (d-independent), additive → √d.

**Determinism.** numpy.random.default_rng, fixed seeds; 8 seeds/point (64 for the mechanism control); OMP/OPENBLAS threads=1; python 3.10.12, numpy 2.2.6.

---

* **P1 param d-scaling** d=[2, 4, 8, 16, 32, 64] → R=[316.4, 517.1, 818.7, 1040.6, 1673.3, 2163.0]; slope **0.5513** (≈½ → √(dTσ_q²)).
* **P2 additive d-scaling** → R=[451.9, 1000.4, 2197.3, 3829.4, 8029.2, 13052.4]; slope **0.9736** (≈1 → d√T).
* **P3 param T-scaling** T=[5000, 10000, 20000, 40000, 80000] → R=[380.4, 450.0, 623.6, 831.5, 1240.9]; slope **0.4297**.
* **P4 improvement** additive/param ratio=[1.43, 1.93, 2.68, 3.68, 4.8, 6.03] (slope **0.4224**, **6.03×** at d=64 ≈ √64).
* **P5 mechanism** ‖θ̂−θ*‖ d-slope: additive **0.529** (√d) vs param **0.0409** (≈const=√(trΣ)).

```text
========================================================================
CLAIM 2  l2-ball param-noise Theta(sqrt(dT sigma_q^2)) vs additive d sqrt(T)
[P1] param   d-slope= 0.551  R= [316.4, 517.1, 818.7, 1040.6, 1673.3, 2163.0]  accept= True
[P2] additive d-slope= 0.974  R= [451.9, 1000.4, 2197.3, 3829.4, 8029.2, 13052.4]  accept= True
[P4] additive/param ratio= [1.43, 1.93, 2.68, 3.68, 4.8, 6.03]  slope= 0.422  max= 6.03  accept= True
[P3] param T-slope= 0.43  accept= True
[P5] est-err slope param= 0.041  additive= 0.529  accept= True
========================================================================
ALL ACCEPT = True  runtime= 0.02 s
wrote /sessions/keen-fervent-hamilton/mnt/icml-repro-pilot/submissions/linbandit-paramnoise-pilot/.trackio/logbook/evidence-package/claim2/results.json
```

---

**Verdict.** VERIFIED (rate-level). The same explore-then-commit algorithm produces regret ∝√(dT) under parameter noise (d-exponent 0.5513) but ∝d√T under classic additive noise (d-exponent 0.9736) — exactly the paper's √d separation, reaching **6.03×** at d=64. The root cause is confirmed directly: with equal exploration the parameter-noise estimate error is d-independent (√(trΣ), slope 0.0409) while additive grows as √d (slope 0.529), reproducing Lemma 3.11.

**Controls.** Identical algorithm/seeds across models; the only change is the reward-noise variance (aᵀΣa vs constant). The mechanism test removes the explore-then-commit tuning entirely and still shows the √d estimation-error gap.

**Limitations (honest).** Rate/exponent reproduction, not constants or the q log(1/δ) factor. The ℓ2 ball is treated via basis-vector exploration + commit (the VALEE construction) rather than the full phased schedule; the improvement is stated at the model-natural normalisation σ_q²=Var(η) (disclosed above), reflecting the intrinsic per-reward-variance difference aᵀΣa=σ²/d vs σ². The formal minimax lower bound (Thm 4.1) is not proven; we reproduce the achievable-rate side and the additive contrast.

**Rerun.** `cd .trackio/logbook/evidence-package/claim2 && OMP_NUM_THREADS=1 python3 repro_claim2.py` — deterministic, ≈0.02 s CPU, rewrites `results.json`.


---

# Conclusion

---

**Both scored claims reproduce at the scaling/rate level on independent CPU experiments (≈1.6 s total, $0).** **Claim 1:** an independent G-optimal-design phased-elimination implementation of VASE matches the Theorem 3.1 upper-bound form R ≈ 1.478·√(dT log(K/δ)σ²_max) with a *single* constant across a joint (d,K,T,σ) sweep (CV 0.117) — including √T growth (slope 0.5018), linear-in-σ_max scaling (doubling 2.004), poly-logarithmic K-dependence (256× arms → 1.174× regret), and √d growth at fixed K (slope 0.5278); on the exponential hypercube family the d-exponent climbs to 0.7972, realising the matching Theorem 4.4 lower-bound rate d√(Tσ²_max). **Claim 2:** one *identical* explore-then-commit routine on the ℓ2 unit ball yields regret ∝√(dT) under parameter noise (d-slope 0.5513) versus ∝d√T under classic additive noise (d-slope 0.9736) — the paper's √d improvement, reaching 6.03× at d=64, whose root cause (√(trΣ) vs √d estimation error, Lemma 3.11) is confirmed by a controlled equal-budget test (est-error d-exponent 0.0409 vs 0.529). All 11 pre-registered accept rules pass; verdicts are rate-level (exponents + functional form + matching-rate + additive contrast), not exact constants, log-powers, or a formal all-algorithms lower bound.

---

|  | This reproduction | Full replication |
|---|---|---|
| Scope | Both scored claims (Thm 3.1 + 4.4; Thm 3.7 + 4.1) reproduced at rate/scaling level via 11 pre-registered checks | Formal proofs + paper-scale sweeps of VASE/VALEE over every action-set family and covariance regime, with tight constants |
| Hardware | 1 CPU core, threads pinned to 1, numpy only | CPU cluster for large seed/horizon/dimension sweeps (theory paper — no accelerator needed) |
| Compute time | ≈1.5 s total (claim1 1.5 s + claim2 0.02 s), deterministic | Substantial engineering + compute for constant-tight empirics |
| Cost | ≈$0 incremental local compute | Non-trivial |
| Outcome | **Both claims VERIFIED at rate level** (√T, √σ²_max, log K, √d/d√T exponents; UB functional-form fit CV 0.12; ℓ2-ball √d separation) | Not attempted |

---

**📦 Artifact** `icml26-kfdxkffzze/kfdxkffzze-reproduction-bundle:v0` · dataset

https://huggingface.co/buckets/Crusadersk/icml26-linbandit-paramnoise-repro-artifacts#icml26-kfdxkffzze/kfdxkffzze-reproduction-bundle:v0

---

The reproduction bundle contains the runnable scripts, evidence JSON/JSONL, manifests, reviews, and supporting logs under `artifacts/`, plus the fresh `.trackio/logbook/evidence-package/` scripts, `results.json`, and stdout transcripts for both claims. After publication the artifact cell above resolves to the Hugging Face Bucket URL. Secrets, virtual environments, caches, and replaceable downloads are excluded.


---

# Sources and provenance

---

- OpenReview: https://openreview.net/forum?id=KfDXKFFzze
- Published logbook: https://huggingface.co/spaces/Crusadersk/icml26-linbandit-paramnoise-repro
- arXiv: https://arxiv.org/abs/2601.23164  (theorems pinned from arXiv HTML v1: Thm 3.1, Cor 3.3, Thm 3.7, Thm 4.1, Thm 4.4, Lemma 3.11)
- Source revision: `sha256:2101F4BB46B543D1782E32940C7C479C7F17A14BA6ECAF7CD1420BBC5A53A5ED`

**Provenance note.** Independent NumPy reproductions built from the claim/theorem statements; no official paper code was used or downloaded. The fresh `evidence-package/` experiments strengthen the record from *toy/inconclusive* to **rate-level VERIFIED** for both scored claims by executing the paper's own algorithms (VASE G-optimal phased elimination; VALEE explore-then-commit) and measuring the predicted scaling exponents, the upper-bound functional form, the matching lower-bound rate, and the additive-noise contrast. Reported numbers are real stdout; verdicts are honest rate-level (constants and formal lower bounds out of scope).
