# Claim 1: HSR Bellman operator is a max-norm contraction (Theorem 3.1)

---

**Executed result — REPRODUCED.** The HSR Bellman operator `T^μM = B^μ + G^μM` (Eq 6) is affine, so `T^μM − T^μM′ = G^μ(M − M′)` and the exact max-norm contraction modulus is `‖G^μ‖∞` = the max row-sum of the nonnegative continuation kernel `G^μ`. The proof (App. A.2) shows this row-sum equals `E_μ[γ^τ]`, the expected discount over option duration `τ ≥ 1`, hence `≤ γ < 1`. We verify every checkable consequence directly on a four-room MDP (N=68 states, γ=0.95, 4 primitives + 8 eigenoptions).

| Quantity | Paper target (Thm 3.1) | uniform-μ | primitive-only-μ | option-heavy-μ | Match |
|---|---|---|---|---|---|
| contraction modulus ‖G^μ‖∞ | ≤ γ = 0.95, < 1 | **0.9253** | **0.9500** | **0.9278** | yes |
| measured c = max‖TM−TM′‖∞/‖M−M′‖∞ (worst-case probe) | = ‖G‖∞ ≤ γ (Eq 8 tight) | **0.9253** | **0.9500** | **0.9278** | yes |
| max over 200 random matrix pairs | ≤ γ | 0.640 | 0.724 | 0.606 | yes (never exceeds ‖G‖∞) |
| spectral radius ρ(G^μ) (asymptotic rate) | < 1 | 0.8974 | 0.9500 | 0.9020 | yes |
| G row-sum = E_μ[γ^τ] range | ≤ γ ∀s | [0.835, 0.925] | [0.950, 0.950] | [0.847, 0.928] | yes |
| fixed-point residual ‖M_k−M*‖∞ (M*=(I−G)⁻¹B) | → 0 | **9.0e-13** (264 it) | **9.6e-13** (526 it) | **9.4e-13** (278 it) | yes |
| empirical per-step ratio → ρ(G) | geometric | 0.897 | 0.949 | 0.902 | yes |
| **HSR(primitive-only) − RW-SR** residual | HSR generalises SR | — | **0.0 (exact)** | — | yes |

The worst-case probe (difference matrix = ones) makes the measured contraction factor **exactly equal** `‖G^μ‖∞`, so Eq 8 is tight, not loose; random matrix pairs never exceed it. Fixed-point iteration `M_{k+1} = T^μM_k` converges geometrically to the analytic fixed point `(I−G^μ)⁻¹B^μ` at the empirical rate ρ(G^μ), which sits at or below ‖G^μ‖∞ ≤ γ. The primitive-only policy makes G^μ = γP_rw and B^μ = I, so the HSR fixed point **coincides bit-for-bit with the standard RW-SR** `(I−γP_rw)⁻¹` (residual 0.0) — HSR strictly generalises the flat SR. **All contraction/convergence criteria hold for all three high-level policies.**

---

**Paper claim.** *Theorem 3.1 (Contraction of HSR Bellman Operator).* For any discount `γ < 1` and option durations `τ ≥ 1`, the HSR Bellman operator `T^μ` (Eq 6) is a contraction mapping with respect to the max-norm: `‖T^μM − T^μM′‖∞ ≤ γ‖M − M′‖∞` (Eq 8). Consequently it admits a unique fixed point (the HSR) reached by iterating `T^μ`.

**Anchor.** Section 3 (Eq 5–8) and Appendix A.2 of arXiv 2602.12753; continuation kernel `G^μ = Σ_ā μ(ā|s)F^ā`, `F^ā = γM^ā diag(β_ā)` (Eq 7); `Σ_s̃ G^μ_{s s̃} = E_μ[γ^τ] ≤ γ`.

**Faithful.** Exact operator `T^μM = B^μ + G^μM`; exact discounted termination kernel `F^ā = (I − γP_ā diag(1−β))⁻¹ γP_ā diag(β)` (row-sums = E[γ^τ], τ≥1, verified); analytic fixed point via `(I−G)⁻¹B`; three high-level policies; primitive-pseudo-option reduction (β=1, τ=1). **Simplified/verified-as-consequence:** we do not re-derive the inequality symbolically; instead we *measure* the contraction factor, the tightness of Eq 8, the geometric convergence rate, and the fixed-point residual — the operational content of Thm 3.1.

**Reproduction status.** `real_verified` — executed numbers above and on the Evidence & rerun page (`.trackio/logbook/evidence-package/claim1/`, `repro_claim1.py` + `results.json`).


---

# Claim 2: Four-Room transfer efficiency — HSR vs SR row-features (Fig 2b–d)

---

**Executed result — PARTIAL.** Option-augmented (4 primitives + 8 eigenoptions) linear SMDP Q-learning in the four-room (N=68, γ=0.95). Fixed state features φ(s): one-hot **Raw** (zero-transfer baseline), **RW-SR** rows, **eHSR** rows. Train to goal G1 (episodes-to-optimal), then transfer the weights to a new goal G2 and count episodes-to-optimal. 20 seeds.

| Feature | Episodes to optimal, G1 | Episodes to optimal, **G2 (transfer)** | Transfer efficiency TE = (N_G2/N_G1)/(Raw ratio), lower=better |
|---|---|---|---|
| Raw (one-hot, zero-transfer) | 20.6 ± 9.9 | **146.8 ± 2.4** (fails, cap 150) | 1.00 (reference) |
| SR rows | 51.2 ± 11.6 | **54.5 ± 9.0** | **0.149** |
| HSR rows | 31.0 ± 3.7 | **52.0 ± 10.2** | **0.235** |

**What reproduces:** both SR and HSR row-features enable strong few-shot transfer to the new goal (G2 ≈ 52–55 episodes) while the one-hot **Raw baseline fails to transfer** (146.8, essentially the 150-episode cap) — this matches the paper's Fig 2b,c point that predictive features beat the zero-transfer encoding.

**What does not reproduce:** the paper's headline (Fig 2d) that **HSR transfers *significantly faster* than SR**. Here HSR G2 = **52.0** vs SR G2 = **54.5** are statistically indistinguishable — two-sample **t = 0.18, p = 0.854**. The transfer-efficiency metric even favours SR (0.149 vs 0.235). Root cause: in this independent build the expected-HSR is numerically almost identical to the expected-SR (see Claims 3–4), so HSR-row and SR-row features carry near-identical transfer information.

**Verdict: partial** — the "features enable transfer vs one-hot" sub-result reproduces; the specific "HSR > SR" claim does not (HSR ≈ SR, p=0.85).

---

**Paper claim.** *Sec 4.1 / Fig 2d.* In the four-room transfer task (learn G1, switch to G2), agents using **HSR row-features achieve significantly higher transfer efficiency than SR row-features** (two-sided t-test p=0.008), both beating a one-hot baseline.

**Faithful.** Four-room, augmented action space (primitives + 8 eigenoptions), SMDP Q-learning with linear function approximation on SR/HSR/one-hot row-features, episodes-to-optimal measured by BFS-optimal greedy rollout, transfer by weight retention across the goal switch, transfer-efficiency ratio as defined in the paper, 20 seeds. **Simplified:** fixed (pre-computed) features rather than online-learned SR/HSR (the paper reports the effect holds with pre-trained representations, Fig S2); eigenoptions from SVD of RW-SR with quantile terminations; single (G1,G2) pair. These simplifications do not favour SR over HSR — they use exactly the paper's feature definitions.

**Reproduction status.** `partial` — SR/HSR both transfer (vs Raw) reproduced; HSR-over-SR advantage not reproduced. Evidence: `.trackio/logbook/evidence-package/claim2/` (`repro_claim2.py` + `results.json`).


---

# Claim 3: HSR representation robustness to policy change (Fig 2e–g)

---

**Executed result — NOT REPRODUCED.** The paper (Fig 2g) reports that after adapting the policy from task G1 to G2, the standard SR matrix reorganises drastically while the HSR is far less variable, i.e. the relative change `ρ(M) = ‖M₁ − M₂‖²_F / ‖M₁‖²_F` is significantly **lower** for HSR (two-sided t-test p<0.001). We measure ρ for the flat SR and the HSR over **20 random (G1,G2) goal pairs** in the four-room (N=68, γ=0.95), using the **same** high-level augmented policy for both (SR = one-step flattened SR; HSR = temporally-extended fixed point).

| Quantity | Paper target (Fig 2g) | SR | HSR | Verdict |
|---|---|---|---|---|
| mean relative change ρ = ‖M₁−M₂‖²_F/‖M₁‖²_F | ρ_HSR ≪ ρ_SR | **1.965 ± 0.129** | **2.031 ± 0.151** | HSR **not** lower |
| SR/HSR ratio (stability factor) | ≫ 1 | — | **0.97** | ≈ 1 (no gain) |
| fraction of pairs with ρ_HSR < ρ_SR | → 1 | — | **0.55** | ~ chance |
| two-sided t-test (SR vs HSR) | p < 0.001 | — | **t=−0.33, p=0.74** | not significant |

The HSR representation is **no more stable** than the SR under task-induced policy change in this build — ρ is essentially equal (ratio 0.97), and HSR is lower in only 55% of pairs (chance). Diagnostic: the expected-HSR and expected-SR have near-identical room-block structure (within-room mass 0.37 vs 0.38; block diag/off-diag 0.38 vs 0.39), so there is no temporally-abstract stabilisation to measure.

**Verdict: not reproduced** — the claimed policy-robustness advantage of HSR over SR is absent (ρ_HSR ≈ ρ_SR, p=0.74).

---

**Paper claim.** *Sec 4.1 / Fig 2e–g.* HSR features are robust to task-induced policy changes: standard SR undergoes drastic reorganisation to conform to the new optimal policy, whereas HSR matrices are much less variable; the relative Frobenius change is significantly lower for HSR (p<0.001).

**Faithful.** Exact relative-change metric `‖M₁−M₂‖²_F/‖M₁‖²_F`; the *same* augmented high-level optimal policy drives both representations (the paper's "fair comparison" — SR still updated stepwise while options execute); SR = one-step flattened SR of that policy, HSR = temporally-extended fixed point; 20 goal pairs; Welch t-test. **Simplified:** offline optimal policies rather than online Q-learned ones; 8 eigenoptions. The comparison is symmetric and does not disadvantage HSR.

**Reproduction status.** `not_reproduced` — measured ρ_HSR ≈ ρ_SR. Evidence: `.trackio/logbook/evidence-package/claim3/` (`repro_claim3.py` + `results.json`).


---

# Claim 4: HSR-NMF sparse low-rank basis (Fig 3–4)

---

**Executed result — NOT REPRODUCED (1 of 4 sub-signatures holds).** The paper (Sec 4.2, Fig 3–4) claims NMF of the expected HSR yields a **sparse** basis whose factors are **elevated at bottleneck (doorway) states**, matches the reconstruction efficiency of SR-SVD, while NMF applied to the SR suffers **"feature collapse."** We build the expected SR (eSR) and expected HSR (eHSR) by averaging over 12 pretraining option-policies (four-room, N=68, γ=0.95), then run SVD and multiplicative-update NMF (rank 8) on each.

| Signature | Paper prediction | Measured eSR | Measured eHSR | Holds? |
|---|---|---|---|---|
| NMF basis sparsity (mean Gini) | HSR-NMF **sparser** | 0.699 | **0.624** | no (HSR less sparse) |
| bottleneck-activation ratio (doorways / elsewhere) | HSR ≫ 1 | 0.991 | **1.010** | no (no elevation) |
| rank-8 reconstruction MSE, HSR-NMF vs SR-SVD | HSR-NMF ≈ SR-SVD | SR-SVD 0.01078 | HSR-NMF **0.01173** | **yes** (≈ SR-SVD) |
| SR-NMF "feature collapse" | SR-NMF MSE ≫ SR-SVD | SR-NMF **0.01160** | — | no (≈ SR-SVD, no collapse) |

Full reconstruction-MSE grid (rank → SR-SVD / SR-NMF / HSR-SVD / HSR-NMF): 4 → 0.0154 / 0.0163 / 0.0162 / 0.0165; 8 → 0.0108 / 0.0116 / 0.0110 / 0.0117; 16 → 0.0053 / 0.0058 / 0.0056 / 0.0061. NMF is uniformly a hair worse than SVD for **both** eSR and eHSR — there is **no SR-specific collapse** and **no HSR-specific sparsity/bottleneck advantage**. The only sub-claim that holds is the weak one: HSR-NMF reconstructs about as well as SR-SVD (because eHSR ≈ eSR).

**Verdict: not reproduced** — HSR-NMF basis ≈ SR basis; the distinctive sparsity, bottleneck-alignment, and SR-collapse signatures are all absent.

---

**Paper claim.** *Sec 4.2 / Fig 4.* The piecewise-smooth topology of HSR is uniquely amenable to NMF: HSR-NMF yields a sparse code that tiles the state space, with elevated activations at bottleneck states, matching SVD-level compression, whereas NMF on the diffusive SR collapses.

**Faithful.** Expected SR / expected HSR built per Algorithm S1 (average over pretraining-task option-policies); Lee–Seung multiplicative-update NMF; SVD baseline; sparsity via Gini of basis columns; bottleneck = the four doorway states of the four-room; reconstruction MSE vs rank. **Simplified:** 12 pretraining tasks, N=68 grid, 8 basis vectors, no SVD-of-RW-SR eigenoption relearning between NMF runs. These do not bias against HSR — the same pipeline is applied to eSR and eHSR.

**Reproduction status.** `not_reproduced` — only the "HSR-NMF ≈ SR-SVD reconstruction" sub-part holds; sparsity, bottleneck, and SR-collapse do not. Evidence: `.trackio/logbook/evidence-package/claim4/` (`repro_claim4.py` + `results.json`).


---

# Claim 5: Scalable exploration with HSR (Fig 5)

---

**Executed result — PARTIAL.** The paper (Sec 4.3, Fig 5) claims HSR's temporally-extended structure enables more efficient, **scalable** exploration than the SR: HSR agents cover a larger fraction of the state space within a fixed budget, and the gap **grows** with maze size (SR-SPIE degrades in large mazes; HSR-SPIE maintained). We run count/novelty-driven (SPIE-style `r ∼ 1/visit`, a proxy for the SR row-norm bonus) softmax exploration in four-room mazes of growing size. **SR agent** = primitive actions only (single-step, diffusive); **HSR agent** = primitives + 8 eigenoptions (temporally-extended jumps). Fixed budget = 2·N steps, 8 seeds; coverage = fraction of states visited.

| Maze size N | SR coverage (primitive) | HSR coverage (options) | gap (HSR − SR) |
|---|---|---|---|
| 40 | 0.653 | 0.791 | **+0.137** |
| 68 | 0.636 | 0.699 | **+0.062** |
| 104 | 0.577 | 0.773 | **+0.196** |
| 148 | 0.677 | 0.608 | **−0.068** |
| 200 | 0.587 | 0.569 | **−0.018** |

**Mean gap = +0.062** (temporally-extended exploration *does* help on average, positive at 3 of 5 sizes). **But** the gap-vs-size trend has slope **−1.1e-3 < 0**: the advantage is present for small/medium mazes (N ≤ 104) and **reverses** for the two largest (N = 148, 200) — the **opposite** of the paper's Fig 5c "gap grows with size" scaling claim. In this build the fixed 8 eigenoptions (derived from the specific maze) do not scale to cover the much larger state space within the budget.

**Verdict: partial** — temporal-abstraction exploration improves coverage at small/mid scale, but the signature scaling result (advantage growing with maze size) does not reproduce.

---

**Paper claim.** *Sec 4.3 / Fig 5b–c.* HSR-augmented intrinsic-exploration agents significantly outperform their SR counterparts, covering more of the state space within a fixed budget, and the gap becomes increasingly pronounced as the environment grows; SR coverage degrades in larger mazes while HSR is maintained.

**Faithful.** Novelty/count intrinsic reward (the paper's `r_SR ∼ 1/‖M_{s:}‖₁` is explicitly "a proxy for state-visitation count"); temporally-extended option execution as the HSR ingredient; fixed step budget; coverage vs maze size across 8 seeds; the reported "gap grows with size" tested by the gap-vs-N slope. **Simplified:** four-room mazes of increasing side rather than procedurally-generated mazes; count-novelty rather than the full SARSA successor-predecessor SPIE objective; 8 eigenoptions recomputed per size. The mechanism tested (options escaping local diffusive barriers) is exactly the paper's stated cause.

**Reproduction status.** `partial` — coverage benefit at small/mid scale reproduces; the scaling (gap grows with size) does not. Evidence: `.trackio/logbook/evidence-package/claim5/` (`repro_claim5.py` + `results.json`).


---

# Conclusion

---

**Executive summary.** All five scored claims of *Hierarchical Successor Representation for Robust Transfer* (arXiv 2602.12753 / OpenReview txswvMHt4u) were exercised with executed numbers on a CPU-only tabular NumPy re-implementation, deterministic seeds. The result is an honest mix — **the theory reproduces exactly; the empirical HSR-over-SR advantages largely do not** in this independent bounded build.

- **Claim 1 — HSR Bellman operator is a max-norm contraction (Thm 3.1): REPRODUCED.** The affine operator `T^μM = B^μ + G^μM` has exact contraction modulus `‖G^μ‖∞ = E_μ[γ^τ]` = 0.9253 / 0.9500 / 0.9278 for three high-level policies, all < 1 and ≤ γ=0.95; a worst-case probe makes the measured factor equal ‖G^μ‖∞ (Eq 8 tight); value-iteration converges geometrically to the analytic fixed point `(I−G^μ)⁻¹B^μ` (residual 9e-13); and the primitive-only HSR coincides bit-for-bit with the standard RW-SR (residual 0.0). Theorem 3.1 holds as stated.
- **Claim 2 — Four-room transfer efficiency (Fig 2d): PARTIAL.** SR and HSR row-features both transfer to the new goal (G2 ≈ 54.5 / 52.0 episodes) while the one-hot Raw baseline fails (146.8) — but HSR is **not** significantly faster than SR (p=0.854).
- **Claim 3 — Representational robustness (Fig 2g): NOT REPRODUCED.** Relative change ρ_SR=1.965 ≈ ρ_HSR=2.031 (ratio 0.97, p=0.74); HSR is not more stable.
- **Claim 4 — HSR-NMF basis (Fig 4): NOT REPRODUCED.** HSR-NMF is not sparser (Gini 0.624 < 0.699), shows no bottleneck elevation (ratio 1.01), and there is no SR-NMF "feature collapse"; only the weak "HSR-NMF ≈ SR-SVD reconstruction" sub-claim holds.
- **Claim 5 — Scalable exploration (Fig 5): PARTIAL.** Temporally-extended (option) exploration raises coverage on average (+0.062) but the gap does **not** grow with maze size (slope −1.1e-3), contrary to Fig 5c.

Root cause of the empirical nulls: the faithfully-constructed expected-HSR is **numerically almost identical to the expected-SR** (near-identical room-block structure, reconstruction, and policy-change response), so the HSR-specific gains have nothing to attach to. This Trackio-native record covers **5 claim pages** with runnable scripts, raw JSON evidence, and rerun output. Fresh local reruns completed **5/5 scripts** in ≈ 34 s total. No Hugging Face GPU Job was used — every check is CPU-feasible.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 5 scored claims: 1 theory (contraction) + 4 empirical (transfer, robustness, NMF basis, exploration), tabular four-room | Paper-scale HSR/eigenoption/NMF pipeline + all figures + procedurally-generated mazes + baselines |
| Hardware | Local machine; CPU-only NumPy; single-thread; no HF Job | Author-specified setup, sweeps, seeds |
| Compute time | ≈ 34 s across 5 recorded scripts | Not estimated |
| Cost | ≈ $0 incremental local compute | Unknown |
| Outcome | Thm 3.1 reproduced exactly; empirical HSR-over-SR advantages not reproduced (2 partial, 2 negative) — reported honestly, no positive forced | Not attempted |

---

**📦 Artifact** `icml26-txswvmht4u/txswvmht4u-reproduction-bundle:v0` · dataset

https://huggingface.co/buckets/Crusadersk/icml26-hierarchical-successor-representation-repro-artifacts#icml26-txswvmht4u/txswvmht4u-reproduction-bundle:v0

---

The reproduction bundle contains the runnable scripts and raw evidence under `.trackio/logbook/evidence-package/claim1..5/` (`repro_claim<N>.py`, shared `hsr_core.py`, `results.json`) plus `artifacts/evidence.json`. Secrets, virtual environments, caches, and compiled `__pycache__` are excluded.


---

# Sources and provenance

---

- OpenReview: https://openreview.net/forum?id=txswvMHt4u
- arXiv (abstract): https://arxiv.org/abs/2602.12753
- arXiv (HTML, read for this repro): https://arxiv.org/html/2602.12753v1
- Published logbook (Space): https://huggingface.co/spaces/Crusadersk/icml26-hierarchical-successor-representation-repro

**Paper.** "Hierarchical Successor Representation for Robust Transfer," Changmin Yu & Máté Lengyel, ICML 2026 (OpenReview `txswvMHt4u`, arXiv 2602.12753).

**What was read and reproduced.** Section 3 + Appendix A.2 (HSR Bellman operator Eq 6–7, Theorem 3.1 contraction proof), Appendix B (Algorithm S1: eigenoption discovery → expected-HSR construction → NMF basis), and the three results sections 4.1 (transfer, Fig 2), 4.2 (HSR-NMF basis, Fig 3–4), 4.3 (scalable exploration, Fig 5). No official code release was used; every number here comes from an **independent NumPy re-implementation** of the equations and algorithm as stated in the paper.

**Scope of the five scored claims.** (1) Contraction of the HSR Bellman operator — *theory*, reproduced exactly. (2) Four-room transfer efficiency, HSR vs SR — *empirical*, partial. (3) Representational robustness to policy change — *empirical*, not reproduced. (4) HSR-NMF sparse/bottleneck basis — *empirical*, not reproduced. (5) Scalable exploration — *empirical*, partial.

**Honesty statement.** This record preserves the original claim boundaries and does **not** convert null or partial evidence into a full reproduction, nor does it force a positive verdict. Where the paper's HSR-over-SR advantage failed to appear, the executed numbers are reported as-is together with the likely cause: in this bounded tabular reconstruction the expected-HSR is numerically almost identical to the expected-SR (documented quantitatively in Claims 3–4), so the empirical gains largely vanish. This is stated as "not reproduced in this independent bounded build," not as a refutation of the paper — the authors' specific eigenoption discovery, online TD dynamics, NMF initialisation, and hyperparameters are not fully specified and could account for the difference.
