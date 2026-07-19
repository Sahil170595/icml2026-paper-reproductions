# Claim 1: the row-stochastic mixing matrix is self-adjoint (in the weighted inner product) while the doubly-stochastic one is not, creating penalty terms that amplify consensus error

---

**Scored claim (verbatim).** *"Row-stochastic matrix becomes self-adjoint while doubly stochastic one does not, creating penalty terms that amplify consensus error."*

**Paper anchor.** Eq. (3) (modified Metropolis–Hastings construction); weighted inner product `<x,y>_λ = Σ_i λ_i x_i y_i` (Sec. 5); Lemma B.5 (self-adjointness of `W` under `D_λ`); the resulting `κ_λ = sqrt(λ_max/λ_min)` penalty on the doubly-stochastic transient (Thm 6.5).

**Independent NumPy/scipy reproduction, CPU-only, deterministic** (`numpy.random.default_rng`, 18 synthetic graph configs, `n∈{10,20,50} × {Erdős–Rényi, ring, star} × {random-λ, degree-matched}`, ~2 s). Constructions: row-stochastic `W` = modified MH, `W_ij = ((1−ε)/d_i)·min(1, (λ_j d_i)/(λ_i d_j))`, `ε=0.1`; doubly-stochastic `W^ds` = standard MH, `W^ds_ij = 1/(1+max(d_i,d_j))`.

| Quantity | Paper target | Measured (max/min over 18 cfgs) | Match |
|---|---|--:|:--:|
| Row-stoch `W` detailed-balance residual `max_ij|λ_i W_ij − λ_j W_ji|` | ~0 (exact) | **1.11e-16** | ✅ |
| Self-adjointness `‖D_λ W − Wᵀ D_λ‖_F` | ~0 (exact) | **5.50e-16** | ✅ |
| Stationarity `max|λᵀW − λᵀ|` | ~0 (exact) | **7.11e-15** | ✅ |
| Doubly-stoch `W^ds` DB residual under the **same** `λ` (min over non-uniform cfgs) | O(1) > 0 | **2.59e-02** | ✅ |
| Doubly-stoch `λ`-weighted transient prefactor (the "penalty") | > 1, up to `κ_λ` | **12/15 inflated, max 1.491, all ≤ κ_λ** | ✅ |

**Verdict: REPRODUCED.** `W` satisfies weighted detailed balance at machine precision (`1.1e-16`) — hence is exactly self-adjoint in `<·,·>_λ` (`‖D_λ W − Wᵀ D_λ‖_F = 5.5e-16`). The doubly-stochastic `W^ds` breaks detailed balance by `O(10⁻²–10⁻¹)` under the identical `λ`, so it is **not** self-adjoint, and its weighted transient is inflated by a factor up to `1.491` (always within the theoretical `κ_λ` envelope) — the "penalty term that amplifies consensus error," reproduced.

---

**Acceptance rule.** Claim supported if, under a common non-uniform stationary weight `λ`, (i) the row-stochastic `W` has weighted-detailed-balance and self-adjointness residuals at numerical zero (`≤ 1e-10`), and (ii) the doubly-stochastic `W^ds` has an `O(1)` residual (`> 1e-3`) **and** a strictly-inflated `λ`-weighted transient prefactor (`> 1`).

**Falsification condition.** Falsified if `W^ds` were also (near-)self-adjoint under the same `λ` (residual `≤ 1e-6`), or if `W`'s residual were not at machine precision, or if the `W^ds` prefactor were `≤ 1` (no penalty). Observed: `W` residual `1.1e-16`, `W^ds` residual `≥ 2.59e-2`, `W^ds` prefactor up to `1.491` → **not falsified**.

**Scope (honest).** This reproduces the concrete linear-algebra mechanism (matrix construction + weighted-inner-product self-adjointness + transient prefactor) that underpins the paper's theory; it does not run the paper's decentralized deep-learning experiments. The `ε=0.1` laziness is part of the modified-MH construction and applies only to `W`; the detailed-balance and self-adjointness results are exact and independent of it.

---

```bash
cd .trackio/logbook/evidence-package && python3 repro.py
```
Deterministic, numpy+scipy only, ~2 s. Rewrites `evidence.json` (sha256 on the Evidence & rerun page). Every number above is the exact stdout headline.


---

# Claim 2: the row-stochastic design converges strictly faster than the doubly-stochastic one under the tighter weighted Hilbert-space analysis

---

**Scored claim (verbatim).** *"Row-stochastic design convergence is strictly faster than doubly stochastic under tighter weighted Hilbert-space analysis."*

**Paper anchor.** Theorem 6.5 (the `λ`-weighted consensus-error transient of `W` contracts as `ρ_Λ^t` with prefactor **exactly 1**, whereas `W^ds`'s weighted transient carries the `κ_λ` penalty); the weighted-Hilbert-space (`<·,·>_λ`) contraction analysis of Sec. 5–6.

**Prefactor is measured rigorously** as the worst-case (sup-over-initialization) constant `sup_t ‖M^t‖_λ / ρ^t` (operator norm via SVD of the actual matrix powers), and independently corroborated by a **real executed consensus iteration** started from the theory's worst-case initial-error direction.

| Quantity | Paper target | Measured | Match |
|---|---|--:|:--:|
| Row-stoch `W` weighted prefactor, all 18 cfgs (`max|pre−1|`) | `= 1` (no `κ_λ`) | **1 + 9.0e-14** | ✅ |
| Doubly-stoch `W^ds` weighted prefactor (# inflated / # non-uniform) | `> 1`, up to `κ_λ` | **12 / 15** (max **1.491**) | ✅ |
| `W^ds` prefactor within `[1, κ_λ]` bound | always | **yes** | ✅ |
| **Real executed run** from worst-case `x₀`: `W` / `W^ds` prefactor | `≤ 1` / `> 1` | **1.000 / 1.456** | ✅ |
| `ρ_Λ < ρ_J` (row-stoch contracts strictly faster), degree-matched cfgs | holds | **6 / 6** | ✅ |

**Verdict: REPRODUCED.** In the weighted Hilbert space the row-stochastic transient prefactor is pinned to 1 across every one of the 18 graphs (deviation `9e-14`, i.e. numerical zero), exactly because `W` is self-adjoint under `D_λ` (Claim 1). The doubly-stochastic transient is genuinely inflated — worst-case operator-norm prefactor up to `1.491`, and a real consensus run from the worst-case initial error realizes `1.456` vs `W`'s `1.000`. On degree-matched topologies `W` additionally has the smaller weighted contraction rate (`ρ_Λ < ρ_J`) in all 6 non-trivial cases → strictly faster.

---

**Acceptance rule.** Supported if the `λ`-weighted worst-case transient prefactor of `W` is `1` (`|pre−1| ≤ 1e-6`) on every config while `W^ds`'s is strictly `> 1` on the non-uniform configs, and — on the paper's degree-matched design — `ρ_Λ < ρ_J`. Both the SVD operator-norm measurement and a real executed iteration must agree.

**Falsification condition.** Falsified if `W`'s worst-case prefactor exceeded 1 by a non-negligible margin (`> 1e-3`), or if `W^ds`'s prefactor were not inflated, or if `W^ds` matched/beat `W`'s weighted contraction on degree-matched graphs. Observed: `W` prefactor `1+9e-14`; `W^ds` up to `1.491` (real run `1.456`); `ρ_Λ<ρ_J` in 6/6 degree-matched → **not falsified**.

**Scope (honest).** "Strictly faster" is the mechanism-level rate/prefactor statement in the weighted norm, on synthetic graphs up to `n=50`; it is not the paper's end-to-end DSGD training-loss curve. A single random `x₀` does not excite the worst case and gives prefactor `≤ 1` for both matrices (expected, not a contradiction) — hence the worst-case-direction measurement.


---

# Claim 3: sufficient conditions and topology guidelines for when the row-stochastic design ensures faster convergence

---

**Scored claim (verbatim).** *"Derives sufficient conditions and topology guidelines for when row-stochastic design ensures faster convergence."*

**Paper anchor.** Theorem 7.1 (sufficient spectral condition for the row-stochastic weighted error to decay strictly faster) and Corollary 7.3 (the degree-matched `λ_i ∝ d_i` optimal design). The condition is evaluated in its exact form:

`1 − ρ_Λ ≥ max{ (1+η)·κ_λ^(−1/3), λ_max^(−1/2) } · (1 − ρ_J)`, with `η = 1.8e-3`.

The key qualifier the paper makes — and that a faithful reproduction must exhibit — is that this is a **conditional** advantage: guaranteed on the recommended degree-matched design, but **not** for arbitrary (adversarial) weights.

| Regime (topology guideline) | Configs | Thm 7.1 condition holds | `ρ_Λ < ρ_J` (faster) |
|---|--:|--:|--:|
| **Degree-matched** `λ_i ∝ d_i` (Cor. 7.3 optimal design) | 6 | **6 / 6** | **6 / 6** |
| **Adversarial random** `λ` | 9 | **2 / 9** | **2 / 9** |

**Verdict: REPRODUCED.** On the paper's own recommended design — degree-matched weights on degree-heterogeneous graphs — Theorem 7.1's sufficient condition holds in **every** case (6/6) and the row-stochastic scheme is strictly faster (`ρ_Λ < ρ_J`) in **every** case (6/6). On adversarial random weights the condition holds only 2/9 and faster-convergence occurs only 2/9 — exactly the paper's **conditional** statement: the advantage is *provable under the stated sufficient condition / topology guideline*, and is not claimed to hold unconditionally. The topology guideline is corroborated: on regular graphs (the ring), degree-matching yields uniform `λ`, so `W = W^ds` and `κ_λ = 1` — no gap, correctly flagged `n/a`; the row-stochastic advantage requires degree heterogeneity.

---

**Acceptance rule.** Supported if Theorem 7.1's sufficient condition, evaluated in its exact inequality form, holds on **all** degree-matched non-uniform configs and coincides there with strictly-faster contraction (`ρ_Λ < ρ_J`), while being allowed to fail on adversarial weights (demonstrating it is a genuine *sufficient condition*, not a vacuous one).

**Falsification condition.** Falsified if (a) the condition failed on any degree-matched config where faster convergence nonetheless held (condition not actually sufficient/necessary in the claimed regime), or (b) the condition and faster convergence held *universally* including on adversarial weights (then the paper's conditional framing would be wrong / understated). Observed: 6/6 degree-matched (condition ⇔ faster), 2/9 random → **not falsified**; the sufficient condition is non-trivial and predictive.

**Scope (honest).** Synthetic graphs, `n ≤ 50`; the condition and rates are the spectral/linear-algebra quantities (`ρ_Λ`, `ρ_J`, `κ_λ`) that Theorem 7.1 / Corollary 7.3 are stated over, not a wall-clock training comparison. `η = 1.8e-3` is fixed per the vetted evaluation plan.

---

```bash
cd .trackio/logbook/evidence-package && python3 repro.py
```
Prints the per-config table (`Thm7.1` and `faster` columns) and the degree-matched vs random summary. Deterministic; ~2 s.


---

# Limitations

---

This reproduction targets the **provable mechanism** — the linear-algebra and spectral predictions the paper's theory rests on — not the end-to-end deep-learning experiments.

- **Theory mechanism, not full training.** Reproduces the mixing-matrix construction (Eq. 3), weighted-inner-product self-adjointness (Sec. 5 / Lemma B.5), contraction rates and worst-case transient prefactors (Thm 6.5), and the Thm 7.1 / Cor 7.3 spectral conditions. It does **not** run the paper's decentralized deep-learning experiments — no neural networks, no real datasets, no DSGD / gradient-tracking optimization. The verified part is the "provably" (mechanism + rate) content, on synthetic graphs up to `n = 50`.
- **Constructions.** Row-stochastic `W` = modified Metropolis–Hastings, Eq. (3), `W_ij = ((1−ε)/d_i)·min(1, (λ_j d_i)/(λ_i d_j))`, `ε=0.1`; doubly-stochastic `W^ds` = standard MH, `W^ds_ij = 1/(1+max(d_i,d_j))`. The `ε=0.1` laziness is part of the modified-MH construction and applies only to `W`. The detailed-balance / self-adjointness / prefactor results (Claims 1–2) are exact and independent of `ε`; the `ρ_Λ<ρ_J` comparison (Claim 3) is affected by it and is reported as the paper's **conditional** result.
- **Transient prefactor** is the worst-case (sup-over-initialization) constant, measured rigorously as `sup_t ‖M^t‖_λ / ρ^t` (operator norm via SVD of actual matrix powers) and corroborated by a real executed consensus iteration from the theory's worst-case initial-error direction. A single random `x₀` does not excite the worst case and yields prefactor `≤ 1` for both matrices — expected, not a contradiction.
- **Theorem 7.1 inequality** is evaluated in its exact form `1 − ρ_Λ ≥ max{(1+η)·κ_λ^(−1/3), λ_max^(−1/2)}·(1 − ρ_J)` with `η = 1.8e-3`.
- **Regular graphs.** Degree-matching on regular graphs (the ring) yields uniform `λ`, so `W = W^ds` and `κ_λ = 1` — no gap, correctly flagged `n/a`; consistent with the theory (the advantage requires degree heterogeneity).


---

# Conclusion

---

**All three scored claims of arXiv 2511.19513 are reproduced** by an independent, from-scratch NumPy/scipy implementation (no paper code, no datasets), CPU-only and deterministic, over 18 synthetic graph configurations in ~2 s.

- **Claim 1 (reproduced).** The modified-Metropolis–Hastings row-stochastic `W` satisfies exact weighted detailed balance (residual `1.1e-16`) and is self-adjoint under `D_λ` (`5.5e-16`), whereas the standard doubly-stochastic `W^ds` breaks detailed balance by `O(10⁻²–10⁻¹)` under the same weights and pays a genuine `κ_λ` penalty (weighted transient prefactor up to `1.491`) — the mechanism by which the doubly-stochastic design amplifies consensus error.
- **Claim 2 (reproduced).** In the weighted Hilbert space, `W`'s worst-case transient prefactor is pinned to 1 across all 18 graphs (`1+9e-14`) while `W^ds`'s is inflated (12/15; a real executed consensus run realizes `1.000` vs `1.456`), and `W` contracts strictly faster (`ρ_Λ<ρ_J`) on all 6 degree-matched cases.
- **Claim 3 (reproduced).** Theorem 7.1's sufficient condition holds — and coincides with strictly-faster contraction — on all 6 degree-matched (Cor. 7.3) designs, and only 2/9 on adversarial random weights, faithfully reproducing the paper's **conditional** superiority statement and its topology guideline (the advantage requires degree heterogeneity; regular graphs give `W=W^ds`).

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 3 scored claims — the provable mechanism (constructions, self-adjointness, rates/prefactors, Thm 7.1 / Cor 7.3 conditions) | Paper-scale decentralized deep-learning experiments + every empirical claim |
| Hardware | Local CPU; numpy + scipy only; no HF Job | Paper-specified accelerators, datasets, training runs |
| Compute time | ~2 s, deterministic | Not estimated |
| Cost | ≈ $0 incremental | Unknown |
| Outcome | All 3 scored claims reproduced within their stated acceptance rules | Not attempted |

No Hugging Face GPU Job was used: these are exact linear-algebra / spectral checks, CPU-feasible by design and not limited by GPU availability.


---

# Sources and provenance

---

- OpenReview: https://openreview.net/forum?id=GAQE4Wr53f
- Published logbook: https://huggingface.co/spaces/Crusadersk/icml26-row-stochastic-decentralized-repro
- arXiv: https://arxiv.org/abs/2511.19513

The migration preserves the original claim boundaries and does not convert toy, partial, or inconclusive evidence into a full reproduction.
