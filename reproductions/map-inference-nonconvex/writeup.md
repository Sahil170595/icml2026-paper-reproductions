# Claim 1 — Theorem 4.5: MpMap solves treewidth-one MAP(LRA) exactly

**Scored claim (paper).** *Theorem 4.5 (Tractability of MAP(LRA)).* If the global graph of a MAP(LRA) problem has treewidth one (and bounded diameter) and the density fulfils the Tractable MAP Conditions, then the constrained MAP can be computed exactly and tractably. The proof is by construction: the message-passing algorithm MpMap.

**Checkable consequence (this repro).** On tree-structured (treewidth-one) factor graphs with **non-convex SMT constraints** and **non-log-concave Ω^PP densities**, the value and argmax returned by our MpMap implementation must equal the true constrained MAP. We cross-check three ways: (a) against an **independent** multi-start SciPy optimizer of the joint objective; (b) MpMap's value must be **attained at its returned feasible assignment** (`eval_joint(assignment)`); (c) MpMap must **dominate** a dense brute-force grid.

## Measured vs target

| Quantity | Target | Measured |
|---|---|---|
| named instances: max rel \|MpMap − independent global opt\| | ~0 (exact) | **1.21e-09** |
| 30 random instances: max rel \|MpMap − independent global opt\| | ~0 (exact) | **1.97e-09** |
| max \|MpMap value − eval_joint(assignment)\| | 0 (attained exactly) | **4.37e-11** |
| min (MpMap value − brute-grid value) [dominance] | ≥ 0 | **+2.10e-04** |
| assignments feasible | all | **33/33 True** |
| Verdict | | **reproduced** |

Named instances: `I1_chain2` (2-var chain, MpMap 38.847287 vs SciPy 38.847287), `I2_chain3` (3-var chain, 2101.5174 vs 2101.5174), `I3_star4` (4-var star, 86736.6815 vs 86736.6815). The non-convex constraints genuinely bite: each edge encodes a disjunction `x_i ≤ x_j−s ∨ x_i ≥ x_j+s` (an excluded diagonal band), so the per-variable unconstrained modes are jointly infeasible and the constrained optimum sits in a different, disconnected polytope — yet MpMap recovers it exactly.

## Method

MpMap performs the upward pass of Eqs (4)–(6): variable→factor messages are point-wise products of incoming messages and the local density (Alg 2), and factor→variable messages are symbolic maximizations `m_{F_ij→X_j}(x_j)=max_{x_i} F_ij(x_i,x_j)·m_{X_i→F_ij}(x_i)` (Alg 3). Densities factorize as `p_ij(x_i,x_j)=A(x_i)·B(x_j)` (Ω^PP); constraints Δ_ij are unions of convex cells (a DNF ⇒ non-convex). Because densities are non-negative, `max` distributes over the product and the inner maximization reduces to the univariate `max-outPP` primitive (Claim 3). Argmax is recovered by back-tracking the fixed parent value into each child sub-tree. The independent reference is SciPy Nelder-Mead + Powell multi-start on the joint `eval_joint`.

**Verdict (from executed numbers).** Reproduced. MpMap equals an independent global optimizer to **1.2e-09** relative, attains its value at a **feasible** assignment (self-consistency **4.4e-11**), and dominates dense grids — exactly the exactness guaranteed by Theorem 4.5, across chains and stars with non-convex constraints and multimodal Ω^PP densities.

## Rerun
````bash
cd .trackio/logbook/evidence-package/claim1 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B repro_claim1.py
````
Deterministic; ≈ 22 s on one CPU core; writes `results.json`.


---

# Claim 2 — MpMap: recursive message passing over Ω^PP tree factors

**Scored claim (paper).** The **MpMap** algorithm (Alg 1–3, Eqs 4–6) performs recursive message passing over tree-shaped factor graphs using piecewise-polynomial (Ω^PP) representations, computing the constrained MAP by exploiting the tree structure (conditioning on a variable renders its children independent, so maximization decomposes over sub-trees).

**Checkable consequences (this repro).** (A) the single most-challenging step — the factor→variable message of Eq (5), `m_{F_ij→X_j}(x_j)=max_{x_i} F_ij(x_i,x_j)·m_{X_i→F_ij}(x_i)` — must equal a direct per-`x_j` maximization over the feasible `x_i` set; (B) the full recursive scheme must be exact on the paper's three benchmark tree families **STAR**, **SNOW** (ternary tree), **PATH** (linear chain).

## Measured vs target

| Quantity | Target | Measured |
|---|---|---|
| (A) message Eq (5) vs exact per-`x_j` reference, max abs err (20 edges, 4820 pts) | 0 | **5.68e-14** |
| (B) STAR (n=6): max rel \|MpMap − independent opt\| / max self-err | ~0 | **2.72e-09 / 5.96e-08** |
| (B) SNOW/ternary (n=7): max rel / max self-err | ~0 | **2.34e-09 / 7.15e-07** |
| (B) PATH/chain (n=4): max rel / max self-err | ~0 | **1.06e-09 / 5.82e-11** |
| Verdict | | **reproduced** |

## Method

The factor→variable reference (independent of MpMap's band construction) computes, for each `x_j`, the feasible `x_i`-intervals from the SMT cells and **exactly** maximizes `A(x_i)·m_{X_i→F_ij}(x_i)` over them (per-piece endpoint + stationary evaluation), then multiplies by `B(x_j)`. MpMap's `compute_msg` instead builds the message symbolically by (i) enumerating the constraint's affine bands, (ii) invoking `max-outPP` per band, (iii) point-wise-maxing the band contributions and multiplying by `B`. The two agree to **5.7e-14**. End-to-end, MpMap runs the post-order upward pass and root maximization on 15 random instances across the three topologies, all exact.

**Verdict.** Reproduced. The critical Eq (5) message operation matches an exact reference to machine precision, and recursive MpMap is exact on STAR, SNOW and PATH graphs — the recursive tree decomposition of Sec 4 works as claimed.

## Rerun
````bash
cd .trackio/logbook/evidence-package/claim2 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B repro_claim2.py
````
Deterministic; ≈ 13 s; writes `results.json`.


---

# Claim 3 — Correctness of the Ω^PP piecewise-polynomial message operations

**Scored claim (paper).** The message operations underlying MpMap are correct over piecewise-polynomial (Ω^PP) representations. In particular *Theorem A.5 (max-outPP is correct)*: for any piecewise polynomial `q` and affine bounds `l,u`, `sup_{x∈[l(y),u(y)]} q(x) = max-outPP(q,l,u)(y)`; the point-wise product and point-wise maximum are exact and the message piece-count is bounded (*Proposition A.6*: a symbolic maximum of a degree-`q`, `m`-piece polynomial over affine bounds has at most `8mq+4m+4` pieces).

**Checkable consequences (this repro).** Evaluate each operation against an **independent exact reference** on hundreds of random piecewise polynomials and affine bounds.

## Measured vs target

| Operation | Target | Measured |
|---|---|---|
| **max-outPP** (Thm A.5) vs exact per-interval sup, max abs err (300 inst., 4746 pts) | 0 | **3.55e-15** |
| **product** closure `pp_product(f,g)` vs `f·g`, max abs err (200 pairs) | 0 | **2.84e-14** |
| **pointwise max** `pp_pointmax(f,g)` vs `max(f,g)`, max abs err (200 pairs) | 0 | **0.00e+00** |
| **Prop A.6** piece bound `#pieces ≤ 8mq+4m+4` holds (200 inst.) | holds | **True** (worst ratio 0.125; max 6 pieces) |
| Verdict | | **reproduced** |

## Method

The `max-outPP` reference is *independent of the symbolic-in-`y` construction*: for each test `y` it fixes the numeric interval `[l(y),u(y)]` and maximizes the piecewise polynomial exactly (per-piece endpoints + derivative roots). Our `max_out` builds `m(y)` symbolically by enumerating `y`-breakpoints (where `l(y)` or `u(y)` crosses a breakpoint/stationary point of `q`, or where `l(y)=u(y)`) and, on each `y`-cell, taking the point-wise max of the candidate value-functions `q∘l`, `q∘u` and interior-stationary constants. Agreement is **3.55e-15** (machine precision) over 4746 comparisons. The product and point-wise max are exact by construction (`np.polymul`; root-splitting of `f−g`). The observed piece counts are far below the worst-case bound (ratio 0.125), consistent with the paper's own remark that the `8mq+4m+4` worst case rarely materializes.

**Verdict.** Reproduced. Every Ω^PP message primitive matches an exact reference to machine precision, and the Prop A.6 piece bound holds on all tested instances.

## Rerun
````bash
cd .trackio/logbook/evidence-package/claim3 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B repro_claim3.py
````
Deterministic; < 1 s; writes `results.json`.


---

# Claim 4 — Tractable MAP Conditions hold for Ω^PP; WMI ≠ MAP(LRA)

**Scored claim (paper).** *Definition 4.2 / Theorem 4.2 (Tractable MAP Conditions, TMC).* A function family Ω admits tractable MpMap if (i) it is **closed under product**, (ii) it has a **tractable symbolic supremum** (for bivariate `f∈Ω` and LRA bounds `l,u`, `sup_{x_j∈[l,u]} f ∈ Ω`), and (iii) a **tractable pointwise maximum** (for univariate `f,g∈Ω`, `max{f,g}∈Ω`). The paper identifies Ω^PP (piecewise polynomials that **factorize** into products of univariate polynomials) as satisfying TMC, and notes that **general** piecewise polynomials (the tractable-WMI class) can **violate (ii)** — so WMI and MAP(LRA) are incomparable.

**Checkable consequences (this repro).** Verify (i)(ii)(iii) numerically for Ω^PP; then demonstrate a **non-factorized** bivariate whose symbolic supremum is **not a polynomial** (property (ii) fails without factorization).

## Measured vs target

| Property | Target | Measured |
|---|---|---|
| (i) product closure error (150 pairs) | 0 | **8.53e-14** |
| (ii) symbolic supremum error, `sup = a(x_i)·max-out(b)` (120 inst.) | 0 | **7.11e-15** |
| (iii) pointwise maximum error (150 pairs) | 0 | **0.00e+00** |
| (iv) non-factorized `f=x_j³−3x_i x_j`: deg-8 poly-fit RMSE of `sup_{x_j} f` | ≫ 0 (not a polynomial) | **3.79e-02** |
| (iv) that sup equals the algebraic form `2·x_i^{3/2}`, RMSE | ~0 | **1.20e-07** |
| (iv) factorized control: sup is a polynomial (poly-fit RMSE) | ~0 | **4.09e-16** |
| Verdict | | **reproduced** |

## Method

For the factorized bivariate `f(x_i,x_j)=a(x_i)·b(x_j)` with `a≥0`, the symbolic supremum over `x_j∈[l(x_i),u(x_i)]` equals `a(x_i)·max-outPP(b,l,u)(x_i)`, a piecewise polynomial in `x_i` — in Ω^PP — and it matches the exact reference to **7e-15**. For the **non-factorized** cubic `f=x_j³−3x_i x_j`, the interior maximizer `x_j*=−√x_i` gives `sup = 2·x_i^{3/2}`: an **algebraic, non-polynomial** function of `x_i`. No finite-degree polynomial represents it (deg-8 least-squares RMSE **0.038**), while it matches `2·x_i^{3/2}` to **1.2e-07** — a direct demonstration that property (ii) fails for general piecewise polynomials, hence Ω^PP must factorize and the tractable-WMI class ≠ the tractable-MAP(LRA) class.

**Verdict.** Reproduced. TMC (i)(ii)(iii) hold exactly for Ω^PP, and the WMI≠MAP(LRA) incomparability is demonstrated by an explicit non-factorized counterexample whose symbolic supremum is algebraic, not polynomial.

## Rerun
````bash
cd .trackio/logbook/evidence-package/claim4 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B repro_claim4.py
````
Deterministic; < 1 s; writes `results.json`.


---

# Claim 5 — Complexity & scalability: tractable, diameter-sensitive, exact

**Scored claim (paper).** MpMap is a **scalable** exact solver for the tractable fragment (Prop A.7 bounds the message piece-count; Thm A.8 bounds the total message size; Q1/Fig 5 shows MpMap out-scaling approximate optimizers on STAR/SNOW/PATH). For **bounded-diameter** trees the cost is polynomial (Thm 4.5 tractable regime); the worst case is exponential **in the graph diameter** (Thm A.8) — e.g. PATH graphs with maximal diameter `d=N−1`.

**Checkable consequences (this repro).** Measure MpMap runtime/piece-count growth in `n`; contrast **STAR** (diameter 2) vs **PATH** (diameter `n−1`) vs **brute-force** exhaustive grid (cost `grid^n`), while confirming exactness.

## Measured vs target

| Quantity | Target | Measured |
|---|---|---|
| STAR (diam 2) runtime exponent `n^a` | polynomial (`a<3`) | **n^1.19** |
| STAR exactness vs independent optimizer (n=4,6,8) | ~0 | **7.5e-12** |
| PATH (diam `n−1`) runtime exponent `n^a` | ≥ STAR (diameter cost) | **n^1.46** |
| brute-force grid MAP: `log(time)/variable` | > 0 (exponential) | **2.23** (base ≈ **9.3**) |
| MpMap solves STAR n=20 | tractable | **0.52 s** (vs brute `grid¹¹·²⁰ ≈ 6.7e20` evals) |
| Verdict | | **reproduced** |

STAR runtimes (s): n=3→0.055, 5→0.116, 8→0.195, 12→0.319, 16→0.452, 20→0.520 — max message pieces constant at 12. PATH runtimes (s): n=3→0.086, 8→0.534, 12→0.879, 16→1.205, 20→1.473 — max pieces constant at 16, cost driven by the growing polynomial degree along the chain. Brute-force grid MAP already needs 0.174 s at n=5 (grid 15) and is astronomically infeasible by n=20.

## Method & honest scope

All problems use multimodal Ω^PP densities and a 3-band "comb" non-convex constraint per edge. STAR (bounded diameter) scales as **n^1.19** and stays exact (**7.5e-12** vs an independent optimizer); PATH scales at the **strictly higher** exponent **n^1.46**, reproducing the diameter-sensitivity of Thm A.8. Brute force is empirically exponential (`log(time) ≈ 2.23·n`, base ≈ 9.3 ≈ the grid resolution). **Nuance:** on these structured (non-adversarial) instances MpMap remains polynomial for **both** topologies — the Thm A.8 exponential-in-diameter blow-up is a worst case that the paper itself notes "can play out with significantly less complexity in practice." What is verified here is the paper's operative claim: MpMap is polynomial and exact for the tractable (bounded-diameter, treewidth-one) regime, is diameter-sensitive, and vastly out-scales exact brute force.

**Verdict.** Reproduced. Polynomial + exact for bounded-diameter trees, higher polynomial exponent for PATH (diameter cost), brute force exponential; MpMap solves n=20 in half a second where exhaustive search needs ~10²⁰ evaluations.

## Rerun
````bash
cd .trackio/logbook/evidence-package/claim5 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B repro_claim5.py
````
Deterministic; ≈ 7 s; writes `results.json`.


---

# Claim 6 — PaMAP: convex-polytope decomposition for general non-convex MAP

**Scored claim (paper).** *Contribution C2 / Sec 5 / Fig 1, 7.* When MpMap's tractability conditions do not apply, **PaMAP** (Alg 4) approximates MAP(LRA) over arbitrary SMT(LRA) constraints by **decomposing the non-convex feasible region into convex polytopes**, running a local convex-polytope optimizer on each, and keeping the best — with **upper-bound pruning** (Fig 6) to skip polytopes that cannot improve the incumbent. It recovers the global constrained optimum where constraint-agnostic optimizers (e.g. Adam) return infeasible or sub-optimal points.

**Checkable consequences (this repro).** (a) PaMAP == fine-grid ground-truth constrained MAP, including the paper's **Example 2.2** constrained optimum ≈ (1.83, 1.83); (b) a constraint-agnostic optimizer lands **infeasible/suboptimal**; (c) valid upper-bound pruning skips polytopes while staying exact.

## Measured vs target

| Quantity | Target | Measured |
|---|---|---|
| Example 2.2: PaMAP rel vs ground-truth MAP | ~0 | **3.18e-05** (value 2.01548 at **(1.82, 1.83)**, feasible) |
| Example 2.2: constraint-agnostic optimum | infeasible | **infeasible** at (1.16, 1.41), p=2.644 |
| battery (N=12, excluded central region): PaMAP == ground truth | all | **12/12** (worst rel 3.7e-04) |
| battery: constraint-agnostic infeasible/suboptimal | most | **11/12** |
| upper-bound pruning: polytopes skipped (of 4) while exact | > 0 | **avg 1.75/4** |
| Verdict | | **reproduced** |

## Method

PaMAP enumerates the convex cells whose union is the SMT feasible region, and for each runs SciPy **SLSQP** (a convex-polytope-constrained local optimizer) from a grid of starts, keeping the best feasible value (Alg 4). Pruning uses a **valid** density upper bound per cell — for a Gaussian-mixture density, `Σ_k h_k·exp(−w_k·dist(cell, c_k)²)` where `dist` is the projection distance of mode `c_k` onto the (convex) cell — so a cell whose bound is below the incumbent is safely skipped without losing exactness. **Example 2.2** uses the paper's verbatim constraint `[0,2]² ∧ (x₂≤1 ∨ x₂>2x₁ ∨ x₂>4.75−2x₁)`; PaMAP relocates the optimum from the infeasible unconstrained mode to the feasible corner **(1.82, 1.83)** — matching the paper's reported (1.83, 1.83). The battery uses a `[0,3]²` box with an excluded central square `(1,2)²` (a non-convex hole ⇒ 4 convex cells) and places the tallest density mode inside the hole, so the constraint-agnostic optimizer is driven infeasible.

**Verdict.** Reproduced. PaMAP recovers the global constrained MAP (Example 2.2 to 3e-05; 12/12 on the battery) where constraint-agnostic optimization fails (11/12 infeasible/suboptimal), and upper-bound pruning skips ~44% of polytopes while remaining exact.

## Rerun
````bash
cd .trackio/logbook/evidence-package/claim6 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B repro_claim6.py
````
Deterministic; ≈ 3 s; writes `results.json`.


---

# Conclusion

---

**Executive summary.** All **6 scored claims** of *The Theory and Practice of MAP Inference over Non-Convex Constraints* (arXiv 2602.08681 / OpenReview jIZqAemuqk) reproduce with executed numbers on an independent CPU-only NumPy/SciPy re-implementation, verified against brute-force MAP and independent optimizers on small factor graphs.

- **Claim 1 — Theorem 4.5 (exact on treewidth-one):** MpMap equals an independent global optimizer to **1.2e-09** relative, attains its value at a **feasible** assignment (self-consistency **4.4e-11**), and dominates dense grids — across chains and stars with non-convex SMT constraints and non-log-concave Ω^PP densities (33/33 feasible).
- **Claim 2 — MpMap recursive message passing:** the Eq (5) factor→variable message matches an exact per-`x_j` reference to **5.7e-14**; end-to-end MpMap is exact on **STAR, SNOW and PATH** (max rel **2.7e-09**).
- **Claim 3 — Ω^PP message operations (Thm A.5):** `max-outPP` **3.6e-15**, product **2.8e-14**, pointwise-max **0.0** vs exact references; the Prop A.6 piece bound `8mq+4m+4` holds.
- **Claim 4 — TMC & WMI≠MAP(LRA):** the three closures hold to **~1e-14**, and a non-factorized bivariate has an **algebraic (non-polynomial)** symbolic supremum `2·x^{3/2}` (poly-fit RMSE **3.8e-2**), confirming Ω^PP must factorize.
- **Claim 5 — complexity/scalability:** MpMap is polynomial (**n^1.19**) and exact (**7.5e-12**) for bounded-diameter STAR, diameter-sensitive (PATH **n^1.46**), and out-scales exponential brute force (n=20 in **0.52 s** vs `~6.7e20` grid evaluations).
- **Claim 6 — PaMAP:** convex-polytope decomposition recovers the global constrained MAP (Example 2.2 to **3.2e-05**, at (1.82, 1.83) ≈ the paper's (1.83, 1.83); battery **12/12**) where a constraint-agnostic optimizer is **infeasible/suboptimal** (11/12), with valid upper-bound pruning skipping ~44% of polytopes.

This Trackio-native record covers **6 claim pages** with runnable scripts, raw `results.json`, and SHA-256. Fresh local reruns completed **6/6** scripts in ≈ **47 s** total on one CPU core. No GPU was used: these checks are CPU-feasible by design; the paper's large-scale benchmark suite and OMT/baseline comparisons are out of scope.

---

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 6 scored claims: Thm 4.5 exactness, MpMap algorithm, Ω^PP message-op correctness (Thm A.5), TMC/WMI incomparability, complexity/scalability (Thm A.8), PaMAP | Paper-scale implementation + every empirical claim (1059 STAR/SNOW/PATH instances, SDD trajectories, VAE imputation, OMT baselines) |
| Method | From-scratch NumPy MpMap + Ω^PP algebra + PaMAP; brute-force / SciPy / exact-interval references | wmi-pa enumeration, SHGO / SoS-moment optimizers, OptiMathSAT, CDCL-OCAC, PAL densities |
| Hardware | Local CPU-only, single thread; no GPU/network | Paper compute for the full benchmark sweeps |
| Compute time | ≈ 47 s across 6 deterministic scripts | Not estimated |
| Cost | ≈ $0 incremental local compute | Unknown; substantial |
| Outcome | **6/6 scored claims reproduced** within stated acceptance rules, with independent references and honest scope notes | Not attempted |

---

The reproduction bundle contains the runnable scripts and raw evidence under `.trackio/logbook/evidence-package/` (`mpmap_core.py` + `claim{1..6}/repro_claim{1..6}.py` + `results.json`) and a copy under `artifacts/`. Secrets, virtual environments, and caches are excluded. SHA-256 for every script and result is on the *Evidence and rerun* page.


---

# Sources and provenance

## Paper
- **Title:** The Theory and Practice of MAP Inference over Non-Convex Constraints
- **Authors:** Leander Kurscheidt, Gabriele Masina, Roberto Sebastiani, Antonio Vergari
- **arXiv:** 2602.08681 (v1, submitted 9 Feb 2026, cs.LG/stat.ML) · **OpenReview:** jIZqAemuqk
- **PDF SHA-256:** `c118419b7da2b634a1d71c04e25f624c1e251b0d3e166678087779871e43ed49` (39 pages; no arXiv HTML available, text extracted via `pdftotext`).

## Claims covered (6 scored)
1. **Theorem 4.5** — treewidth-one MAP(LRA) is exact/tractable via MpMap (proof by construction).
2. **MpMap** algorithm — recursive message passing over Ω^PP tree factors (Alg 1–3, Eqs 4–6, Sec 4).
3. **Ω^PP message operations** — Theorem A.5 (max-outPP correct), TMC product & pointwise-max, Prop A.6 piece bound.
4. **Tractable MAP Conditions** (Def 4.2 / Thm 4.2) for Ω^PP; general PP violate (ii) ⇒ WMI ≠ MAP(LRA) (Sec A.1).
5. **Complexity / scalability** — Prop A.7, Thm A.8, Q1 / Fig 5 (polynomial for bounded diameter, diameter-sensitive; brute exponential).
6. **PaMAP** — convex-polytope decomposition with pruning (Sec 5 / Alg 4 / Fig 1, 6, 7); Example 2.2 constrained optimum ≈ (1.83, 1.83).

## What we implemented independently
A from-scratch NumPy/SciPy re-implementation: univariate piecewise-polynomial algebra (product, pointwise-max, affine composition), the `max-outPP` symbolic supremum, the MpMap upward pass + argmax back-tracking for tree factor graphs with non-convex SMT (union-of-cells) constraints and factorized Ω^PP densities, and PaMAP (cell enumeration + SLSQP per polytope + valid Gaussian-distance upper-bound pruning). Independent references: exhaustive/vectorized grid MAP, exact per-interval maximization, and SciPy multi-start optimization.

## Deliberately out of scope
The paper's large-scale empirical study — the 1059-instance STAR/SNOW/PATH benchmark suite, the Stanford Drone Dataset trajectory experiments, VAE data-imputation, and the OMT baselines (OptiMathSAT, CDCL-OCAC), SHGO/SoS-moment optimizers, and PAL densities — are not reproduced. This logbook targets the **scored theoretical claims** and the **core algorithmic mechanisms**, verified on small factor graphs where an exact/independent reference exists, which is what is CPU-feasible and decisive.

## Provenance note
Every measured value originates from real stdout of the six scripts (captured into `results.json`). No numbers are transcribed from the paper as if measured; paper targets are labelled as targets. Self-reported "verified" booleans in the JSON are convenience flags — the load-bearing evidence is the measured error/agreement values in the tables.
