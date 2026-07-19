# Claim 1 — Definition 1: evolutionary selection model as a DAG G^(T)

**Measured vs target (executed).** Independent NumPy/networkx build of G^(T) and contrast with a one-shot static selection model.

| Quantity | Target / rule | Measured | Match |
|---|---|---|---|
| DAG acyclic + Definition-1 inventory | traits X^(0..T), factors eps^(0..T), selection S^(0..T-1); each S^(t) a leaf child of X^(t) | acyclic ✓; **4** traits, **12** heritable factors, **3** selection nodes; every S^(t) leaf-child of X^(t) ✓; 24 edges | ✅ |
| Faithful realization (unselected data, n=400k) | d-separated pairs \|corr\|≈0 (<0.02); adjacent pairs \|corr\|>0.05 | max\|corr\|(48 d-sep pairs) = **0.0043**; min\|corr\|(21 adjacent) = **0.534** | ✅ |
| Distinct from one-shot static model | # d-sep relations where evolutionary ≠ static > 0 | **2283 / 12720 = 17.95%** of tested relations differ (evol has 3 selection nodes vs 1) | ✅ |

**Verdict: VERIFIED.** G^(T) is a well-formed DAG matching Definition 1, faithfully realized by the SCM, and provably NOT equivalent to a one-shot static selection model — e.g. `eps_0^(0) ⟂̸ eps_1^(0) | eps_0^(1)` holds under evolutionary selection but is an independence under static selection.

---

**Paper claim (verbatim).** "Definition 1 formalizes an evolutionary selection model as a DAG G^(T) over trait variables X^(0)...X^(T), heritable factors epsilon^(0)...epsilon^(T), and reproduction/selection indicators S^(0)...S^(T-1), distinguishing it from one-shot static selection models (Section 2, Definition 1)."

**Checkable consequences + acceptance rule.** Definition 1 is definitional, so we verify its structural content:
- **(A)** the constructed graph is a DAG (acyclic) whose node/edge inventory matches Definition 1 exactly: T+1 trait generations, K heritable-factor components per generation, T selection nodes, inheritance edges eps_k^(t-1)→eps_k^(t), factor→trait edges eps_k^(t)→X^(t), and trait→selection edges X^(t)→S^(t);
- **(B)** the SCM faithfully realizes G^(T): on **un-selected** data every marginally d-separated pair is empirically independent (\|corr\|<0.02) and every adjacent pair is dependent (\|corr\|>0.05);
- **(C)** the evolutionary model is genuinely distinct from a one-shot static selection model (single global selection node): the two graphs disagree on a strictly positive number of d-separation relations once selection is conditioned.

**Falsification.** The claim fails if the graph is cyclic / mis-structured, or if the SCM is unfaithful, or if the evolutionary and static models imply identical CI constraints (disagreements = 0).

---

**Model (Definition 1).** Over T+1 generations, each generation t has K heritable-factor components eps_k^(t) (the components of epsilon^(t)), a phenotype/trait X^(t), and (for t<T) a reproduction/selection indicator S^(t). Structural equations (linear-Gaussian): eps_k^(0)∼N(0,1); eps_k^(t)=ρ·eps_k^(t-1)+√(1−ρ²)·u (per-component inheritance chain, ρ=0.6); X^(t)=Σ_k eps_k^(t)+env (env∼N(0,0.7²)) — so X^(t) is a **collider** of its heritable factors; S^(t) is a child of X^(t) (selection depends on the trait). Here K=3, T=3.

**Static comparison model.** Same traits/factors but a single one-shot global selection node S\* that is a child of the final trait X^(T) only (classic one-shot selection-bias graph).

**Method.** Build both graphs in `networkx`; check acyclicity and inventory; simulate 400k un-selected individuals and test marginal correlations to confirm faithfulness; enumerate d-separations over all observed pairs and conditioning sets up to size 2, conditioning on all selection nodes (evolutionary) vs the single S\* (static), and count disagreements.

---

**Controls.** The faithfulness check is a positive+negative control: d-separated pairs measure ≈0 (max 0.0043) while adjacent pairs are strongly dependent (min 0.534), confirming the simulator realizes exactly the stated DAG. The static model is the falsification foil — a 17.95% disagreement rate shows the two selection regimes are not interchangeable.

**Limitations (honest scope).** Definition 1 is a definition; we verify its checkable structural content, not a numerical bound. The heritable factor epsilon^(t) is instantiated as K=3 additive Gaussian components (natural in quantitative genetics); the phenotype is an additive collider; selection is on the trait. Nonlinear traits and more elaborate selection functions are not exercised here (they are addressed indirectly by the Gaussian-stabilizing-selection choice used in Claims 2–6).

**Rerun.**
```
cd .trackio/logbook/evidence-package/claim1 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 repro_claim1.py
```
Deterministic (seed 0), ≈2.1 s on one CPU core; writes `results.json`.


---

# Claim 2 — Lemma 1: repeated selection induces dependencies absent under static selection

**Measured vs target (executed).** Bulmer-effect selection-induced association, measured on 1.5M-individual populations under Gaussian stabilizing selection.

| Quantity | Target / rule | Measured | Match |
|---|---|---|---|
| corr(eps_i^(t), eps_j^(t)) — un-selected (causally NON-adjacent) | ≈ 0 (\|corr\|<0.02) | max\|corr\| = **0.0009** | ✅ |
| corr — under **evolutionary** selection (all S conditioned) | induced, \|corr\|>0.05, Fisher-z p<1e-6 | \|corr\| = **0.14–0.28**, p = **0.0** | ✅ |
| gen-0 pair: **static** (one-shot) vs **evolutionary** | static ≈0, evolutionary large | static max\|corr\| = **0.029** vs evolutionary min\|corr\| = **0.276** | ✅ |
| spurious "causal" factor–factor edges (static-blind skeleton) | evolutionary ≫ un-selected | none / static / evolutionary = **0 / 3 / 29** | ✅ |

**Verdict: VERIFIED.** Repeated (per-generation) selection makes heritable factors that are *marginally independent and causally non-adjacent* strongly dependent (corr ≈ −0.28 at generation 0, p=0), while one-shot static selection barely touches early generations (−0.029). A selection-blind graphical model run on evolutionary data hallucinates **29** spurious causal edges among the factors — Lemma 1's "false causal discoveries."

---

**Paper claim (verbatim).** "Lemma 1 shows that repeated evolutionary selection induces conditional dependencies among variables that are absent under static selection models, so applying static-selection graphical models to evolutionary data can yield false causal discoveries (Section 2, Lemma 1)."

**Mechanism.** The phenotype X^(t)=Σ_k eps_k^(t)+env is a **collider** of its heritable factors. Selecting on X^(t) conditions on a descendant of that collider, inducing (negative) dependence among the eps_k^(t) — the classical **Bulmer effect** / selection-induced association. Repeated selection at every generation spreads this dependence across generations; one-shot static selection (a single event) does not reach the early generations.

**Acceptance rule.** (i) the induced cross-factor pairs are ≈0 un-selected (max\|corr\|<0.02) but dependent under evolutionary selection (min\|corr\|>0.05, Fisher-z p<1e-6); (ii) a static-blind skeleton search draws strictly more spurious factor–factor edges on evolutionary data than on un-selected data. **Falsification:** the pairs stay independent under evolutionary selection, or evolutionary produces no more spurious edges than un-selected.

---

**Setup.** T=3 (four trait generations), K=3 heritable factors, ρ=0.6, env sd 0.7. Gaussian stabilizing selection: reproduce with probability exp(−(X−1.5)²/(2·1.2²)); this keeps the selected joint exactly Gaussian, so Fisher-z partial correlation is an exact CI test. Three regimes from the same generator: **none** (no selection), **evolutionary** (select every generation 0..T−1), **static** (select only the final generation). Partial correlations from a correlation matrix built once per regime; Fisher-z two-sided p-value.

**Measured cross-factor associations (causally non-adjacent pairs).**

| pair | none | static | evolutionary | p (evol) |
|---|---|---|---|---|
| eps_0^(0)–eps_1^(0) | −0.00003 | −0.027 | **−0.276** | 0 |
| eps_0^(0)–eps_2^(0) | −0.0009 | −0.027 | **−0.282** | 0 |
| eps_0^(1)–eps_1^(1) | −0.0004 | −0.079 | **−0.291** | 0 |
| eps_0^(0)–eps_1^(2) | −0.0007 | −0.083 | **−0.137** | 0 |

**Controls.** The *none* column is the negative control (≈0 confirms the factors are genuinely independent without selection). The *static* column is the discriminating control: at generation 0 it stays near zero (−0.027), so the strong evolutionary dependence (−0.276) is specific to **repeated** selection, exactly Lemma 1.

---

**False causal discoveries.** A static-blind skeleton search (order-≤1 Fisher-z CI, α=1e-6) over the heritable factors draws an edge whenever two factors cannot be made independent. On the true causal DAG, cross-index factors are never adjacent, so any such edge is spurious: **0** on un-selected data, **3** under static selection, **29** under evolutionary selection. These 29 selection-induced "edges" are precisely what a static-selection graphical model would mis-report as causal.

**Limitations.** Linear-Gaussian additive-genetics SCM with stabilizing selection; the Bulmer effect is exercised on the additive components. Real evolutionary data would need a non-Gaussian-robust CI test — here the Gaussian-selection choice makes the linear test exact by construction, isolating the selection phenomenon from test mis-specification.

**Rerun.**
```
cd .trackio/logbook/evidence-package/claim2 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 repro_claim2.py
```
Deterministic (seed 0), ≈1.7 s; writes `results.json`.


---

# Claim 3 — Theorem 1 / Definition 2: clique-augmented G^+ captures ALL d-separations

**Measured vs target (executed).** Exhaustive enumeration of every d-separation relation over the observed variables.

| Quantity | Target / rule | Measured | Match |
|---|---|---|---|
| d-sep(G^+) == selected-model d-sep, ALL pairs × ALL conditioning subsets | fraction = **1.0** (exact) | **67584 / 67584 = 1.000000** | ✅ |
| naive selection-blind DAG vs selected model | < 1.0 (selection matters) | **59232 / 67584 = 0.8764** (fails **8352**) | ✅ |
| empirical Fisher-z CI vs G^+ d-sep (selected data, n=336959) | ≥ 0.95 | **462 / 462 = 1.0000** | ✅ |

**Verdict: VERIFIED (decisive).** The clique-augmented graph G^+ — which drops the selection variables and instead makes the heritable factors of each selected phenotype a clique — reproduces the conditional-independence model of the selected distribution **exactly**: all 67,584 (pair, conditioning-set) triples agree. The selection-blind DAG disagrees on 8,352 of them, so the augmentation is necessary, not cosmetic. Empirical CI tests on simulated selected data confirm the graph prediction on every one of 462 sampled triples.

---

**Paper claim (verbatim).** "Theorem 1 proves that the clique-augmented DAG G^+ (Definition 2) fully captures all d-separation/conditional-independence constraints implied by the evolutionary selection model, without needing to explicitly model the selection variables (Section 3, Definition 2, Theorem 1)."

**Ground truth.** The conditional-independence model of the selected distribution P(observed \| S^(0..T-1)=1) is, by the Markov property of a conditional, exactly **d-separation in the full DAG G^(T) with the selection nodes S placed in the conditioning set**. We verify that G^+ (which contains **no** S nodes) reproduces this exactly.

**G^+ construction (Definition 2).** Keep all directed edges over the observed nodes {eps, X}; drop the selection nodes; for every generation t that underwent selection, the heritable factors feeding the selected phenotype X^(t) form a **clique** (undirected selection edges) — because conditioning on a descendant of the collider X^(t) couples its parents inseparably. m-separation of this mixed graph is computed via its canonical DAG (each selection clique rendered as a common child W^(t) held in the conditioning set).

**Acceptance rule.** exhaustive d-sep agreement G^+ vs selected model **= 1.0**; naive DAG **< 1.0**; empirical CI-vs-G^+ agreement **≥ 0.95**. **Falsification:** any single (pair, conditioning-set) triple where G^+ disagrees with the ground-truth selected model.

---

**Setup.** T=2 (trait generations 0,1,2; selection at 0,1; generation 2 un-selected), K=3 heritable factors → **12 observed nodes**. Exhaustive enumeration: for every one of the 66 observed pairs and every one of the 2^10 conditioning subsets of the remaining nodes = **67,584 triples**, compare (a) `is_d_separator` in G^+ (conditioning on the clique nodes W), (b) `is_d_separator` in G^(T) conditioning on {S_0, S_1} (ground truth), (c) the naive DAG that ignores selection.

**Why the naive graph fails (8352 triples).** Selection at X^(t) does two things the naive DAG misses: (i) it couples the factors of generation t (the clique), and (ii) it turns X^(t) into an always-open collider, so cross-generation paths like `eps_i^(0) → eps_i^(1) → X^(1) ← eps_j^(1)` become active. The clique/W construction reproduces both effects; the correlation-only naive graph reproduces neither, hence the 12.4% disagreement.

---

**Empirical confirmation.** On 336,959 selected individuals (Gaussian stabilizing selection ⇒ exact Gaussian CI), 462 sampled (pair, conditioning-set) triples were tested with a Fisher-z partial-correlation CI test (α=1e-3) and compared to the G^+ d-separation prediction: **462/462 agree**. So the graph-theoretic equivalence is also visible in finite data.

**Controls / robustness.** The naive-DAG comparison is the built-in control: it shares the same directed skeleton but omits the clique augmentation and scores only 0.8764, proving the exhaustive 1.0 for G^+ is non-trivial. The empirical layer guards against a purely formal match.

**Limitations.** Exhaustive enumeration is run at a tractable size (12 observed nodes); the equivalence is a graph-theoretic identity and does not depend on the specific ρ, env-variance, or selection strength (those only affect the empirical layer, which is exact under Gaussian selection).

**Rerun.**
```
cd .trackio/logbook/evidence-package/claim3 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 repro_claim3.py
```
Deterministic, ≈5.9 s; writes `results.json`.


---

# Claim 4 — Theorem 2 / Algorithm 1: PC/GES on G^+ is sound and complete

**Measured vs target (executed).** Independent PC run on the evolutionary-selection distribution, compared to the G^+ ground truth.

| Layer | Quantity | Target | Measured | Match |
|---|---|---|---|---|
| oracle | skeleton precision / recall / SHD | 1 / 1 / 0 | **1.000 / 1.000 / 0** | ✅ |
| oracle | orientation soundness (directed = true causal) | 1.0 | **3/3 = 1.000** | ✅ |
| oracle | v-structures eps_k^(T)→X^(T) recovered | K=3 | **3/3** | ✅ |
| oracle | selection-clique edges unoriented / mis-oriented | 6 / 0 | **6 / 0** | ✅ |
| finite (n=2000) | precision / recall / SHD / soundness | ~1 / ~1 / low / ~1 | **1.0 / 1.0 / 0 / 1.0** | ✅ |
| finite (n=5000) | precision / recall / SHD / soundness | ~1 / ~1 / low / ~1 | **1.0 / 1.0 / 0 / 1.0** | ✅ |

**Verdict: VERIFIED.** With the correct CI oracle (the meaning of "sound & complete" in the PC/GES literature), PC on G^+ recovers the skeleton exactly (SHD=0), orients every identifiable causal edge correctly (soundness 1.0), and leaves all 6 selection-clique edges unoriented — precisely the claim's dichotomy: *oriented edges = true causal relations, unoriented edges reflect selection.* The finite-sample run (conservative-PC collider rule) matches this at n=2000 and n=5000.

---

**Paper claim (verbatim).** "Theorem 2 establishes that applying standard constraint-based algorithms such as PC or GES (Algorithm 1) to G^+ is sound and complete: oriented edges correspond to true causal relations, while unoriented edges may reflect the presence of selection (Section 3, Theorem 2, Algorithm 1)."

**What "sound & complete" means.** In causal discovery this is an **oracle** property: given a correct conditional-independence oracle, the algorithm returns the correct Markov-equivalence class (CPDAG). We verify it in two layers: **Layer 1** runs PC with the true d-separation oracle of G^+ (exact); **Layer 2** runs the same PC with a finite-sample Fisher-z test on simulated selected data.

**Ground truth** over observed {eps, X}: causal directed edges eps_k^(t-1)→eps_k^(t) and eps_k^(t)→X^(t); selection-clique (undirected) edges among the factors of each selected generation. The final generation T is un-selected, so eps_k^(T)→X^(T) are genuine, identifiable v-structures.

**Acceptance rule (Layer 1).** skeleton precision=recall=1, orientation soundness=1, #oriented>0, selection edges oriented=0, all K v-structure edges recovered. **Falsification:** any directed edge contradicting G^(T), any selection edge oriented as causal, or skeleton ≠ G^+.

---

**Setup.** T=2, K=3 → 12 observed nodes; G^+ has 21 edges (15 causal + 6 selection). PC is an independent implementation: PC-stable skeleton search + collider orientation + Meek rules R1/R2. Layer 1 uses `networkx` d-separation on the G^+ canonical graph as the CI oracle; Layer 2 uses Fisher-z on Gaussian-stabilizing-selection data (5 seeds each at n≈2000 and 5000).

**Finite-sample subtlety (documented honestly).** Vanilla PC picks a single separating set per pair, which under dense selection cliques can mis-place a collider and mis-orient a selection edge. The fix is the standard **Conservative-PC** collider rule (orient a→c←b only if c is in *none* of the sets that separate a and b). With it, finite-sample orientation soundness rises to **1.0** with **0** selection edges mis-oriented, matching the oracle. Gaussian stabilizing selection keeps the selected joint exactly Gaussian, so the Fisher-z test is valid.

**Controls.** The recovered directed edges are exactly {eps_0^(2)→X_2, eps_1^(2)→X_2, eps_2^(2)→X_2} — the un-selected-generation v-structures — while every selection-clique edge stays undirected. This is the built-in soundness/selection control.

---

**Verdict.** Both layers pass: oracle SHD=0 and soundness=1.0; finite-sample SHD=0 and soundness=1.0 (conservative rule) at n=2000 and 5000. Theorem 2's soundness (oriented ⇒ true causal) and completeness (skeleton = G^+, identifiable orientations recovered) hold, with selection edges correctly left unoriented.

**Limitations.** GES is not separately re-implemented; PC and GES return the same CPDAG under a correct oracle, so the oracle layer certifies both. The demonstration graph is 12 nodes; scaling behaviour is measured in Claim 6. Orientation of the *causal* edges at selected generations remains (correctly) unidentifiable because their v-structures are shielded by the selection clique — this is the "unoriented edges reflect selection" half of the claim.

**Rerun.**
```
cd .trackio/logbook/evidence-package/claim4 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 repro_claim4.py
```
Deterministic, ≈1.5 s; writes `results.json`.


---

# Claim 5 — Theorem 4 / Algorithm 2: multi-environment CDNOD improves identifiability

**Measured vs target (executed).** Correctly-oriented true-causal edges and SHD, single-environment vs pooled CDNOD.

| Layer | Metric | Single-env | CDNOD (multi-env) | Improved? |
|---|---|---|---|---|
| oracle | oriented causal edges (of 15) | **3 / 15** | **15 / 15** | ✅ |
| oracle | SHD to true DAG | **12** | **0** | ✅ |
| oracle | wrong-direction / selection-as-causal | 0 / 0 | **0 / 0** | ✅ (sound) |
| finite (E=4, n≈15k/env) | mean oriented causal edges | **3.0 / 15** | **15.0 / 15** | ✅ |
| finite | mean SHD / total wrong-direction | 12.0 / 0 | **0.0 / 0** | ✅ |

**Verdict: VERIFIED.** Pooling E environments and adding a domain index C (a known exogenous root) lets Meek propagation orient the inheritance chains that are Markov-equivalent — hence unoriented — with single-environment data. Oriented causal edges jump from **3/15 to 15/15** (SHD 12→0) with **zero** wrong-direction edges and **zero** selection edges mis-oriented, at both the oracle level and in finite samples (E=4, 4 seeds).

---

**Paper claim (verbatim).** "Theorem 4 shows that combining heterogeneous data from multiple environments/domains via the CDNOD-based procedure (Algorithm 2) improves identifiability of the evolutionary selection model compared to single-environment data (Section 4, Theorem 4, Algorithm 2)."

**CDNOD mechanism.** Pool E environments and add a domain index C, a known exogenous root. C becomes adjacent to any variable whose causal mechanism changes across environments. The natural heterogeneity here: different populations start from different genetic means, so the root heritable factors eps_k^(0) have environment-specific means (C→eps_k^(0)); the inheritance and trait mechanisms are invariant. Because C is a known root, orienting C→eps^(0) and applying Meek R1 orients the whole inheritance chain eps_k^(t)→eps_k^(t+1). Selection-clique edges are a known edge type in G^+ and are never oriented as causal.

**Acceptance rule.** oracle CDNOD oriented-causal > single-env, both sound (0 wrong, 0 selection-as-causal), CDNOD SHD < single-env; finite-sample CDNOD also orients strictly more with 0 wrong-direction edges. **Falsification:** CDNOD fails to orient more, or introduces wrong-direction / selection-as-causal edges.

---

**Setup.** T=2, K=3 (15 causal edges, 6 selection edges). **Oracle layer:** single-env G^+ vs CDNOD-augmented G^+ (add C→roots); run PC with the d-separation oracle, C as an exogenous root, selection edges frozen (never oriented); count correctly-oriented causal edges. **Finite layer:** E=4 environments with root means {−1.8, −0.6, 0.6, 1.8}, n≈15k selected/env, pooled with numeric domain index C; Fisher-z CI (α=1e-2), conservative collider rule, 4 seeds.

**Why single-env leaves 12 causal edges unoriented.** In one environment only the un-selected-generation v-structures (3 edges eps_k^(2)→X_2) are identifiable; the inheritance chains and the selected-generation factor→trait edges are Markov-equivalent and stay undirected. The domain index breaks that equivalence for the inheritance chains.

**Controls.** Soundness is tracked explicitly: across every oracle and finite-sample run, wrong-direction edges = 0 and selection-clique edges oriented as causal = 0. So the identifiability gain is not bought with false orientations — the built-in falsification control never fires.

**Limitations.** Heterogeneity is modelled as environment-specific root means (detectable by a linear CI test); CDNOD's kernel machinery for non-Gaussian / variance-only mechanism changes is not re-implemented. The factor→trait edges at selected generations remain (correctly) unidentifiable — CDNOD improves identifiability of the inheritance backbone, not the selection-shielded edges.

**Rerun.**
```
cd .trackio/logbook/evidence-package/claim5 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 repro_claim5.py
```
Deterministic, ≈1.8 s; writes `results.json`.


---

# Claim 6 — Section 5: validation on synthetic graphs of varying size (+ real data)

**Measured vs target (executed).** G^+ skeleton F1 and SHD vs a selection-blind PC baseline, over graphs of growing size (5 seeds each).

| graph (T,K) | nodes | selected n | G^+ skeleton F1 | SHD proposed | SHD naive baseline | proposed wins |
|---|---|---|---|---|---|---|
| (1,2) | 6 | 18134 | **1.000** | **0.0** | 1.0 | ✅ |
| (2,2) | 9 | 9935 | **1.000** | **0.0** | 2.0 | ✅ |
| (2,3) | 12 | 8955 | **1.000** | **0.0** | 6.0 | ✅ |
| (3,2) | 12 | 5571 | **1.000** | **0.0** | 3.0 | ✅ |
| (3,3) | 16 | 4785 | **1.000** | **0.0** | 9.0 | ✅ |
| (4,3) | 20 | 2564 | **1.000** | **0.0** | 12.0 | ✅ |

**Verdict: VERIFIED (synthetic-scale); real-data = toy (not CPU-accessible).** The proposed selection-aware procedure recovers the G^+ skeleton perfectly (F1 = 1.000) and the true causal skeleton with SHD = 0 across 6–20-node graphs, beating the selection-blind PC baseline (whose SHD grows 1→12 as it mis-reads selection cliques as causal) at every size. The paper's seven real-world datasets are **not available offline** and are reported honestly as toy/protocol-only — no dataset numbers are fabricated.

---

**Paper claim (verbatim).** "The proposed identification procedure is validated on synthetic graphs of varying size and on seven real-world datasets spanning biology, agriculture, and social science (Section 5)."

**Scope of this reproduction (honest).**
- **Synthetic-scale (real executed evidence):** the validation *protocol* on evolutionary-selection graphs of increasing size, reporting recovery accuracy (skeleton F1) and Structural Hamming Distance vs a selection-blind baseline, over 5 seeds per size. Fully CPU-reproducible.
- **Seven real-world datasets (biology / agriculture / social science):** **not available offline** on a CPU sandbox. We do **not** fabricate dataset numbers; that sub-claim is reported as **not-CPU-accessible (toy / protocol-only)**.

**Acceptance rule (synthetic protocol).** mean G^+ skeleton F1 ≥ 0.95 across sizes AND proposed SHD < baseline SHD at every size. **Falsification:** proposed does not beat the baseline, or skeleton F1 collapses with size.

---

**Protocol per size.** Simulate Gaussian stabilizing-selection data (selected joint stays Gaussian ⇒ Fisher-z valid); **proposed (G^+/selection-aware):** recover the G^+ skeleton, classify the known selection-clique edges as selection (not causal), report the recovered causal skeleton; **baseline (selection-blind PC):** recover a skeleton and interpret every adjacency as causal. Metrics: G^+ skeleton F1 (recovery accuracy) and SHD of each method's causal skeleton to the true causal skeleton. n≈8000/graph, α=1e-2, 5 seeds/size.

**Result.** mean G^+ skeleton F1 = **1.000** across all sizes; proposed SHD = **0** everywhere; the selection-blind baseline's SHD grows with the number of selection-clique edges (1 → 12), because it mistakes selection-induced association for causal adjacency — the same failure mode as Claim 2, now measured at scale.

**Seven real-world datasets — honest status.** Offline CPU sandbox: the biology/agriculture/social-science datasets referenced in Section 5 are not retrievable and are **not** reproduced. No numbers are invented for them. Only the synthetic-scale protocol above is executed with real measurements. This sub-claim is therefore **toy / not-CPU-accessible**, reported as such rather than forced to a positive.

**Rerun.**
```
cd .trackio/logbook/evidence-package/claim6 && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 repro_claim6.py
```
Deterministic, ≈2.0 s; writes `results.json`.


---

# Conclusion

---

**Executive summary.** All **6 scored claims** of "Causal Modeling of Selection in Evolution" (OpenReview mOcTXKawFY; Dai, Tang, Spirtes, Zhang) are covered by **executed numbers** from an independent NumPy/networkx/SciPy implementation, CPU-only, deterministic. The reproduction builds a Gaussian-stabilizing-selection evolutionary SCM (heritable factors → collider phenotype → selection), its DAG G^(T), and the clique-augmented graph G^+, then verifies each theorem's checkable consequence.

- **Claim 1 (Def 1):** G^(T) is a valid DAG with the Definition-1 inventory (4 traits / 12 factors / 3 selection nodes), faithfully realized (max\|corr\|_dsep=0.004), and distinct from a static model on 17.9% of d-separation relations. **verified**
- **Claim 2 (Lemma 1):** repeated selection induces the Bulmer dependence (corr −0.28, p=0) that static selection does not (−0.03); a selection-blind search hallucinates 0/3/29 spurious causal edges (none/static/evolutionary). **verified**
- **Claim 3 (Thm 1):** G^+ reproduces the selected distribution's d-separation model **exactly — 67584/67584 = 1.000000** over all pairs and conditioning sets; the naive DAG scores 0.876; empirical CI 462/462. **verified**
- **Claim 4 (Thm 2):** PC on G^+ is sound & complete — oracle SHD=0, orientation soundness 1.0, all selection edges left unoriented; finite-sample (n=2k,5k, conservative rule) SHD=0, soundness 1.0. **verified**
- **Claim 5 (Thm 4):** CDNOD lifts oriented causal edges from 3/15 to 15/15 (SHD 12→0) with zero wrong-direction / selection-as-causal edges, oracle and finite-sample. **verified**
- **Claim 6 (§5):** on synthetic graphs of 6–20 nodes the proposed method recovers structure with F1=1.000 and SHD=0, beating the selection-blind baseline (SHD up to 12) at every size; the seven real-world datasets are not CPU-accessible and are reported as **toy/protocol-only** (no fabrication). **verified (synthetic); real-data toy**

Fresh local reruns completed **6/6** commands in **≈15 s** total (C1 2.1 · C2 1.7 · C3 5.9 · C4 1.5 · C5 1.8 · C6 2.0 s). No GPU was used: these checks are CPU-feasible pure theory. The only sub-claim not reproduced (Claim 6's real datasets) is blocked by offline data availability, not compute, and is labelled honestly rather than forced positive.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 6 scored claims (Def 1, Lemma 1, Thm 1, Thm 2, Thm 4, §5 validation) verified by their checkable consequences on evolutionary-selection SCMs | Paper-scale theory + all synthetic sweeps + seven real biology/agriculture/social-science datasets |
| Hardware | Local machine; CPU-only NumPy/networkx; single-thread; no HF Job | Same theory (CPU) + access to the seven real datasets |
| Compute time | ≈15 s across 6 recorded commands | Hours (real-data preprocessing, discovery, and evaluation) |
| Cost | ≈ $0 incremental local compute | Modest, dominated by real-data acquisition/curation |
| Outcome | 6/6 claims reproduced with executed numbers (Claim 6 real-data half = toy, honestly) | Not attempted for the real datasets (offline) |


---

# Sources and provenance

- **OpenReview:** https://openreview.net/forum?id=mOcTXKawFY
- **Paper:** "Causal Modeling of Selection in Evolution" — Haoyue Dai, Zeyu Tang, Peter Spirtes, Kun Zhang (ICML 2026, spotlight). arXiv id 2606.05689.
- **Area:** Causality / causal discovery under selection bias (DAGs, d-separation, PC/GES, CDNOD).

**Independent-implementation note.** This is a from-scratch NumPy/networkx/SciPy reproduction. No paper code was used; the evolutionary-selection SCM, the clique-augmented graph G^+, the PC/CDNOD implementations, and all CI tests were written independently from the six scored claim statements. The scored claims are pure CPU theory (causal DAGs / d-separation / PC / GES / CDNOD), so the checkable consequences are verified directly: constructing the SCM and its DAG, enumerating d-separations of G^(T) and G^+, running PC on data from G^+, and measuring multi-environment (CDNOD) identifiability.

**Faithfulness of the modelling choices.**
- The phenotype-as-collider + selection = the standard collider/selection-bias structure (Spirtes–Zhang selection theory), which the paper's authors work in.
- Selection-induced association among heritable components = the classical **Bulmer effect** of quantitative genetics, derived exactly under Gaussian stabilizing selection — the choice used here.
- G^+ = the clique augmentation that replaces conditioned selection variables by cliques among co-parents; verified to reproduce the selected distribution's CI model exactly (Claim 3).

**Non-fabrication.** Every reported number is real stdout from the executed scripts. Claim 6's seven real-world datasets are not available offline and are reported as toy/protocol-only; no dataset results are invented. Verdicts are stated honestly (verified / toy) — a falsification, had any occurred, would have been reported as such.

**Published logbook space id:** `Crusadersk/icml26-causal-selection-evolution-repro`.
