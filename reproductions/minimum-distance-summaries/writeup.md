# Claim 1 - Plug-in MMD summary independent of NPE

---

## Exact challenge claim

> Minimum-distance summaries provide a plug-in robust neural posterior estimation method that adapts test-time summaries independently of pretrained NPE.

## Verdict: REPRODUCED

The paper defines the summary objective `s*(Q) = argmin_s D(P_{x|s}, Q)` at
`arxiv_main.tex:236` (Eq. `eq:mds_def`), specializes `D` to MMD and its RFF
approximation at lines 261-285, and returns `q_ψ(θ | s*)` in **Algorithm 1**
(lines 288-314). The pretrained NPE `q_ψ` is **queried, not retrained** — MDS depends on
the observations only through the summary `s`, so the adapted `s*` is fed straight into
the frozen NPE (`arxiv_main.tex:222`, `285`).

### The decisive test: two genuine frozen NPEs, hash-checked before/after every adaptation

`run_gaussian.py` and `run_oup.py` train two real conditional-density NPEs
`q_ψ(θ|s)` (neural conditional-Gaussian, mean + Cholesky covariance from an MLP of the
summary, trained by exact negative log-likelihood):

| mechanism | NPE params | training sims | frozen NPE SHA-256 (state dict) | adaptations | hash identical? |
|---|---:|---:|---|---:|---:|
| Gaussian | **4,677** | 20,000 | `66c325f2b7309ffad5a092b37bf9f8de659a4cb0240f1339ecd2007f3e590406` | 300 | **300/300 YES** |
| OUP | **17,669** | 10,000 | `658cce1ce85a4611231f59d4fddf7bc58088ff0445497f3c5d2910b4bf5c8df2` | 250 | **250/250 YES** |

Before **and** after **every single** MDS adaptation the NPE's tensor-state SHA-256 is
recomputed and asserted equal to the frozen baseline. Across all **550** adaptations the
hash never changes: MDS never receives, reads, optimizes, or mutates a single NPE weight.
It optimizes only the query summary `s` (a separate leaf tensor). This is the plug-in
separation, demonstrated directly rather than by proxy.

### The NPE is a genuine, correct posterior estimator

For the conjugate Gaussian task the learned NPE's posterior mean matches the analytic
conjugate posterior mean to **RMSE 0.02505** on 200 held-out clean datasets — the network
really learned the Bayesian posterior, it is not a look-up of the summary.

### The plug-in mechanism, per dataset

For each held-out test dataset the **same frozen** posterior is queried **twice**:
`q_ψ(θ | s_obs)` with the ordinary contaminated summary, and `q_ψ(θ | s*)` with the
MDS-adapted summary. Only the input `s` differs. 300 paired Gaussian queries + 250 paired
OUP queries. The RMSE gap between the two queries (Claim 3) is produced entirely by the
change of summary, with the network held constant.

**Source integrity:** arXiv TeX SHA-256 `ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2`
(pinned in `evidence-package/artifacts/arxiv_main.tex`); official code
`github.com/Shermjj/Minimum-Distance-Summaries` @ `45158124f0cbdc2f6c1ac602c9fc5501dce20af3`.


---

# Claim 2 - RFF efficiency and lightweight adaptation

---

## Exact challenge claim

> The algorithm is implemented efficiently with random Fourier feature approximations, yielding a lightweight, model-free test-time adaptation procedure.

## Verdict: REPRODUCED

The paper (`arxiv_main.tex:276-285`, Eq. `eq:mmd-rff`) approximates the MMD by the
Euclidean distance between finite RFF mean embeddings, learns the conditional mean
embedding `μ_ω(s)` by MSE regression (Algorithm 1), and optimizes `s* = argmin_s
‖μ_ω(s) − (1/N)Σ z(x̃_n)‖²` at test time with **L-BFGS** initialised at the observed
summary. Implementation details are fixed by the appendix (`arxiv_main.tex:944`):
scikit-learn RFF, **median-heuristic bandwidth**, **512** features, a **2×256**
fully-connected mean-embedding regressor, PyTorch L-BFGS with line search.

### What was executed (both mechanisms, exactly these settings)

| | Gaussian | OUP |
|---|---:|---:|
| RFF features `K` | **512** | **512** |
| RFF library | scikit-learn `RBFSampler` | scikit-learn `RBFSampler` |
| bandwidth (median heuristic) | σ_med = 2.375, γ = 0.0886 | σ_med = 11.934, γ = 0.00351 |
| decoder regressor | 2×256 FC → 512 | 2×256 FC → 512 |
| decoder final MSE | **8.50e-4** | **6.54e-4** |
| test-time optimizer | PyTorch L-BFGS (strong-Wolfe) | PyTorch L-BFGS (strong-Wolfe) |
| adaptation median / p95 (ms, CPU) | see table below | see table below |

### Lightweight: per-adaptation wall-clock on a single CPU thread

| eps | Gaussian median / p95 (ms) | OUP median / p95 (ms) |
|---:|---:|---:|
| 0.1 | 3.982 / 12.555 | 42.996 / 69.725 |
| 0.2 | 4.763 / 14.606 | 38.532 / 70.454 |
| 0.3 | 6.434 / 17.239 | 50.262 / 80.453 |
| 0.4 | 7.848 / 18.536 | 64.003 / 103.889 |

Gaussian adaptation is single-digit milliseconds; OUP (a 25-dim trajectory embedding with a
harder landscape) is tens of milliseconds. Both are lightweight test-time procedures — the
whole adaptation is a deterministic L-BFGS solve over a 2- or 3-dimensional summary against
a frozen regressor.

### RFF approximates the exact MMD (measured gap, not asserted)

Over held-out (clean, contaminated) dataset pairs I compare the RFF distance
`‖z̄_A − z̄_B‖²` against the **exact** RBF-kernel MMD² (closed-form Gram matrices, same γ):

| mechanism | mean |exact − RFF| | max gap | corr(exact, RFF) | mean exact MMD² | mean RFF MMD² |
|---|---:|---:|---:|---:|---:|
| Gaussian | **0.00123** | 0.00559 | **0.972** | 0.06310 | 0.06296 |
| OUP | **0.00249** | 0.00461 | 0.726 | 0.05768 | 0.05969 |

The 512-feature approximation tracks the exact MMD to ~1-2×10⁻³ in absolute value — a direct
approximation-quality check, not merely a runtime measurement.

### Model-free, as the paper means it

"Model-free" = no likelihood and no explicit error/contamination model is used at test time;
the adaptation is a data-space distance minimization. It does **not** mean there is no offline
learned object — the conditional mean-embedding regressor `μ_ω` is trained once, offline, on the
same clean simulator pairs used for the NPE (no extra simulations).


---

# Claim 3 - Robustness gains with low overhead

---

## Exact challenge claim

> The method demonstrates substantial robustness gains with minimal additional overhead.

## Verdict: REPRODUCED ON BOTH MECHANISMS

Under Huber ε-contamination (`arxiv_main.tex:334`, `Q_{ε,y}=(1-ε)Q+εδ_y`) I measure the
**posterior-mean RMSE** of the frozen NPE against the true parameter, querying once with the
contaminated observed summary and once with the MDS-adapted summary. Same network, same test
seeds across the two queries.

### Gaussian (bivariate location, N=100, outlier magnitude δ=8)

| eps | trials | NPE RMSE | NPE + MDS RMSE | reduction | wins | median ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 50 | 0.1484 | 0.1525 | -2.74% | 24/50 | 2.341 |
| 0.1 | 50 | 0.7771 | 0.1569 | **79.80%** | 50/50 | 3.982 |
| 0.2 | 50 | 1.5678 | 0.1573 | **89.97%** | 50/50 | 4.763 |
| 0.3 | 50 | 2.3290 | 0.2025 | **91.30%** | 50/50 | 6.434 |
| 0.4 | 50 | 3.1096 | 0.5088 | **83.64%** | 50/50 | 7.848 |
| 0.5 | 50 | 3.6003 | 2.5005 | 30.55% | 40/50 | 7.671 |

At every prespecified non-severe level (ε=0.1-0.4) the frozen-NPE RMSE falls
**79.80%-91.30%** (mean **86.18%**) and **all 50/50** paired tests improve. The uncontaminated
NPE accuracy (ε=0) is preserved — MDS is a near-no-op there. Adaptation is single-digit ms.

### OUP (T=25, N=100 trajectories, out-of-prior contamination θ_c=(-0.5, 1.0), σ²_c=0.5)

| eps | trials | NPE RMSE | NPE + MDS RMSE | reduction | wins | median ms | summary→oracle gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 50 | 0.6284 | 0.6322 | -0.60% | 19/50 | 7.698 | 0.076 |
| 0.1 | 50 | 3.4420 | 1.3301 | **61.36%** | 44/50 | 42.996 | 1.277 |
| 0.2 | 50 | 3.5784 | 1.1738 | **67.20%** | 45/50 | 38.532 | 1.442 |
| 0.3 | 50 | 3.4250 | 1.4127 | **58.75%** | 45/50 | 50.262 | 2.177 |
| 0.4 | 50 | 3.5394 | 1.1994 | **66.11%** | 44/50 | 64.003 | 2.165 |

MDS reduces frozen-NPE RMSE at **4/4** contamination levels by **58.75%-67.20%**
(mean **63.36%**), win rate **44-45/50**. Even though the objective aligns in *data space*,
the adapted summary stays close to the clean oracle summary (gap ≈ 1.3-2.2 in a 3-D summary
whose contaminated-vs-clean shift is far larger), exactly as the paper reports for the OUP task
(`arxiv_main.tex:419`).

### Falsification boundary (disclosed, not discarded)

Gaussian ε=0.5 is **not** called a success: RMSE falls only **30.55%**, the win rate drops to
37→40/50, and the single-start L-BFGS can lock onto the wrong mode when outliers dominate. The
paper itself states MDS is guaranteed only for relatively small contamination and "can degrade"
under severe proportion (`arxiv_main.tex:404`). The reproduction verifies **substantial** gains
at moderate contamination on two mechanisms with **low absolute CPU overhead**; it does not
claim robustness at arbitrary contamination, and it does not rerun the paper's full comparator
suite (SIR, cryo-EM, NPE-PFN, NNPE).


---

# Claim 4 - Robustness and consistency guarantees

---

## Exact challenge claim

> Theoretical guarantees for the robustness of the algorithm are provided.

## Verdict: VERIFIED WITHIN THE PAPER'S STATED SCOPE

The paper provides two theorems. **Theorem 4.1** (`thm:robustness`, `arxiv_main.tex:337`):
under the ten robustness assumptions, `sup_y d/dε KL[P_{θ|s*(Q)}, P_{θ|s*(Q_{ε,y})}]|_{ε=0} < ∞`
— infinitesimal Huber contamination causes only a proportional KL change in the MDS posterior.
**Theorem 4.2** (`thm:consistency`, `arxiv_main.tex:368`): under the four consistency assumptions,
if the ordinary summary posterior is consistent then so is the MDS posterior.

Rather than re-prove theorems, `evidence-package/claim4_audit.py` runs a **fail-closed audit**
of the pinned source and two numerical corroborations.

## Independent dependency certificate

The audit pins the 1,197-line TeX (SHA-256
`ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2`), counts
**10 robustness assumptions** (lines 497-560; three enumerate blocks of 2 + 3 + 5 items) and
**4 consistency assumptions** (lines 686-705), resolves the equation/lemma/assumption anchors,
and asserts an eight-step dependency DAG is complete — else it exits non-zero.

| Step | Source | Assumptions / audited implication |
|---|---|---|
| **R0** perturbation path | Thm 4.1 (337-346), `eq:mds_def` (238), Huber `Q_{ε,y}` (334) | population MDS argmin exists; contamination model defined |
| **R1** bounded influence | Lemma `lem:s_influence` (571-604), L601 | `infl1,infl2,deriv_bound,non_sing` ⇒ `IF=M⁻¹∇ξ`, `‖∇ξ‖ ≤ 4·sup k·Σᵢ∫|∂ᵢp_s|` uniformly finite |
| **R2** summary → likelihood | `eq:rob1` (618), `eq:rob2` (625) | mean-value theorem + `convexity`, `log_likelihood_sensitivity` ⇒ `‖Φ_s−Φ_t‖_{L¹} ≤ k₁‖s−t‖` |
| **R3** likelihood → KL | `eq:rob3` (635), `eq:robustness` (340) | Sprungk Thm 11 + `sprungk_first/last` ⇒ `KL/ε ≤ k₁k₂‖Δs‖/ε`, bounded via R1 |
| **C1** posterior → predictive | Part 1 (744-759), `eq:s_consistency` | `generative2` + continuous mapping + dominated convergence ⇒ weak predictive convergence |
| **C2** weak → MMD | Part 2 (761-773), `rem:metrize` | bounded characteristic-kernel (`kernel`) metrizes weak convergence ⇒ `B_N→0` |
| **C3** argmin contraction | Part 3 (775-792) | optimality ⇒ `0 ≤ A_N ≤ B_N → 0` |
| **C4** identifiability | Part 4 (794-798) | strong mixture `identifiability` ⇒ `P_{θ|s*_N} ⇒ δ_{θ₀}` |

Audit output: `10 robustness + 4 consistency assumptions; 8/8 DAG steps complete;
robustness chain R0→R1→R2→R3 complete: True; consistency chain C1→C2→C3→C4 complete: True`.
The proof retains arbitrary finite `d_s`, `d_x` — it is a dimension-general dependency/inequality
audit, not a mechanized prover, and it does not re-prove the imported Sprungk / Briol results.

## Numerical corroborations (closed-form Gaussian MDS)

- **Bounded summary influence (Lemma).** Contaminating the target by a point mass at `y` and
  computing `IF(y) ≈ [s*(Q_{ε,y}) − s*(Q)]/ε` for a bounded RBF kernel: `s*(clean)=[0,0]`,
  `sup_y ‖IF(y)‖ = 1.832730` (finite → bounded influence), and `‖IF‖ → 0` as `‖y‖ → 60`.
  The influence *redescends* to zero for far outliers — exactly the App.-D observation that
  large-shift outliers affect MDS *less* than moderate ones.
- **Monotone posterior contraction (consistency direction).** Conjugate posterior RMS radius over
  N = 10,20,50,100,200,500,1000 is `[0.4264, 0.3086, 0.1980, 0.1407, 0.0998, 0.0632, 0.0447]`,
  strictly decreasing, log-log slope **-0.4910** (Bayesian-CLT rate ≈ -1/2). Concentration of the
  base posterior is the precondition Theorem 4.2 transfers to the MDS posterior.

## Honest TeX findings (recorded, non-fatal)

- **L601** writes `sup_{z,z'} k(z,z')` without an absolute value; licensed by boundedness
  (`itm:infl1`) since the RBF kernel is non-negative, so `sup|k| = sup k`.
- **L656** the second `Φ` subscript is `s(Q_{ε,y})` and drops the argmin star (should be `s*`);
  typographical, does not affect the argument.
- **L720** the consistency proof explicitly states it "does not consider approximation error" —
  it assumes exact conditionals, not the learned NPE/decoder.
- Theorem 4.1 is a one-sided derivative `d/dε KL|_{ε=0}` and relies on `KL(0)=0` plus existence of
  the influence-function limit (the Lemma).

None of these introduces an extra assumption or a stronger global-robustness claim. The guarantee
is **local/infinitesimal** (Thm 4.1) and **asymptotic under exact conditionals** (Thm 4.2); it does
not promise arbitrary-contamination robustness of an approximate learned decoder — the same boundary
the empirical ε=0.5 result exhibits.
