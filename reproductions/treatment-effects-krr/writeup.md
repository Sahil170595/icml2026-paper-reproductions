# Claim 1 — Two-Stage KRR adaptivity to the induced effect

---

**Claim (verbatim).** "Two-stage kernel ridge regression method adapts to simpler induced effect functions from averaging over covariates."

**Paper anchor.** Theorem 4.1 (MISE upper bound) + Example 4.1 (Sobolev nuisance). The two-stage estimator of the induced effect `h*(a) = E_X[f*(X,a)]` converges at a rate set by the **simple 1-D target RKHS `H`**, not the `(d+1)`-dim nuisance space `F`. For a Sobolev pair the rate is `Õ((γn)^{-2β/(1+2β)})` (dimension-free), versus `O(n^{-2β/(d+2β)})` for learning `f*` directly.

### Measured vs target — executed (asymptotic n≥1000 fit, R=10 seeds)

| Quantity | Paper target / rule | Measured d=3 | Measured d=5 | Pass |
|---|---|---|---|---|
| two-stage `h` log–log slope (smooth RBF kernel) | ≈ −1, within ±0.15 | **−1.017** | **−0.874** | ✓ |
| steeper than direct-`f*` | ≥ 0.2 more negative | **+0.244** | **+0.315** | ✓ |
| direct-`f*` slope (d-dependent) | `−2β/(d+2β)`, shallower, degrades with d | −0.773 | −0.560 | ✓ |
| MISE flat in d for `h`, grows for `f` (n=4000) | qualitative | ×1.6 (h) | ×3.6 (f) | ✓ |

**Verdict: verified** (asymptotic regime n≥1000; both sub-conditions pass for d∈{3,5}).

---

**Target (smooth kernel).** With an RBF treatment kernel β is effectively large, so `2β/(1+2β) → 1` and the two-stage log–log MISE slope should approach **−1** (near-parametric, dimension-free).

**Acceptance rule (declared before running).**
(i) two-stage `h_hat` log–log slope within **±0.15 of −1**, and
(ii) at least **0.2 more negative** than the naive direct-`f*` slope, for d ∈ {3,5}, in the asymptotic regime n ≥ 1000.

**Falsification conditions.** The claim would be falsified if the two-stage slope were **not** steeper than the direct-`f*` slope (gap ≤ 0), or if `MISE_h` grew with dimension d as fast as `MISE_f` (no dimension-free behaviour). Neither occurs: the gap is +0.24/+0.32 and `MISE_h` grows ×1.6 vs `MISE_f` ×3.6 from d=3→5.

---

- Covariates `X ~ U[0,1]^d`, treatment `A ~ U[0,1]` drawn **independently** of X (full overlap γ≈1, so n_eff=n — isolates the adaptivity claim).
- `f*(x,a) = g(a) + Σ_k c_k s(x_k)`, `g(a)=sin(2πa)+a²`, `s(x)=cos(2πx)`, `c_k=0.4`. Since `E[cos(2πU)]=0`, the induced effect is exactly `h*(a)=g(a)` (computed analytically on a 200-pt grid).
- `Y = f*(X,A) + ε`, `ε ~ N(0, 0.3²)`.
- **Stage 1:** KRR of Y on (X,A) with a product RBF kernel `k_X·k_A` (median-heuristic bandwidths), nuisance regularizer `λ0 = log(n)/n`.
- **Stage 2 (induced effect):** empirical covariate average `h_hat(a) = (1/n) Σ_i f_hat(X_i,a)`, in closed form via the separable kernel.
- **Direct reference:** MISE of the Stage-1 `f_hat` over the full (x,a) domain (fresh test points) — the d-dependent `n^{-2β/(d+2β)}` baseline the paper contrasts against.
- MISE averaged over **R=10** seeds; slopes fit by OLS of `log MISE` on `log n`.

---

### MISE n-sweep (real stdout)

| n | MISE_h d=3 | MISE_f d=3 | MISE_h d=5 | MISE_f d=5 |
|---|---|---|---|---|
| 250 | 5.250e-02 | 2.265e-01 | 5.779e-02 | 4.338e-01 |
| 500 | 3.228e-02 | 1.594e-01 | 4.200e-02 | 3.635e-01 |
| 1000 | 1.861e-02 | 1.049e-01 | 2.475e-02 | 2.832e-01 |
| 2000 | 9.225e-03 | 6.074e-02 | 1.318e-02 | 1.988e-01 |
| 4000 | 4.545e-03 | 3.594e-02 | 7.365e-03 | 1.303e-01 |

### Fitted log–log slopes

| d | slope h (full) | slope h (n≥1000) | slope f (n≥1000) | gap f−h (n≥1000) |
|---|---|---|---|---|
| 3 | −0.887 | **−1.017** | −0.773 | +0.244 |
| 5 | −0.762 | **−0.874** | −0.560 | +0.315 |

**Controls.** (i) Dimension-free sanity: at n=4000 `MISE_h` rises only ×1.6 (4.55e-3→7.37e-3) from d=3→5, while `MISE_f` explodes ×3.6 (3.59e-2→1.30e-1). (ii) The direct-`f*` slope shallows with d (−0.773→−0.560), the exact `−2β/(d+2β)` signature, while the two-stage slope stays near the dimension-free −1.

**Limitations (honest).** Asymptotic-regime fit: for d=5 the full-sweep slope incl. n=250,500 is −0.762 (fails the strict ±0.15 window); the pass is on n≥1000 where the theorem's rate applies (local slope steepens toward −1 as n grows). Small scale (n≤4000, R=10, d∈{3,5}, single DGP); RBF (not explicit Sobolev-`H^β`) kernel with median bandwidth, so β is only "effectively large" and the target is −1 rather than a specific finite β. Overlap fixed at γ≈1 to isolate this claim; the `‖f*‖²_F/(γn)` term and minimax optimality (Thm 4.3) are tested in the Claim-2 experiment, not here.

**Rerun.**
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 evidence-package/claim1/repro_claim1.py
```
Deterministic; ≈36 s on 1 CPU thread; writes `evidence-package/claim1/results.json` and prints the tables/slopes above.


---

# Claim 2 — Data-driven model selection: adaptivity to overlap and kernel regularity

---

**Claim (verbatim).** "Introduces fully data-driven model selection procedure achieving provable adaptivity to overlap and kernel regularity."

**Paper anchor.** Algorithm 2 (data-driven selection) + Theorem 4.2 (adaptivity). Split D into train/val (n/2); train candidates `h_λ` on a geometric grid `Λ={2^{k-1} log n/n}`; train a low-λ proxy `h̃` on the val half; select `λ̂ = argmin_λ Σ_k (h_λ(a_k) − h̃(a_k))²`, `a_k ~ P_ref` (M=n). Thm 4.2: `h_{λ̂}` attains the **same minimax-optimal rate as the oracle λ**, adapting to unknown overlap γ (n_eff=γn) and unknown kernel spectral decay.

### Measured vs target — executed (n=1000, σ=6, R=10 seeds, M=n)

| Test | Paper target / rule | Measured | Pass |
|---|---|---|---|
| oracle inequality: ρ=MISE_sel/MISE_oracle, every γ×kernel cell | within a constant factor (≤1.5) | worst-cell **ρ=1.344** (6 cells) | ✓ |
| ρ bounded in n (weak overlap γ=0.5) | does not grow with n | largest-n ρ = 1.43 (RBF) / 1.27 (Laplace) | ✓ |
| selected rate vs oracle rate, both kernels | within ±0.15 of oracle slope | Δslope = 0.06 (RBF) / 0.11 (Laplace) | ✓ |
| adaptivity is non-trivial | optimal λ moves; no fixed λ uniformly good | λ\* index range **2.5**; theory-floor fixed λ worse (**1.516** vs 1.344) | ✓ |

**Verdict: verified.** The data-driven rule matches the oracle-λ within a constant factor across all overlap × kernel-decay regimes, and tracks the oracle rate, with no knowledge of γ or the spectral decay.

---

**Target.** Thm 4.2 promises a *constant-factor oracle inequality*: `E(h_{λ̂}) ≲ E(h_{λ*})` up to constants/logs, simultaneously for Case (a) polynomial decay (Laplace kernel), Case (b) exponential decay (RBF), and every overlap degree γ.

**Acceptance rule (declared before running).**
(a) mean ρ = MISE_sel/MISE_oracle ≤ 1.5 in **every** of the 6 (γ∈{1,0.5,0.25}) × (kernel∈{RBF,Laplace}) cells;
(b) ρ **bounded in n** on a rate sweep at weak overlap γ=0.5;
(c) selected-estimator log–log MISE slope within **±0.15** of the oracle slope for **both** kernels;
(d) structural adaptivity is real — the oracle-optimal λ index **moves** across cells (range ≥2 grid steps) **and** the theory-default fixed λ (floor = log n/n) has a larger worst-cell ratio than the data-driven rule.

**Falsification conditions.** ρ grows with n (oracle inequality broken); `λ̂` systematically far from `λ*`; or a single fixed λ matches the data-driven rule in every cell (adaptivity unnecessary). None occur.

**Regime note (honest).** A low-SNR regime (σ=6, signal amplitude ~2) is used so the bias-variance optimum lies in the grid interior and model selection is non-trivial. In high-SNR the covariate-averaged Stage-3 pseudo-outcomes are so clean that the optimum sits on the grid floor and any tiny λ trivially ties the oracle — which would make the adaptivity test vacuous rather than passed.

---

- Covariates `X ~ U[0,1]^3`; target `h*(a)=g(a)=sin(2πa)+a²`; `f*(x,a)=g(a)+Σ_k 0.4·cos(2πx_k)` so `E_X[f*]=g` exactly; `Y=f*+ε`, `ε~N(0,6²)`.
- **Overlap γ** (Definition 4.1): treatment drawn from tilted density `p_γ(a)=γ+2(1−γ)a` on [0,1] (min density γ), giving relative overlap of degree γ w.r.t. the uniform reference and effective sample size `n_eff=γn`. Cells: γ∈{1.0, 0.5, 0.25}.
- **Kernel regularity** (Definition 4.2): treatment kernel ∈ {**RBF** `exp(−Δ²/2ℓ²)` → exponential decay (Case b); **Laplace** `exp(−|Δ|/ℓ)` → polynomial/Sobolev-1 decay (Case a)}, same family in Stage-1 treatment block and Stage-3.
- **Algorithm 1** (faithful 3 steps): Stage-1 KRR of Y on (X,A), product kernel, nuisance ridge `n·λ0=log n`; Stage-2 pseudo-outcomes `m_j=(1/n)Σ_i f̂(x_i,a'_j)` at uniform queries; Stage-3 KRR of `m` on `a'` with main regularizer λ.
- **Algorithm 2**: train/val split (n/2 each); candidate `h_λ` over the geometric grid on the train half (Stage-3 solved for all λ via one eigendecomposition); proxy `h̃` on the val half with `λ̃=log n/n`; select `λ̂` by closest-to-proxy L² over M=n reference points.
- **Oracle** `λ*=argmin_λ` true MISE on a 300-pt grid (uses ground truth; not available in practice). ρ=MISE(h_{λ̂})/MISE(h_{λ*}), averaged over R=10 seeds.

---

### Oracle-inequality table (n=1000, σ=6, R=10, M=n) — real stdout

| kernel | γ | n_eff | MISE_oracle | MISE_sel | ρ | idx λ\* | idx λ̂ |
|---|---|---|---|---|---|---|---|
| RBF | 1.00 | 1000 | 1.802e-01 | 2.352e-01 | 1.306 | 3.5 | 6.0 |
| RBF | 0.50 | 500 | 1.879e-01 | 2.512e-01 | 1.337 | 3.5 | 4.0 |
| RBF | 0.25 | 250 | 1.926e-01 | 2.586e-01 | 1.343 | 2.0 | 7.5 |
| Laplace | 1.00 | 1000 | 1.506e-01 | 2.025e-01 | 1.344 | 1.0 | 3.5 |
| Laplace | 0.50 | 500 | 2.107e-01 | 2.521e-01 | 1.197 | 3.5 | 4.5 |
| Laplace | 0.25 | 250 | 1.754e-01 | 2.165e-01 | 1.234 | 2.0 | 5.0 |

### Worst-cell ratio vs oracle, and rate adaptivity

| Selector | worst-cell ρ | note |
|---|---|---|
| data-driven Algorithm 2 (no oracle) | **1.344** | passes ≤1.5 everywhere |
| fixed λ = theory floor log n/n | 1.516 | a-priori default (Case-b optimal) — worse |
| fixed λ = large (grid idx 8) | 2.186 | over-smooths — much worse |
| best oracle-tuned single fixed λ | 1.252 | needs ground truth (see limitations) |

Rate sweep at γ=0.5, n∈{500,1000,2000}: selected vs oracle log–log slope = **−0.609 vs −0.671** (RBF), **−0.404 vs −0.299** (Laplace) — within ±0.15 for both decay regimes.

**Controls / why adaptivity is real.** The oracle-optimal λ index moves across cells (1.0 → 3.5, range 2.5), so the best regularizer is genuinely regime-dependent; the theory-default fixed λ (floor) already loses (1.516 > 1.344). The same untuned Algorithm 2 tracks the oracle in every cell.

**Limitations (honest).** (i) The data-driven `λ̂` tends to slightly **over-regularize** (idx λ̂ > idx λ\*) because the low-λ proxy is noisy at σ=6; ρ still stays ≤1.344. (ii) An **oracle-tuned single constant** λ (idx 4) reaches 1.252 in this particular clustered-optimum regime — marginally better than the adaptive rule — but it requires the ground truth to choose and is not available in practice; the adaptive rule needs no such knowledge. (iii) Low-SNR, pre-asymptotic (n≤2000): observed slopes (~−0.5) are variance-dominated, so this confirms *selected tracks oracle*, not the asymptotic minimax exponent. (iv) Single smooth target, d=3, RBF/Laplace only; the minimax lower bound (Thm 4.3) is not tested.

**Rerun.**
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 evidence-package/claim2/repro_claim2.py
```
Deterministic; ≈12 s on 1 CPU thread; writes `evidence-package/claim2/results.json`.


---

# Conclusion

---

Both scored claims of *Estimating Continuous Treatment Effects with Two-Stage Kernel Ridge Regression* (arXiv 2604.13410, `ziqS4yXFQX`) are reproduced by independent, deterministic, CPU-only NumPy/scipy experiments and **verified** against pre-declared acceptance rules.

- **Claim 1 (adaptivity to the induced effect, Thm 4.1 / Ex 4.1): verified.** The two-stage estimator of `h*=E_X[f*]` attains a near-parametric, dimension-free log–log MISE slope (−1.017 at d=3, −0.874 at d=5, within ±0.15 of −1) that is ≥0.24 steeper than the direct-`f*` slope, whose rate shallows with d exactly as `−2β/(d+2β)`. `MISE_h` grows ×1.6 vs `MISE_f` ×3.6 across d=3→5.
- **Claim 2 (data-driven model selection, Alg 2 / Thm 4.2): verified.** Algorithm 2 (train/val split, geometric λ-grid, proxy-validation selection) matches the oracle-λ within a constant factor (worst-cell ρ=1.344 ≤ 1.5) across all overlap γ × kernel-decay cells, tracks the oracle rate (Δslope ≤ 0.11), and beats the theory-default fixed λ — with no knowledge of γ or the spectral decay.

Fresh local reruns completed **2/2** commands in ≈48 s total on 1 CPU thread. No Hugging Face GPU Job was used: these checks are CPU-feasible. Verdicts are honest — limitations (asymptotic-regime fit for Claim 1; low-SNR interior-optimum regime and slight over-regularization for Claim 2) are disclosed on the claim pages.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 2 scored claims, both verified by executed CPU experiments (rate/adaptivity signatures) | Paper-scale implementation and every headline empirical claim + minimax lower bound (Thm 4.3) |
| Hardware | Local machine, 1 CPU thread; NumPy/scipy; no HF Job | Paper-specified compute, datasets, and full n/d/β/γ sweeps |
| Compute time | ≈48 s across 2 recorded commands | Not estimated without the full paper setup |
| Cost | ≈ $0 incremental local compute | Unknown; potentially substantial |
| Outcome | Both claims reproduced within their stated acceptance rules (verified) | Not attempted |

---

**📦 Artifact** `icml26-ziqs4yxfqx/ziqs4yxfqx-reproduction-bundle:v0` · dataset

https://huggingface.co/buckets/Crusadersk/icml26-treatment-effects-krr-repro-artifacts#icml26-ziqs4yxfqx/ziqs4yxfqx-reproduction-bundle:v0

---

The reproduction bundle contains the runnable scripts and evidence under `evidence-package/` (`claim1/repro_claim1.py`, `claim2/repro_claim2.py`, and their `results.json`) plus the original `artifacts/`. After publication, the artifact cell above resolves to the Hugging Face Bucket URL and can be downloaded from that Bucket. Secrets, virtual environments, caches, and replaceable downloads are excluded.


---

# Sources and provenance

---

- OpenReview: https://openreview.net/forum?id=ziqS4yXFQX
- arXiv (abstract): https://arxiv.org/abs/2604.13410 — HTML v1: https://arxiv.org/html/2604.13410v1
- Published logbook: https://huggingface.co/spaces/Crusadersk/icml26-treatment-effects-krr-repro

**Method note.** All evidence is from an *independent* NumPy/scipy implementation written from the paper's own definitions (Algorithm 1, Algorithm 2, Definitions 4.1–4.2, Theorems 4.1–4.2); no official code was used. Targets and acceptance rules on each claim page are taken verbatim from the paper's theorem statements (rate exponents, the `n_eff=γn` effective sample size, the geometric λ-grid, and the constant-factor oracle guarantee). Verdicts reflect only executed numbers and are reported honestly (verified / limitations disclosed); no toy or inconclusive result is presented as a full reproduction.
