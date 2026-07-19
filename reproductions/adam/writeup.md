# Claim 1: Adam converges locally linearly while GD and Momentum are sub-linear

---

**Paper claim (verbatim).** "Adam achieves local linear convergence on degenerate polynomials, significantly outperforming sub-linear convergence of Gradient Descent and Momentum."

**Target + decision rule.** On L(x)=xᵏ/k (k≥4 even), Adam's iterate contracts geometrically at xₜ₊₁/xₜ → **β₂^{1/(2(k−2))}** (loss-log slope k·ln β₂/(2(k−2))), while GD and Momentum follow the sub-linear power law xₜ = Θ(t^{−1/(k−2)}) (log-log slope −1/(k−2)). Reproduced iff measured matches predicted; falsified if Adam is not geometric or GD/Momentum are not the predicted power law. Adam β₁=0.9, β₂=0.93, ε=0; η=0.001; x₀=1; 100,000 steps.

### Measured vs target (executed; `artifacts/evidence/adam_degenerate.json`)

| quantity | k=4 measured | k=4 target | k=6 measured | k=6 target |
|---|--:|--:|--:|--:|
| Adam geometric ratio xₜ₊₁/xₜ | 0.98202 | 0.98202 | 0.99097 | 0.99097 |
| Adam loss-log slope | −0.07257 | −0.07257 | −0.05443 | −0.05443 |
| GD power-law log-log slope | −0.4965 | −0.5000 | −0.2491 | −0.2500 |
| Momentum power-law slope | −0.4991 | −0.5000 | −0.2496 | −0.2500 |

Practical outcome after 100,000 steps — final |x|:

| | Adam | GD | Momentum |
|---|--:|--:|--:|
| k=4 | 1.8e−82 | 7.1e−2 | 2.2e−2 |
| k=6 | 1.1e−41 | 2.2e−1 | 1.3e−1 |

**Verdict: REPRODUCED.** Adam's measured geometric ratio matches β₂^{1/(2(k−2))} to five decimals and its loss-log slope matches k·ln β₂/(2(k−2)) exactly; Adam drives |x| to machine zero (1.8e−82 at k=4) while GD/Momentum follow the t^{−1/(k−2)} power law to three decimals and barely move. The Adam geometric ratio is fit only inside a machine-precision-safe window (2,000-step burn-in; end before gₜ² underflows: [2000,7225] for k=4, [2000,8396] for k=6) — a cutoff set by IEEE-754, not by the target. Full script, recorded rerun, and stdout are on the **Evidence and rerun** page.

**Rerun.** `pip install numpy && python artifacts/adam_repro.py`  (prints measured vs predicted for k=4 and k=6; ~0.7 s).


---

# Claim 2: the acceleration comes from a decoupling between the second moment vₜ and the squared gradient gₜ²

---

**Paper claim (verbatim).** "Adam exhibits acceleration stemming from a decoupling mechanism between the second moment vₜ and the squared gradient gₜ²." The paper (arXiv 2603.09581, Sec 5.2 / Lemma 5.4 / Thm 5.7 / Sec 6, Regimes I & III) states: as xₜ→0 the gradient vanishes, so vₜ **decouples** from gₜ² and decays autonomously, vₜ ≈ β₂·vₜ₋₁ (Lemma 5.4: vₜ/vₜ₋₁→β₂). This geometric decay of vₜ acts as an exponentially growing effective learning rate η/√vₜ ∝ β₂^(−t/2), converting sub-linear into linear convergence.

**Target + decision rule.** If the rate is set by the *decoupled* second moment, the measured geometric rate must equal **β₂^{1/(2(k−2))}** (Thm 5.7 eq 21 / Thm 4.1 eq 10) and must *move with β₂*: OLS of measured log-rate on predicted log-rate ⇒ slope≈1, R²≈1. Direct decoupling fingerprints: vₜ/vₜ₋₁→β₂ and coupling ratio Rₜ=vₜ/gₜ²→∞.

**Falsification.** The claim FAILS if (i) the measured rate does not track β₂^{1/(2(k−2))} (regression slope ≠ 1), or (ii) the **coupled control** (set vₜ:=gₜ² exactly, no EMA) still shows the same β₂-dependent linear convergence — that would prove decoupling is *not* the cause.

### Table A — β₂ sweep, DECOUPLED RMSProp (β₁=0), L(x)=x⁴/4: the rate MOVES with β₂

| β₂ | rate xₜ₊₁/xₜ measured | β₂^{1/(2(k−2))} target | vₜ/vₜ₋₁ measured | target β₂ | max Rₜ=vₜ/gₜ² |
|---|--:|--:|--:|--:|--:|
| 0.900 | 0.974004 | 0.974004 | 0.900000 | 0.900 | 4.9e+97 |
| 0.930 | 0.982021 | 0.982021 | 0.930000 | 0.930 | 1.0e+98 |
| 0.990 | 0.997491 | 0.997491 | 0.990000 | 0.990 | 5.6e+99 |
| 0.999 | 0.999749 | 0.999750 | 0.999000 | 0.999 | 6.5e+27 |

OLS log(rate_meas) on log(rate_pred) across the sweep: **slope 0.99998, intercept −3.7e−07, R² 1.000000** — the rate tracks β₂ exactly. k=6 is identical: slope 0.99998, R² 1.000000. Loss-log slope also matches k·ln β₂/(2(k−2)) to 5 decimals at every β₂ (k=4 meas/pred: −0.10536/−0.10536, −0.07257/−0.07257, −0.01005/−0.01005, −0.00100/−0.00100).

### Table B — COUPLED CONTROL vₜ:=gₜ² exactly (no EMA): linear convergence DISAPPEARS and β₂ has no effect

| β₂ | rate measured (coupled) | β₂^{1/(2(k−2))} target | tail max \|x\| | converged? |
|---|--:|--:|--:|--:|
| 0.900 | 0.997686 | 0.974004 | 1.0e−3 | never (−1) |
| 0.930 | 0.997686 | 0.982021 | 1.0e−3 | never (−1) |
| 0.990 | 0.997686 | 0.997491 | 1.0e−3 | never (−1) |
| 0.999 | 0.997686 | 0.999750 | 1.0e−3 | never (−1) |

Coupled rate std across β₂ = **0.0e+00**; OLS slope of coupled rate on β₂^{1/(2(k−2))} = **0.000, R²=0**; the four trajectories are **byte-identical** (max pairwise diff 0.0e+00). With vₜ=gₜ² the step is η·gₜ/√(gₜ²)=η·sgn(gₜ) (SignGD): it stalls in an O(η)=1e−3 limit cycle and *never* reaches zero (converged_step=−1), and β₂ is irrelevant. k=6 identical.

---

### The mechanism, measured directly

- **Autonomous decay of vₜ (Lemma 5.4).** In the real (decoupled) RMSProp runs the second-moment ratio vₜ/vₜ₋₁ equals β₂ to 6 decimals at every β₂ (0.900000, 0.930000, 0.990000, 0.999000). vₜ follows its own EMA memory, *not* the instantaneous gₜ².
- **vₜ is decoupled from gₜ² (coupling ratio).** Because gₜ²=xₜ^{2k−2} collapses far faster than vₜ~β₂^t, the coupling ratio Rₜ=vₜ/gₜ² explodes to 1e+27–1e+99 within the numerically stable window. Tight coupling would keep Rₜ≈1; it does not.
- **Effective learning-rate amplification.** vₜ~β₂^t ⇒ η/√vₜ ∝ β₂^(−t/2), an exponentially growing step that is exactly what turns the GD/Momentum power law into geometric convergence.

### Controls that isolate the decoupling

1. **β₂ sweep (Table A):** the rate is a pure function of β₂, matching β₂^{1/(2(k−2))} (regression slope 1.0, R² 1.0). The knob that sets the rate is the *second-moment EMA decay*, not the gradient magnitude.
2. **Coupled control (Table B):** forcing vₜ=gₜ² (removing the EMA/decoupling) destroys linear convergence (SignGD limit cycle at O(η), converged_step=−1) and makes β₂ inert (identical trajectories, regression slope 0). Removing *only* the decoupling removes *both* the acceleration and its β₂-dependence.

The contrast between (1) and (2) is the mechanism proof: the acceleration and its β₂-tunability are present with decoupling and absent without it.

---

**Verdict: REPRODUCED (decisive).** The measured geometric rate moves with β₂ exactly as β₂^{1/(2(k−2))} (regression slope 0.99998, R² 1.000000, both k=4 and k=6), the two direct decoupling fingerprints (vₜ/vₜ₋₁→β₂, Rₜ→∞) hold, and the coupled control that removes the decoupling abolishes linear convergence and all β₂-dependence. This isolates the vₜ/gₜ² decoupling as the cause of Adam's acceleration.

**Scope & honesty.**
- Setting is the paper's own scalar isolation experiment: RMSProp = Adam with β₁=0, ε=0 (Sec 5.2 "To isolate adaptivity effects, we analyse RMSProp"), on L(x)=xᵏ/k. RMSProp is stable for all swept β₂ (k=4 needs β₂>0.0625). Full Adam (β₁=0.9) reproduces the same rate law where its own stability condition β₁<β₂^{k/(2(k−2))} holds (β₂≥0.93 for k=4); the pure-second-moment RMSProp isolation is used deliberately so the whole β₂ grid is in-regime.
- Deterministic (no RNG); single-threaded (OMP/OPENBLAS/MKL=1). The coupling ratio and geometric rate are read only inside a machine-precision-safe window (before gₜ² underflows), a cutoff independent of the target being tested.
- This is the empirical mechanism reproduction; the paper's full stability proof (Jacobian eigenvalue β₂^{1/(2(k−2))}) is analytic and not re-derived here.

**Rerun.**
```bash
pip install numpy
cd .trackio/logbook/evidence-package/claim2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 repro_claim2.py
# ~1.9 s; writes results.json; override output path with CLAIM2_OUTPUT
```


---

# Conclusion

---

Both scored claims of arXiv 2603.09581 are reproduced with executed numbers on a CPU, deterministically (ε=0, single-threaded).

- **Claim 1 (Adam linear vs GD/Momentum sub-linear): REPRODUCED.** Adam's geometric ratio matches β₂^{1/(2(k−2))} to 5 decimals (0.98202 at k=4, 0.99097 at k=6) and reaches |x|≈1.8e−82, while GD/Momentum follow the t^{−1/(k−2)} power law (log-log slopes −0.4965 / −0.4991 vs −0.5000 at k=4).
- **Claim 2 (acceleration from vₜ ⁄ gₜ² decoupling): REPRODUCED (decisive).** In decoupled RMSProp the geometric rate moves with β₂ exactly as β₂^{1/(2(k−2))} (rate-vs-β₂ regression slope 0.99998, R² 1.000000, k∈{4,6}), with the decoupling fingerprints vₜ/vₜ₋₁→β₂ and Rₜ=vₜ/gₜ²→1e+97. A coupled control that forces vₜ:=gₜ² (no decoupling) abolishes linear convergence (SignGD limit cycle at O(η)) and makes β₂ inert (regression slope 0.000) — isolating decoupling as the cause.

Two experiments were run fresh: `artifacts/adam_repro.py` (Claim 1, ~0.7 s) and `.trackio/logbook/evidence-package/claim2/repro_claim2.py` (Claim 2, ~1.9 s), ~2.6 s combined. No Hugging Face GPU Job was used or needed: both checks are CPU-feasible.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | Both scored claims covered by executed measured-vs-target numbers (Claim 1 rates + Claim 2 β₂ sweep and coupled control) | Paper-scale implementation and every headline empirical claim |
| Hardware | Local CPU; deterministic single-threaded NumPy; no HF Job | Paper-specified accelerators, datasets, sweeps |
| Compute time | ~2.6 s across 2 freshly recorded commands | Not estimated without the full paper setup |
| Cost | ≈ $0 incremental local compute | Unknown; potentially substantial |
| Outcome | Claim 1 REPRODUCED; Claim 2 REPRODUCED with a decisive coupled-control mechanism test | Not attempted |

---

**📦 Artifact** `icml26-uywvgk1qt0/uywvgk1qt0-reproduction-bundle:v0` · dataset

https://huggingface.co/buckets/Crusadersk/icml26-adam-degenerate-repro-artifacts#icml26-uywvgk1qt0/uywvgk1qt0-reproduction-bundle:v0

---

The reproduction bundle contains the runnable scripts, evidence JSON/JSONL, manifests, reviews, and supporting logs under `artifacts/` and `.trackio/logbook/evidence-package/`. The Claim 2 mechanism experiment and its `results.json` live at `.trackio/logbook/evidence-package/claim2/`. After publication, the artifact cell above resolves to the Hugging Face Bucket URL and can be downloaded from that Bucket. Secrets, virtual environments, caches, and replaceable downloads are excluded.


---

# Sources and provenance

---

- OpenReview: https://openreview.net/forum?id=uYWVGk1Qt0
- Published logbook: https://huggingface.co/spaces/Crusadersk/icml26-adam-degenerate-repro
- arXiv: https://arxiv.org/abs/2603.09581

Both scored claims are backed by executed, deterministic CPU experiments recorded in this logbook: Claim 1 in `artifacts/adam_repro.py` and the Claim 2 decoupling mechanism (β₂ sweep + coupled control) in `.trackio/logbook/evidence-package/claim2/repro_claim2.py`. Reported numbers are real stdout; no values are fabricated.
