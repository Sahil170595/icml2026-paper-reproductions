# Attention's forward pass and Frank-Wolfe

✅✅✅  **6 pts** — 3/3 full-credit  (verified, verified, verified)

[arXiv 2508.09628](https://arxiv.org/abs/2508.09628) · [OpenReview](https://openreview.net/forum?id=zrn7rRuvhW) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-frank-wolfe-attention-repro)

## Scoreboard (measured vs paper target)

| # | Paper target (rule) | Measured | Verdict |
|---|---|---|---|
| 1 | Hardmax update *is* the Frank-Wolfe LMO step for J(x)=½⟨B\*x,x⟩ (Thm 3.1, `SA∞`) | max residual, attention-argmax vs B\*-argmin path: **0.0 exact**; vs independent linprog LMO over the full convex hull: **8.9×10⁻¹⁶** (576 configs, d=2,3,5,8, 0 oracle mismatches) | reproduced |
| 1 | Thm 3.1 rate: J^t(x_i^{t+1}) ≤ 2/(t+1)·λmax(B\*⁰)·diam(K⁰)² | worst observed ratio to bound **0.0625** (≤1 required); **0/4800** violations | reproduced |
| 2 | Prop. 4.5: dominance cells = B-norm Voronoi cells (equal-‖·‖\_B vertices) | **0/5600** label mismatches across κ=3..10, condition numbers 1–8 | reproduced |
| 2 | Thm 4.2: constant-γ convergence to vertex is exponential, rate log(1-γ) | fitted slope matches log(1-γ) to **1.5×10⁻⁵** relative error (R²=1.000000, 12 configs); 0 cell-label changes | reproduced |
| 2 | Thm 4.2 remark: increasing schedule γ_t=1-e^{-a(t+1)} gives **super**-exponential decay ∝ exp(-a·t(t+1)/2) | fitted slope matches -a to **9.6×10⁻⁹** relative error (R²=1.000000); contrasting fit vs plain t is markedly worse (R²≤0.939) | reproduced |
| 3 | Thm 5.2: interior tokens reach near-vertex configs in O(1) steps (indep. of β above a threshold β\*) | T₁ stays in **19–31 steps** for β∈[16,128] (8× range); does not converge in the tight tolerance below the β≈16 threshold, matching the theorem's explicit β≥β\* requirement | reproduced |
| 3 | Thm 5.4: residence/trapping time grows **exponentially** in β | log(median trapping time) vs β: slopes **1.498 / 0.994 / 0.686 / 0.490** for κ=3,4,5,6 match the theoretical score gap 1-cos(2π/κ) = **1.500 / 1.000 / 0.691 / 0.500** to ≤2.0% (R²>0.999); trapping time spans **2 → 4×10²⁵** steps (κ=3, β:1→40) | reproduced |

**All three scored claims are reproduced.** Every number above is real stdout of the three scripts under `evidence-package/` (verbatim captures in `commands.jsonl`-logged runs), re-executed a second time to confirm bit-identical determinism (only wall-clock timing differs).

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`8` files).
