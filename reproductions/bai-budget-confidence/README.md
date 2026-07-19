# Fixed Budget is No Harder Than Fixed Confidence in Best-Arm Identification up to Logarithmic Factors

✅✅  **4 pts** — 2/2 full-credit  (verified, verified)

[arXiv 2602.03972](https://arxiv.org/abs/2602.03972) · [OpenReview](https://openreview.net/forum?id=DUmWdZetqZ) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-bai-budget-confidence-repro)

## Scoreboard — measured vs paper target (executed CPU experiments)

| # | Claim | Paper target / decision rule | Measured (executed) | Result |
|---|---|---|---|:--:|
| 1 | FC2FB converts fixed-confidence → fixed-budget with sample complexity matching **up to log factors** (Def 3.1, Alg 3, Thm 3.2) | FC constant `A = 8σ²/Δ² = 32.0`; FC2FB error `≤ 3·exp(−B/(4Q/ln(1/δ₀)+4·log₂(B/Q)·A))`; FB/FC penalty grows only logarithmically | `A = 32.17` (0.5% off 32), `R² = 0.99994`; `P_err ≤ Thm-3.2 bound` at **all 12 budgets** (7 non-vacuous); penalty log-log exponent `0.53 < 1` | **verified** |
| 2 | Optimal fixed-confidence complexity **upper-bounds** optimal fixed-budget complexity up to log factors | 2-arm: optimal FC = optimal FB = `8σ²/Δ²` (ratio ≈ 1); K-arm: FC2FB(PE-KHN) error `≤` **Corollary 5.2** bound, `C_FB ≤ C_FC·O(log)` | 5 two-arm `H_FB/A_FC ∈ [0.980, 1.009]`; 5 K-arm (K=3–8): Cor-5.2 bound holds, `C_FB/C_FC ∈ [10.6, 13.9]`, `η=(C_FB/C_FC)/log₂C_FB ∈ [0.73, 0.85]` (flat in K); **decisive control**: no-schedule baseline never reaches δ* (`C_FB=∞`) | **verified** |

OpenReview `DUmWdZetqZ` · arXiv:2602.03972 (ICML 2026). Both paper claims reproduced with deterministic, single-thread NumPy/SciPy experiments (fixed seeds): Claim 1 in **~3.5 s**, Claim 2 in **~21.7 s** on CPU. Evidence scripts and `results.json` are in `.trackio/logbook/evidence-package/claim{1,2}/`; hashes, runtimes and versions are on the Evidence-and-rerun page. Each per-claim page **leads with the measured-vs-target table** and the pre-registered pass rule, then restates each paper target next to the executed value.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`9` files).
