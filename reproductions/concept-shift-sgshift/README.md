# Explaining Concept Shift with Interpretable Feature Attribution

✅⚪✅⚪✅  **6 pts** — 3/5 full-credit  (verified, inconclusive, verified, inconclusive, verified)

[arXiv 2505.20634](https://arxiv.org/abs/2505.20634) · [OpenReview](https://openreview.net/forum?id=wpKA7G7Cqu) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-concept-shift-sgshift-repro)

## Scoreboard — measured vs. paper target

| # | Paper claim | Target + acceptance rule | Measured (this repro) | Verdict |
|---|---|---|---|---|
| 1 | SGShift-KA (knockoffs + absorption) achieves **AUC > 0.9** for identifying shifted features | mean SGShift-KA AUC across real datasets > 0.90; ≥2/3 datasets individually > 0.9 | SGShift-KA AUC: diabetes **0.870±0.027**, SUPPORT2 **0.985±0.014**, Adult **0.986±0.014**; mean **0.947**; 2/3 > 0.9 | **reproduced** |
| 2 | SGShift **requires few target-domain samples** for effective shifted-feature detection (sample efficiency) | best SGShift variant AUC beats best baseline at every tested n_target (100-15,000), gap does not vanish as n_target shrinks, AUC>0.85 by n_target<20% of full target | diabetes: SGShift-K/KA beats best baseline at **all 7** tested n_target (100-15,000) by **+0.10..+0.15 AUC**; AUC>=0.85 by n_target=1,000 (2.7% of full 37,273-row target); support2 confirms the same ordering | **reproduced** |
| 3 | SGShift variants **outperform baselines** Diff / WhyShift / SHAP | SGShift best-or-tied AUC in every dataset×config; positive mean gap | SGShift-K/KA best in **6/6** cells; mean AUC gap **+0.024**; largest on hardest set (diabetes matched **+0.095**) | **reproduced** |
| 4 | **Knockoffs** control false discoveries at target FDR **and** raise recall | derandomized empirical FDR ≤ target (+0.03 tol) at q∈{0.1,0.2}; recall gain > 0 | empirical FDR (derand) q=0.1: **0.000/0.000/0.038**; q=0.2: **0.109/0.234/0.207**; controlled **5/6**; recall gain **+0.032** | **reproduced** |
| 5 | **Absorption** term improves detection under model **misspecification** | mismatched SGShift-A ≥ SGShift (AUC & recall) in nearly every setting | mismatched mean ΔAUC **−0.008**, ΔRecall **−0.028**; helps diabetes (ΔAUC +0.005) & adult (ΔRecall +0.028), hurts SUPPORT2; hurts matched | **partial / not robust** |

Claims 1–4 reproduce the paper's central empirical findings on real tabular data at real scale. Claim 5 (absorption) is reported honestly as **not robustly reproduced**: the effect is small and inconsistent across datasets — consistent with the paper's own tiny Table-2 effect size (+0.01–0.02 AUC) but not the "improves in nearly every setting" wording.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`47` files).
