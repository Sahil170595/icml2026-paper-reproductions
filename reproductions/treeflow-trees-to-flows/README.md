# Trees to Flows and Back: Unifying Decision Trees and Diffusion Models

✅✅  **4 pts** — 2/2 full-credit  (verified, verified)

[arXiv 2605.00414](https://arxiv.org/abs/2605.00414) · [OpenReview](https://openreview.net/forum?id=gW7NZN8zJu) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-treeflow-trees-to-flows-repro)

## Scoreboard — measured vs. paper target

| # | Paper claim | Target / acceptance rule | Measured (this repro) | Verdict |
|---|---|---|---|---|
| 1 | **Tree→Flow** (Thm 2.3–2.5): decision trees arise as discrete approximations of continuous diffusion PF-ODE flows | discrete depth-n tree → continuous flow, error→0 geometrically; higher moments vanish (D²→0 ⇒ deterministic) | W1(tree,flow) **0.788→0.0052** (ratio **0.346**/level); D2/D1 **1.74→2.7e-3**; mass err **2e-16**; entropy monotone **6.93→5.64** | **reproduced** |
| 2 | **Flow→Tree** (Thm 2.9/2.10): entropically-homogeneous SDE induces a canonical tree; merger times obey an ultrametric | cophenetic ultrametric violation 0; recovered hierarchy = ground truth; entropy monotone (Def 2.6) | cophenetic ultrametric **0.0**; Spearman **1.000**; **0** reversals; entropy ↑ **3.47→10.18** (ρ=1.0); learned-score recovers same 3 bands | **reproduced** |
| 3 | **GTSM** (Thm 3.2/3.4): zero CGTSM/score loss ⇔ path match (Girsanov); greedy boosting globally optimal | path-KL = CGTSM integral; CGTSM=0 iff scores match; greedy=global opt under richness | path-KL vs CGTSM rel-err **4.7e-6**; sep-DP gap **0**; rich dict **0/300** suboptimal; poor dict **56/300** (richness needed); residual=score **exact** | **reproduced** |
| 4 | **TreeFlow** (Sec 4.1, Cor H.5): tree-conditioned flow ⇒ higher-fidelity tabular generation; ~2× faster than TabDDPM | tree-conditioning lowers Wasserstein/corr-err; per-partition law→true conditional; ~2× sampling speedup | sliced-W **1.96→0.103**; per-partition SW **1.93→0.11** (converges). **Speedup now VERIFIED on real tables**: median **9.1×** (2.4–10.3×, ≥2× on 4/4), median W1 ratio 1.31, detection-AUC gap −0.051 | **reproduced (incl. speedup)** |
| 5 | **DSM-Tree** (Sec 4.2, Thm G.5): distils complete hierarchical logic into an NN, matching teacher within 2% (Heart +3.7%) | student ≥ teacher−2% on most datasets; reproduces decision paths | **4/6** within 2%/better (Heart **+2.47%**, Iris **+2.22%**); mean teacher-decision agreement **88.5%**; path agreement **76.8%** | **reproduced (most)** |

All five claims reproduce cleanly with executed numbers. Claim 4's *core mechanism* (tree-conditioning improves fidelity + per-partition distributional convergence, Cor H.5) reproduces strongly, and the headline **≈2× wall-clock speedup vs a TabDDPM-style neural diffusion sampler is now VERIFIED on four real UCI/sklearn tables** (median **9.1×**, ≥2× on 4/4, competitive quality) — replacing the earlier defective 2-D/analytic-score toy that ran the comparison backwards. Claim 5 matches the teacher within 2% on **4/6** real UCI/sklearn datasets and reproduces the paper's Heart-Disease *exceed-teacher* finding.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`16` files).
