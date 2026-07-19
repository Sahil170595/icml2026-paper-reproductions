# The Theory and Practice of MAP Inference over Non-Convex Constraints

✅✅  **4 pts** — 2/2 full-credit  (verified, verified)

[arXiv 2602.08681](https://arxiv.org/abs/2602.08681) · [OpenReview](https://openreview.net/forum?id=jIZqAemuqk) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-map-inference-nonconvex-repro)

## Scoreboard — measured vs. paper target

| # | Scored claim | Target + acceptance rule | Measured (this repro) | Verdict |
|---|---|---|---|---|
| 1 | **Theorem 4.5** — treewidth-one MAP(LRA) is exact/tractable via MpMap | MpMap value == exact constrained MAP; assignment feasible | max rel vs independent global opt **1.2e-09** (named) / **2.0e-09** (30 random); \|MpMap−eval(assign)\| **4.4e-11**; **33/33 feasible**, dominates brute grid | **reproduced** |
| 2 | **MpMap** recursive message passing (Alg 1-3, Eqs 4-6) over Ω^PP tree factors | factor→var msg (Eq 5) == per-x_j brute; exact on STAR/SNOW/PATH | message-op err **5.7e-14** (20 edges, 4820 pts); 3 topologies max rel **2.7e-09** | **reproduced** |
| 3 | **Ω^PP message ops correct** (Thm A.5 max-outPP; product; pointwise-max; Prop A.6) | ops == exact reference; #pieces ≤ 8mq+4m+4 | max-outPP **3.6e-15** (4746 pts); product **2.8e-14**; ptwise-max **0.0**; Prop A.6 bound **holds** (ratio 0.125) | **reproduced** |
| 4 | **TMC** (Def 4.2) holds for Ω^PP; general PP violate (ii) ⇒ WMI≠MAP(LRA) | (i)(ii)(iii) closures exact; non-factorized sup is non-polynomial | closures **8.5e-14 / 7.1e-15 / 0.0**; non-fact. sup poly-fit RMSE **3.8e-2** (=2·x^1.5, 1.2e-7) | **reproduced** |
| 5 | **Complexity/scalability** (Prop A.7, Thm A.8, Fig 5) — tractable, diameter-sensitive; brute exponential | STAR polynomial+exact; PATH higher exponent; brute exp | STAR **n^1.19**, exact **7.5e-12**; PATH **n^1.46**; brute exp base **9.3**; n=20 in **0.52s** vs **6.7e20** grid evals | **reproduced** |
| 6 | **PaMAP** — convex-polytope decomposition recovers global constrained MAP where constraint-agnostic fails | PaMAP == ground-truth MAP; agnostic infeasible/suboptimal | Example 2.2 rel **3.2e-05** at (1.82,1.83) feasible; agnostic **infeasible**; battery **12/12** match, agnostic fails **11/12** | **reproduced** |

**All 6 scored claims reproduce** within their acceptance rules with executed numbers. The two theory pillars (MpMap exactness on treewidth-one graphs, Thm 4.5; correctness of the piecewise-polynomial Ω^PP message operations, Thm A.5) match brute-force / exact references to **machine precision** (3.6e-15 – 5.7e-14). The tractability/complexity picture (Prop A.7 / Thm A.8) and the practical PaMAP scheme (Sec 5) both reproduce, including PaMAP recovering the paper's Example 2.2 constrained optimum ≈ (1.83, 1.83).

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`23` files).
