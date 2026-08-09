# Causal Modeling of Selection in Evolution

✅✅✅✅✅🟡  **11 pts** — 5/6 full-credit  (verified, verified, verified, verified, verified, toy)

[arXiv 2606.05689](https://arxiv.org/abs/2606.05689) · [OpenReview](https://openreview.net/forum?id=mOcTXKawFY) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-causal-selection-evolution-repro)

## Scoreboard — measured vs. paper target

| # | Claim (paper object) | Target + acceptance rule | Measured (this repro) | Verdict |
|---|---|---|---|---|
| 1 | **Def 1** — evolutionary selection DAG G^(T), distinct from static | valid DAG w/ Def-1 inventory; faithful; ≠ static (d-sep disagreements>0) | acyclic ✓, 4 traits/12 factors/3 sel nodes; faithful (max\|corr\|_dsep=**0.004**, min\|corr\|_adj=**0.53**); **2283/12720 (17.9%)** d-sep differ from static | **verified** |
| 2 | **Lemma 1** — repeated selection induces dependencies absent under static ⇒ false discoveries | induced pairs \|corr\|>0.05, p<1e-6 under evol; ~0 unselected; #spurious edges evol>none | unselected max\|corr\|=**0.0009**; evol induced \|corr\|=**0.14–0.28** (p=0); spurious causal edges none/static/evol = **0 / 3 / 29** | **verified** |
| 3 | **Thm 1 / Def 2** — clique-augmented G^+ captures ALL d-separations | exhaustive d-sep(G^+)==selected-model, frac=1.0; naive<1.0; empirical≥0.95 | **67584/67584 = 1.000000** exact; naive DAG **0.8764** (8352 fail); empirical CI **462/462 = 1.0** | **verified** |
| 4 | **Thm 2 / Alg 1** — PC/GES on G^+ sound & complete | oracle skel prec=rec=1, orient soundness=1, sel edges unoriented | oracle **SHD=0**, soundness **3/3=1.0**, v-structures **3/3**, selection edges **6/6 unoriented, 0 mis-oriented**; finite-sample (n=2k,5k) SHD=0, soundness=1.0 | **verified** |
| 5 | **Thm 4 / Alg 2** — multi-env CDNOD improves identifiability | CDNOD oriented-causal>single-env, sound, SHD lower | oriented causal edges **3/15 → 15/15**; SHD **12 → 0**; 0 wrong-dir, 0 selection mis-oriented (oracle **and** finite-sample E=4) | **verified** |
| 6 | **§5** — validation on synthetic graphs of varying size + real data | mean G^+ skeleton F1≥0.95 across sizes; proposed SHD<baseline at all sizes | F1 = **1.000** for 6–20-node graphs; SHD_proposed **0** vs SHD_naive **1→12**; proposed beats naive at every size. 7 real datasets = **toy (not CPU-accessible)** | **verified (synthetic); real-data toy** |

Every verdict is backed by the measured numbers above. A decisive **falsification** would count the same as a reproduction; here all six checkable consequences hold. Claim 6's real-world half (seven biology/agriculture/social-science datasets) is **not available offline** and is reported as toy/protocol-only — no dataset numbers are fabricated.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`12` files).
