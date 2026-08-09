# Finite and Corruption-Robust Regret Bounds in Online Inverse Linear Optimization under M-Convex Action Sets

✅✅✅✅✅  **10 pts** — 5/5 full-credit  (verified, verified, verified, verified, verified)

[arXiv 2602.01682](https://arxiv.org/abs/2602.01682) · [OpenReview](https://openreview.net/forum?id=g7LE4mukGq) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-mconvex-inverse-linear-regret-repro)

## Scoreboard — measured vs. paper target

| # | Paper claim | Target + acceptance rule | Measured (this repro) | Result |
|---|---|---|---|---|
| 1 | **Thm 3.1** — Algorithm 1 (topological-sort) regret **O(d²)** over M-convex sets | worst-case regret ≤ C(d,2)=d(d−1)/2 ∀d; log-log slope α₁∈[1.8,2.2] | regret **= C(d,2) exactly** for d=4…64 (6,15,…,2016); **α₁=2.088** (R²=0.9997); bound never exceeded | reproduced |
| 2 | **Thm 4.2** — Algorithm 1 with **center of gravity** improves to **O(d log d)** | regret ≤ log_{e/(e−1)}(d!); regret/(d·ln d) flat while /d grows, /d² shrinks; Lemma 4.1 vol-cut ≤ 1−1/e | centroid ≤ Grünbaum ∀d; **regret/(d ln d) band 1.30** (/d ↑×2.97, /d² ↓×2.02); **vol ratio max 0.568 ≤ 0.632**; ≪ toposort (×1.5→3.9) | reproduced |
| 3 | **Thm 5.3** — Algorithm 2 (restart) **O((C+1) d log d)** under C corruptions, no knowledge of C | regret linear in (C+1), slope≈base; #restarts ≤ C; base ~ d log d | regret(C)/base = **1,2,3,5,9,13.05** vs C+1=1,2,3,5,9,13; fit **38.1·C+37.7 (R²=1.000)**; **restarts=C**; base/(d ln d) flat | reproduced |
| 4 | **Thm 6.1** — **Ω(d)** lower bound on M-convex hard instance (tight up to log d) | E[R]/d ~const∈[0.4,0.6]; slope∈[0.4,0.6], R²>0.999; no learner sublinear | **E[R]/d ≈ 0.50** for 3 learners across d=16…256; **slope 0.498–0.501**, R²≈1.0; none sublinear | reproduced |
| 5 | **Headline/abstract** — regret **FINITE (independent of T)**, resolving the open problem (prior O(d log T)) | R_{10⁶}=R_{10³} (plateau, ratio∈[0.98,1.02]); R_T≤C(d,2)/Grünbaum; prior d·ln T grows ≥1.5× | R_T **plateaus** (topo 46, cent 22 at d=20 for T=10³…10⁶, ratio 1.00) ≤ caps; prior d·ln T grows **×2.0** | reproduced |

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`10` files).
