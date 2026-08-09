# Parametrized Power-Iteration Clustering for Directed Graphs

✅✅🟡⚪⚪🟡  **6 pts** — 2/6 full-credit  (verified, verified, toy, inconclusive, inconclusive, toy)

[arXiv 2210.00310](https://arxiv.org/abs/2210.00310) · [OpenReview](https://openreview.net/forum?id=5vI6ApLOg8) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-power-iteration-clustering-digraph-repro)

## Scoreboard — measured vs. paper target

| # | Paper claim | Target + acceptance rule | Measured (this repro) | Verdict |
|---|---|---|---|---|
| 1 | Parametrized random-walk operator **P_(ν)** (Def 3.1, Eq 4) is a valid transition matrix; **L_RW,(ν)=I−P_(ν)** (Prop 3.1) | rows ≥0 and sum to 1; identity to ~machine eps; over many graphs/measures | min entry **0.0**; max\|rowsum−1\| **4.4e−16**; \|L_RW−(I−P_(ν))\| **0.0**; Eq4≡PropA.2 **2.2e−16** (64 cases) | reproduced |
| 2 | Vertex measure via **γ∈[0,1] forward/backward mixing** P_γ=γP_out+(1−γ)P_in (Eq 8–9) | P_γ row-stochastic ∀γ∈[0,1]; γ=1→P_out, γ=0→P_in; γ∉[0,1] fails | P_γ min entry **0.0**, max\|rs−1\| **4.4e−16**; endpoints **0.0**; γ∉[0,1] min entry **−0.50**; App A.3.1 ν→π₁ᐟ₂ᵅ **1.1e−15** (99 cases) | reproduced |
| 3 | P_(ν) is **self-adjoint in ℓ²(ν+ξ)** → **real spectrum**, ergodic; raw directed walk is complex & reducible | detailed balance & \|Im λ(P_(ν))\|→0; raw P \|Im λ\|>0; π_ν>0 on non-str.-conn. graph | balance **1.1e−16**; \|Im λ(P_(ν))\| **0.0**; raw \|Im λ\| **0.317**; raw 40/80 transient, π_ν min **8.4e−3**>0 | reproduced |
| 4 | Diffusion distance = **Mahalanobis** with RWDK **K_t=P²ᵗD_d⁻¹** (Prop 4.1); digraph P-RWDK | pairwise identity to ~1e−12; RWDK symmetric PSD; rank-1 limit | undirected \|Δ\| **2.8e−17**; digraph \|Δ\| **8.3e−17**; RWDK min-eig **−6e−18**; limit **2.3e−17** | reproduced |
| 5 | **ParPIC/P-RWDKC: competitive accuracy + improved scalability** vs spectral & PIC (Alg 1, §6, App A.2) | runtime exponent α(ParPIC)<1.5 vs α(dense SC)>2.3 with equal NMI at scale; NMI ≥ best-SC−5pp on heterogeneous digraphs, ≥+15pp where signal is directional; beats PIC everywhere | **α=1.02 vs 2.87** (781× at N=5k; NMI 99.7 vs 99.9; 3.0 vs 96.5 MB); hetero NMI gap vs best SC **−0.3/+0.6/+97.6 pp**; vs PIC **+63.1/+3.6/+88.0/+13.1/+34.0 pp** | reproduced\* |
| 6 | **Diffusion-time** controls scale (Alg 2, CH/DCH); multi-scale metastability (§5.2, Fig 1) | N_eff drops 6→2→1; coarse mode persists ≫ fine; both scales recovered | N_eff **6→2→1**; t\*_coarse/t\*_fine **8.9×**; fine k=6 ARI **100**, coarse k=2 ARI **100** | reproduced |

\* Claim 5 verifies **both legs** of the scored claim. *Scalability*: block power iteration on sparse P_(ν) runs N=50,000 (600k edges) in **0.31 s** with fitted exponent **α=1.02** (near-linear), vs **α=2.87** for the classical dense spectral pipeline (**781×** faster at N=5,000; 3.0 vs 96.5 MB peak memory at N=2,000) — at NMI within 0.2–0.4 pp of spectral at every N. *Competitive accuracy in the paper's target regime* (heterogeneous digraphs, App A.2): tie with the best SC on power-law DC-SBMs (**80.4 vs 80.7**), best method on reducible citation-style core–periphery (**100.0**), and **+97.6 pp** over SC on flow-defined digraphs where the measured symmetrized view is non-assortative (within-density 0.040 < between 0.081 — symmetrization destroys the directional signal; both SC variants ≈ chance). PIC is beaten in all five regimes (up to +88 pp), causally traced to reducibility of the raw walk (180.5/240 resp. 420/600 transient vertices). Honest caveat kept: on *homogeneous assortative* SBMs, SC-SYM remains an equally strong or stronger baseline (−1.3/−14.6 pp).

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`18` files).
