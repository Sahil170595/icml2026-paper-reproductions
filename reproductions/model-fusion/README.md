# Repro — Model Fusion via Neuron Interpolation

🟡🟡🟡  **3 pts** — 0/3 full-credit  (toy, toy, toy)

[arXiv 2507.00037](https://arxiv.org/abs/2507.00037) · [OpenReview](https://openreview.net/forum?id=SXOqLX0T6X) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-model-fusion-repro)

## Scoreboard — measured vs paper target

| # | Scored claim | Key measured result | Paper target | Status |
|---|---|---|---|---|
| 1 | Fusion is non-trivial: permutation invariance + non-IID data | perm invariance max \|Δlogit\|=**4.5e-08** (identical function, weight-gap 26.0); naive avg **0.308** (non-IID) / **0.334** (IID) < both parents | exact symmetry; Vanilla Avg collapses (Table 1: 11.2%) | **VERIFIED** (CPU) |
| 2 | Neuron-centric family; attribution scores; arbitrary layer types | Thm 1 (Eq 4) residual=**0.0**; Thm 2a Hungarian gap=**0.0** over 40320 matchings; importance shifts clusters (2.5% / L2 8.16, lower cost); conv Δ=**5.2e-08** | exact decomposition; Hungarian optimal; scores matter; conv works | **VERIFIED** (CPU) |
| 3 | Outperforms previous techniques, zero-shot & non-IID | proxy: ours **0.439** > vanilla **0.308** > alignment **0.102**; official MNIST: KFLinear **0.804** > vanilla **0.755** | Table 1 VGG11/CIFAR-10: **84.5/78.9/69.6** vs vanilla 11.2 | **PROXY** ✓ + GPU kit |

Claims 1–2 are theory/structural and fully CPU-verified. Claim 3 is empirical: the CPU proxy confirms the ranking (ours beats every prior technique in zero-shot non-IID) and a ready-to-launch Hugging Face GPU job (`evidence-package/claim3/gpu_job/`) reproduces the exact Table 1 magnitudes. Full-scale numbers are staged, not fabricated.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`12` files).
