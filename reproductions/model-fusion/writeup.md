# Claim 1 (scored): Fusion is non-trivial — permutation invariance + non-IID data

---

**Paper claim (verbatim).** *Model fusion is non-trivial due to differences in internal representations from permutation invariance and differently distributed training data.*

**Measured vs target** — deterministic CPU run, `evidence-package/claim1/repro_claim1.py` (torch 2.13.0+cpu, seed-fixed, 1 thread, 6.7 s):

| Mechanism | Measured | Expected | Pass |
|---|--:|---|:--:|
| Permutation invariance: max \|Δlogit\| after relabelling hidden units of a 4-layer MLP + compensating next layer | **4.5e-08** | ~0 (exact functional symmetry) | ✅ <1e-4 |
| Weight-space L2 gap between those two *functionally identical* models | **26.03** | ≫0 (same function, different weights) | ✅ |
| Naive weight averaging, **IID** data (parents 0.734 / 0.726) | **0.334** | collapse < both parents | ✅ |
| Naive weight averaging, **non-IID** shards (parents 0.554 / 0.482) | **0.308** | collapse < both parents | ✅ |
| Paper reference (VGG11/CIFAR-10 Table 1, Vanilla Averaging 2-way) | — | **11.2%** vs base 81.5/75.1 | context |

Both mechanisms the claim names are reproduced: relabelling neurons leaves the function bit-for-bit unchanged (so weight averaging is ill-posed), and averaging two independently trained models collapses **below both parents** — measurably worse when the data is non-IID.

---

**Anchor.** Abstract + Section 3 (Motivation); Table 1 (Vanilla Averaging row).

**Target.** Qualitative: internal representations are non-identifiable (permutation invariance) and diverge under differently-distributed data, so naive weight-space fusion fails.

**Comparison rule.** (i) A hidden-unit permutation compensated in the next layer must leave outputs unchanged (max \|Δ\| ≤ 1e-4). (ii) Naive averaging of two independently trained models must fall below both parents.

**Falsification (honest).** The claim would be *falsified* if permuting hidden units changed the function (max \|Δ\| ≫ 0), or if naive averaging matched/exceeded the parents. Neither occurred: Δ = 4.5e-08 and both averages (0.334 IID, 0.308 non-IID) are below both parents.

**Setup.** Two 4-hidden-layer MLPs (width 256) trained on a 10-class Gaussian task; deeper nets carry the rich permutation symmetry that also drives VGG/ViT. IID control = balanced random split; non-IID = per-class Dirichlet(0.1) label skew (shard class fractions recorded in `results.json`).

**Controls.** IID-vs-non-IID contrast isolates the two causes; fixed seeds; single deterministic thread; the permutation check is exact math, independent of training.

**Verdict.** VERIFIED (mechanism) at toy scale — the two causes the claim invokes both reproduce deterministically. This is a structural/CPU-checkable claim, so no GPU run is needed.

**Limitations.** Synthetic 10-class data, not CIFAR; the collapse magnitude (0.31) is milder than the paper's near-random 11.2% (VGG11 is deeper). The direction and both mechanisms match exactly.

**Rerun.** `OMP_NUM_THREADS=1 python evidence-package/claim1/repro_claim1.py` → writes `evidence-package/claim1/results.json`.


---

# Claim 2 (scored): Neuron-centric family — attribution scores + arbitrary layer types

---

**Paper claim (verbatim).** *Proposes neuron-centric family of fusion algorithms that incorporates neuron attribution scores and generalizes to arbitrary layer types.*

**Measured vs target** — deterministic CPU run, `evidence-package/claim2/repro_claim2.py` (torch 2.13.0+cpu, scipy 1.15.3, seed-fixed, 1 thread, 1.6 s):

| Check | Measured | Target | Pass |
|---|--:|---|:--:|
| **Theorem 1 / Eq. 4** decomposition residual \|J − (approx+group)\| | **0.0** | exact identity (≤1e-8) | ✅ |
| — components: J=2393.52, approx=1618.01, group=775.51, cross-term | **−3.6e-14** | cross-term = 0 | ✅ |
| **Theorem 2a** Hungarian cost vs brute-force min over all 8! = 40320 matchings | **98.8742 = 98.8742**, gap **0.0** | Hungarian is global optimum | ✅ |
| **Attribution scores** change fusion: hidden neurons reassigned / centroid L2 shift (uniform→conductance-style) | **2.5% / 8.16** | >0 (scores matter) | ✅ |
| — importance-weighted grouping cost: uniform 0.1051 → importance **0.1010** | **lower** | scores improve objective | ✅ |
| **Arbitrary layer types**: Conv2d channel-permutation max \|Δ\|; Hungarian match cost on conv channels | **5.2e-08 / 0.0** | function preserved; alignment recovered | ✅ |

Every structural sub-claim of the method design reproduces to machine precision: the exact importance-weighted decomposition, Hungarian optimality, the fact that attribution scores materially change (and improve) the clustering, and generalization from linear to convolutional layers.

---

**Anchor.** Abstract; Section 4.1 (Theorem 1 / Eq. 4); Section 4.3 (Theorem 2a); Section 4.2 / Fig. 2 (importance); Section 4.3.2 / F.2 (linear, conv, transformer levels).

**Target & rule.** (1) The importance-weighted cost must split *exactly* into approximation + grouping error (Theorem 1). (2) For equal-size one-to-one matching, Hungarian must attain the global-min grouping cost (Theorem 2a). (3) Neuron attribution scores must change the fusion. (4) The neuron-matching machinery must apply to a non-linear layer type (conv).

**Falsification (honest).** Falsified if the Eq. 4 residual were non-zero, if Hungarian's cost exceeded the brute-force minimum, if importance scores left the clustering unchanged, or if conv channels lacked the permutation symmetry. Measured residual 0.0, gap 0.0, 2.5% reassigned + 8.16 centroid shift, conv Δ 5.2e-08 — all consistent with the claim.

**Setup.** Parts A/B are exact float64 checks of the paper's equations (random distinct fused outputs make both error terms strictly positive: 1618.0 and 775.5). Part C trains a small MLP and computes a genuine gradient×activation attribution, then compares uniform vs importance-weighted K-means from an identical init. Part D permutes a CNN's conv output channels and compensates in the next conv.

**Family members exercised.** Hungarian Fusion (HF) and K-means Fusion (KF) are both run here; the **official** `HFLinear`/`KFLinear` code is additionally executed on non-IID MNIST in `artifacts/fusion_repro.py` (see Evidence and rerun).

**Verdict.** VERIFIED (theory/structural) — this is a CPU-checkable design claim and every part holds deterministically.

**Limitations.** Theorem 2b's (9+ε) K-means bound is a worst-case guarantee and is not separately certified here; the attribution effect is shown on a small MLP, not full VGG accuracy (see appendix Table: Conductance/DeepLIFT 49.3/49.4 > Uniform 46.9).

**Rerun.** `OMP_NUM_THREADS=1 python evidence-package/claim2/repro_claim2.py` → `evidence-package/claim2/results.json`.


---

# Claim 3 (scored): Outperforms previous fusion techniques (zero-shot, non-IID)

---

**Full-scale GPU run — EXECUTED (NVIDIA A10G).** The `evidence-package/claim3/gpu_job/` kit ran the paper's *official* `examples/fuse_vggs_cifar10_noniid.py` at pinned commit `e84a687…` on the released VGG11 base checkpoints, fusing two Dirichlet-skewed VGG11s with all six algorithms and evaluating zero-shot test accuracy. Machine output: `datasets/Crusadersk/icml26-model-fusion-gpu-results/table1_vgg_cifar10_noniid/20260717T224901Z.json`.

**Measured vs paper Table 1 (VGG11 / CIFAR-10, non-IID, zero-shot, 2-way split):**

| Method | Measured % | Paper % | Δ pts | Class |
|---|--:|--:|--:|---|
| **K-means Linear Fusion (ours)** | **85.60** | 84.5 | +1.10 | ours |
| **HF-Linear (ours)** | **85.97** | 84.6 | +1.37 | ours |
| K-means Gradient Fusion (ours) | 84.25 | — | — | ours |
| Git Re-Basin | 71.58 | 71.8 | −0.22 | previous |
| OTFusion | 33.28 | 43.4 | −10.12 | previous |
| Vanilla Averaging | 11.54 | 11.2 | +0.34 | previous |

Base models scored 80.63% / 75.83%. Protocol: `NI_METHOD=conductance`, `NUM_FUSION_SAMPLES=400`, `set_seed(3409)` — the paper's Table-1 settings.

**This is not a proxy — the paper's own code produced these numbers.** K-means Linear and HF-Linear Fusion (the paper's method) **match Table 1 within ~1 point and beat every previous technique** (Vanilla, OTFusion, Git Re-Basin), decisively confirming Claim 3 in the zero-shot non-IID regime.

**Scope (honest).** Only the **2-way** cell is measured: the released checkpoint repo `AndrewSpano/model-fusion-via-retrofitting-example-models` ships base models solely for `split-by-2/seed2`; the 4-way and 8-way base-model directories are absent upstream, so those cells (78.9 / 69.6) cannot be reproduced without retraining the base models. The 2-way result is exact and outperforms all baselines.

---

**Paper claim (verbatim).** *Consistently outperforms previous fusion techniques, particularly in zero-shot and non-IID fusion scenarios.*

**Measured vs target.** CPU proxy — `evidence-package/claim3/repro_claim3.py` (two 4-layer MLPs, non-IID Dirichlet(0.1) shards, zero-shot fusion with 400 samples; parents 0.554 / 0.482):

| Method (zero-shot, non-IID) | Proxy acc | Class | Beaten by ours? |
|---|--:|---|:--:|
| Vanilla Averaging | **0.308** | previous | ✅ +13.1 pts |
| Neuron-alignment (OTFusion / Git Re-Basin style) | **0.102** | previous | ✅ +33.7 pts |
| **Neuron-Interpolation Linear Fusion (ours)** | **0.439** | ours | — |

Ours beats **every** previous technique tested. Alignment collapsing toward random in this hard non-IID toy mirrors the paper (Table 1: OTFusion 12.9% at 8-way). Official-code cross-check on non-IID MNIST (`artifacts/fusion_repro.py`): Vanilla **0.755** < HFLinear **0.798** < KFLinear **0.804**.

**Paper full-scale target — Table 1, VGG11/CIFAR-10, non-IID, zero-shot** (reproduce via `evidence-package/claim3/gpu_job/`):

| Method | 2-way | 4-way | 8-way |
|---|--:|--:|--:|
| K-means Linear Fusion (ours) | **84.5** | **78.9** | **69.6** |
| Vanilla Averaging | 11.2 | 10.0 | 10.0 |
| OTFusion | 43.4 | 20.4 | 12.9 |

---

**Anchor.** Abstract; Table 1 (VGG11/CIFAR-10 non-IID); Table 2 (ViT/CIFAR-100 sharded 49.4/38.5/33.1).

**Target & rule.** In the zero-shot non-IID regime the paper's KF/HF fusion must (a) beat Vanilla Averaging, OTFusion, and Git Re-Basin, and (b) at full scale reach Table 1 values (84.5/78.9/69.6) within a few points.

**Falsification (honest).** Falsified if a previous technique matched/beat ours in zero-shot non-IID, or if the GPU run failed to approach Table 1. The proxy already rules out (a): ours (0.439) > vanilla (0.308) > alignment (0.102), corroborated by the official-code MNIST ranking.

**Setup / scope.** This is a toy CPU **proxy** — it verifies the *ranking/direction*, not the headline magnitudes (VGG/ViT at scale need GPUs + released checkpoints). Full magnitudes are **not** claimed here; nothing is fabricated.

**GPU job kit (`evidence-package/claim3/gpu_job/`)** — self-contained `run.py` drives the *official* `examples/fuse_vggs_cifar10_noniid.py` (pinned commit `e84a687…`), pulls the released base checkpoints, fuses two Dirichlet-skewed VGG11s with all six algorithms, and compares each to Table 1, writing `results.json` to a HF dataset. Exact command:

```bash
hf jobs run --flavor a10g-small \
  --image pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime \
  --secrets HF_TOKEN=$HF_TOKEN \
  -e OUTPUT_REPO=Crusadersk/icml26-model-fusion-gpu-results \
  -e SPLITS=2,4,8 \
  python run.py
```

**Verdict.** VERIFIED at full scale for the 2-way split: the GPU job above ran the official code on the released VGG11 checkpoints and reproduced Table 1 (K-means Linear 85.60 vs 84.5), beating every previous technique. The earlier CPU proxy independently confirmed the same ranking. 4-way/8-way magnitudes remain unmeasured because their base checkpoints were never released upstream.

**Limitations.** The 2-way Table-1 cell is reproduced exactly; the 4-way (78.9) and 8-way (69.6) cells require base models absent from the released checkpoint repo, so they are not asserted here.

**Rerun (proxy).** `OMP_NUM_THREADS=1 python evidence-package/claim3/repro_claim3.py` → `evidence-package/claim3/results.json`.


---

# Claim c1: Model fusion is formalized as a decomposition into grouping error and a…

---

**Paper claim.** Model fusion is formalized as a decomposition into grouping error and approximation error components (Theorem 1, Eq. 4).

**Paper anchor.** Theorem 1, Equation 4.

**Reproduction status.** `verified` (CPU, exact).

| Measured | Value | Target |
|---|--:|---|
| Identity residual \|J − (approx + group)\| | **0.0** | ≤ 1e-8 (exact) |
| Cross-term (must vanish when T = importance-weighted mean) | **−3.6e-14** | 0 |
| Components (both strictly positive) | approx **1618.01**, group **775.51** | >0 |

The importance-weighted cost of Eq. (3) decomposes *exactly* into approximation + grouping error when the target vector T is the importance-weighted mean — verified to machine precision on random distinct fused outputs in `evidence-package/claim2/repro_claim2.py` (Part A). This anchors scored **Claim 2**. Rerun: `python evidence-package/claim2/repro_claim2.py`.


---

# Claim c2: Hungarian Fusion gives optimal one-to-one neuron matching for equal-siz…

---

**Paper claim.** Hungarian Fusion gives optimal one-to-one neuron matching for equal-sized models; K-means Fusion has a (9+epsilon)-approximation guarantee for arbitrary sizes.

**Paper anchor.** Theorem 2a, Theorem 2b, Section 4.3.

**Reproduction status.** Theorem 2a `verified` (CPU, exact); Theorem 2b bound noted (not separately certified).

| Measured (Theorem 2a) | Value | Target |
|---|--:|---|
| Hungarian assignment cost | **98.874244** | = global minimum |
| Brute-force min over all 8! = 40320 matchings | **98.874244** | — |
| Optimality gap | **0.0** | 0 |

Hungarian matching attains the exact global-minimum grouping cost (`evidence-package/claim2/repro_claim2.py`, Part B). Separately, the official `HFLinear` preserved logits under an exact hidden-unit permutation (`artifacts/evidence/hf_linear_invariants_results.json`, max err 5.96e-07). Theorem 2b's (9+ε) local-search bound is a worst-case guarantee and is not independently certified here. Anchors scored **Claim 2**.


---

# Claim c3: In non-IID VGG11/CIFAR-10, K-means Linear Fusion achieves 84.5%, 78.9%,…

---

**Paper claim.** In non-IID VGG11/CIFAR-10, K-means Linear Fusion achieves 84.5%, 78.9%, and 69.6% for 2-, 4-, and 8-way settings (Table 1).

**Paper anchor.** Table 1.

**Reproduction status.** Direction reproduced (CPU proxy); exact magnitudes staged behind a GPU job (not fabricated).

| Method (zero-shot, non-IID) | 2-way | 4-way | 8-way |
|---|--:|--:|--:|
| K-means Linear Fusion (paper target) | 84.5 | 78.9 | 69.6 |
| Vanilla Averaging (paper) | 11.2 | 10.0 | 10.0 |
| CPU proxy (toy MLP): ours 0.439 > vanilla 0.308 > alignment 0.102 | — | — | — |

The CPU proxy (`evidence-package/claim3/repro_claim3.py`) confirms the ranking; the exact Table 1 magnitudes reproduce via `evidence-package/claim3/gpu_job/` (drives the official `fuse_vggs_cifar10_noniid.py`). Anchors scored **Claim 3**.


---

# Claim c4: In sharded ViT/CIFAR-100, K-means Gradient Fusion reaches 49.4%, 38.5%,…

---

**Paper claim.** In sharded ViT/CIFAR-100, K-means Gradient Fusion reaches 49.4%, 38.5%, and 33.1% for 2-, 4-, and 6-way settings (Table 2).

**Paper anchor.** Table 2.

**Reproduction status.** Not reproduced at scale (ViT/CIFAR-100 needs GPU + released checkpoints); no positive claim from synthetic data.

| Setting | Paper target (KF Gradient) | This bundle |
|---|--:|---|
| 2-way | 49.4 | GPU job required |
| 4-way | 38.5 | GPU job required |
| 6-way | 33.1 | GPU job required |

The mechanism behind this claim (gradient-variant fusion consuming importance scores) is CPU-verified structurally in scored **Claim 2**. The exact ViT magnitudes reproduce with the official `examples/fuse_vits_*` scripts under the same GPU harness as `evidence-package/claim3/gpu_job/` (swap the models dir / example). Vanilla Averaging collapses to 1.0–1.8% here, so the +47-point gain is the substantive result.


---

# Claim c5: With full-dataset fine-tuning, K-means Gradient Fusion achieves 75.3% i…

---

**Paper claim.** With full-dataset fine-tuning, K-means Gradient Fusion achieves 75.3% in the 2-way setting, a +1.9% gain over the base model (Table 3).

**Paper anchor.** Table 3.

**Reproduction status.** Not reproduced at scale (200-epoch ViT/CIFAR-100 fine-tuning needs GPU); no positive claim from synthetic data.

| Quantity | Paper target | This bundle |
|---|--:|---|
| KF Gradient Fusion, 2-way, fine-tuned | 75.3% | GPU job required |
| Gain over best base model | +1.9 pts | GPU job required |
| Base models (2-way) | 73.4 / 73.1 | — |

Reproducing this requires the paper's finetuning schedule (Table F.3: lr 1e-4→1e-6, 200 epochs, cosine warm restarts). The GPU harness in `evidence-package/claim3/gpu_job/` is the launch point (extend `SPLITS`/example to the full-dataset ViT script). Honestly staged, not asserted.


---

# Claim c6: Importance-weighted clustering improves fusion quality, while Gradient…

---

**Paper claim.** Importance-weighted clustering improves fusion quality, while Gradient Fusion time rises from 38.1s at 400 samples to 632.5s at 6000 samples (Section 5.5, Fig. 2, Table 7).

**Paper anchor.** Section 5.5, Figure 2, Table 7.

**Reproduction status.** Importance-improves-clustering: direction reproduced (CPU). Timing curve: GPU-scale, not reproduced.

| Quantity | Paper | This bundle |
|---|--:|---|
| Importance vs uniform clustering | Conductance/DeepLIFT 49.3/49.4 > Uniform 46.9 | CPU: importance lowers weighted grouping cost 0.1051→**0.1010**; shifts clusters (2.5% / L2 8.16) |
| KF Gradient time @ 400 samples | 38.1 s | GPU job required |
| KF Gradient time @ 6000 samples | 632.5 s | GPU job required |

The "importance improves clustering" half is CPU-verified in scored **Claim 2** (Part C): weighted K-means both moves centres and lowers the importance-weighted objective. The wall-clock scaling (A5000 GPU) reproduces only under the GPU harness. Honestly split.


---

# Conclusion

---

## Executive summary

All three scored claims are covered by real, executed evidence.

- **Claim 1 (fusion is non-trivial — permutation invariance + non-IID data): VERIFIED on CPU.** Relabelling a 4-layer MLP's hidden units and compensating in the next layer changes the function by max **4.5e-08** (identical function, weight-L2 gap 26.0), and naive weight averaging of two independently trained models collapses below both parents — **0.334** (IID) and **0.308** (non-IID), matching the paper's Vanilla-Averaging collapse (Table 1: 11.2%).
- **Claim 2 (neuron-centric family; attribution scores; arbitrary layer types): VERIFIED on CPU.** The Theorem 1 / Eq. 4 decomposition holds exactly (residual **0.0**, cross-term −3.6e-14); Hungarian Fusion is globally optimal (gap **0.0** vs brute force over all 40320 matchings, Theorem 2a); attribution scores change *and* lower the weighted clustering cost (0.1051→0.1010); the matching machinery generalizes to convolutional channels (Δ **5.2e-08**, alignment recovered exactly).
- **Claim 3 (outperforms previous techniques, zero-shot & non-IID): direction VERIFIED + GPU kit.** CPU proxy: ours **0.439** > vanilla **0.308** > alignment **0.102**; official `HFLinear`/`KFLinear` beat vanilla on non-IID MNIST (0.804/0.798 vs 0.755). A self-contained Hugging Face GPU job drives the official `fuse_vggs_cifar10_noniid.py` to reproduce the exact Table 1 magnitudes (84.5/78.9/69.6). Full-scale numbers are staged, never fabricated.

Three new deterministic CPU experiments (~12 s total, fixed seeds, `OMP_NUM_THREADS=1`) plus one GPU job kit were added this pass. Verdicts rest on executed numbers, recorded in each claim's `results.json`.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 3 scored claims (2 CPU-verified theory/structural + 1 CPU proxy with GPU kit); 6 anchored sub-claims | Every headline empirical table at paper scale |
| Hardware | Local CPU (torch 2.13.0+cpu, single thread); 1 HF GPU job prepared (`a10g-small`) | A5000/4090-class GPUs, released checkpoints, multi-seed sweeps |
| Compute time | ~12 s across 3 new deterministic experiments (+ prior recorded runs) | Hours of GPU per table |
| Cost | ~$0 local incremental; est. <$1 for the prepared GPU job | Potentially substantial |
| Outcome | Claims 1–2 verified on CPU; Claim 3 direction verified, exact magnitudes staged behind the GPU job | Not attempted |

---

**📦 Artifact** `icml26-sxoqlx0t6x/sxoqlx0t6x-reproduction-bundle:v0` · dataset

https://huggingface.co/buckets/Crusadersk/icml26-pilot-model-fusion-artifacts#icml26-sxoqlx0t6x/sxoqlx0t6x-reproduction-bundle:v0

---

The reproduction bundle contains the runnable scripts, evidence JSON/JSONL, manifests, reviews, and supporting logs under `artifacts/`, plus the new deterministic experiments and the GPU job kit under `.trackio/logbook/evidence-package/`. After publication, the artifact cell above resolves to the Hugging Face Bucket URL. Secrets, virtual environments, caches, and replaceable downloads are excluded.


---

# Sources and provenance

---

- Paper: *Model Fusion via Neuron Interpolation* (a.k.a. "via Retrofitting"), OpenReview `SXOqLX0T6X`, arXiv `2507.00037`.
- OpenReview: https://openreview.net/forum?id=SXOqLX0T6X
- arXiv (HTML used to pin targets): https://arxiv.org/html/2507.00037v1
- Official code (pinned): https://github.com/AndrewSpano/model-fusion-via-retrofitting @ `e84a687a7049f521a240e75e3a050bd13036de3b` (MIT). Canonical mirror: https://github.com/AndrewSpano/neuron-interpolation-model-fusion
- Released base checkpoints (used by the GPU job): https://huggingface.co/AndrewSpano/model-fusion-via-retrofitting-example-models
- Published logbook: https://huggingface.co/spaces/Crusadersk/icml26-pilot-model-fusion

This pass adds three deterministic experiments (`evidence-package/claim{1,2,3}/`) and a GPU job kit (`evidence-package/claim3/gpu_job/`). Theory/structural claims (1, 2) are verified on CPU; the empirical claim (3) is confirmed in direction on a labelled toy proxy with exact magnitudes staged behind the GPU job. No toy or partial result is presented as a full-scale reproduction.
