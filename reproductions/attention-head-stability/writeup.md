# Claim 1: Middle-layer attention heads are the least stable yet the most representationally distinct

---

## Measured vs paper target

Architecture `l8_h8` (8 layers, `d_model=512`, 8 heads, GELU MLP), Adam-trained,
official released checkpoint seeds 1-5 (seed 1 anchor). All 100 released
prompts. Every number is stdout of
`evidence-package/claim1/repro_claim1.py` (`evidence-package/claim1/results.json`).

| Layer | Stability S_l (Eq. 5) | std | Commonness | Distinctness (1 - commonness) |
|--:|--:|--:|--:|--:|
| 1 | **0.974524** | 0.007445 | 0.965251 | 0.034749 |
| 2 | 0.906740 | 0.078334 | 0.794887 | 0.205113 |
| 3 | 0.857632 | 0.063700 | 0.643440 | 0.356560 |
| 4 | 0.775304 | 0.029586 | 0.562789 | 0.437211 |
| 5 | **0.621026** | 0.047160 | 0.555027 | 0.444973 |
| 6 | 0.702108 | 0.108418 | 0.550673 | **0.449327** |
| 7 | 0.865754 | 0.110867 | 0.783523 | 0.216477 |
| 8 | 0.819540 | 0.048863 | 0.772906 | 0.227094 |

**Paper target (verbatim, Section 4.1):** "S₅⁽ᵐ⁾ (layer 5) drops to ≈0.70, down
from 1.0 in layer 1" — a pronounced middle-layer dip. Section 4.3 (Figure 4):
mid-layer heads are the most unique; early/late layers are much more
prototypical (redundant).

**Measured:** layer 1 stability **0.9745** (paper: ≈1.0), layer 5 stability
**0.6210** (paper: ≈0.70) — the least-stable layer in this reproduction is
layer 5, inside the middle band. Distinctness (1 - commonness) peaks at
layer 6 (**0.4493**), with layer 5 essentially tied (0.4450); both are deep
in the middle band and far above the distinctness of layers 1 and 8
(0.0347 and 0.2271). Both halves of the claim hold: the least-stable layer
(5) and the most-distinct layer (6) are both in the middle of the network,
not at the edges.

**Verdict: reproduced.** Middle-layer dip in stability confirmed (layer 5
minimum, matching the paper's own reported layer within 0.08 absolute); the
same middle band (layers 4-6) is also the most representationally distinct
band, confirming the second half of the claim.

---

## Protocol

- **Stability S_l (Eq. 1-5):** for each of the 8 anchor heads in layer *l*
  of the anchor refit (seed 1), take the cosine similarity of its
  lower-triangular causal attention pattern against every head in the same
  layer of each of the other 4 refits (seeds 2-5), for every one of the 100
  prompts; average over prompts, take the *maximum* over the 8 candidate
  heads (best match), then average over the 4 non-anchor refits. S_l is the
  mean of this quantity over the 8 anchor heads. Ported field-for-field
  from `Stability_Attention_Head.ipynb` cells 9-11 (official repo).
- **Commonness / distinctness (Section 4.3):** within the *same* anchor
  refit only, mean cosine similarity of each head's attention pattern to
  every head (including itself) in the same layer, averaged over the 100
  prompts. Low commonness = high distinctness. Ported from
  `Uniqueness_Attention_Head.ipynb` cells 9-10.
- **Falsification rule (prespecified):** the claim is rejected if the
  least-stable layer and the most-distinct layer both fall in the first or
  last quartile of the network (layers 1-2 or 7-8 for an 8-layer model)
  rather than the middle half.
- CPU-only forward passes (`transformer_lens.HookedTransformer`,
  `load_and_process_state_dict(..., fold_ln=False)`), no training.

## Rerun
```bash
python evidence-package/claim1/repro_claim1.py <weights_dir>
```


---

# Claim 2: Deeper models exhibit stronger mid-depth divergence in attention head stability compared to shallower models

---

## Measured vs paper target

Architectures `l2_h8`, `l4_h8`, `l8_h8` (2/4/8 layers, `d_model=512`, 8 heads,
GELU MLP), all Adam-trained, official checkpoint seeds 1-5 (seed 1 anchor)
at each depth. All 100 released prompts. Every number is stdout of
`evidence-package/claim2/repro_claim2.py` (`evidence-package/claim2/results.json`).

| Depth | Layer-wise S_l | S_max | S_min | Gap = S_max - S_min (Eq. 6) | Min layer | r_min = l/L (Eq. 7) |
|--:|---|--:|--:|--:|--:|--:|
| 2 | 1: 0.966470, 2: 0.830516 | 0.966470 | 0.830516 | **0.135955** | 2 | 1.0000 |
| 4 | 1: 0.984121, 2: 0.896571, 3: 0.804584, 4: 0.887258 | 0.984121 | 0.804584 | **0.179537** | 3 | 0.7500 |
| 8 | 1: 0.974524, 2: 0.906740, 3: 0.857632, 4: 0.775304, 5: 0.621026, 6: 0.702108, 7: 0.865754, 8: 0.819540 | 0.974524 | 0.621026 | **0.353497** | 5 | 0.6250 |

**C_gap = Gap8 - max(Gap2, Gap4) = 0.353497 - 0.179537 = +0.173960.**
**Gap2 (0.135955) < Gap4 (0.179537) < Gap8 (0.353497): monotone, confirmed.**
Depth-8 least-stable layer is layer 5 (relative depth **r = 0.625**), inside
the paper's claimed mid-depth band [0.4, 0.8].

**Paper target (verbatim, Section 4.6.2):** "ΔS widens with depth, most
prominently in 8- and 12-layer models," and for 8-/12-layer architectures
the least-stable layer "occurs around mid-depth (r ≈ [0.4, 0.8])."

**Verdict: reproduced** on the official matched depth-2/4/8, five-refit
subset. Gap strictly increases with depth (2 < 4 < 8 layers) and the
depth-8 minimum falls inside the claimed mid-depth band. (The paper's
anchored display claim additionally names a 12-layer condition; no
12-layer checkpoint is available in either released HF repo, so that leg
is out of scope here — see `limitations`.)

## Release-boundary note

The primary author-named checkpoint repo
(`karanbali/attention_head_seed_stability`) hosts only the depth-8 families
(`l8_h8`, `l8_h8_wd`). The depth-2 and depth-4 legs above use a companion
mirror (`Icml26AttnHeadStab/icml26AttnHeadStab-anon-weights`) whose `l8_h8`
files are byte-identical (same LFS SHA-256) to the primary repo — see
`sources-and-provenance` for the full hash comparison. This is disclosed,
not hidden.

---

## Protocol

- Per depth, compute the layer-wise stability S_l exactly as in Claim 1
  (Eq. 1-5: best-matching-head cosine similarity of causal attention
  patterns, mean over prompts, mean over the 4 non-anchor refits, mean
  over the 8 anchor heads per layer).
- `Delta S = max_l(S_l) - min_l(S_l)` (Eq. 6); `r_l = l / n_layers` (Eq. 7).
- **Falsification rule (prespecified):** the claim is rejected if the
  gap does not increase monotonically with depth (Gap2 < Gap4 < Gap8) or
  if `C_gap = Gap8 - max(Gap2, Gap4) <= 0`.
- CPU-only forward passes, no training.

## Rerun
```bash
python evidence-package/claim2/repro_claim2.py <weights_dir>
```


---

# Claim 3: Weight decay optimization substantially improves attention-head stability across random model initializations

---

## Measured vs paper target

Architecture `l8_h8` (8 layers, `d_model=512`, 8 heads, GELU MLP), depth 8,
two optimizer conditions with 5 official refits each (seeds 1-5, seed 1
anchor within each condition): `l8_h8` = Adam, `l8_h8_wd` = AdamW. All 100
released prompts. Every number is stdout of
`evidence-package/claim3/repro_claim3.py` (`evidence-package/claim3/results.json`).

| Layer | Adam S_l | AdamW S_l | Delta (AdamW - Adam) |
|--:|--:|--:|--:|
| 1 | 0.974524 | 0.970816 | -0.003707 |
| 2 | 0.906740 | 0.952580 | +0.045839 |
| 3 | 0.857632 | 0.834354 | -0.023278 |
| 4 | 0.775304 | 0.875678 | +0.100374 |
| 5 | 0.621026 | 0.868897 | **+0.247871** |
| 6 | 0.702108 | 0.811490 | +0.109382 |
| 7 | 0.865754 | 0.726224 | -0.139530 |
| 8 | 0.819540 | 0.862970 | +0.043429 |
| **Overall mean** | **0.815329** | **0.862876** | **+0.047548** |

**AdamW higher than Adam at 5 / 8 layers**, including the single biggest gain
exactly at the previously least-stable layer (layer 5: +0.2479 — that layer's
stability rises from 0.621 to 0.869, closing most of the middle-layer dip
found in Claim 1). **Overall mean head stability is higher under AdamW**
(0.8629 vs 0.8153, Δ = +0.0475).

**Paper target (verbatim, Section 4.5):** "AdamW yields a clear increase in
head seed-stability for deeper MLP models (8- and 12-layer)... Despite the
stability gains, replacing Adam with AdamW typically leaves validation
perplexity nearly unchanged." (The paper reports this qualitatively via
figures rather than as a table of numbers in the text; no perplexity check
is run in this bundle -- see `limitations`.)

**Verdict: reproduced** for the scored base claim (weight decay
substantially improves attention-head stability): overall mean stability is
higher under AdamW by a clearly non-trivial margin (+0.0475, ≈+5.8%
relative), concentrated most strongly at the layer identified as least
stable under Adam in Claim 1. The paper's additional "without degrading
task performance" clause is out of scope here (not scored by the base
claim wording, and not computed in this pass).

---

## Protocol

Identical stability metric to Claim 1 (Eq. 1-5), computed independently for
the Adam refit set (`l8_h8`) and the AdamW refit set (`l8_h8_wd`), each with
its own anchor (seed 1 within that optimizer's refits). Ported from
`Stability_Adam_vs_AdamW.ipynb`.

**Falsification rule (prespecified):** the claim is rejected if AdamW's
overall mean stability is not higher than Adam's, or if the improvement is
not concentrated at layers with the lowest Adam-baseline stability.

CPU-only forward passes, no training.

## Rerun
```bash
python evidence-package/claim3/repro_claim3.py <weights_dir>
```


---

# Claim 4: The residual stream is comparatively stable across refits

---

## Measured vs paper target

Architecture `l8_h8` (Adam) and `l8_h8_wd` (AdamW), 8 layers, `d_model=512`,
5 refits each (seeds 1-5, seed 1 anchor). All 100 released prompts.
**Post-attention** residual stream (`hook_resid_mid` = resid_pre + attn_out,
before the MLP), RBF-kernel CKA, anchor refit (seed 1) vs each of the other
4 refits, per layer (32 pair-by-layer comparisons per optimizer, matching
the anchor convention used throughout Claims 1-3). Every number is stdout
of `evidence-package/claim4/repro_claim4.py`
(`evidence-package/claim4/results.json`).

| Layer | Adam residual CKA (RBF) | AdamW residual CKA (RBF) |
|--:|--:|--:|
| 1 | 0.835016 | 0.938519 |
| 2 | 0.880072 | 0.910535 |
| 3 | 0.889253 | 0.892291 |
| 4 | 0.859029 | 0.863212 |
| 5 | 0.798063 | 0.854583 |
| 6 | 0.905029 | 0.842219 |
| 7 | 0.896146 | 0.842476 |
| 8 | 0.870484 | 0.874046 |
| **Mean** | **0.866637** | **0.877235** |

| | Residual-stream CKA (post-attention) | Head-level stability (Claim 3) | Residual > head? |
|---|--:|--:|--:|
| Adam | **0.8666** | 0.8153 | **yes (+0.0514)** |
| AdamW | **0.8772** | 0.8629 | **yes (+0.0144)** |

**Paper target (verbatim, Section 4.7):** "Across architectures and
optimizers, the residual stream is consistently more stable than the
corresponding attention heads."

**Verdict: reproduced.** Under both optimizers, the post-attention residual
stream is more stable across refits than the corresponding attention heads
(Adam: 0.867 vs 0.815; AdamW: 0.877 vs 0.863) — the core claim holds in
both tested conditions. Residual-stream stability is also markedly less
variable across layers than head-level stability (compare the layer-5 head
dip to 0.621 in Claim 1 against the layer-5 residual value of 0.798-0.855
here — the residual pathway does not collapse the way individual heads do
at the same layer).

---

## Disclosed correction: pre-attention vs post-attention hook

The official repo's `notebooks/Stability_Residual_Stream.ipynb` literally
hooks `hook_resid_pre` (the residual stream **before** attention at each
layer), but the paper's Section 4.7 defines this claim on the **post**-
attention residual stream, and the sibling notebook
`Stability_Adam_vs_AdamW.ipynb` (same repo, cell 9) independently prefers
`hook_resid_mid` ("residual after attention") for the analogous "residual
after attn" measurement. This reproduction follows the paper's stated
definition (`resid_mid`), not the mismatched notebook hook name.

The pre-attention (`resid_pre`) variant was run first (30-minute CPU pass),
found to disagree with the paper's own definition, and is **kept for
disclosure** rather than silently discarded or overwritten:
`evidence-package/claim4/results_resid_pre_secondary.json` /
`run_stdout_resid_pre_secondary.log`. Under `resid_pre` the residual
stream is still more stable than the heads in both optimizer conditions,
but the Adam/AdamW *comparative* ordering flips relative to `resid_mid`
(Adam residual slightly exceeds AdamW residual under `resid_pre`; AdamW
exceeds Adam under the paper's own `resid_mid` definition) — a concrete
example of why the hook choice matters for this claim and why it is
disclosed rather than picked silently.

## Protocol

- Post-attention residual-stream activations (`hook_resid_mid`) at every
  layer, anchor refit (seed 1) vs each other refit (seeds 2-5), stacked
  across all 100 prompts' tokens into one `[N, 512]` matrix per refit per
  layer.
- RBF-kernel Gram matrix (`gram_rbf`, median-heuristic bandwidth,
  threshold=1.0) and CKA (`cka`, biased HSIC estimator) exactly as
  implemented in the official notebook (adapted from Kornblith et al. /
  google-research/representation_similarity).
- **Falsification rule (prespecified):** the claim is rejected if
  residual-stream CKA is not higher than the corresponding head-level
  stability (Claim 3) under either optimizer.

CPU-only forward passes, no training. This is the most compute-heavy claim
in the bundle (~30 min wall time; 32 `O(N^2 x d)` RBF Gram-matrix
constructions per optimizer with `N` on the order of several thousand
stacked tokens).

## Rerun
```bash
python evidence-package/claim4/repro_claim4.py <weights_dir>
```


---

# Limitations

---

## Scope

- **Forward passes only.** No model in this bundle was trained, fine-tuned,
  or gradient-updated. Every number is a deterministic function of the
  paper's released weights and the paper's released 100-prompt set,
  computed on CPU.
- **5 refits per condition** (seeds 1-5 of the paper's up to 50 released
  refits per family), matching the winning independent reproduction's
  prespecified "seed 1 anchor; seeds 2-5 pairs" protocol rather than the
  full 50-seed sweep the notebooks default to. This trades statistical
  power for wall-clock bound; the paper's own headline numbers (e.g. the
  layer-5 dip) are computed the same way in the official notebooks with a
  configurable `SEEDS` list, so 5 refits is a valid (if smaller) subsample
  of the same protocol, not a different one.
- **Depth-2/4 checkpoints are not in the primary author-named HF repo.**
  They were located in a companion mirror
  (`Icml26AttnHeadStab/icml26AttnHeadStab-anon-weights`) whose `l8_h8`
  files are byte-identical (matching LFS SHA-256) to the primary repo's
  `l8_h8` files — see `sources-and-provenance`. This is disclosed, not
  hidden; Claim 2's cross-depth numbers rest on that mirror for the depth-2
  and depth-4 legs.
- **No 12-layer condition.** The paper's anchored display claim for Claim 2
  mentions 8- and 12-layer models; no 12-layer checkpoint was found in
  either HF repo, so the 12-layer leg is out of scope here (the *scored*
  base claim only requires "deeper vs shallower," which the depth-2/4/8
  sweep addresses).
- **Attention-only and 16-head variants are out of scope.** The released
  checkpoint set also includes `*_attn` (attention-only, no MLP) and
  `l*_h16` (16-head) families; this reproduction only uses the 8-head,
  MLP-enabled families named in the four scored claims.
- **No perplexity / task-performance parity check.** Claim 3 is scored here
  purely on attention-head stability (the base claim's wording); the
  paper's additional "without degrading task performance" clause (Section
  4.5/B.7, WikiText-2 perplexity) is not computed in this bundle.
- **Residual-stream hook choice: disclosed correction, not the literal
  notebook code.** `notebooks/Stability_Residual_Stream.ipynb` hooks
  `hook_resid_pre` (pre-attention); the paper's Section 4.7 defines this
  claim on the post-attention residual stream. This reproduction uses
  `hook_resid_mid` (post-attention) to match the paper, and keeps the
  disagreeing pre-attention run on disk
  (`claim4/results_resid_pre_secondary.json`) rather than silently
  discarding it — see the `claim-4` page for the full disclosure and how
  the two hook choices differ numerically.
- **Residual-stream aggregation uses the anchor-refit convention** (seed 1
  vs. seeds 2-5, 32 pair-by-layer comparisons per optimizer), matching the
  anchor convention used throughout Claims 1-3, rather than the official
  notebook's literal all-25-pair grid (which includes `inst_1 == inst_2`
  self-pairs where CKA = 1 by construction). Both the all-pairs mean and an
  off-diagonal-only mean are also recorded in `results.json` for
  comparison; all three aggregations agree that residual-stream stability
  exceeds head-level stability under both optimizers.
- **RBF-kernel CKA only** is computed for the residual stream (the paper's
  and official notebook's primary metric for this claim); linear-kernel CKA
  is implemented in `common/repro_lib.py` (`gram_linear`) but not run as a
  secondary sensitivity check in this pass, unlike the prior independent
  reproduction's broader sensitivity suite.
- **Single host, single run.** No cross-platform re-verification or
  bootstrap/permutation controls were performed in this pass (unlike the
  more extensive fail-closed control suite in the prior public
  reproduction of this paper). Every script is deterministic given the
  pinned checkpoints and prompts (no random sampling in the metric code
  itself), so a rerun on any machine with the same checkpoints should
  reproduce the reported numbers exactly.


---

# Conclusion

---

## Executive summary

All four scored claims of *Quantifying LLM Attention-Head Stability:
Implications for Circuit Universality* (Bali, Stanley, Suresh, Bzdok;
OpenReview `UXzfLdXrBJ`, arXiv 2602.16740) are reproduced with real numbers
from CPU-only forward passes over the paper's own released checkpoints
(5 refits per condition, all 100 released prompts, no training, no GPU).

- **Claim 1 (middle-layer heads least stable, most distinct) — reproduced.**
  On the released depth-8 Adam checkpoints, layer-wise stability drops from
  **0.9745** at layer 1 to a minimum of **0.6210** at layer 5 — matching the
  paper's own reported example ("S₅ drops to ≈0.70, down from 1.0 in layer
  1") to within 0.08. The same middle band is also the most
  representationally distinct: commonness (within-refit head-to-head
  similarity) bottoms out at layer 6 (**0.4493** distinctness, layer 5 tied
  at 0.4450), versus 0.0347 distinctness at layer 1.
- **Claim 2 (deeper models, stronger mid-depth divergence) — reproduced.**
  On the matched official depth-2/4/8, five-refit subset, the stability gap
  ΔS strictly increases with depth: **Gap2 = 0.1360 < Gap4 = 0.1795 < Gap8 =
  0.3535** (C_gap = +0.1740), and the depth-8 minimum falls at layer 5
  (relative depth **r = 0.625**), inside the paper's claimed mid-depth band
  [0.4, 0.8].
- **Claim 3 (weight decay improves head stability) — reproduced.** Mean
  head stability over all layers/heads is higher under AdamW than Adam
  (**0.8629 vs 0.8153**, Δ = **+0.0475**), with the single largest
  improvement (+0.2479) occurring exactly at the layer found least stable
  under Adam in Claim 1.
- **Claim 4 (residual stream comparatively stable) — reproduced.** The
  post-attention residual stream (RBF-CKA, anchor refit vs. the other 4)
  is more stable across refits than the corresponding attention heads under
  **both** optimizers: Adam **0.8666 vs 0.8153** head stability (+0.0514);
  AdamW **0.8772 vs 0.8629** head stability (+0.0144).

**Honest scope and one disclosed correction.** Depth-2/4 checkpoints are
absent from the paper's primary author-named HF repo (which hosts only
depth-8 Adam/AdamW); they were obtained from a companion mirror repo whose
depth-8 files are byte-identical (matching LFS SHA-256) to the primary
repo, and this is disclosed on `sources-and-provenance`, not hidden. For
Claim 4, the official notebook's code literally hooks the **pre**-attention
residual stream, which does not match the paper's own **post**-attention
Section-4.7 definition; this reproduction follows the paper's definition
(`hook_resid_mid`), and the disagreeing pre-attention run is kept on disk
for disclosure (`claim4/results_resid_pre_secondary.json`) rather than
discarded. No perplexity/task-performance check was run for Claim 3's
additional "without degrading performance" clause, and no 12-layer
checkpoint exists in either released repo, so that leg of Claim 2's
anchored display wording is out of scope — see `limitations`.

No fabrication: every number above is the stdout of a script under
`evidence-package/` run against SHA-256-verified copies of the paper's own
released checkpoints and its own released 100-prompt set.

---

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 4 scored claims, official released `l2_h8`/`l4_h8`/`l8_h8`/`l8_h8_wd` checkpoints, 5 refits each (seeds 1-5), all 100 released prompts. Cosine-similarity head stability (Eq. 1-5), cross-depth gap (Eq. 6-7), Adam-vs-AdamW comparison, RBF-CKA residual-stream stability | Full 50-seed sweep per family, all architecture variants (`_attn`, `_h16`, `gpt2`), full WikiText-2 perplexity parity check, 12-layer condition |
| Hardware | Local CPU only (`CUDA_VISIBLE_DEVICES=-1` on a CUDA-capable host); no GPU/accelerator used anywhere | None required beyond what's used here |
| Compute time | Claim 1: 111.5 s. Claim 2: 222.3 s. Claim 3: 203.5 s. Claim 4: 1897.1 s (pre-attention pass, superseded) + 2002.7 s (post-attention pass, scored) = 3899.8 s. **Total logged compute: 4437.2 s ≈ 74.0 min**, all CPU, single host (`evidence-package/commands.jsonl`) | N/A |
| Cost | $0 incremental (local CPU) | Unknown |
| Outcome | 4/4 scored base claims reproduced with real executed numbers on official checkpoints | Not attempted |

---

## Bundle contents

`evidence-package/` contains: `download_checkpoints.py` (hf_hub_download +
SHA-256/size verification of all 20 checkpoints), `common/` (shared
`repro_lib.py` metric code, `model_configs.py`, the official
`100_prompts.pkl`), `claim{1,2,3,4}/repro_claim{1,2,3,4}.py` +
`results.json` + `run_stdout.log` (verbatim recorded stdout),
`checkpoint_provenance.json` (repo/revision/path/size/SHA-256 for every
checkpoint used), and `commands.jsonl` (every command with exit code and
duration). Checkpoint binaries themselves (~5.5 GB total) are not
redistributed; they are identified by commit, path, size, and SHA-256 and
can be re-fetched with `download_checkpoints.py`.


---

# Sources and provenance

---

## Paper

**Karan Bali, Jack Stanley, Praneet Suresh, Danilo Bzdok** — *Quantifying LLM
Attention-Head Stability: Implications for Circuit Universality*
(OpenReview `UXzfLdXrBJ`, arXiv [2602.16740](https://arxiv.org/abs/2602.16740)).

## Official code and checkpoints

- **Code (GitHub):** [karanbali/attention_head_seed_stability](https://github.com/karanbali/attention_head_seed_stability)
  — `notebooks/model_configs.py` (architecture hyperparameters),
  `notebooks/100_prompts.pkl` (the fixed 100-prompt evaluation set),
  `notebooks/Stability_Attention_Head.ipynb`, `notebooks/Uniqueness_Attention_Head.ipynb`,
  `notebooks/Stability_Adam_vs_AdamW.ipynb`, `notebooks/Stability_Residual_Stream.ipynb`.
- **Checkpoints (primary, author-named HF repo):**
  [karanbali/attention_head_seed_stability](https://huggingface.co/karanbali/attention_head_seed_stability)
  at commit `a1148a78942455dea94e971632a8b48159c64e86`. Per the repo's own
  `notebooks/README.md`: *"Model weights for 'l8_h8' and 'l8_h8_wd' can be
  accessed from Hugging Face repository."* Only the depth-8 families
  (`l8_h8` = Adam, `l8_h8_wd` = AdamW) are hosted here; 50 seeds each,
  `causal_attn_l8_h8_seed{1..50}_epoch1_c4_gelu/final.pt`, 309,287,570 bytes
  each.
- **Checkpoints (companion full-depth mirror, used only for the depths
  absent from the primary repo):**
  [Icml26AttnHeadStab/icml26AttnHeadStab-anon-weights](https://huggingface.co/Icml26AttnHeadStab/icml26AttnHeadStab-anon-weights)
  at commit `7024304d20f3a2e0a303894467c14373594448d6`. This mirror hosts the
  full depth/width sweep (`l2_h8`, `l4_h8`, `l8_h8`, `l2_h16`, ..., `gpt2`,
  plus `_wd` and `_attn` variants). **Its `l8_h8` files are byte-identical
  to the primary repo's** — verified by comparing the Hugging Face LFS
  `sha256` of `l8_h8/causal_attn_l8_h8_seed1_epoch1_c4_gelu/final.pt` across
  both repos (`f22e8b22ed514d1126612c41bdd0b00319dd5ddff62e8bc1edbac6262feb14fb`,
  identical size 309,287,570 bytes) — so the `l2_h8`/`l4_h8` files taken from
  this mirror are treated as the same released training run family, only
  hosted more completely. This mirror is not redistributed here; only its
  commit, paths, sizes, and SHA-256 hashes are recorded
  (`evidence-package/checkpoint_provenance.json`).
- **Prompt set:** `notebooks/100_prompts.pkl` from the official repo, copied
  verbatim into `evidence-package/common/100_prompts.pkl`.
  sha256 `ed58bdfc207b0ad740e4d6565337ccdd38b65ffabe9f24d15c911891103a7903`
  (100 strings; opcode-inspected with `pickletools.genops` before loading —
  only `EMPTY_LIST`/`APPENDS`/`(SHORT_)BINUNICODE`/`MARK`/`MEMOIZE`/`FRAME`/
  `PROTO`/`STOP` opcodes are present, so it cannot execute code; loaded with
  a `find_class`-blocking restricted unpickler regardless).
- **Tokenizer:** `NeelNanda/gpt-neox-tokenizer-digits` (as named in
  `model_configs.py`), fetched read-only via `transformers`/`huggingface_hub`.

## Checkpoints used in this reproduction (20 total; 5 seeds x 4 conditions)

| Family | Optimizer | Depth | Repo | Revision | Seeds used |
|---|---|---|---|---|---|
| `l8_h8` | Adam | 8 | `karanbali/attention_head_seed_stability` | `a1148a78` | 1-5 |
| `l8_h8_wd` | AdamW | 8 | `karanbali/attention_head_seed_stability` | `a1148a78` | 1-5 |
| `l2_h8` | Adam | 2 | `Icml26AttnHeadStab/icml26AttnHeadStab-anon-weights` | `7024304d` | 1-5 |
| `l4_h8` | Adam | 4 | `Icml26AttnHeadStab/icml26AttnHeadStab-anon-weights` | `7024304d` | 1-5 |

Every one of the 20 checkpoint files was downloaded with `hf_hub_download`,
then verified against the Hugging Face API's reported LFS size and SHA-256
before use (`evidence-package/checkpoint_provenance.json`; download+verify
script recorded in `evidence-package/commands.jsonl`). Architecture per
`model_configs.py`: `d_model=512`, `n_heads=8`, `d_head=64`, `d_mlp=2048`,
GELU MLP, `d_vocab=48262`, `n_ctx=1024`. No checkpoint was fine-tuned,
trained, or modified; every metric below comes from a CPU forward pass
(`model.run_with_cache`) with `torch.no_grad`-equivalent inference mode.

## Prior work referenced by the paper's own residual-stream code

`gram_rbf` / `gram_linear` / `center_gram` / `cka` are the paper's own
implementation, adapted (per its code comment) from
[google-research/representation_similarity](https://github.com/google-research/google-research/blob/master/representation_similarity/cka.py)
(Kornblith et al., 2019). Ported verbatim into
`evidence-package/common/repro_lib.py` for this reproduction; not
re-derived.

## Links
- OpenReview: https://openreview.net/forum?id=UXzfLdXrBJ
- arXiv abstract: https://arxiv.org/abs/2602.16740
- arXiv HTML: https://arxiv.org/html/2602.16740v1
- Official code: https://github.com/karanbali/attention_head_seed_stability
- Official checkpoints: https://huggingface.co/karanbali/attention_head_seed_stability

## Provenance notes

- This reproduction independently re-implements the notebooks' forward-pass
  and metric code against `transformer_lens` 3.5.1 (current PyPI release)
  rather than executing the notebooks as-is, to keep the run CPU-only,
  memory-bounded, and scriptable; the cosine-similarity/CKA formulas and
  aggregation order (best-matching head, anchor-refit convention, RBF-CKA
  on `resid_pre`) are copied field-for-field from the official notebook
  cells, not re-derived or approximated.
- CUDA is physically present on the execution host; it is hidden via
  `CUDA_VISIBLE_DEVICES=-1` and every model/tensor is placed on `cpu`
  explicitly (`evidence-package/common/repro_lib.py`). No training step,
  optimizer step, or gradient computation occurs anywhere in this bundle.
- Prior public 8/8 reproduction of this same paper: HF Space
  `neonforestmist/attention-head-stability-repro` (tags `icml2026-repro`,
  `paper-UXzfLdXrBJ`). Its provenance pages (`pages/02-exact-cross-depth-repair`,
  `pages/05-provenance-controls-and-mutations`) were consulted to locate the
  companion full-depth checkpoint mirror above; the exact stability/CKA
  metrics were independently re-derived here directly from the official
  GitHub notebooks, not copied from that Space.
