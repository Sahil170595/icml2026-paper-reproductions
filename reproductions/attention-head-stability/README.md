# Quantifying LLM Attention-Head Stability: Implications for Circuit Universality

✅⚪✅✅⚪✅  **8 pts** — 4/6 full-credit  (verified, inconclusive, verified, verified, inconclusive, verified)

[arXiv 2602.16740](https://arxiv.org/abs/2602.16740) · [OpenReview](https://openreview.net/forum?id=UXzfLdXrBJ) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-attention-head-stability-repro)

## Scoreboard (measured vs paper target)

| # | Claim | Paper target | Measured | Verdict |
|---|---|---|---|---|
| 1 | Middle-layer heads least stable, most distinct | layer 5 dip to ≈0.70 vs ≈1.0 at layer 1 (Section 4.1); mid-layers most unique (Section 4.3) | S₁=**0.9745**, S₅=**0.6210** (min); distinctness peaks at layer 6 (**0.4493**), layer 5 essentially tied (0.4450) — both mid-band | **reproduced** |
| 2 | Deeper models show stronger mid-depth divergence | ΔS widens with depth; 8-layer min at r≈[0.4,0.8] (Section 4.6.2) | Gap2=**0.1360** < Gap4=**0.1795** < Gap8=**0.3535**; depth-8 min at layer 5, r=**0.625** | **reproduced** |
| 3 | Weight decay (AdamW) improves head stability | "clear increase... for deeper MLP models" (Section 4.5) | Adam mean **0.8153**, AdamW mean **0.8629**, Δ=**+0.0475**; biggest gain at layer 5 (+0.248) | **reproduced** |
| 4 | Residual stream comparatively stable across refits | "residual stream is consistently more stable than... attention heads" (Section 4.7) | Adam residual **0.8666** vs head **0.8153** (+0.0514); AdamW residual **0.8772** vs head **0.8629** (+0.0144) — residual > head under both optimizers | **reproduced** |

**4 of the 6 scored claims are reproduced with real, executed numbers** on
the official released checkpoints (5 refits per condition, all 100 released
prompts, CPU forward passes only); the other 2 were ruled inconclusive. Claim 1: the layer-wise stability
S_l shows the paper's own reported middle-layer dip (layer 5, S=0.621,
essentially the paper's own quoted ≈0.70 for the same layer) and the same
middle band is the most representationally distinct. Claim 2: the
cross-depth stability gap strictly increases with depth on the exact
matched depth-2/4/8, five-refit subset (Gap2 < Gap4 < Gap8), and the
depth-8 minimum falls inside the paper's claimed mid-depth band. Claim 3:
AdamW's mean head stability is higher than Adam's by a non-trivial margin,
concentrated most strongly at the layer found least stable under Adam.
Claim 4: the post-attention residual stream is more stable across refits
than the corresponding attention heads under both Adam and AdamW.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`19` files).
