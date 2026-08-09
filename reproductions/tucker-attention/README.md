# Tucker Attention: A generalization of approximate attention mechanisms

✅✅✅✅✅⚪  **10 pts** — 5/6 full-credit  (verified, verified, verified, verified, verified, inconclusive)

[arXiv 2603.30033](https://arxiv.org/abs/2603.30033) · [OpenReview](https://openreview.net/forum?id=ErcPPRZaiq) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-tucker-attention-repro)

## Scoreboard — measured vs. paper target

| # | Paper claim | Target + acceptance rule | Measured (this repro) | Verdict |
|---|---|---|---|---|
| 1 | **MHA** is an exact special case of Tucker Attention (Thm B.2) | tensor reformulation (Eq 6) and Tucker forward reproduce MHA to <1e-9; ranks (n_H,d,d) | Eq-6 err **4.6e-13**, Tucker→MHA **6.0e-14**, ranks **(4,20,20)**✓ | reproduced |
| 2 | **GQA/MQA** are exact special cases (Thm B.3), key-mode rank **n_KV·d_H** | Tucker output = GQA to <1e-9; measured key-mode rank = n_KV·d_H | out **4–7e-14**; key-rank **8 / 16 / 4** = n_KV·d_H (n_KV=2/4/1)✓ | reproduced |
| 3 | **MLA** is an exact special case (Thm B.4), ranks (n_H,d_c^Q,d_c^K) | Tucker output = MLA to <1e-9; ranks exact | out **0.9–1.1e-13**; ranks (n_H,d_c^Q,d_c^K) & (n_H,d,d_c^K)✓ | reproduced |
| 4 | **Parameter savings** O(r·d) vs O(d²); Tables 1–3; "18% of MHA, 39% of MLA, up to 9×" | reproduce GPT2/LLaMA MB values; ratios; Tucker slope≈1 vs MHA≈2 | GPT2 Table-2 max dev **0.05 MB**; **18.1% / 39.4% / 8.99×**; slopes Tucker **0.78** vs MHA **2.00** | reproduced |
| 5 | **Fully compatible with flash-attention and RoPE** (abstract; Sec. 3.2–3.3): latent RoPE rel-position + MLA RoPE without decoupling (Cor 3.2.1) **+ flash-attention tiled/online-softmax exactness** (Sec. 3.3) | RoPE: R(m)R(n)ᵀ=R(m−n); score=f(m−n); query fusion exact. Flash: tiled == naive to <1e-9 across generic/MHA/GQA/MLA + RoPE; tiled peak memory O(block²) vs naive O(N²) | RoPE **1.6e-15 / 2.3e-13 / 2.6e-13**; control err **1335**. Flash: tiled-vs-naive **3.1e-12** (generic), **7.2e-12** (MHA/GQA/MLA vs reference), **5.7e-14** (+RoPE); memory ratio up to **4096×** at N=4096 | reproduced |
| 6 | **ViT**: Tucker matches GQA/MLA val perf with ~10× fewer params (Fig 3) | CPU: layer correct+differentiable & param axis; accuracy via GPU kit | layer fwd **1.1e-18**, gradcheck **True**, **23.9× / 12.2×** fewer params vs GQA/MLA; accuracy → GPU kit | layer verified; accuracy = GPU kit |

Claims 1–5 are decisive CPU-exact verifications (machine precision or ≤0.05 MB). Claim 6's headline **val-accuracy** number is **not** CPU-reproducible without fabrication, so it is deferred to the accompanying `hf jobs run` kit (real ViT-S/16 on the Imagenette ImageNet subset, Tucker vs GQA vs MLA, low-rank-initialized from pretrained MHA); the CPU checks confirm the Tucker layer is correct, differentiable, and — the other half of the claim — an order of magnitude smaller than GQA/MLA at the ViT-B.16 config.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`23` files).
