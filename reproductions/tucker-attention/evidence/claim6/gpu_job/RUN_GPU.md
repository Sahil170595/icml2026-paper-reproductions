# GPU job kit — Claim 6 (ViT: Tucker Attention vs GQA vs MLA)

Reproduces the **validation-accuracy vs attention-parameter** result of Figure 3 /
Section 4.1 of *Tucker Attention* (arXiv 2603.30033 / OpenReview ErcPPRZaiq) on a
**real Vision Transformer** and a **real ImageNet subset** (Imagenette — 10 native
ImageNet classes, full-resolution photographs). This is the part of Claim 6 that
cannot be verified on CPU without fabrication; the CPU sanity checks (layer
correctness, gradcheck, parameter count) are in `../repro_claim6.py`.

`run.py` loads a **pretrained** ViT (timm), replaces every attention layer with the
chosen variant **initialized from the pretrained MHA weights** (SVD for GQA/MLA,
HOSVD for Tucker — Appendix C.1.4), fine-tunes (AdamW + cosine schedule w/ warmup,
Table 5), and writes `val_top1`, `val_top5`, and `attn_params_total` to a JSON file.
One method per job; sweeping the ranks reproduces the Pareto frontier of Figure 3.

## Exact command (Hugging Face Jobs)

`run.py` carries PEP-723 inline dependencies, so `hf jobs uv run` installs them on
the GPU automatically. Authenticate first (`hf auth login`) and set `HF_TOKEN`.

```bash
# Tucker Attention, ranks [r1,r2,r3]=[8,32,32], 10 epochs on an A10G:
hf jobs uv run --flavor a10g-large --timeout 2h \
  --secrets HF_TOKEN=$HF_TOKEN \
  run.py -- \
  --method tucker --r1 8 --r2 32 --r3 32 \
  --model vit_small_patch16_224 --dataset frgfm/imagenette --dataset_config 320px \
  --epochs 10 --batch 128 --lr 1e-4 --out out_tucker_8_32_32.json
```

## Full sweep to build the Figure-3 Pareto frontier

```bash
# baseline
hf jobs uv run --flavor a10g-large --timeout 2h --secrets HF_TOKEN=$HF_TOKEN run.py -- \
  --method mha --epochs 10 --out out_mha.json

# GQA family  (n_kv = 1 is MQA)
for K in 1 2 3; do
  hf jobs uv run --flavor a10g-large --timeout 2h --secrets HF_TOKEN=$HF_TOKEN run.py -- \
    --method gqa --n_kv $K --epochs 10 --out out_gqa_$K.json
done

# MLA family
for DC in 16 32 64; do
  hf jobs uv run --flavor a10g-large --timeout 2h --secrets HF_TOKEN=$HF_TOKEN run.py -- \
    --method mla --dc $DC --epochs 10 --out out_mla_$DC.json
done

# Tucker family  (r1=head rank, r2=query/output rank, r3=key/value rank)
for R in "8 16 16" "8 32 32" "4 32 32" "8 64 64"; do
  set -- $R
  hf jobs uv run --flavor a10g-large --timeout 2h --secrets HF_TOKEN=$HF_TOKEN run.py -- \
    --method tucker --r1 $1 --r2 $2 --r3 $3 --epochs 10 --out out_tucker_${1}_${2}_${3}.json
done
```

## Alternative (explicit CUDA image + pip)

```bash
hf jobs run --flavor a10g-large --timeout 2h --secrets HF_TOKEN=$HF_TOKEN \
  pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
  bash -c "pip install -q timm datasets torchvision pillow && python run.py \
           --method tucker --r1 8 --r2 32 --r3 32 --epochs 10 --out out_tucker.json"
```
(`run.py` must be made available in the container, e.g. via a small HF model/dataset
repo or `--repo`; `hf jobs uv run` above avoids this by uploading the local script.)

## Expected output (per job)

```
[tucker] epoch 10/10  val top1=XX.XX  top5=XX.XX  loss=Y.YYY  (Zs)
RESULT {"method": "tucker", ..., "val_top1": ..., "val_top5": ...,
        "attn_params_total": ..., "attn_params_MB_bf16": ...}
```

## What the sweep should show (paper claim to be tested)

Plot `val_top5` (y) against `attn_params_total` (x, log scale) across all JSON
outputs. The paper's claim (Fig 3): **the Tucker points sit up and to the left of
the GQA and MLA points** — i.e. Tucker matches GQA/MLA top-5 accuracy while using
**almost an order of magnitude fewer** attention parameters. Acceptance rule for
this reproduction: at matched top-5 (within ~0.5%), the smallest Tucker config uses
≤ ~1/5 the attention parameters of the smallest GQA/MLA config at that accuracy,
and no Tucker config is Pareto-dominated by GQA/MLA.

## Cost / hardware

Single A10G (24 GB). ViT-S/16 on Imagenette (~9.5k train / ~3.9k val), 10 epochs
≈ 15–25 min/job. Full 11-job sweep ≈ 3–4.5 GPU-hours (order ~$5–10 at typical
A10G pricing). Scale to ViT-L/32 + ImageNet-1k (`--model vit_large_patch32_224
--dataset imagenet-1k --dataset_config default`) to match the paper exactly (Table
5: batch 512, 10 epochs, lr 1e-3), which needs an A100 and is materially costlier.
