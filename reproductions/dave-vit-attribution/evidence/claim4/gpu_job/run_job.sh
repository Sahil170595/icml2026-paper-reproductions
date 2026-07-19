#!/usr/bin/env bash
# Launch the full Table-1 (GridPG/EnergyPG, conventional ViTs) benchmark as a Hugging Face GPU Job.
# Requires: `pip install huggingface_hub` and `hf auth login`; an ImageNet-1k val mount with bboxes.
set -euo pipefail
hf jobs run \
  --flavor a10g-large \
  --image "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --secret HF_TOKEN \
  -- \
  bash -lc "pip install -q timm kornia && \
            python benchmark.py --models vit_b deit_b deit3_b dino_b \
              --imagenet /data/imagenet/val --bbox /data/imagenet/bboxes.json \
              --n_images 2000 --n_samples 50 --out results_table1.json"
# Expected DAVE targets (paper Table 1): GridPG ViT-B 60.19 / DeiT-B 63.52 / DeiT-III-B 65.76 /
# DINO-B 51.33 ; EnergyPG DeiT-B 82.23 / DeiT-III-B 82.43 / DINO-B 83.38.
