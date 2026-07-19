#!/usr/bin/env bash
# Full Table-2 (GridPG/EnergyPG, B-cos ViTs) benchmark as a Hugging Face GPU Job.
set -euo pipefail
hf jobs run \
  --flavor a10g-large \
  --image "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime" \
  --env HF_TOKEN="${HF_TOKEN}" --secret HF_TOKEN \
  -- \
  bash -lc "pip install -q timm kornia bcos && \
            python benchmark.py --models bcos_vit bcos_vit_c \
              --imagenet /data/imagenet/val --n_images 2000 --n_samples 50 --out results_table2.json"
# Expected DAVE (paper Table 2): B-cos-ViT GridPG 84.00 / EnergyPG 78.55 ;
#                                B-cos-ViT-C GridPG 88.43 / EnergyPG 79.63.
