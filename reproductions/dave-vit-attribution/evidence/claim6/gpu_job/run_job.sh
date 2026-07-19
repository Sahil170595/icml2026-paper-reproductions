#!/usr/bin/env bash
# Full Figure-6 (pixel-deletion faithfulness) benchmark as a Hugging Face GPU Job.
set -euo pipefail
hf jobs run \
  --flavor a10g-large \
  --image "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime" \
  --env HF_TOKEN="${HF_TOKEN}" --secret HF_TOKEN \
  -- \
  bash -lc "pip install -q timm kornia && \
            python benchmark.py --models vit_b deit3_b \
              --imagenet /data/imagenet/val --n_images 2000 --out results_fig6.json"
# Acceptance: DAVE deletion AUC (target prob, least->most-important removal) is the highest
# (flattest curve) vs all six baselines on ViT-B/16 and DeiT-III-B/16 (paper Figure 6).
