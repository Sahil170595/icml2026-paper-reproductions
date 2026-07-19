# GPU job — Claim 1 (training speedup at scale)

**Measures.** The 4-extra-parameters and eval-mode-identity sub-claims are already
**decisively verified on CPU** (`../repro_claim1.py`). This job measures the one part
that needs scale: **end-to-end training wall-clock** for the Informer backbone **with
vs without** `DropoutTSCallback` under the paper's 100-epoch + `EarlyStopping(10)`
protocol on ETTh2, and re-counts the added parameters at real-backbone scale.

**Paper target.** **1.12x–1.45x** training speedup (`baseline_wall / dropoutts_wall`),
achieved via faster convergence → earlier early-stopping. `run.py` writes the measured
speedup next to this range. **No number is fabricated.**

## Prerequisites
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
hf repo create <user>/dropoutts-repro-results --repo-type dataset
```

## Exact command
```bash
hf jobs run \
  --flavor a10g-small \
  --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
  --secret HF_TOKEN \
  --env HF_RESULTS_REPO=<user>/dropoutts-repro-results \
  --timeout 4h \
  bash -lc '
    set -e
    pip install -q "huggingface_hub[cli]"
    hf download Crusadersk/icml26-pilot-dropoutts --repo-type space \
      --include ".trackio/logbook/evidence-package/claim1/gpu_job/*" \
      --include ".trackio/logbook/evidence-package/dropout_ts.py" \
      --local-dir /tmp/job
    cd /tmp/job/.trackio/logbook/evidence-package/claim1/gpu_job
    pip install -q -r requirements.txt
    python run.py --model Informer --dataset ETTh2 --num_features 7 --horizon 96 --epochs 100
  '
```

Reports `baseline_wall_seconds`, `dropoutts_wall_seconds`, `training_speedup_x`, and the
re-counted extra-parameter total, uploaded as `claim1/results.json`. ~1–3 GPU-h.
