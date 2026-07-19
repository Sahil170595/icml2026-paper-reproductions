# GPU job — Claim 3 (real-world robustness at scale)

**Measures.** Trains **with vs without** the official `DropoutTSCallback` on the paper's
real datasets/backbones (100 epochs, `EarlyStopping(10)`, `use_clean_targets=True`,
seed 42) and reports test MSE/MAE plus the **% MSE improvement** per (dataset, model, H):
- `ETTh2` × Informer, `Electricity` × Informer, `Weather` × TimeMixer.

**Paper targets.** up to **68.0%** MSE (Electricity), **47.6%** (ETTh2), **13.8%**
(Weather). `run.py` prints the measured best improvement per dataset next to these
targets. **No number is fabricated** — metrics come only from executed training.

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
  --timeout 12h \
  bash -lc '
    set -e
    pip install -q "huggingface_hub[cli]"
    hf download Crusadersk/icml26-pilot-dropoutts --repo-type space \
      --include ".trackio/logbook/evidence-package/claim3/gpu_job/*" \
      --include ".trackio/logbook/evidence-package/dropout_ts.py" \
      --local-dir /tmp/job
    cd /tmp/job/.trackio/logbook/evidence-package/claim3/gpu_job
    pip install -q -r requirements.txt
    python run.py --out results.json
  '
```

The job clones the official repo at the pinned commit, runs each dataset's
`scripts/data_preparation/<Dataset>/generate_training_data.py` (downloads + windows the
raw data), trains both arms, and uploads `claim3/results.json`. Electricity (321 vars)
is the heaviest cell; start with `--horizons 96` and the ETTh2 task to validate, then
scale up. Estimated 6–12 GPU-h on a10g-small for all three datasets × 4 horizons.
