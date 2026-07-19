# GPU job — Claim 2 (Synth-12 robustness at scale)

**Measures.** For the Informer backbone on the paper's synthetic noise datasets
`SyntheticTS_noise{0.1,0.3,0.5,0.7,0.9}` and horizons `H in {96,192,336,720}`, trains
the backbone **with vs without** the official `DropoutTSCallback` (100 epochs,
`EarlyStopping(patience=10)`, `use_clean_targets=True`, seed 42), then reports test
MSE/MAE and the **% MSE/MAE improvement**.

**Paper target.** 46.0% avg MSE / 24.5% avg MAE improvement over Informer; peak 48.2%
MSE at sigma=0.3. `run.py` writes the measured averages next to these targets. **No
number is fabricated** — metrics come only from executed training.

## Prerequisites
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login                       # sets HF_TOKEN
hf repo create <user>/dropoutts-repro-results --repo-type dataset   # results sink
```

## Exact command
```bash
hf jobs run \
  --flavor a10g-small \
  --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
  --secret HF_TOKEN \
  --env HF_RESULTS_REPO=<user>/dropoutts-repro-results \
  --timeout 6h \
  bash -lc '
    set -e
    pip install -q "huggingface_hub[cli]"
    hf download Crusadersk/icml26-pilot-dropoutts --repo-type space \
      --include ".trackio/logbook/evidence-package/claim2/gpu_job/*" \
      --include ".trackio/logbook/evidence-package/dropout_ts.py" \
      --local-dir /tmp/job
    cd /tmp/job/.trackio/logbook/evidence-package/claim2/gpu_job
    pip install -q -r requirements.txt
    python run.py --model Informer --out results.json
  '
```

The job clones `https://github.com/CityMind-Lab/DropoutTS.git` at commit
`64a096ec6801d9506ab3a30541b6f1b6dbbd7f40`, generates each synthetic dataset with the
repo's own generator, trains both arms, and uploads `claim2/results.json` to the
`HF_RESULTS_REPO` dataset. Estimated 2–5 GPU-h on a10g-small for the full sweep; pass
`--sigmas 0.3 --horizons 96` for a fast single-cell smoke test first.
