# Claim 2 - full-scale GPU job (Categorical VAE on MNIST, paper Table 3)

`run.py` trains the paper's actual Categorical VAE (convolutional encoder ->
`N x K` categorical scores -> Gumbel-Softmax straight-through -> transposed-conv
decoder, ELBO objective, temperature annealed 1.0 -> 0.5 with exp decay 3e-5)
and reports the paper's exact Table-3 metric: **test-set NLL estimated with 512
importance samples (lower is better)**, for the three parameterizations
`softmax`, `catnat_sigmoid`, `catnat_nu`, swept over `N in {10,20,30}` and
`K in {8,16,32}` on MNIST and binarized MNIST.

This is the **verified-evidence path**. It is NOT run yet: the current logbook
evidence for Claim 2 is the deterministic CPU proxy in `../repro_claim2.py`
(toy-scale exact-ELBO VAE). No full-scale number is reported until this job is
executed and writes `results.json`.

## What it measures (paper target it will confirm or falsify)

Paper Table 3 finding: *both* catnat parameterizations beat softmax on test NLL
across every `(N,K)` setting, with the natural activation best in the majority.
The job reproduces the exact `test_nll_mean +/- std` per `(N,K,method)` so the
claim can be checked head-to-head against the paper's numbers.

## Exact command (Hugging Face Jobs, A10G)

```bash
# 1) pick a HF dataset repo to receive results (created automatically)
export HF_RESULT_REPO=Crusadersk/icml26-catnat-claim2-results
export HF_TOKEN=hf_xxx   # a token with write access

# 2) launch the GPU job
hf jobs run \
  --flavor a10g-small \
  --timeout 6h \
  -s HF_TOKEN=$HF_TOKEN \
  -e HF_RESULT_REPO=$HF_RESULT_REPO \
  pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
  bash -c "pip install -q torchvision numpy huggingface_hub && \
    curl -sL https://huggingface.co/spaces/Crusadersk/icml26-catnat-beyond-softmax-repro/resolve/main/.trackio/logbook/evidence-package/claim2/gpu_job/run.py -o run.py && \
    python run.py --hf-repo $HF_RESULT_REPO --epochs 100 && \
    python run.py --hf-repo $HF_RESULT_REPO --epochs 100 --binarize"
```

Results are uploaded to `HF_RESULT_REPO:claim2/results.json` and are also
checkpointed after every `(N,K,method)` config, so partial progress survives a
time-out.

## Fast smoke test (single setting, ~10-15 min on A10G)

```bash
hf jobs run --flavor a10g-small --timeout 40m \
  -s HF_TOKEN=$HF_TOKEN -e HF_RESULT_REPO=$HF_RESULT_REPO \
  pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
  bash -c "pip install -q torchvision numpy huggingface_hub && \
    curl -sL https://huggingface.co/spaces/Crusadersk/icml26-catnat-beyond-softmax-repro/resolve/main/.trackio/logbook/evidence-package/claim2/gpu_job/run.py -o run.py && \
    python run.py --hf-repo $HF_RESULT_REPO --N-list 10 --K-list 8 --seeds 0 1 2 --epochs 60"
```

## Local run (if you have a CUDA box instead of HF Jobs)

```bash
pip install -r requirements.txt
python run.py --epochs 100 --out results.json                 # MNIST sweep
python run.py --epochs 100 --binarize --out results_bin.json  # binarized MNIST
```

## CLI options

| flag | default | meaning |
|---|---|---|
| `--N-list` | `10 20 30` | numbers of categorical latents |
| `--K-list` | `8 16 32` | categories per latent (catnat needs powers of 2) |
| `--seeds` | `0 1 2` | random seeds (paper uses more; increase for tighter CIs) |
| `--methods` | `softmax catnat_sigmoid catnat_nu` | parameterizations compared |
| `--epochs` | `100` | training epochs per config |
| `--n-importance` | `512` | importance samples for the test-NLL estimate (paper protocol) |
| `--binarize` | off | use binarized MNIST (threshold 0.5) |
| `--hf-repo` | `$HF_RESULT_REPO` | HF dataset repo to upload `results.json` |

Approx. cost: the full `3 x 3 x 3` sweep on both datasets is a few A10G-hours;
`a10g-small` is ~$1/hr, so a full replication is on the order of a few dollars.
