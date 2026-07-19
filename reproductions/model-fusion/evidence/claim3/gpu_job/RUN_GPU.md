# GPU job — reproduce Table 1 (VGG11 / CIFAR-10, non-IID, zero-shot)

`run.py` drives the **official** fusion example
`examples/fuse_vggs_cifar10_noniid.py` from the pinned paper repo
(`AndrewSpano/model-fusion-via-retrofitting` @ `e84a687a7049f521a240e75e3a050bd13036de3b`).
It clones the repo, pulls the released base checkpoints (~1 GB) from
`AndrewSpano/model-fusion-via-retrofitting-example-models`, fuses two Dirichlet-skewed
VGG11s with all six algorithms using the paper's exact configs, evaluates zero-shot test
accuracy, compares each number to Table 1, writes `results.json`, and (optionally) pushes
it to a Hugging Face dataset. It fabricates nothing — it records what the official code prints.

## Targets it checks (Table 1, %)

| Algorithm | 2-way | 4-way | 8-way | class |
|---|--:|--:|--:|---|
| K-means Linear Fusion (Ours) | 84.5 | 78.9 | 69.6 | paper method |
| Vanilla Averaging | 11.2 | 10.0 | 10.0 | previous |
| OTFusion | 43.4 | 20.4 | 12.9 | previous |
| Git Re-Basin | 71.8 | — | — | previous |

Success = KF/HF-Linear land near the paper values AND beat every "previous" baseline.

## Run it (Hugging Face Jobs)

Exact command (A10G, ~$1/hr; the 2-way split runs in a few minutes):

```bash
hf jobs run --flavor a10g-small \
  --image pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime \
  --secrets HF_TOKEN=$HF_TOKEN \
  -e OUTPUT_REPO=Crusadersk/icml26-model-fusion-gpu-results \
  -e SPLITS=2,4,8 \
  python run.py
```

`hf jobs run` executes the command inside the image; make `run.py` + `requirements.txt`
available first. The supported one-liner that uploads the local script and installs deps is:

```bash
hf jobs uv run --flavor a10g-small \
  --image pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime \
  --with-requirements requirements.txt \
  --secrets HF_TOKEN=$HF_TOKEN \
  -e OUTPUT_REPO=Crusadersk/icml26-model-fusion-gpu-results \
  -e SPLITS=2,4,8 \
  run.py
```

(If the image lacks the paper's deps, `run.py` installs them at start via `pip install -e <repo>`.)

## Output

`results.json` holds, per split, `{measured_accuracy: {...}, vs_table1: {algo: {measured_pct,
paper_pct, abs_diff_pts}}}`, plus GPU name and runtime. With `OUTPUT_REPO`+`HF_TOKEN` set it
is pushed to `datasets/<OUTPUT_REPO>/table1_vgg_cifar10_noniid/<timestamp>.json`.

## Cost / footprint

- Flavor `a10g-small` (24 GB A10G). 2-way ≈ 2–3 min; 2,4,8 splits ≈ 10–15 min incl. the 1 GB
  checkpoint download. Est. well under $1.
- Deterministic seed (`set_seed(3409)`); `NI_METHOD=conductance`; `NUM_FUSION_SAMPLES=400`
  (matches the paper's Table 1 protocol).
