#!/usr/bin/env python3
"""GPU job: full-scale REAL-WORLD reproduction for the scored robustness claim.
Targets (paper, Real-World Benchmarks, 7 datasets):
  - Electricity: up to 68.0% MSE improvement (Informer backbone)
  - ETTh2:       up to 47.6% MSE improvement (Informer backbone)
  - Weather:     13.8% MSE improvement (TimeMixer backbone)

Trains the OFFICIAL DropoutTS repo (pinned commit) at scale: for each (dataset, model,
horizon) it runs the backbone WITHOUT DropoutTS (baseline) and WITH DropoutTS
(DropoutTSCallback), reports MSE/MAE and the % improvement. No numbers are fabricated --
metrics come only from executed training. Results -> results.json and (if HF_TOKEN set)
pushed to a Hugging Face dataset repo. See RUN_GPU.md for the exact `hf jobs run` command."""
import argparse, json, os, subprocess, sys, time, glob, re
from pathlib import Path

REPO = "https://github.com/CityMind-Lab/DropoutTS.git"
COMMIT = "64a096ec6801d9506ab3a30541b6f1b6dbbd7f40"

# (dataset, num_features, model) triples mirroring the paper's headline real-world cells
DEFAULT_TASKS = [
    ("ETTh2", 7, "Informer"),
    ("Electricity", 321, "Informer"),
    ("Weather", 21, "TimeMixer"),
]

# Raw benchmark CSVs (standard LTSF format: `date` column + channels), from the
# THUML Time-Series-Library mirror. The repo cannot download data itself: its
# datasets/README.md does not exist at the pinned commit and the dataset class's
# `local=False` download path is a TODO stub.
RAW_URLS = {
    "ETTh2": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh2.csv",
    "Electricity": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv",
    "Weather": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv",
}

def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)

def import_module_from(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def setup(work: Path):
    src = work / "DropoutTS"
    if not src.exists():
        sh(["git", "clone", REPO, str(src)])
        sh(["git", "-C", str(src), "checkout", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(src / "requirements.txt")])
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", str(src)])
    return src

def prepare_dataset(src: Path, dataset: str):
    """Fetch the raw CSV, then run the repo's per-dataset preparation script,
    which windows it into datasets/<name>/{train,valid,test}_data.npy."""
    if (src / "datasets" / dataset / "train_data.npy").exists():
        return
    raw = src / "datasets" / "raw_data" / dataset
    raw.mkdir(parents=True, exist_ok=True)
    csv = raw / f"{dataset}.csv"
    if not csv.exists():
        import urllib.request
        print(f"+ download {RAW_URLS[dataset]}", flush=True)
        urllib.request.urlretrieve(RAW_URLS[dataset], csv)
    sh([sys.executable, str(src / "scripts" / "data_preparation" / dataset / "generate_training_data.py")],
       cwd=str(src))

def parse_metrics(save_dir: Path):
    mse = mae = None
    # Read the runner's authoritative test_metrics.json ({"overall": {"MSE":..,
    # "MAE":..}}), written to ckpt_save_dir/<md5>/test_metrics.json. This is
    # exact; the previous log-regex approach mis-parsed multi-"test/" log lines.
    for f in glob.glob(str(save_dir / "**" / "test_metrics.json"), recursive=True):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        overall = d.get("overall", d) if isinstance(d, dict) else {}
        for k, val in (overall.items() if isinstance(overall, dict) else []):
            if not isinstance(val, (int, float)):
                continue
            if k.upper() == "MSE": mse = float(val)
            elif k.upper() == "MAE": mae = float(val)
    return mse, mae

def train_one(src, rb, model, dataset, num_features, L, Hout, enable_dts):
    from basicts.configs import BasicTSForecastingConfig
    from basicts.runners.callback import EarlyStopping, DropoutTSCallback
    from basicts import BasicTSLauncher
    model_class, model_config, use_ts = rb.get_model_config(model, L, Hout, num_features, dataset)
    callbacks = [EarlyStopping(patience=10)]
    if enable_dts:
        callbacks.insert(0, DropoutTSCallback(p_min=0.05, p_max=0.5, init_alpha=10.0,
                                              init_sensitivity=5.0,
                                              enable_visualization=False, enable_statistics=False))
    save_dir = Path(f"runs/{model}_{dataset}_{Hout}_{'dts' if enable_dts else 'base'}").resolve()
    cfg = BasicTSForecastingConfig(
        model=model_class, model_config=model_config,
        dataset_name=dataset, input_len=L, output_len=Hout,
        use_timestamps=use_ts, use_clean_targets=True,
        gpus="0", num_epochs=100, batch_size=64, callbacks=callbacks, seed=42,
    )
    # Point the runner's checkpoint/log dir at save_dir so parse_metrics reads
    # the training_log this exact run produced (the default is a shared
    # checkpoints/ tree keyed by config md5, which parse_metrics cannot see).
    cfg.ckpt_save_dir = str(save_dir)
    BasicTSLauncher.launch_training(cfg)
    return parse_metrics(save_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    ap.add_argument("--input_len", type=int, default=96)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--hf_repo", default=os.environ.get("HF_RESULTS_REPO", ""))
    args = ap.parse_args()

    work = Path("work").resolve(); work.mkdir(exist_ok=True)
    src = setup(work)
    sys.path.insert(0, str(src / "src"))
    rb = import_module_from(src / "run_baselines.py", "rb")
    out_path = Path(args.out).resolve()

    def checkpoint(payload):
        """Write and (best-effort) upload partial results so a job timeout
        cannot zero out completed rows."""
        out_path.write_text(json.dumps(payload, indent=2))
        if args.hf_repo:
            try:
                from huggingface_hub import HfApi
                HfApi().upload_file(path_or_fileobj=str(out_path),
                                    path_in_repo=f"claim3/{out_path.name}",
                                    repo_id=args.hf_repo, repo_type="dataset")
            except Exception as e:
                print("[hf] partial upload failed (continuing):", e, flush=True)

    os.chdir(src)  # BasicTS resolves "datasets/<name>" relative to the CWD
    rows = []
    for dataset, nf, model in DEFAULT_TASKS:
        prepare_dataset(src, dataset)
        for Hout in args.horizons:
            L = 24 if dataset == "Illness" else args.input_len
            base_mse, base_mae = train_one(src, rb, model, dataset, nf, L, Hout, False)
            dts_mse, dts_mae   = train_one(src, rb, model, dataset, nf, L, Hout, True)
            row = {"dataset": dataset, "model": model, "H": Hout,
                   "baseline_mse": base_mse, "dropoutts_mse": dts_mse,
                   "baseline_mae": base_mae, "dropoutts_mae": dts_mae,
                   "mse_improvement_pct": (None if not base_mse or dts_mse is None else round((base_mse - dts_mse)/base_mse*100, 2))}
            rows.append(row); print(json.dumps(row), flush=True)
            checkpoint({"partial": True, "commit": COMMIT, "rows": rows})

    best = {}
    for r in rows:
        if r["mse_improvement_pct"] is None:
            continue
        best[r["dataset"]] = max(best.get(r["dataset"], -1e9), r["mse_improvement_pct"])
    summary = {"commit": COMMIT, "rows": rows, "best_mse_improvement_pct_by_dataset": best,
               "paper_targets": {"Electricity": 68.0, "ETTh2": 47.6, "Weather": 13.8},
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    checkpoint(summary)
    print("WROTE", str(out_path), flush=True)
    if args.hf_repo:
        print("UPLOADED to", args.hf_repo, flush=True)

if __name__ == "__main__":
    main()
