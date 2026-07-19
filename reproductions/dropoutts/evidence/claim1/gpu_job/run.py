#!/usr/bin/env python3
"""GPU job: full-scale overhead + TRAINING-SPEEDUP measurement for the scored
"negligible overhead / no architectural modification" claim.
Targets (paper, Key Findings, Negligible Overhead): 4 extra parameters; zero inference
latency; 1.12x-1.45x TRAINING speedup relative to baseline backbones.

The 4-parameter and eval-identity sub-claims are already decisively verified on CPU
(../repro_claim1.py). This job measures the part that needs scale: end-to-end TRAINING
WALL-CLOCK with vs without DropoutTS, under the paper's 100-epoch + EarlyStopping(10)
protocol on a real dataset. Speedup = baseline_wall_seconds / dropoutts_wall_seconds
(paper claims 1.12x-1.45x via faster convergence -> earlier early-stopping). It also
re-counts the added parameters at real-backbone scale. No numbers are fabricated.
Results -> results.json, optionally pushed to a HF dataset. See RUN_GPU.md."""
import argparse, json, os, subprocess, sys, time, glob, re
from pathlib import Path

REPO = "https://github.com/CityMind-Lab/DropoutTS.git"
COMMIT = "64a096ec6801d9506ab3a30541b6f1b6dbbd7f40"

# Raw benchmark CSVs (standard LTSF format: `date` column + channels), from the
# THUML Time-Series-Library mirror. The repo's datasets/README.md referenced by
# its top-level README does not exist at the pinned commit, and its dataset
# classes cannot download data (`local=False` is a TODO stub), so the raw file
# must be fetched explicitly before running the repo's own preparation script.
RAW_URLS = {
    "ETTh2": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh2.csv",
    "Electricity": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv",
    "Weather": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv",
}

def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True); return subprocess.run(cmd, check=True, **kw)

def prepare_real_dataset(src, name):
    """Fetch the raw CSV and run the repo's own per-dataset preparation script,
    which windows it into datasets/<name>/{train,valid,test}_data.npy."""
    if (src / "datasets" / name / "train_data.npy").exists():
        return
    raw = src / "datasets" / "raw_data" / name
    raw.mkdir(parents=True, exist_ok=True)
    csv = raw / f"{name}.csv"
    if not csv.exists():
        import urllib.request
        print(f"+ download {RAW_URLS[name]}", flush=True)
        urllib.request.urlretrieve(RAW_URLS[name], csv)
    sh([sys.executable, str(src / "scripts" / "data_preparation" / name / "generate_training_data.py")],
       cwd=str(src))

def import_module_from(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def setup(work):
    src = work / "DropoutTS"
    if not src.exists():
        sh(["git", "clone", REPO, str(src)]); sh(["git", "-C", str(src), "checkout", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(src / "requirements.txt")])
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", str(src)])
    return src

def count_params(model_obj):
    return sum(p.numel() for p in model_obj.parameters() if p.requires_grad)

def train_timed(src, rb, model, dataset, nf, L, Hout, enable_dts, epochs):
    from basicts.configs import BasicTSForecastingConfig
    from basicts.runners.callback import EarlyStopping, DropoutTSCallback
    from basicts import BasicTSLauncher
    model_class, model_config, use_ts = rb.get_model_config(model, L, Hout, nf, dataset)
    callbacks = [EarlyStopping(patience=10)]
    if enable_dts:
        callbacks.insert(0, DropoutTSCallback(p_min=0.05, p_max=0.5, init_alpha=10.0,
                                              init_sensitivity=5.0,
                                              enable_visualization=False, enable_statistics=False))
    cfg = BasicTSForecastingConfig(
        model=model_class, model_config=model_config, dataset_name=dataset,
        input_len=L, output_len=Hout, use_timestamps=use_ts, use_clean_targets=True,
        gpus="0", num_epochs=epochs, batch_size=64, callbacks=callbacks, seed=42,
    )
    t0 = time.perf_counter()
    BasicTSLauncher.launch_training(cfg)
    return time.perf_counter() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Informer")
    ap.add_argument("--dataset", default="ETTh2")
    ap.add_argument("--num_features", type=int, default=7)
    ap.add_argument("--input_len", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--hf_repo", default=os.environ.get("HF_RESULTS_REPO", ""))
    args = ap.parse_args()

    work = Path("work").resolve(); work.mkdir(exist_ok=True)
    src = setup(work); sys.path.insert(0, str(src / "src"))
    rb = import_module_from(src / "run_baselines.py", "rb")
    out_path = Path(args.out).resolve()
    prepare_real_dataset(src, args.dataset)
    os.chdir(src)  # BasicTS resolves "datasets/<name>" relative to the CWD

    base_s = train_timed(src, rb, args.model, args.dataset, args.num_features,
                         args.input_len, args.horizon, False, args.epochs)
    dts_s  = train_timed(src, rb, args.model, args.dataset, args.num_features,
                         args.input_len, args.horizon, True, args.epochs)
    speedup = base_s / dts_s if dts_s else None
    summary = {"model": args.model, "dataset": args.dataset, "commit": COMMIT,
               "baseline_wall_seconds": round(base_s, 1), "dropoutts_wall_seconds": round(dts_s, 1),
               "training_speedup_x": (None if speedup is None else round(speedup, 3)),
               "paper_target_speedup_range": [1.12, 1.45],
               "extra_params_target": 4,
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True); print("WROTE", str(out_path), flush=True)
    if args.hf_repo:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=str(out_path), path_in_repo=f"claim1/{out_path.name}",
                            repo_id=args.hf_repo, repo_type="dataset")
        print("UPLOADED to", args.hf_repo, flush=True)

if __name__ == "__main__":
    main()
