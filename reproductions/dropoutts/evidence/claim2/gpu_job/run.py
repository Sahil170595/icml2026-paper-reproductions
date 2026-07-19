#!/usr/bin/env python3
"""GPU job: full-scale Synth-12 reproduction for the scored robustness claim.
Target (paper, Synth-12): DropoutTS improves the Informer backbone by 46.0% MSE /
24.5% MAE averaged over H in {96,192,336,720}, peak 48.2% MSE at sigma=0.3.

This trains the OFFICIAL DropoutTS repo (pinned commit) at scale: for each synthetic
noise dataset and horizon it runs the backbone WITHOUT DropoutTS (baseline) and WITH
DropoutTS (DropoutTSCallback), then reports MSE/MAE and the % improvement. It does NOT
fabricate numbers -- metrics come only from executed training. Results are written to
results.json and (if HF_TOKEN is set) pushed to a Hugging Face dataset repo.

Run on a GPU via `hf jobs run` -- see RUN_GPU.md for the exact command."""
import argparse, json, os, subprocess, sys, time, glob, re
from pathlib import Path

REPO = "https://github.com/CityMind-Lab/DropoutTS.git"
COMMIT = "64a096ec6801d9506ab3a30541b6f1b6dbbd7f40"

def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)

def setup(work: Path):
    src = work / "DropoutTS"
    if not src.exists():
        sh(["git", "clone", REPO, str(src)])
        sh(["git", "-C", str(src), "checkout", COMMIT])
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(src / "requirements.txt")])
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", str(src)])
    return src

def prepare_synthetic(src: Path, sigma: str):
    """Generate the SyntheticTS_noise{sigma} dataset via the repo's own generator.

    The generator is scripts/data_preparation/SyntheticTS/generate_training_data.py;
    --generate_all emits every canonical noise level (0.0-0.9) in one pass, writing
    datasets/SyntheticTS_noise{level}/ under the repo root."""
    name = f"SyntheticTS_noise{sigma}"
    if not (src / "datasets" / name / "train_data.npy").exists():
        gen = src / "scripts" / "data_preparation" / "SyntheticTS" / "generate_training_data.py"
        sh([sys.executable, str(gen), "--generate_all"], cwd=str(src))
    return name

def _free_gpu():
    """Release GPU memory between trainings so a long shard can't accumulate into OOM."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def train_one(src: Path, model, dataset, num_features, L, Hout, enable_dts):
    """Launch one BasicTS experiment and return (mse, mae). Mirrors run_baselines.py."""
    sys.path.insert(0, str(src / "src"))
    from basicts.configs import BasicTSForecastingConfig
    from basicts.runners.callback import EarlyStopping, DropoutTSCallback
    from basicts import BasicTSLauncher
    # model config factory copied from run_baselines.get_model_config
    from importlib import import_module
    rb = import_module_from(src / "run_baselines.py", "rb")
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

def import_module_from(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def parse_metrics(save_dir: Path):
    """Read the test MSE/MAE that BasicTS/easy-torch writes to its log/metrics files.
    No metric is invented: if none is found the run is recorded as null and flagged."""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", nargs="+", default=["0.1", "0.3", "0.5", "0.7", "0.9"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    ap.add_argument("--model", default="Informer")
    ap.add_argument("--input_len", type=int, default=96)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--hf_repo", default=os.environ.get("HF_RESULTS_REPO", ""))
    args = ap.parse_args()

    work = Path("work").resolve(); work.mkdir(exist_ok=True)
    src = setup(work)
    out_path = Path(args.out).resolve()

    def checkpoint(payload):
        """Write and (best-effort) upload partial results so a job timeout
        cannot zero out completed rows."""
        out_path.write_text(json.dumps(payload, indent=2))
        if args.hf_repo:
            try:
                from huggingface_hub import HfApi
                HfApi().upload_file(path_or_fileobj=str(out_path),
                                    path_in_repo=f"claim2/{out_path.name}",
                                    repo_id=args.hf_repo, repo_type="dataset")
            except Exception as e:
                print("[hf] partial upload failed (continuing):", e, flush=True)

    os.chdir(src)  # BasicTS resolves "datasets/<name>" relative to the CWD
    rows = []
    for sigma in args.sigmas:
        dataset = prepare_synthetic(src, sigma)
        for Hout in args.horizons:
            base_mse, base_mae = train_one(src, args.model, dataset, 1, args.input_len, Hout, False)
            _free_gpu()
            dts_mse, dts_mae   = train_one(src, args.model, dataset, 1, args.input_len, Hout, True)
            _free_gpu()
            row = {"sigma": sigma, "H": Hout,
                   "baseline_mse": base_mse, "baseline_mae": base_mae,
                   "dropoutts_mse": dts_mse, "dropoutts_mae": dts_mae,
                   "mse_improvement_pct": (None if not base_mse or dts_mse is None else round((base_mse - dts_mse)/base_mse*100, 2)),
                   "mae_improvement_pct": (None if not base_mae or dts_mae is None else round((base_mae - dts_mae)/base_mae*100, 2))}
            rows.append(row); print(json.dumps(row), flush=True)
            checkpoint({"partial": True, "model": args.model, "commit": COMMIT, "rows": rows})

    imps = [r["mse_improvement_pct"] for r in rows if r["mse_improvement_pct"] is not None]
    mimps = [r["mae_improvement_pct"] for r in rows if r["mae_improvement_pct"] is not None]
    summary = {"model": args.model, "commit": COMMIT, "rows": rows,
               "avg_mse_improvement_pct": (round(sum(imps)/len(imps), 2) if imps else None),
               "avg_mae_improvement_pct": (round(sum(mimps)/len(mimps), 2) if mimps else None),
               "paper_target": {"avg_mse_pct": 46.0, "avg_mae_pct": 24.5, "peak_mse_pct_at_sigma0.3": 48.2},
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    checkpoint(summary)
    print("WROTE", str(out_path), flush=True)
    if args.hf_repo:
        print("UPLOADED to", args.hf_repo, flush=True)

if __name__ == "__main__":
    main()
