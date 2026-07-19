#!/usr/bin/env python3
"""GPU job: reproduce Table 1 of "Model Fusion via Neuron Interpolation"
(OpenReview SXOqLX0T6X, arXiv 2507.00037) - VGG11 / CIFAR-10, NON-IID, zero-shot.

It drives the OFFICIAL example examples/fuse_vggs_cifar10_noniid.py (pinned commit
e84a687a7049f521a240e75e3a050bd13036de3b), which fuses two Dirichlet-skewed VGG11s with
Git Re-Basin, OTFusion, HF-Linear, KF-Linear, HF/KF-Gradient using the paper's exact
configs and released base checkpoints, then compares the measured accuracies to Table 1:

    K-means Linear Fusion (Ours):  2-way 84.5 | 4-way 78.9 | 8-way 69.6
    Vanilla Averaging (previous):  2-way 11.2 | 4-way 10.0 | 8-way 10.0
    OTFusion (previous):           2-way 43.4 | 4-way 20.4 | 8-way 12.9
    Git Re-Basin (previous):       2-way 71.8

Results (measured vs target, per algorithm and split) are written to results.json and,
if OUTPUT_REPO + HF_TOKEN are set, pushed to a Hugging Face dataset. NOTHING is fabricated;
this script only runs the official code and records what it prints.

Env vars:
  OUTPUT_REPO  HF dataset id to upload results to     (default: unset -> local only)
  SPLITS       comma list of n-way splits to attempt  (default: "2")  e.g. "2,4,8"
  MODELS_SEED  seed subfolder for base checkpoints     (default: "seed2" for 2-way)
"""
import os, sys, json, time, subprocess, traceback
from pathlib import Path

COMMIT   = "e84a687a7049f521a240e75e3a050bd13036de3b"
REPO_URL = "https://github.com/AndrewSpano/model-fusion-via-retrofitting.git"
HF_MODELS = "AndrewSpano/model-fusion-via-retrofitting-example-models"
WORK = Path(os.environ.get("WORK_DIR", "/tmp/model-fusion")); WORK.mkdir(parents=True, exist_ok=True)
REPO = WORK / "repo"
SPLITS = [s.strip() for s in os.environ.get("SPLITS", "2").split(",") if s.strip()]
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "").strip()

TABLE1 = {  # paper targets (%). Git Re-Basin / HF-Linear are pairwise (2-way) only.
    "2": {"kf-linear": 84.5, "hf-linear": 84.6, "git-rebasin": 71.8, "otf-acts": 43.4, "vanilla": 11.2},
    "4": {"kf-linear": 78.9, "otf-acts": 20.4, "vanilla": 10.0},
    "8": {"kf-linear": 69.6, "otf-acts": 12.9, "vanilla": 10.0},
}

def sh(cmd): print("+", " ".join(cmd), flush=True); subprocess.check_call(cmd)

def bootstrap():
    if not (REPO / ".git").exists():
        sh(["git", "clone", REPO_URL, str(REPO)])
        sh(["git", "-C", str(REPO), "checkout", COMMIT])
    sys.path.insert(0, str(REPO))
    # The repo is flat-layout with multiple top-level packages and is NOT
    # pip-installable (setuptools auto-discovery refuses); it is imported via
    # sys.path. Its deps come from the kit requirements.txt, installed by the
    # job command before this script runs.
    sh([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_MODELS, repo_type="model", local_dir=str(REPO / "example-base-models"))

def vanilla_average(base_models, ex):
    import torch, copy
    f = copy.deepcopy(base_models[0]); sd = f.state_dict()
    for k in sd:
        sd[k] = sum(m.state_dict()[k].float() for m in base_models) / len(base_models)
    f.load_state_dict(sd); return f

def run_split(nway, ex):
    import torch
    root = REPO / "example-base-models" / "non-iid" / "VGGs" / "CIFAR-10" / f"split-by-{nway}"
    seed_dirs = sorted(root.glob("seed*")) if root.exists() else []
    if not seed_dirs:
        return {"error": f"no base-model dir for split-by-{nway} under {root}"}
    models_dir = seed_dirs[0]; ex.MODELS_DIR = models_dir
    model_paths = sorted(models_dir.glob("model_*.pt"))
    base = [ex.load_model(p, ex.DEVICE, fold_model=False) for p in model_paths]
    X, y, test_dl, fusion_dl, train_ds, val_ds = ex.load_data(model_paths)
    ni_layer = ex.ni_scores(ex.layer_models(base), model_paths, "layer", ex.NI_METHOD, train_ds, val_ds)
    ni_level = ex.ni_scores(ex.level_models(base), model_paths, "level", ex.NI_METHOD, train_ds, val_ds)
    res = {"models_dir": str(models_dir), "num_base_models": len(base)}
    for i, m in enumerate(base):
        res[f"model_{i}"] = float(ex.eval_model(m, test_dl, ex.DEVICE, num_classes=ex.NUM_CLASSES)["accuracy"])
    res["vanilla"] = float(ex.eval_model(vanilla_average(base, ex), test_dl, ex.DEVICE, num_classes=ex.NUM_CLASSES)["accuracy"])
    algos = {"otf-acts":   lambda: ex.fuse_otf(base, X, ni_layer),
             "kf-linear":  lambda: ex.fuse_kf_linear(base, X, ni_layer),
             "kf-gradient":lambda: ex.fuse_kf_gradient(base, X, y, ni_level)}
    if len(base) == 2:  # Git Re-Basin and HF-Linear are pairwise (2-model) only
        algos["git-rebasin"] = lambda: ex.fuse_gr(base, X, ni_layer)
        algos["hf-linear"]   = lambda: ex.fuse_hf_linear(base, X, ni_layer)
    for name, fn in algos.items():
        try:
            fused = fn()
            res[name] = float(ex.eval_model(fused, test_dl, ex.DEVICE, num_classes=ex.NUM_CLASSES)["accuracy"])
            print(f"[split-{nway}] {name}: {res[name]:.4f}", flush=True)
        except Exception as e:
            res[name] = f"ERROR: {e}"; print(f"[split-{nway}] {name} FAILED: {e}", flush=True)
    return res

def main():
    t0 = time.time()
    bootstrap()
    os.chdir(REPO)
    import importlib.util, torch
    spec = importlib.util.spec_from_file_location("fuse_noniid", REPO / "examples" / "fuse_vggs_cifar10_noniid.py")
    ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
    out = {"paper": "SXOqLX0T6X / arXiv 2507.00037", "experiment": "Table 1: VGG11/CIFAR-10 non-IID zero-shot",
           "source_commit": COMMIT, "device": str(ex.DEVICE), "cuda": torch.cuda.is_available(),
           "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
           "ni_method": ex.NI_METHOD, "num_fusion_samples": ex.NUM_FUSION_SAMPLES, "splits": {}}
    for nway in SPLITS:
        print(f"\n===== split-by-{nway} =====", flush=True)
        try:
            measured = run_split(nway, ex)
        except Exception as e:
            measured = {"error": f"{e}", "traceback": traceback.format_exc()}
        targets = TABLE1.get(nway, {})
        comparison = {}
        for algo, tgt in targets.items():
            m = measured.get(algo)
            if isinstance(m, (int, float)):
                comparison[algo] = {"measured_pct": round(100 * m, 2), "paper_pct": tgt,
                                    "abs_diff_pts": round(100 * m - tgt, 2)}
        out["splits"][nway] = {"measured_accuracy": measured, "vs_table1": comparison}
    out["runtime_s"] = round(time.time() - t0, 1)
    Path("results.json").write_text(json.dumps(out, indent=2))
    print("\nWROTE results.json\n" + json.dumps(out, indent=2), flush=True)
    if OUTPUT_REPO and os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(OUTPUT_REPO, repo_type="dataset", exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        api.upload_file(path_or_fileobj="results.json", path_in_repo=f"table1_vgg_cifar10_noniid/{stamp}.json",
                        repo_id=OUTPUT_REPO, repo_type="dataset")
        print(f"UPLOADED results to hf.co/datasets/{OUTPUT_REPO}", flush=True)

if __name__ == "__main__":
    main()
