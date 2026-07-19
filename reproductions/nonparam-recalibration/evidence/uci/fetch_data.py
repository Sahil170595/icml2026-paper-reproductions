#!/usr/bin/env python3
r"""
fetch_data.py -- fetch the paper's EXACT UCI benchmark data (predefined splits).

Paper: "Nonparametric Distribution Regression Re-calibration" (ICML 2026,
OpenReview fTl7NXYtAB, arXiv 2602.13362).
Official experiment repo (pinned):
  https://github.com/adamgnuj/recalibration_experiment.git
  revision 12b4a203d5a259cf13e621fccdd0b9a4ab073fa0 (master HEAD, 2026-07-17)
Its data pipeline (experiment/data/uci_datasets/prepare_uci.ipynb) clones
  https://github.com/yaringal/DropoutUncertaintyExps.git
and uses the Hernandez-Lobato & Adams UCI datasets WITH THE PREDEFINED
train/test split index files (20 splits, 10% test), validation = last 20%
of the train rows (in predefined order, no shuffle).

This script copies exactly those files from the PINNED revision
  DATA_REV = 6eb4497628d12b0f300f4b4f6bdc386bebad565c
for the 5 smallest of the paper's 9 UCI datasets (CPU budget; no multi-GB
downloads -- the whole data repo is ~20 MB of plain text), records SHA-256
of every file into data_checksums.json, and caches under data_cache/.

Source resolution order:
  1. a local `git clone --depth 1` of DropoutUncertaintyExps whose HEAD equals
     DATA_REV, passed as argv[1] (raw.githubusercontent returned HTTP 503 at
     run time, so the recorded run used this path);
  2. raw.githubusercontent.com pinned to DATA_REV.

Run:  python fetch_data.py [path-to-local-DropoutUncertaintyExps-clone]
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data_cache"
DATA_REV = "6eb4497628d12b0f300f4b4f6bdc386bebad565c"  # yaringal/DropoutUncertaintyExps
BASE = ("https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/"
        f"{DATA_REV}/UCI_Datasets")

# 5 of the paper's 9 UCI datasets (Table 1 / Sec. 6), smallest-first for CPU budget
DATASETS = ["yacht", "bostonHousing", "energy", "concrete", "wine-quality-red"]
N_SPLITS_USED = 20  # download all 20 predefined split index files


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch(url: str, dest: Path, local_root: Path | None, rel: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    if local_root is not None:
        src = local_root / "UCI_Datasets" / rel
        shutil.copyfile(src, dest)
        return "copied-from-pinned-clone"
    with urllib.request.urlopen(url, timeout=60) as r:
        dest.write_bytes(r.read())
    return "downloaded"


def main():
    t0 = time.time()
    local_root = None
    if len(sys.argv) > 1:
        local_root = Path(sys.argv[1]).resolve()
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=local_root,
                              capture_output=True, text=True).stdout.strip()
        if head != DATA_REV:
            raise SystemExit(f"local clone HEAD {head} != pinned {DATA_REV}")
        print(f"using local clone {local_root} @ {head}")
    checks = {"data_repo": "yaringal/DropoutUncertaintyExps",
              "data_revision": DATA_REV,
              "official_repo": "adamgnuj/recalibration_experiment",
              "official_revision": "12b4a203d5a259cf13e621fccdd0b9a4ab073fa0",
              "files": {}}
    for ds in DATASETS:
        names = ["data.txt", "index_features.txt", "index_target.txt", "n_splits.txt"]
        names += [f"index_train_{i}.txt" for i in range(N_SPLITS_USED)]
        names += [f"index_test_{i}.txt" for i in range(N_SPLITS_USED)]
        for name in names:
            url = f"{BASE}/{ds}/data/{name}"
            rel = f"{ds}/data/{name}"
            dest = CACHE / ds / name
            status = fetch(url, dest, local_root, rel)
            checks["files"][f"{ds}/{name}"] = {
                "sha256": sha256(dest), "bytes": dest.stat().st_size, "status": status}
    for k in list(checks["files"])[:4]:
        print(k, checks["files"][k]["sha256"][:16], checks["files"][k]["bytes"], "B")
    out = HERE / "data_checksums.json"
    out.write_text(json.dumps(checks, indent=2))
    print(f"wrote {out.name} ({len(checks['files'])} files) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
