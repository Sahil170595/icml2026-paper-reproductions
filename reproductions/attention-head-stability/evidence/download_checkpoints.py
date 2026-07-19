"""
Download the paper's released checkpoints (CPU forward-pass reproduction, no
training, no HF writes). Read-only hf_hub_download of the paper's own
released weights, then a fail-closed SHA-256/size check against what the HF
API itself reports for each file before anything is used.

Official author repo: karanbali/attention_head_seed_stability (l8_h8, l8_h8_wd only)
Companion full-depth mirror: Icml26AttnHeadStab/icml26AttnHeadStab-anon-weights
  (byte-identical l8_h8 content verified via LFS sha256; also hosts l2_h8, l4_h8
  which are absent from the primary author repo per its own README).

Usage: python download_checkpoints.py <weights_dir>
Writes <weights_dir>/checkpoint_manifest.json, consumed by the claim{1..4}
scripts as their checkpoint path index.
"""
import hashlib
import json
import os
import sys
from huggingface_hub import HfApi, hf_hub_download

SEEDS = [1, 2, 3, 4, 5]

# (family, repo_id, revision)
JOBS = [
    ("l8_h8", "karanbali/attention_head_seed_stability", "a1148a78942455dea94e971632a8b48159c64e86"),
    ("l8_h8_wd", "karanbali/attention_head_seed_stability", "a1148a78942455dea94e971632a8b48159c64e86"),
    ("l2_h8", "Icml26AttnHeadStab/icml26AttnHeadStab-anon-weights", "7024304d20f3a2e0a303894467c14373594448d6"),
    ("l4_h8", "Icml26AttnHeadStab/icml26AttnHeadStab-anon-weights", "7024304d20f3a2e0a303894467c14373594448d6"),
]

OUT_ROOT = sys.argv[1] if len(sys.argv) > 1 else "weights"
os.makedirs(OUT_ROOT, exist_ok=True)

api = HfApi()
manifest = {}


def path_for(family, seed):
    # both families use "l{N}_h8" folder naming, and l8_h8_wd checkpoints are
    # physically stored under path prefix "l8_h8_wd" but filenames still say l8_h8
    n_layers = family.split("_")[0][1:]
    return f"{family}/causal_attn_l{n_layers}_h8_seed{seed}_epoch1_c4_gelu/final.pt"


for family, repo_id, revision in JOBS:
    for seed in SEEDS:
        rel_path = path_for(family, seed)
        print(f"Fetching {family} seed {seed} from {repo_id}@{revision[:8]} -> {rel_path}")
        info = api.get_paths_info(repo_id=repo_id, paths=[rel_path], repo_type="model", revision=revision)
        assert len(info) == 1, f"path not found: {rel_path} in {repo_id}"
        expected_sha256 = info[0].lfs.sha256
        expected_size = info[0].size

        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            filename=rel_path,
        )
        actual_size = os.path.getsize(local_path)
        assert actual_size == expected_size, f"size mismatch {rel_path}: {actual_size} != {expected_size}"
        h = hashlib.sha256()
        with open(local_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual_sha256 = h.hexdigest()
        assert actual_sha256 == expected_sha256, f"sha256 mismatch {rel_path}: {actual_sha256} != {expected_sha256}"

        key = f"{family}/seed{seed}"
        manifest[key] = {
            "family": family,
            "seed": seed,
            "repo_id": repo_id,
            "revision": revision,
            "rel_path": rel_path,
            "local_cache_path": local_path,
            "size_bytes": actual_size,
            "sha256": actual_sha256,
        }
        print(f"  OK size={actual_size} sha256={actual_sha256}")

with open(os.path.join(OUT_ROOT, "checkpoint_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("\nAll checkpoints verified. Manifest written to", os.path.join(OUT_ROOT, "checkpoint_manifest.json"))
