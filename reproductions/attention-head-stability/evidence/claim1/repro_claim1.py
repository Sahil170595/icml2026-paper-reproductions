"""
Claim 1: "Middle-layer attention heads are the least stable yet the most
representationally distinct across different training runs."

Protocol (matches the paper's official notebooks exactly):
  - Architecture: l8_h8 (8 layers, d_model=512, 8 heads, GELU MLP), Adam-trained.
  - 5 official refits (seeds 1-5); seed 1 is the anchor refit.
  - All 100 released prompts, in repository order.
  - Stability S_l: cosine similarity of the lower-triangular causal attention
    pattern between the anchor head and its best-matching head (max over
    head_pair) in each of the other 4 refits, mean over prompts, mean over
    refits (Stability_Attention_Head.ipynb, Eq. 5 in the paper).
  - Distinctness: within the anchor refit only, mean cosine similarity of
    each head's attention pattern to every head in the same layer (including
    itself), mean over prompts (Uniqueness_Attention_Head.ipynb). This is
    "commonness"; distinctness = 1 - commonness.

CPU-only, forward passes only, no training. Deterministic given the released
checkpoints and the fixed 100-prompt set.
"""
import json
import sys
import time
from pathlib import Path

COMMON = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON))
import repro_lib as R  # noqa: E402

WEIGHTS_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("weights")
OUT_DIR = Path(__file__).resolve().parent

FAMILY = "l8_h8"


def checkpoint_path(manifest, family, seed):
    return manifest[f"{family}/seed{seed}"]["local_cache_path"]


def main():
    t0 = time.time()
    manifest = json.loads((WEIGHTS_ROOT / "checkpoint_manifest.json").read_text())
    prompts = R.load_prompts()
    print(f"Loaded {len(prompts)} official prompts (sha256-verified).")

    print(f"Building and loading {FAMILY} seeds {R.SEEDS} ...")
    models = []
    for seed in R.SEEDS:
        ckpt = checkpoint_path(manifest, FAMILY, seed)
        m = R.build_and_load(FAMILY, seed, ckpt)
        models.append(m)
        print(f"  seed {seed}: loaded from {ckpt}")

    num_layers = models[0].cfg.n_layers
    num_heads = models[0].cfg.n_heads
    print(f"Architecture: n_layers={num_layers}, n_heads={num_heads}, d_model={models[0].cfg.d_model}")

    print("Running forward passes over 100 prompts x 5 refits (attention patterns only) ...")
    prompts_cache = R.run_prompts_cache(models, prompts, names_filter=R.attention_pattern_names_filter)

    print("Computing cross-refit head stability (Eq. 5) ...")
    stability_rows = R.compute_head_stability(models, prompts_cache, anchor=R.ANCHOR_INDEX)

    print("Computing within-refit head commonness (distinctness = 1 - commonness) ...")
    anchor_cache_only = [cache[R.ANCHOR_INDEX] for cache in prompts_cache]
    commonness_rows = R.compute_head_commonness(models[R.ANCHOR_INDEX], anchor_cache_only, num_layers, num_heads)

    # --- Aggregate to layer level ---
    layer_stability_mean = {}
    layer_stability_std = {}
    for layer in range(1, num_layers + 1):
        vals = [r["cos_sim"] for r in stability_rows if r["layer"] == layer]
        layer_stability_mean[layer] = sum(vals) / len(vals)
        mean = layer_stability_mean[layer]
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        layer_stability_std[layer] = var ** 0.5

    layer_commonness_mean = {}
    for layer in range(1, num_layers + 1):
        vals = [r["commonness"] for r in commonness_rows if r["layer"] == layer]
        layer_commonness_mean[layer] = sum(vals) / len(vals)

    layer_distinctness_mean = {l: 1.0 - v for l, v in layer_commonness_mean.items()}

    min_stability_layer = min(layer_stability_mean, key=layer_stability_mean.get)
    max_stability_layer = max(layer_stability_mean, key=layer_stability_mean.get)
    max_distinctness_layer = max(layer_distinctness_mean, key=layer_distinctness_mean.get)
    min_distinctness_layer = min(layer_distinctness_mean, key=layer_distinctness_mean.get)

    mid_layers = [l for l in range(1, num_layers + 1) if 0.25 <= (l / num_layers) <= 0.875]

    wall_time = time.time() - t0

    print("\n=== Layer-wise stability S_l (mean over 8 anchor heads) ===")
    for layer in range(1, num_layers + 1):
        print(f"  layer {layer}: S_l = {layer_stability_mean[layer]:.6f} "
              f"(std {layer_stability_std[layer]:.6f})  "
              f"distinctness = {layer_distinctness_mean[layer]:.6f}")

    print(f"\nLeast-stable layer: {min_stability_layer} (S_l={layer_stability_mean[min_stability_layer]:.6f})")
    print(f"Most-stable layer:  {max_stability_layer} (S_l={layer_stability_mean[max_stability_layer]:.6f})")
    print(f"Most-distinct layer (lowest commonness): {max_distinctness_layer} "
          f"(distinctness={layer_distinctness_mean[max_distinctness_layer]:.6f})")
    print(f"Least-distinct layer (highest commonness): {min_distinctness_layer} "
          f"(distinctness={layer_distinctness_mean[min_distinctness_layer]:.6f})")

    middle_layer_is_least_stable = min_stability_layer in mid_layers
    middle_layer_is_most_distinct = max_distinctness_layer in mid_layers
    print(f"\nLeast-stable layer in middle band {mid_layers}: {middle_layer_is_least_stable}")
    print(f"Most-distinct layer in middle band {mid_layers}: {middle_layer_is_most_distinct}")
    print(f"Wall time: {wall_time:.3f}s")

    results = {
        "claim": "Middle-layer attention heads are the least stable yet the most "
                 "representationally distinct across different training runs.",
        "arch": FAMILY,
        "n_layers": num_layers,
        "n_heads": num_heads,
        "seeds": R.SEEDS,
        "anchor_seed": R.SEEDS[R.ANCHOR_INDEX],
        "n_prompts": len(prompts),
        "layer_stability_mean": layer_stability_mean,
        "layer_stability_std": layer_stability_std,
        "layer_commonness_mean": layer_commonness_mean,
        "layer_distinctness_mean": layer_distinctness_mean,
        "min_stability_layer": min_stability_layer,
        "max_stability_layer": max_stability_layer,
        "max_distinctness_layer": max_distinctness_layer,
        "min_distinctness_layer": min_distinctness_layer,
        "middle_band_layers": mid_layers,
        "middle_layer_is_least_stable": middle_layer_is_least_stable,
        "middle_layer_is_most_distinct": middle_layer_is_most_distinct,
        "per_head_stability_rows": stability_rows,
        "per_head_commonness_rows": commonness_rows,
        "wall_time_s": wall_time,
        "device": "cpu",
        "cuda_available": False,
    }
    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
