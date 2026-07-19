"""
Claim 3: "Weight decay optimization substantially improves attention-head
stability across random model initializations."

Protocol (matches notebooks/Stability_Adam_vs_AdamW.ipynb exactly):
  - Architecture: l8_h8 (8 layers, d_model=512, 8 heads, GELU MLP), depth 8.
  - Two optimizer conditions: l8_h8 (Adam) vs l8_h8_wd (AdamW), 5 refits each
    (seeds 1-5); seed 1 is the anchor refit within each optimizer condition.
  - All 100 released prompts, in repository order.
  - Same layer-wise stability S_l metric as Claim 1 (Eq. 5), computed
    independently for the Adam refit set and the AdamW refit set.
  - Scored contrast: mean head stability averaged across all layers/heads,
    Adam vs AdamW.

CPU-only, forward passes only, no training.
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

FAMILIES = ["l8_h8", "l8_h8_wd"]  # Adam, AdamW


def checkpoint_path(manifest, family, seed):
    return manifest[f"{family}/seed{seed}"]["local_cache_path"]


def stability_for_family(manifest, prompts, family):
    print(f"\n--- {family} ({'AdamW' if family.endswith('_wd') else 'Adam'}) ---")
    models = []
    for seed in R.SEEDS:
        ckpt = checkpoint_path(manifest, family, seed)
        m = R.build_and_load(family, seed, ckpt)
        models.append(m)
        print(f"  seed {seed}: loaded from {ckpt}")

    num_layers = models[0].cfg.n_layers
    num_heads = models[0].cfg.n_heads

    prompts_cache = R.run_prompts_cache(models, prompts, names_filter=R.attention_pattern_names_filter)
    stability_rows = R.compute_head_stability(models, prompts_cache, anchor=R.ANCHOR_INDEX)

    layer_mean = {}
    for layer in range(1, num_layers + 1):
        vals = [r["cos_sim"] for r in stability_rows if r["layer"] == layer]
        layer_mean[layer] = sum(vals) / len(vals)

    all_vals = [r["cos_sim"] for r in stability_rows]
    overall_mean = sum(all_vals) / len(all_vals)

    del models, prompts_cache
    return num_layers, num_heads, layer_mean, overall_mean, stability_rows


def main():
    t0 = time.time()
    manifest = json.loads((WEIGHTS_ROOT / "checkpoint_manifest.json").read_text())
    prompts = R.load_prompts()
    print(f"Loaded {len(prompts)} official prompts (sha256-verified).")

    per_optimizer = {}
    for family in FAMILIES:
        n_layers, n_heads, layer_mean, overall_mean, rows = stability_for_family(manifest, prompts, family)
        per_optimizer[family] = {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "layer_stability_mean": layer_mean,
            "overall_mean_stability": overall_mean,
        }
        print(f"  Layer-wise S_l: " + ", ".join(f"{l}:{v:.6f}" for l, v in layer_mean.items()))
        print(f"  Overall mean stability ({family}): {overall_mean:.6f}")

    adam_mean = per_optimizer["l8_h8"]["overall_mean_stability"]
    adamw_mean = per_optimizer["l8_h8_wd"]["overall_mean_stability"]
    delta = adamw_mean - adam_mean
    adamw_higher = adamw_mean > adam_mean

    per_layer_delta = {
        l: per_optimizer["l8_h8_wd"]["layer_stability_mean"][l] - per_optimizer["l8_h8"]["layer_stability_mean"][l]
        for l in per_optimizer["l8_h8"]["layer_stability_mean"]
    }
    layers_where_adamw_higher = sum(1 for v in per_layer_delta.values() if v > 0)

    wall_time = time.time() - t0

    print("\n=== Adam vs AdamW summary ===")
    print(f"Adam  (l8_h8)    overall mean head stability = {adam_mean:.6f}")
    print(f"AdamW (l8_h8_wd) overall mean head stability = {adamw_mean:.6f}")
    print(f"Delta (AdamW - Adam) = {delta:.6f}")
    print(f"AdamW higher than Adam: {adamw_higher}")
    print(f"Layers where AdamW > Adam: {layers_where_adamw_higher} / {len(per_layer_delta)}")
    print(f"Per-layer delta: " + ", ".join(f"{l}:{v:+.6f}" for l, v in per_layer_delta.items()))
    print(f"Wall time: {wall_time:.3f}s")

    results = {
        "claim": "Weight decay optimization substantially improves attention-head "
                 "stability across random model initializations.",
        "families": FAMILIES,
        "seeds": R.SEEDS,
        "anchor_seed": R.SEEDS[R.ANCHOR_INDEX],
        "n_prompts": len(prompts),
        "per_optimizer": per_optimizer,
        "adam_overall_mean": adam_mean,
        "adamw_overall_mean": adamw_mean,
        "delta_adamw_minus_adam": delta,
        "adamw_higher_than_adam": adamw_higher,
        "per_layer_delta": per_layer_delta,
        "layers_where_adamw_higher": layers_where_adamw_higher,
        "n_layers": len(per_layer_delta),
        "wall_time_s": wall_time,
        "device": "cpu",
        "cuda_available": False,
    }
    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
