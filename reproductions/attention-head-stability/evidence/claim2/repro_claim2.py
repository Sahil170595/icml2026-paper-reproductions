"""
Claim 2: "Deeper models exhibit stronger mid-depth divergence in attention
head stability compared to shallower models."

Protocol (matches the paper's Eq. 6-7 and the official notebooks):
  - Architectures: l2_h8, l4_h8, l8_h8 (2/4/8 layers, d_model=512, 8 heads,
    GELU MLP), all Adam-trained, all with d_head=64.
  - 5 official refits per depth (seeds 1-5); seed 1 is the anchor refit.
  - All 100 released prompts, in repository order.
  - Per depth d, compute layer-wise stability S_l exactly as in Claim 1
    (Stability_Attention_Head.ipynb), then:
      Delta S_d = max_l(S_l) - min_l(S_l)                       (Eq. 6)
      r_min_d   = argmin_l(S_l) / n_layers_d                    (Eq. 7)
  - Scored contrast: C_gap = Delta S_8 - max(Delta S_2, Delta S_4).
    The claim is supported if Gap2 < Gap4 < Gap8 (monotone in depth) and
    C_gap > 0.

Note on release boundary: the paper's primary author-named HF repo
(karanbali/attention_head_seed_stability) hosts ONLY l8_h8 and l8_h8_wd.
l2_h8 and l4_h8 come from a companion mirror
(Icml26AttnHeadStab/icml26AttnHeadStab-anon-weights) whose l8_h8 files are
byte-identical (same LFS sha256) to the primary repo's l8_h8 files -- see
../../checkpoint_provenance.json. This is recorded, not hidden.

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

FAMILIES = ["l2_h8", "l4_h8", "l8_h8"]


def checkpoint_path(manifest, family, seed):
    return manifest[f"{family}/seed{seed}"]["local_cache_path"]


def layer_stability_for_family(manifest, prompts, family):
    print(f"\n--- {family} ---")
    models = []
    for seed in R.SEEDS:
        ckpt = checkpoint_path(manifest, family, seed)
        m = R.build_and_load(family, seed, ckpt)
        models.append(m)
        print(f"  seed {seed}: loaded from {ckpt}")

    num_layers = models[0].cfg.n_layers
    print(f"  n_layers={num_layers}, n_heads={models[0].cfg.n_heads}")

    prompts_cache = R.run_prompts_cache(models, prompts, names_filter=R.attention_pattern_names_filter)
    stability_rows = R.compute_head_stability(models, prompts_cache, anchor=R.ANCHOR_INDEX)

    layer_mean = {}
    for layer in range(1, num_layers + 1):
        vals = [r["cos_sim"] for r in stability_rows if r["layer"] == layer]
        layer_mean[layer] = sum(vals) / len(vals)

    del models, prompts_cache
    return num_layers, layer_mean


def main():
    t0 = time.time()
    manifest = json.loads((WEIGHTS_ROOT / "checkpoint_manifest.json").read_text())
    prompts = R.load_prompts()
    print(f"Loaded {len(prompts)} official prompts (sha256-verified).")

    per_depth = {}
    for family in FAMILIES:
        n_layers, layer_mean = layer_stability_for_family(manifest, prompts, family)
        s_max = max(layer_mean.values())
        s_min = min(layer_mean.values())
        gap = s_max - s_min
        l_min = min(layer_mean, key=layer_mean.get)
        l_max = max(layer_mean, key=layer_mean.get)
        r_min = l_min / n_layers
        per_depth[family] = {
            "n_layers": n_layers,
            "layer_stability_mean": layer_mean,
            "S_max": s_max,
            "S_min": s_min,
            "gap": gap,
            "l_min": l_min,
            "l_max": l_max,
            "r_min": r_min,
        }
        print(f"  Layer-wise S_l: " + ", ".join(f"{l}:{v:.6f}" for l, v in layer_mean.items()))
        print(f"  Gap (S_max - S_min) = {gap:.6f}  (min at layer {l_min}, r_min={r_min:.4f}; "
              f"max at layer {l_max})")

    gap2 = per_depth["l2_h8"]["gap"]
    gap4 = per_depth["l4_h8"]["gap"]
    gap8 = per_depth["l8_h8"]["gap"]
    c_gap = gap8 - max(gap2, gap4)
    monotone = gap2 < gap4 < gap8
    r_min_8 = per_depth["l8_h8"]["r_min"]
    depth8_min_in_mid_band = 0.4 <= r_min_8 <= 0.8

    wall_time = time.time() - t0

    print("\n=== Cross-depth summary (Eq. 6-7) ===")
    print(f"Gap2 = {gap2:.6f}")
    print(f"Gap4 = {gap4:.6f}")
    print(f"Gap8 = {gap8:.6f}")
    print(f"C_gap = Gap8 - max(Gap2, Gap4) = {c_gap:.6f}")
    print(f"Monotone Gap2 < Gap4 < Gap8: {monotone}")
    print(f"Depth-8 minimum layer: {per_depth['l8_h8']['l_min']} / 8  (r_min={r_min_8:.4f}); "
          f"in [0.4, 0.8]: {depth8_min_in_mid_band}")
    print(f"Wall time: {wall_time:.3f}s")

    results = {
        "claim": "Deeper models exhibit stronger mid-depth divergence in attention "
                 "head stability compared to shallower models.",
        "families": FAMILIES,
        "seeds": R.SEEDS,
        "anchor_seed": R.SEEDS[R.ANCHOR_INDEX],
        "n_prompts": len(prompts),
        "per_depth": per_depth,
        "gap2": gap2,
        "gap4": gap4,
        "gap8": gap8,
        "c_gap": c_gap,
        "monotone_gap2_lt_gap4_lt_gap8": monotone,
        "depth8_min_layer": per_depth["l8_h8"]["l_min"],
        "depth8_r_min": r_min_8,
        "depth8_min_in_mid_band_0.4_0.8": depth8_min_in_mid_band,
        "wall_time_s": wall_time,
        "device": "cpu",
        "cuda_available": False,
    }
    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
