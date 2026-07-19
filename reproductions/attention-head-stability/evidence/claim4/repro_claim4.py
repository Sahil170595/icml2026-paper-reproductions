"""
Claim 4: "The residual stream is comparatively stable across refits."

Protocol:
  - Architecture: l8_h8 (Adam) and l8_h8_wd (AdamW), 8 layers, d_model=512,
    5 refits each (seeds 1-5).
  - All 100 released prompts, in repository order.
  - POST-attention residual-stream activations (hook_resid_mid: resid_pre +
    attn_out, i.e. immediately after the attention sublayer is added and
    before the MLP) at every layer, for every refit pair (inst_1, inst_2)
    including self-pairs (no exclusion of the diagonal; both the all-pairs
    and off-diagonal-only means are reported).
  - RBF-kernel CKA (primary metric; the paper's own implementation, adapted
    from Kornblith et al. / google-research/representation_similarity) of
    the stacked [n_prompts * seq_len, d_model] activation matrix between
    inst_1 and inst_2, per layer.
  - Compare the resulting mean residual-stream CKA against the mean
    attention-head stability from Claims 1/3 (same checkpoints, same
    prompts) to test "comparatively stable".

Disclosed hook-choice correction: the official repo's
notebooks/Stability_Residual_Stream.ipynb literally hooks hook_resid_pre
(PRE-attention), but the paper's Section 4.7 defines this claim on the
POST-attention residual stream, and the sibling notebook
Stability_Adam_vs_AdamW.ipynb (same repo) independently picks
hook_resid_mid ("resid after attn") as the correct "residual after
attention" hook. This reproduction follows the paper's stated definition
(resid_mid), not the mismatched notebook hook. The pre-attention variant
was run first, found to disagree with the paper's own definition, and is
kept for disclosure as results_resid_pre_secondary.json /
run_stdout_resid_pre_secondary.log rather than silently discarded.

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


def residual_cka_for_family(manifest, prompts, family):
    print(f"\n--- {family} ({'AdamW' if family.endswith('_wd') else 'Adam'}) ---")
    models = []
    for seed in R.SEEDS:
        ckpt = checkpoint_path(manifest, family, seed)
        m = R.build_and_load(family, seed, ckpt)
        models.append(m)
        print(f"  seed {seed}: loaded from {ckpt}")

    num_layers = models[0].cfg.n_layers
    num_models = len(models)

    prompts_cache = R.run_prompts_cache(models, prompts, names_filter=R.resid_mid_names_filter)

    rows = []
    for inst_1 in range(num_models):
        for inst_2 in range(num_models):
            for layer in range(num_layers):
                cka_sim = R.residual_cka_layer(prompts_cache, inst_1, inst_2, layer, kernel="rbf", hook="resid_mid")
                rows.append({
                    "inst_1": inst_1 + 1, "inst_2": inst_2 + 1,
                    "layer": layer + 1, "cka_rbf": cka_sim,
                })
        print(f"  finished refit {inst_1 + 1}/{num_models} as inst_1")

    layer_mean = {}
    for layer in range(1, num_layers + 1):
        vals = [r["cka_rbf"] for r in rows if r["layer"] == layer]
        layer_mean[layer] = sum(vals) / len(vals)

    off_diag_vals = [r["cka_rbf"] for r in rows if r["inst_1"] != r["inst_2"]]
    overall_mean_all_pairs = sum(r["cka_rbf"] for r in rows) / len(rows)
    overall_mean_off_diag = sum(off_diag_vals) / len(off_diag_vals)

    # Anchor-refit convention (seed 1 = inst_1 index 0, vs the other 4 refits,
    # x n_layers): matches the same anchor convention used throughout Claims
    # 1-3, rather than the official notebook's literal all-pairs grid.
    anchor_vals = [r["cka_rbf"] for r in rows if r["inst_1"] == 1 and r["inst_2"] != 1]
    overall_mean_anchor = sum(anchor_vals) / len(anchor_vals)

    del models, prompts_cache
    return {
        "n_layers": num_layers,
        "layer_mean_cka_rbf": layer_mean,
        "overall_mean_cka_rbf_all_pairs": overall_mean_all_pairs,
        "overall_mean_cka_rbf_off_diagonal": overall_mean_off_diag,
        "overall_mean_cka_rbf_anchor_vs_others": overall_mean_anchor,
        "n_anchor_comparisons": len(anchor_vals),
        "rows": rows,
    }


def main():
    t0 = time.time()
    manifest = json.loads((WEIGHTS_ROOT / "checkpoint_manifest.json").read_text())
    prompts = R.load_prompts()
    print(f"Loaded {len(prompts)} official prompts (sha256-verified).")

    per_optimizer = {}
    for family in FAMILIES:
        per_optimizer[family] = residual_cka_for_family(manifest, prompts, family)
        lm = per_optimizer[family]["layer_mean_cka_rbf"]
        print(f"  Layer-wise residual CKA (RBF): " + ", ".join(f"{l}:{v:.6f}" for l, v in lm.items()))
        print(f"  Overall mean (all pairs incl. self): "
              f"{per_optimizer[family]['overall_mean_cka_rbf_all_pairs']:.6f}")
        print(f"  Overall mean (off-diagonal pairs only): "
              f"{per_optimizer[family]['overall_mean_cka_rbf_off_diagonal']:.6f}")
        print(f"  Overall mean (anchor seed1 vs seeds2-5, {per_optimizer[family]['n_anchor_comparisons']} comparisons): "
              f"{per_optimizer[family]['overall_mean_cka_rbf_anchor_vs_others']:.6f}")

    adam_resid = per_optimizer["l8_h8"]["overall_mean_cka_rbf_all_pairs"]
    adamw_resid = per_optimizer["l8_h8_wd"]["overall_mean_cka_rbf_all_pairs"]
    adam_resid_offdiag = per_optimizer["l8_h8"]["overall_mean_cka_rbf_off_diagonal"]
    adamw_resid_offdiag = per_optimizer["l8_h8_wd"]["overall_mean_cka_rbf_off_diagonal"]
    adam_resid_anchor = per_optimizer["l8_h8"]["overall_mean_cka_rbf_anchor_vs_others"]
    adamw_resid_anchor = per_optimizer["l8_h8_wd"]["overall_mean_cka_rbf_anchor_vs_others"]

    # Cross-reference against Claim 1 / Claim 3 head-level stability if those
    # results.json files already exist (same checkpoints, same prompts).
    claim1_path = OUT_DIR.parents[0] / "claim1" / "results.json"
    claim3_path = OUT_DIR.parents[0] / "claim3" / "results.json"
    head_stability_ref = {}
    if claim3_path.exists():
        c3 = json.loads(claim3_path.read_text())
        head_stability_ref["adam_head_mean_stability"] = c3["adam_overall_mean"]
        head_stability_ref["adamw_head_mean_stability"] = c3["adamw_overall_mean"]

    wall_time = time.time() - t0

    print("\n=== Residual-stream vs. head-level stability summary ===")
    print(f"Adam  residual CKA (RBF, all pairs)   = {adam_resid:.6f}")
    print(f"AdamW residual CKA (RBF, all pairs)   = {adamw_resid:.6f}")
    print(f"Adam  residual CKA (RBF, off-diagonal) = {adam_resid_offdiag:.6f}")
    print(f"AdamW residual CKA (RBF, off-diagonal) = {adamw_resid_offdiag:.6f}")
    print(f"Adam  residual CKA (RBF, anchor vs others) = {adam_resid_anchor:.6f}")
    print(f"AdamW residual CKA (RBF, anchor vs others) = {adamw_resid_anchor:.6f}")
    if head_stability_ref:
        print(f"[cross-ref] Adam head-level mean stability  = {head_stability_ref['adam_head_mean_stability']:.6f}")
        print(f"[cross-ref] AdamW head-level mean stability  = {head_stability_ref['adamw_head_mean_stability']:.6f}")
        print(f"Residual > head stability (Adam):  "
              f"{adam_resid_anchor > head_stability_ref['adam_head_mean_stability']}")
        print(f"Residual > head stability (AdamW): "
              f"{adamw_resid_anchor > head_stability_ref['adamw_head_mean_stability']}")
    print(f"Wall time: {wall_time:.3f}s")

    results = {
        "claim": "The residual stream is comparatively stable across refits.",
        "hook": "resid_mid (post-attention, per paper Section 4.7; see docstring for the disclosed hook-choice correction vs. the official notebook's resid_pre)",
        "families": FAMILIES,
        "seeds": R.SEEDS,
        "n_prompts": len(prompts),
        "per_optimizer": per_optimizer,
        "adam_residual_cka_all_pairs": adam_resid,
        "adamw_residual_cka_all_pairs": adamw_resid,
        "adam_residual_cka_off_diagonal": adam_resid_offdiag,
        "adamw_residual_cka_off_diagonal": adamw_resid_offdiag,
        "adam_residual_cka_anchor_vs_others": adam_resid_anchor,
        "adamw_residual_cka_anchor_vs_others": adamw_resid_anchor,
        "head_stability_cross_reference": head_stability_ref,
        "wall_time_s": wall_time,
        "device": "cpu",
        "cuda_available": False,
    }
    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()


