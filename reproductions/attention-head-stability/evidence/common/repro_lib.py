"""
Shared, deterministic, CPU-only reproduction utilities for
"Quantifying LLM Attention-Head Stability: Implications for Circuit
Universality" (arXiv 2602.16740, OpenReview UXzfLdXrBJ).

Loads the paper's released checkpoints (forward passes only, no training,
no gradient updates, no GPU) and computes the exact stability / CKA metrics
defined in the official notebooks (karanbali/attention_head_seed_stability):
  - notebooks/Stability_Attention_Head.ipynb   (S 4.1: layer-wise stability, Eq. 5)
  - notebooks/Uniqueness_Attention_Head.ipynb  (S 4.3: within-layer "commonness")
  - notebooks/Stability_Adam_vs_AdamW.ipynb    (S 4.5: Adam vs AdamW)
  - notebooks/Stability_Residual_Stream.ipynb  (S 4.7: residual-stream CKA)

Every numeric formula below (lower-triangular cosine similarity of attention
patterns, best-matching-head aggregation, RBF-kernel CKA) is copied/ported
verbatim from those official notebooks; nothing is re-derived or guessed.
"""
import os
import pickle
import hashlib
import importlib.util
from pathlib import Path

# Force CPU-only *before* torch/transformer_lens are imported anywhere in the
# process. The pilot instructions require CPU-only forward passes; this
# sandbox machine does have a physical CUDA device, so it must be hidden
# explicitly rather than relying on the absence of a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.utils import get_act_name

COMMON_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Model configs (ported from the official repo's notebooks/model_configs.py)
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location("model_configs", COMMON_DIR / "model_configs.py")
_model_configs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model_configs)
CONFIGS = _model_configs.CONFIGS

SEEDS = [1, 2, 3, 4, 5]
ANCHOR_INDEX = 0  # seed 1 is the anchor refit, matching every official notebook


# ---------------------------------------------------------------------------
# Safe prompt loading
# ---------------------------------------------------------------------------
class _RestrictedUnpickler(pickle.Unpickler):
    """Refuses to resolve any class/function reference; the official
    100_prompts.pkl contains only EMPTY_LIST/APPENDS/(SHORT_)BINUNICODE/MARK/
    MEMOIZE/FRAME/PROTO/STOP opcodes (verified with pickletools.genops before
    this loader was written), so a plain list of strings is all this can ever
    produce -- but find_class is still hard-blocked defensively since the
    repo is treated as untrusted input per the pilot contract."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked class lookup: {module}.{name}")


def load_prompts():
    path = COMMON_DIR / "100_prompts.pkl"
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    expected = "ed58bdfc207b0ad740e4d6565337ccdd38b65ffabe9f24d15c911891103a7903"
    if sha != expected:
        raise RuntimeError(f"100_prompts.pkl sha256 mismatch: {sha} != {expected}")
    import io
    prompts = _RestrictedUnpickler(io.BytesIO(data)).load()
    assert isinstance(prompts, list) and all(isinstance(p, str) for p in prompts)
    assert len(prompts) == 100
    return prompts


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def build_model(family: str, seed: int) -> HookedTransformer:
    cfg_dict = CONFIGS[family]
    cfg = HookedTransformerConfig(
        n_layers=cfg_dict["n_layers"],
        d_model=cfg_dict["d_model"],
        n_heads=cfg_dict["n_heads"],
        d_head=cfg_dict["d_head"],
        d_mlp=cfg_dict.get("d_mlp", None),
        n_ctx=cfg_dict["n_ctx"],
        act_fn=cfg_dict.get("act_fn", "gelu"),
        d_vocab=cfg_dict["d_vocab"],
        init_weights=True,
        tokenizer_name=cfg_dict["tokenizer_name"],
        model_name=cfg_dict.get("model_name", family),
        attn_only=cfg_dict.get("attn_only", False),
        seed=seed,
        device="cpu",
    )
    model = HookedTransformer(cfg)
    return model


def load_checkpoint_into(model: HookedTransformer, checkpoint_path: str):
    # weights_only=True + mmap=True mirrors the winner's fail-closed, safe
    # tensor-only load. fold_ln=False mirrors every official notebook's
    # load_and_process_state_dict(..., fold_ln=False) call.
    state_dict = t.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    model.load_and_process_state_dict(state_dict, fold_ln=False)
    model.eval()
    return model


def build_and_load(family: str, seed: int, checkpoint_path: str) -> HookedTransformer:
    model = build_model(family, seed)
    load_checkpoint_into(model, checkpoint_path)
    return model


# ---------------------------------------------------------------------------
# S 4.1 / S 4.5: attention-pattern cosine-similarity stability
#   (ported verbatim from Stability_Attention_Head.ipynb / Stability_Adam_vs_AdamW.ipynb)
# ---------------------------------------------------------------------------
_cos = t.nn.CosineSimilarity(dim=1, eps=1e-08)


def lower_triang(mat: t.Tensor) -> t.Tensor:
    lower_triangular_mat = t.tril(mat)
    mask = t.tril(t.ones_like(mat)).bool()
    return lower_triangular_mat[mask]


def run_prompts_cache(models, prompts, names_filter):
    """cache[i_prompt][i_model] = ActivationCache restricted to names_filter."""
    prompts_cache = []
    for prompt in prompts:
        cache_for_prompt = []
        for model in models:
            _, cache_i = model.run_with_cache(
                prompt, remove_batch_dim=True, names_filter=names_filter
            )
            cache_for_prompt.append(cache_i.to("cpu"))
        prompts_cache.append(cache_for_prompt)
    return prompts_cache


def attention_pattern_names_filter(name: str) -> bool:
    return name.endswith("hook_pattern")


def resid_pre_names_filter(name: str) -> bool:
    return "hook_resid_pre" in name


def resid_mid_names_filter(name: str) -> bool:
    return "hook_resid_mid" in name


def resid_all_names_filter(name: str) -> bool:
    return ("hook_resid_pre" in name) or ("hook_resid_mid" in name)


def compute_head_stability(models, prompts_cache, anchor=ANCHOR_INDEX):
    """
    Reproduces the exact aggregation in Stability_Attention_Head.ipynb cell 11:
    for every (layer, anchor_head), find the best-matching head (max cosine
    similarity of the lower-triangular attention pattern, mean over prompts)
    in every non-anchor refit, then mean over refits.

    Returns a DataFrame-equivalent list of dicts: layer (1-based),
    head_anchor (1-based), cos_sim.
    """
    num_layers = models[0].cfg.n_layers
    num_heads = models[0].cfg.n_heads
    num_models = len(models)
    num_prompts = len(prompts_cache)

    prompts_cs_matrix = t.empty((num_layers, num_heads, num_models, num_heads, num_prompts))

    for ind_prompt in range(num_prompts):
        cache = prompts_cache[ind_prompt]
        cache_anchor = cache[anchor]
        for layer in range(num_layers):
            for head_anchor in range(num_heads):
                head_anchor_attn = cache_anchor[get_act_name("pattern", layer, "a")][head_anchor]
                head_anchor_attn = lower_triang(head_anchor_attn).unsqueeze(0)
                for model_i in range(num_models):
                    cache_model_i = cache[model_i]
                    for head_pair in range(num_heads):
                        head_i_attn = cache_model_i[get_act_name("pattern", layer, "a")][head_pair]
                        head_i_attn = lower_triang(head_i_attn).unsqueeze(0)
                        cos_score_i = _cos(head_anchor_attn, head_i_attn)
                        prompts_cs_matrix[layer, head_anchor, model_i, head_pair, ind_prompt] = cos_score_i

    results = []
    for layer_i in range(num_layers):
        csh = prompts_cs_matrix[layer_i, :, :, :]  # [head_anchor, model_i, head_pair, prompts]
        csh = t.mean(csh, -1)                      # mean over prompts
        csh = csh.max(dim=-1, keepdim=True)         # max over head_pair (best match)
        csh = csh.values[:, :, 0]                   # [head_anchor, model_i]
        csh = t.cat([csh[:, :anchor], csh[:, anchor + 1:]], dim=1)  # drop anchor-vs-anchor
        csh = csh.mean(dim=-1).reshape(num_heads, 1)  # mean over the remaining refits
        for head_anchor in range(num_heads):
            results.append({
                "layer": layer_i + 1,
                "head_anchor": head_anchor + 1,
                "cos_sim": csh[head_anchor].item(),
            })
    return results


def compute_head_commonness(model, prompts_cache_single, num_layers, num_heads):
    """
    Reproduces Uniqueness_Attention_Head.ipynb cells 9-10: within ONE model
    instance (inst_1 = inst_2 = the same anchor refit), mean cosine similarity
    of each head's attention pattern against every head (including itself) in
    the same layer, averaged over prompts. Low commonness => high
    representational distinctness.
    """
    num_prompts = len(prompts_cache_single)
    prompts_cs_matrix = t.empty((num_layers, num_heads, num_heads, num_prompts))
    for ind_prompt in range(num_prompts):
        cache = prompts_cache_single[ind_prompt]
        for layer in range(num_layers):
            for head_anchor in range(num_heads):
                head_anchor_attn = cache[get_act_name("pattern", layer, "a")][head_anchor]
                head_anchor_attn = lower_triang(head_anchor_attn).unsqueeze(0)
                for head_pair in range(num_heads):
                    head_i_attn = cache[get_act_name("pattern", layer, "a")][head_pair]
                    head_i_attn = lower_triang(head_i_attn).unsqueeze(0)
                    cos_score_i = _cos(head_anchor_attn, head_i_attn)
                    prompts_cs_matrix[layer, head_anchor, head_pair, ind_prompt] = cos_score_i

    results = []
    for layer_i in range(num_layers):
        csh = prompts_cs_matrix[layer_i, :, :, :]
        csh = t.mean(csh, -1)      # mean over prompts -> [head_anchor, head_pair]
        csh = csh.mean(dim=1)      # mean over head_pair -> [head_anchor]
        for head_anchor in range(num_heads):
            results.append({
                "layer": layer_i + 1,
                "head_anchor": head_anchor + 1,
                "commonness": csh[head_anchor].item(),
            })
    return results


# ---------------------------------------------------------------------------
# S 4.7: residual-stream CKA
#   (ported verbatim from Stability_Residual_Stream.ipynb / google-research CKA)
# ---------------------------------------------------------------------------
def gram_linear(x):
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    if not np.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric matrix.")
    gram = gram.copy()
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]
    return gram


def cka(gram_x, gram_y, debiased=False):
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def residual_cka_layer(prompts_cache, inst_1, inst_2, layer, kernel="rbf", hook="resid_mid"):
    """
    hook="resid_mid" (default): the residual stream immediately AFTER the
    attention sublayer is added (resid_pre + attn_out), matching the paper's
    Section 4.7 "post-attention residual stream" definition.

    hook="resid_pre": the residual stream BEFORE attention at this layer.
    This is what notebooks/Stability_Residual_Stream.ipynb literally hooks
    (get_act_name('resid_pre', layer)); kept here only as a disclosed
    secondary check, since the official notebook's hook choice does not
    match the paper's own stated definition for this claim (also flagged
    independently by the prior public reproduction of this paper).
    """
    activations1, activations2 = [], []
    for cache in prompts_cache:
        activations1.append(cache[inst_1][get_act_name(hook, layer)].numpy())
        activations2.append(cache[inst_2][get_act_name(hook, layer)].numpy())
    a1 = np.vstack(activations1).astype(np.float64)
    a2 = np.vstack(activations2).astype(np.float64)
    if kernel == "rbf":
        g1, g2 = gram_rbf(a1), gram_rbf(a2)
    else:
        g1, g2 = gram_linear(a1), gram_linear(a2)
    g1 = (g1 + g1.T) / 2.0
    g2 = (g2 + g2.T) / 2.0
    return float(cka(g1, g2))
