# Copied verbatim (subset used by this reproduction) from the paper's official
# GitHub repository karanbali/attention_head_seed_stability, file
# notebooks/model_configs.py. Non-model, non-checkpoint utility code (a plain
# Python dict of architecture hyperparameters) -- no weights are embedded here.
#
# Source: https://github.com/karanbali/attention_head_seed_stability
#         (commit dd95fe00996f81943a678a136aa85f09d7f0ba7f)

CONFIGS = {
    # 2 layers, 8 heads, GELU MLP (Adam)
    "l2_h8": dict(
        n_layers=2, d_model=512, n_heads=8, d_head=64, d_mlp=2048,
        n_ctx=1024, act_fn="gelu", d_vocab=48262,
        tokenizer_name="NeelNanda/gpt-neox-tokenizer-digits",
        model_name="GELU_2L512W_C4_Code_8H", attn_only=False, lr=1e-4,
    ),
    # 4 layers, 8 heads, GELU MLP (Adam)
    "l4_h8": dict(
        n_layers=4, d_model=512, n_heads=8, d_head=64, d_mlp=2048,
        n_ctx=1024, act_fn="gelu", d_vocab=48262,
        tokenizer_name="NeelNanda/gpt-neox-tokenizer-digits",
        model_name="GELU_4L512W_C4_Code_8H", attn_only=False, lr=1e-4,
    ),
    # 8 layers, 8 heads, GELU MLP (Adam)
    "l8_h8": dict(
        n_layers=8, d_model=512, n_heads=8, d_head=64, d_mlp=2048,
        n_ctx=1024, act_fn="gelu", d_vocab=48262,
        tokenizer_name="NeelNanda/gpt-neox-tokenizer-digits",
        model_name="GELU_8L512W_C4_Code_8H", attn_only=False, lr=1e-4,
    ),
    # 8 layers, 8 heads, GELU MLP (AdamW / weight decay) -- same architecture
    # as l8_h8; only the training optimizer differs (per the official repo,
    # the "_wd" checkpoint folder holds the AdamW-trained refits).
    "l8_h8_wd": dict(
        n_layers=8, d_model=512, n_heads=8, d_head=64, d_mlp=2048,
        n_ctx=1024, act_fn="gelu", d_vocab=48262,
        tokenizer_name="NeelNanda/gpt-neox-tokenizer-digits",
        model_name="GELU_8L512W_C4_Code_8H", attn_only=False, lr=1e-4,
    ),
}
