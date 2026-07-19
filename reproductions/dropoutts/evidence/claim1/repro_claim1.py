"""CPU proxy for Claim 1 (scored): "DropoutTS achieves advanced robustness with
NEGLIGIBLE PARAMETER OVERHEAD and NO ARCHITECTURAL MODIFICATIONS."
Paper anchor (Key Findings, Negligible Overhead): "DropoutTS adds only 4 extra
parameters and zero inference latency overhead, while providing a 1.12x-1.45x
training speedup relative to baseline models."

This uses the OFFICIAL, unmodified DropoutTS / SampleAdaptiveDropout modules
(vendored dropout_ts.py, sha256 recorded in evidence-and-rerun page) from pinned
commit 64a096ec6801d9506ab3a30541b6f1b6dbbd7f40. Deterministic, CPU, 1 thread.

Measures three sub-claims:
  (A) parameter overhead  -> target: exactly 4 extra trainable scalars
  (B) eval-mode identity  -> target: zero inference transform (bit-exact passthrough)
  (C) per-step train cost -> honest bounded timing (the headline 1.12-1.45x
      wall-clock speedup is a convergence/early-stopping effect at 100-epoch
      scale -> GPU job kit).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, importlib.util
from pathlib import Path
import numpy as np, torch, torch.nn as nn

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
MOD = HERE.parent / "dropout_ts.py"            # vendored official module
spec = importlib.util.spec_from_file_location("dropout_ts", MOD)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
DropoutTS, SampleAdaptiveDropout = m.DropoutTS, m.SampleAdaptiveDropout

torch.manual_seed(0); np.random.seed(0)
L, H = 96, 24     # paper input length L=96

class MLP(nn.Module):
    """Plain forecasting backbone. Identical arch whether or not DropoutTS is used;
    DropoutTS only swaps the dropout call site (drop-in for nn.Dropout)."""
    def __init__(self, adaptive):
        super().__init__()
        self.fc1 = nn.Linear(L, 128); self.act = nn.ReLU(); self.fc2 = nn.Linear(128, H)
        self.adaptive = adaptive
        self.drop = SampleAdaptiveDropout(p=0.3) if adaptive else nn.Dropout(0.3)
        self.dts = DropoutTS(seq_len=L, p_min=0.1, p_max=0.6) if adaptive else None
    def forward(self, x):
        h = self.act(self.fc1(x.squeeze(-1)))
        if self.adaptive:
            rates = self.dts.compute_dropout_rates(x)
            h = self.drop(h, rates)
        else:
            h = self.drop(h)
        return self.fc2(h).unsqueeze(-1)

def backbone_params(mod):
    return sum(p.numel() for n, p in mod.named_parameters() if not n.startswith(("drop", "dts")))
def extra_params(mod):
    return [(n, tuple(p.shape), p.numel()) for n, p in mod.named_parameters() if n.startswith(("drop", "dts"))]

# ---- (A) parameter overhead + no architectural modification --------------------
base = MLP(adaptive=False)
dts  = MLP(adaptive=True)
# trigger DropoutTS lazy init (noise_scorer built on first call) so all params exist
_ = dts.dts.compute_dropout_rates(torch.randn(4, L, 1))
nb_base = backbone_params(base)
nb_dts  = backbone_params(dts)
extra   = extra_params(dts)
n_extra = sum(e[2] for e in extra)
overhead_pct = 100.0 * n_extra / nb_base

# ---- (B) eval-mode identity = zero inference transform -------------------------
sad = SampleAdaptiveDropout(p=0.3)
x = torch.randn(8, 64)
sad.eval()
with torch.no_grad():
    y_eval = sad(x, torch.full((8,), 0.5))
eval_bit_identical = bool(torch.equal(y_eval, x))          # target: True
sad.train()
y_train = sad(x, torch.full((8,), 0.5))
train_transforms = bool(not torch.equal(y_train, x))       # sanity: dropout active in train

# ---- (C) per-step training cost (honest bounded timing) -----------------------
def time_steps(adaptive, n_steps=150, bs=64, seed=0):
    torm = MLP(adaptive); torch.manual_seed(seed)
    opt = torch.optim.SGD(torm.parameters(), 0.01); lf = nn.MSELoss()
    xb = torch.randn(bs, L, 1); yb = torch.randn(bs, H, 1)
    torm.train()
    for _ in range(5):  # warmup
        opt.zero_grad(); lf(torm(xb), yb).backward(); opt.step()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt.zero_grad(); lf(torm(xb), yb).backward(); opt.step()
    return (time.perf_counter() - t0) / n_steps * 1000.0   # ms/step

ms_base = time_steps(False); ms_dts = time_steps(True)
per_step_ratio = ms_dts / ms_base   # >1 means DropoutTS adds per-step compute (FFT scoring)

# ---- inference latency (eval forward), backbone identical ----------------------
def infer_ms(mod, n=100, bs=64):
    mod.eval(); xb = torch.randn(bs, L, 1)
    with torch.no_grad():
        for _ in range(10): mod(xb)
        t0 = time.perf_counter()
        for _ in range(n): mod(xb)
    return (time.perf_counter() - t0) / n * 1000.0
inf_base = infer_ms(base); inf_dts = infer_ms(dts)

res = {
    "backbone_params_baseline": nb_base,
    "backbone_params_with_dropoutts": nb_dts,
    "backbone_identical": nb_base == nb_dts,
    "dropoutts_extra_params": n_extra,
    "dropoutts_extra_params_target": 4,
    "extra_param_inventory": extra,
    "overhead_pct_of_backbone": round(overhead_pct, 4),
    "eval_bit_identical_passthrough": eval_bit_identical,
    "train_mode_applies_dropout": train_transforms,
    "per_step_ms_baseline": round(ms_base, 4),
    "per_step_ms_dropoutts": round(ms_dts, 4),
    "per_step_ratio_dropoutts_over_baseline": round(per_step_ratio, 3),
    "infer_ms_baseline": round(inf_base, 4),
    "infer_ms_dropoutts": round(inf_dts, 4),
}
with (HERE / "results.json").open("w") as f:
    json.dump(res, f, indent=1)

print("=== Claim 1: negligible parameter overhead / no architectural modification ===")
print(f"backbone params (baseline)      : {nb_base}")
print(f"backbone params (with DropoutTS): {nb_dts}   identical backbone: {nb_base==nb_dts}")
print(f"DropoutTS EXTRA trainable params: {n_extra}   (target = 4)")
for n_, sh, ct in extra:
    print(f"    - {n_:<22} shape={sh} numel={ct}")
print(f"overhead vs backbone            : {overhead_pct:.4f}%")
print(f"\n=== zero inference latency overhead (eval-mode transform) ===")
print(f"eval-mode dropout bit-exact passthrough (target True): {eval_bit_identical}")
print(f"train-mode dropout actually applied (sanity True)    : {train_transforms}")
print(f"\n=== bounded per-step training cost (honest; not the 100-epoch speedup) ===")
print(f"ms/step baseline : {ms_base:.3f}")
print(f"ms/step DropoutTS: {ms_dts:.3f}   ratio DropoutTS/baseline = {per_step_ratio:.3f}x")
print(f"(headline 1.12-1.45x TRAINING speedup is a convergence/early-stopping")
print(f" effect over 100 epochs on real data -> see gpu_job/RUN_GPU.md)")
print(f"eval forward ms: baseline {inf_base:.3f}  DropoutTS {inf_dts:.3f}")
print("\nwrote results.json")
