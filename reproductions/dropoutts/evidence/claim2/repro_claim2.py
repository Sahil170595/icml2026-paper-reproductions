"""CPU proxy for Claim 2 (part of scored robustness claim): "DropoutTS consistently
improves time series forecasting robustness across diverse noise regimes."
Paper anchor (Synth-12 Benchmark): averaged across H in {96,192,336,720}, DropoutTS
improves the Informer backbone by 46.0% MSE / 24.5% MAE, peak 48.2% MSE at sigma=0.3.

Mechanism proxy: a small MLP forecaster (L=96 -> H=24) on synthetic multi-sinusoid+
trend series. Inputs are corrupted at one of five noise regimes sigma in
{0.0,0.1,0.3,0.5,0.7}; the target is the CLEAN future (robustness = predict clean
signal from noisy history, mirroring the repo's use_clean_targets=True). Three arms:
no-dropout, standard fixed nn.Dropout(0.3), and the OFFICIAL DropoutTS sample-adaptive
dropout. Reports MSE AND MAE by regime, mean over 3 seeds. Deterministic, CPU, 1 thread.
Toy scale: this substantiates the MECHANISM/DIRECTION, not the paper's Informer-scale
percentages (see gpu_job/RUN_GPU.md)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, importlib.util
from pathlib import Path
import numpy as np, torch, torch.nn as nn

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dropout_ts", HERE.parent / "dropout_ts.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
DropoutTS, SampleAdaptiveDropout = m.DropoutTS, m.SampleAdaptiveDropout

L, H, Ntr, Nte = 96, 24, 3000, 1500
REGIMES = [0.0, 0.1, 0.3, 0.5, 0.7]

def make(n, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, L, 1), np.float32); Y = np.zeros((n, H, 1), np.float32); reg = np.zeros(n, int)
    t = np.arange(L + H)
    for i in range(n):
        k = rng.integers(1, 4); sig = np.zeros(L + H)
        for _ in range(k):
            f = rng.uniform(0.02, 0.4); ph = rng.uniform(0, 2*np.pi); a = rng.uniform(0.5, 1.5)
            sig += a * np.sin(2*np.pi*f*t + ph)
        sig += rng.uniform(-0.5, 0.5) * (t / (L + H))
        r = rng.integers(0, len(REGIMES)); reg[i] = r
        obs = sig + rng.normal(0, REGIMES[r], L + H)
        X[i, :, 0] = obs[:L]; Y[i, :, 0] = sig[L:L+H]   # noisy past -> clean future
    return torch.tensor(X), torch.tensor(Y), torch.tensor(reg)

Xtr, Ytr, _ = make(Ntr, 1); Xte, Yte, Rte = make(Nte, 2)

class Forecaster(nn.Module):
    def __init__(self, mode):        # mode: 'none' | 'fixed' | 'adaptive'
        super().__init__()
        self.fc1 = nn.Linear(L, 128); self.act = nn.ReLU(); self.fc2 = nn.Linear(128, H)
        self.mode = mode
        if mode == 'adaptive':
            self.drop = SampleAdaptiveDropout(p=0.3)
            self.dts = DropoutTS(seq_len=L, p_min=0.1, p_max=0.6)
        elif mode == 'fixed':
            self.drop = nn.Dropout(0.3)
        else:
            self.drop = nn.Identity()
    def forward(self, x):
        h = self.act(self.fc1(x.squeeze(-1)))
        if self.mode == 'adaptive':
            h = self.drop(h, self.dts.compute_dropout_rates(x))
        else:
            h = self.drop(h)
        return self.fc2(h).unsqueeze(-1)

def train_eval(mode, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Forecaster(mode); opt = torch.optim.Adam(net.parameters(), 1e-3); lf = nn.MSELoss()
    net.train()
    for _ in range(55):
        perm = torch.randperm(Ntr)
        for i in range(0, Ntr, 128):
            b = perm[i:i+128]; opt.zero_grad(); lf(net(Xtr[b]), Ytr[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte)
        mse = {}; mae = {}
        for r in range(len(REGIMES)):
            idx = (Rte == r)
            mse[REGIMES[r]] = float(((pred[idx]-Yte[idx])**2).mean())
            mae[REGIMES[r]] = float((pred[idx]-Yte[idx]).abs().mean())
        mse['overall'] = float(((pred-Yte)**2).mean())
        mae['overall'] = float((pred-Yte).abs().mean())
    return mse, mae

SEEDS = [0, 1, 2]
def avg(mode):
    ms, ma = [], []
    for s in SEEDS:
        a, b = train_eval(mode, s); ms.append(a); ma.append(b)
    keys = list(ms[0].keys())
    return ({k: float(np.mean([d[k] for d in ms])) for k in keys},
            {k: float(np.mean([d[k] for d in ma])) for k in keys})

none_mse, none_mae = avg('none')
fix_mse,  fix_mae  = avg('fixed')
dts_mse,  dts_mae  = avg('adaptive')

def imp(base, new): return (base - new) / base * 100.0
keys = [str(r) for r in REGIMES] + ['overall']
regkeys = REGIMES + ['overall']

print("=== Claim 2: robustness across noise regimes (mean over 3 seeds, lower=better) ===")
print(f"{'sigma':>8} {'noDrop MSE':>11} {'fixed MSE':>10} {'DropoutTS':>10} {'impVSfixed':>11} {'impVSnone':>10}")
all_pos_vs_fixed = True
for r in regkeys:
    iv = imp(fix_mse[r], dts_mse[r]); iv2 = imp(none_mse[r], dts_mse[r])
    if r != 'overall': all_pos_vs_fixed = all_pos_vs_fixed and iv > 0
    lbl = 'OVERALL' if r == 'overall' else f'{r}'
    print(f"{lbl:>8} {none_mse[r]:>11.4f} {fix_mse[r]:>10.4f} {dts_mse[r]:>10.4f} {iv:>10.1f}% {iv2:>9.1f}%")
print(f"\n{'sigma':>8} {'fixed MAE':>10} {'DropoutTS':>10} {'impVSfixed':>11}")
for r in regkeys:
    lbl = 'OVERALL' if r == 'overall' else f'{r}'
    print(f"{lbl:>8} {fix_mae[r]:>10.4f} {dts_mae[r]:>10.4f} {imp(fix_mae[r], dts_mae[r]):>10.1f}%")
print(f"\nDropoutTS beats fixed dropout in EVERY noise regime (MSE): {all_pos_vs_fixed}")
print(f"peak MSE improvement vs fixed at sigma="
      f"{max(REGIMES, key=lambda r: imp(fix_mse[r], dts_mse[r]))}")

res = {"regimes": REGIMES, "seeds": SEEDS,
       "mse": {"none": none_mse, "fixed": fix_mse, "dropoutts": dts_mse},
       "mae": {"none": none_mae, "fixed": fix_mae, "dropoutts": dts_mae},
       "mse_improvement_vs_fixed_pct": {str(r): round(imp(fix_mse[r], dts_mse[r]), 2) for r in regkeys},
       "mae_improvement_vs_fixed_pct": {str(r): round(imp(fix_mae[r], dts_mae[r]), 2) for r in regkeys},
       "dropoutts_beats_fixed_every_regime": bool(all_pos_vs_fixed)}
with (HERE / "results.json").open("w") as f:
    json.dump(res, f, indent=1)
print("\nwrote results.json")
