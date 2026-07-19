"""CPU proxy for Claim 3 (part of scored robustness claim), on REAL data:
"On real-world datasets, DropoutTS yields up to 47.6% MSE improvement on ETTh2 with
the Informer backbone" (Real-World Benchmarks, 7 datasets).

Real dataset: ETTh2 (oil-temperature 'OT' channel) -- the exact dataset the paper's
47.6% number is reported on. Standard univariate forecasting, L=96 -> H=24, trained on
the real series (no synthetic target). We use the PAPER's DropoutTS hyper-parameters
from run_baselines.py: p_min=0.05, p_max=0.5, init_alpha=10, init_sensitivity=5.
Three arms: no-dropout, standard fixed nn.Dropout(0.3), OFFICIAL DropoutTS adaptive
dropout. Robustness is probed by evaluating on clean test windows AND test windows
perturbed with input noise (sigma in std units); dropout is OFF at eval for all arms,
so this isolates which model learned more ROBUST weights. Deterministic, CPU, 1 thread.
Toy backbone (MLP), REAL data -> substantiates DIRECTION on the paper's actual dataset,
not the Informer-scale 47.6% (see gpu_job/RUN_GPU.md). One a-priori config; no tuning."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, importlib.util, urllib.request
from pathlib import Path
import numpy as np, torch, torch.nn as nn

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dropout_ts", HERE.parent / "dropout_ts.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
DropoutTS, SampleAdaptiveDropout = m.DropoutTS, m.SampleAdaptiveDropout

CSV = os.environ.get("ETTH2_CSV", "/tmp/ETTh2.csv")
URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
if not Path(CSV).is_file():
    urllib.request.urlretrieve(URL, CSV)
ot = np.loadtxt(CSV, delimiter=",", skiprows=1, usecols=7, dtype=np.float64)  # 'OT'
n = len(ot); n_tr = int(n * 0.6)
mu, sd = ot[:n_tr].mean(), ot[:n_tr].std()
ot = (ot - mu) / sd
train_series, test_series = ot[:n_tr], ot[int(n*0.8):]

L, H = 96, 24
def windows(series, stride):
    xs, ys = [], []
    for i in range(0, len(series) - L - H, stride):
        xs.append(series[i:i+L]); ys.append(series[i+L:i+L+H])
    return (torch.tensor(np.array(xs, np.float32)).unsqueeze(-1),
            torch.tensor(np.array(ys, np.float32)).unsqueeze(-1))
Xtr, Ytr = windows(train_series, 2)
Xte_c, Yte = windows(test_series, 2)
SIGMAS = [0.0, 0.25, 0.5, 0.75]
print(f"ETTh2 OT: {n} points | train windows {Xtr.shape[0]} | test windows {Xte_c.shape[0]}")

class Forecaster(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.fc1 = nn.Linear(L, 128); self.act = nn.ReLU(); self.fc2 = nn.Linear(128, H)
        self.mode = mode
        if mode == 'adaptive':
            self.drop = SampleAdaptiveDropout(p=0.3)
            self.dts = DropoutTS(seq_len=L, p_min=0.05, p_max=0.5,
                                 init_alpha=10.0, init_sensitivity=5.0)  # paper values
        elif mode == 'fixed': self.drop = nn.Dropout(0.3)
        else:                 self.drop = nn.Identity()
    def forward(self, x):
        h = self.act(self.fc1(x.squeeze(-1)))
        if self.mode == 'adaptive': h = self.drop(h, self.dts.compute_dropout_rates(x))
        else:                       h = self.drop(h)
        return self.fc2(h).unsqueeze(-1)

def run(mode, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    net = Forecaster(mode); opt = torch.optim.Adam(net.parameters(), 1e-3); lf = nn.MSELoss()
    N = Xtr.shape[0]; net.train()
    for _ in range(45):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, 128):
            b = perm[i:i+128]; opt.zero_grad(); lf(net(Xtr[b]), Ytr[b]).backward(); opt.step()
    net.eval(); out = {}
    with torch.no_grad():
        for sig in SIGMAS:
            gg = torch.Generator().manual_seed(1234)
            xt = Xte_c + torch.randn(Xte_c.shape, generator=gg) * sig
            pred = net(xt)
            out[sig] = (float(((pred-Yte)**2).mean()), float((pred-Yte).abs().mean()))
    return out

SEEDS = [0, 1, 2]
def avg(mode):
    a = [run(mode, s) for s in SEEDS]
    return {sig: (float(np.mean([x[sig][0] for x in a])),
                  float(np.mean([x[sig][1] for x in a]))) for sig in SIGMAS}

none = avg('none'); fix = avg('fixed'); dts = avg('adaptive')
def imp(b, a): return (b - a) / b * 100.0

print("\n=== Claim 3: REAL ETTh2 forecasting robustness (mean over 3 seeds, lower=better) ===")
print(f"{'testSigma':>9} {'noDrop':>8} {'fixed':>8} {'DropTS':>8} {'impVSfix':>9} {'impVSnone':>10}")
allpos = True; bs = []; ds_ = []
for sig in SIGMAS:
    nb = none[sig][0]; fb = fix[sig][0]; db = dts[sig][0]; iv = imp(fb, db)
    allpos = allpos and iv > 0; bs.append(fb); ds_.append(db)
    print(f"{sig:>9} {nb:>8.4f} {fb:>8.4f} {db:>8.4f} {iv:>8.1f}% {imp(nb,db):>9.1f}%")
ovb = float(np.mean(bs)); ovd = float(np.mean(ds_))
print(f"{'MEAN':>9} {'':>8} {ovb:>8.4f} {ovd:>8.4f} {imp(ovb,ovd):>8.1f}%")
print(f"\nDropoutTS beats fixed dropout (MSE) at every test-noise level: {allpos}")
print(f"DropoutTS vs fixed at highest noise (sigma=0.75): {imp(fix[0.75][0], dts[0.75][0]):+.1f}% MSE")

res = {"dataset": "ETTh2 (OT channel, real)", "url": URL, "L": L, "H": H, "seeds": SEEDS,
       "dropoutts_hparams": {"p_min": 0.05, "p_max": 0.5, "init_alpha": 10.0, "init_sensitivity": 5.0},
       "test_sigmas": SIGMAS,
       "none": {str(s): {"mse": none[s][0], "mae": none[s][1]} for s in SIGMAS},
       "fixed": {str(s): {"mse": fix[s][0], "mae": fix[s][1]} for s in SIGMAS},
       "dropoutts": {str(s): {"mse": dts[s][0], "mae": dts[s][1]} for s in SIGMAS},
       "mse_improvement_vs_fixed_pct": {str(s): round(imp(fix[s][0], dts[s][0]), 2) for s in SIGMAS},
       "mean_mse_improvement_vs_fixed_pct": round(imp(ovb, ovd), 2),
       "dropoutts_beats_fixed_every_level": bool(allpos)}
with (HERE / "results.json").open("w") as f:
    json.dump(res, f, indent=1)
print("\nwrote results.json")
