#!/usr/bin/env python3
"""Aggregate _c2/*.json (REAL datasets x REAL models, now including a real Mixture
Density Network -- the paper's own named model class) + O(n log n) kernel bench.
Honest verdict; writes results_real.json. Deterministic, single thread."""
import json, glob, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.stats import t as student_t
from pathlib import Path
HERE = Path(__file__).resolve().parent

def wss(a):
    a = np.sort(a); n = a.size; i = np.arange(n)
    return 2.0 * np.dot(a, (2 * i - n + 1).astype(np.float64))
def css(x, y):
    x = np.sort(x); n = x.size; pre = np.concatenate(([0.0], np.cumsum(x))); tot = pre[-1]
    k = np.searchsorted(x, y, side="right")
    return float(np.sum((y * k - pre[k]) + ((tot - pre[k]) - y * (n - k))))
def ed_sort(x, y):
    n, m = x.size, y.size; return 2.0 * css(x, y) / (n * m) - wss(x) / (n * n) - wss(y) / (m * m)
def ed_brute(x, y):
    n, m = x.size, y.size
    return 2.0 * np.abs(x[:, None] - y[None, :]).sum() / (n * m) - np.abs(x[:, None] - x[None, :]).sum() / (n * n) - np.abs(y[:, None] - y[None, :]).sum() / (m * m)

def part_a():
    rng = np.random.default_rng(20260717); mr = 0.0
    for _ in range(150):
        n = int(rng.integers(2, 400)); m = int(rng.integers(2, 400))
        x = rng.normal(0, 1, n); y = rng.normal(0.7, 1.3, m)
        mr = max(mr, abs(ed_sort(x, y) - ed_brute(x, y)) / max(abs(ed_brute(x, y)), 1e-300))
    def bench(fn, sizes, reps):
        ns, ts = [], []
        for n in sizes:
            xx = rng.normal(0, 1, n); yy = rng.normal(0.5, 1, n); fn(xx, yy); fn(xx, yy); best = np.inf
            for _ in range(reps):
                t0 = time.perf_counter(); fn(xx, yy); best = min(best, time.perf_counter() - t0)
            ns.append(n); ts.append(best)
        return np.array(ns, float), np.array(ts, float)
    a, b = bench(ed_sort, [2000, 4000, 8000, 16000, 32000, 65536, 131072, 262144], 5)
    c, d = bench(ed_brute, [2000, 3000, 4000, 5000, 6000], 6)
    return dict(max_rel_err=float(mr), sort_sizes=[int(v) for v in a], sort_times_s=[float(v) for v in b],
                sort_slope=float(np.polyfit(np.log(a), np.log(b), 1)[0]),
                brute_sizes=[int(v) for v in c], brute_times_s=[float(v) for v in d],
                brute_slope=float(np.polyfit(np.log(c), np.log(d), 1)[0]))

def ci95(v):
    v = np.asarray(v, float); n = v.size; mu = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mu, mu - float(student_t.ppf(0.975, max(n - 1, 1))) * se, mu + float(student_t.ppf(0.975, max(n - 1, 1))) * se

def meanstd(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0

DS = ["california", "wine_red", "concrete", "energy", "diabetes"]
MD = ["gbm", "rf", "mdn"]; METH = ["raw", "kuleshov", "song", "ckme", "ckme_fx"]
MDNAME = {"gbm": "GBM (heteroscedastic Gradient Boosting)", "rf": "DRF (Random-Forest predictive, kernel-smoothed)",
          "mdn": "MDN (Mixture Density Network, torch, K=3)"}
LAB = {"raw": "raw", "kuleshov": "Kuleshov18", "song": "Song19", "ckme": "CKME", "ckme_fx": "CKME-fullX"}
MET = ["ece", "ks", "condece", "ace", "crps"]

cells = {}
for f in glob.glob(str(HERE / "_c2" / "*.json")):
    d = json.load(open(f)); cells[(d["dataset"], d["model"])] = d
cm = {(ds, md): {m: {mt: float(np.mean(cells[(ds, md)]["per_seed"][m][mt])) for mt in MET} for m in METH} for (ds, md) in cells}
feats = {ds: cells[(ds, "gbm")]["n_features"] for ds in DS}; ntot = {ds: cells[(ds, "gbm")]["n_total"] for ds in DS}
def suite(mt, m): return [cm[(ds, md)][m][mt] for md in MD for ds in DS]
stats = {mt: {m: ci95(suite(mt, m)) for m in METH} for mt in MET}
def bestp(mt): return min(["kuleshov", "song"], key=lambda m: stats[mt][m][0])

# per-dataset (avg models) + per-dataset paired CKME-vs-Kuleshov significance (pool seeds over both models)
perds = {ds: {mt: {m: float(np.mean([cm[(ds, md)][m][mt] for md in MD])) for m in METH} for mt in MET} for ds in DS}
paired = {}
for ds in DS:
    for mt in ["ace", "crps", "condece"]:
        dif = []
        for md in MD:
            k = np.array(cells[(ds, md)]["per_seed"]["kuleshov"][mt]); c = np.array(cells[(ds, md)]["per_seed"]["ckme"][mt])
            dif += list(k - c)
        dif = np.array(dif); z = float(dif.mean() / (dif.std(ddof=1) / np.sqrt(dif.size) + 1e-12))
        paired[f"{ds}/{mt}"] = dict(mean_gain=float(dif.mean()), z=z, n=int(dif.size), ckme_better=int((dif > 0).sum()))

def wins(mt):
    w = sum(min({m: cm[c][m][mt] for m in ["kuleshov", "song", "ckme"]}, key=lambda m: cm[c][m][mt]) == "ckme" for c in cm)
    return w, len(cm)
wc = {mt: wins(mt) for mt in ["ace", "condece", "crps"]}
beat_raw = sum(cm[c]["ckme"]["ace"] < cm[c]["raw"]["ace"] for c in cm)
crps_imp = sum(cm[c]["ckme"]["crps"] < 1.0 for c in cm)
bp = bestp("ace"); dv = np.array([cm[c][bp]["ace"] - cm[c]["ckme"]["ace"] for c in cm])
zc = float(dv.mean() / (dv.std(ddof=1) / np.sqrt(dv.size)))
A = part_a()

lines = []
def P(s): lines.append(s); print(s)
P("=== REAL benchmark datasets x REAL model pipeline: CKME re-calibration vs named priors ===")
P("datasets: " + ", ".join(f"{ds}({ntot[ds]}x{feats[ds]})" for ds in DS))
P("models  : gbm = heteroscedastic GradientBoosting (Gaussian predictive);  rf = RandomForest/DRF-style kernel-smoothed predictive;  mdn = Mixture Density Network (real torch MLP, K=3, trained by NLL)")
P("priors  : Kuleshov 2018 (marginal PIT map), Song 2019 (parametric Beta) ; CKME = this paper (cond. kernel mean embedding)")
P(f"\n-- Suite mean over {len(cm)} (dataset x model) cells [95% CI across cells] --  lower=better")
for mt in ["ace", "condece", "crps", "ece", "ks"]:
    P(f"  {mt.upper():<8}" + "  ".join(f"{LAB[m]}={stats[mt][m][0]:.4f}[{stats[mt][m][1]:.4f},{stats[mt][m][2]:.4f}]" for m in ["raw","kuleshov","song","ckme"]))
P(f"\n-- Per-dataset (mean over {len(MD)} models: {'+'.join(MD)}); ACE=SKCE-family energy auto-cal, CRPS ratio vs raw --")
for ds in DS:
    P(f"  {ds:<11}(f={feats[ds]:<2}) ACE: raw={perds[ds]['ace']['raw']:.4f} K={perds[ds]['ace']['kuleshov']:.4f} S={perds[ds]['ace']['song']:.4f} CKME={perds[ds]['ace']['ckme']:.4f}"
      f"  | CRPS: K={perds[ds]['crps']['kuleshov']:.4f} S={perds[ds]['crps']['song']:.4f} CKME={perds[ds]['crps']['ckme']:.4f}")
P(f"\n-- Per-dataset paired CKME vs Kuleshov (pooled seeds, {len(MD)} models) --")
for ds in DS:
    a = paired[f"{ds}/ace"]; P(f"  {ds:<11} ACE gain={a['mean_gain']:+.4f} z={a['z']:+.1f} ({a['ckme_better']}/{a['n']} seeds CKME better)")
P(f"\nCKME lowest ACE among priors: {wc['ace'][0]}/{wc['ace'][1]} ; condECE {wc['condece'][0]}/{wc['condece'][1]} ; CRPS {wc['crps'][0]}/{wc['crps'][1]}")
P(f"CKME beats RAW on ACE: {beat_raw}/{len(cm)} ; CKME improves CRPS (<raw): {crps_imp}/{len(cm)} ; paired suite ACE z={zc:.1f}, better {int((dv>0).sum())}/{dv.size}")

# ---- Table-1-style per (dataset x model) normalized CRPS, mean+-std across seeds (paper's own reporting format) ----
table1 = {}
P("\n=== Table-1-style normalized CRPS (mean +/- std across seeds; uncalibrated raw = 1.000) ===")
for ds in DS:
    for md in MD:
        if (ds, md) not in cells: continue
        row = {}
        for meth in ["raw", "kuleshov", "song", "ckme"]:
            mu, sd = meanstd(cells[(ds, md)]["per_seed"][meth]["crps"])
            row[meth] = dict(mean=mu, std=sd)
        table1[f"{ds}/{md}"] = row
        P(f"  {ds:<11}/{md:<3} ({MDNAME[md]:<42}) raw=1.000+/-0.000  Kuleshov={row['kuleshov']['mean']:.3f}+/-{row['kuleshov']['std']:.3f}"
          f"  Song={row['song']['mean']:.3f}+/-{row['song']['std']:.3f}  CKME={row['ckme']['mean']:.3f}+/-{row['ckme']['std']:.3f}")
# paper's own quoted Table 1 example format check: "0.594+/-0.178 for a Mixture Density Network on Energy"
if "energy/mdn" in table1:
    r = table1["energy/mdn"]["ckme"]
    P(f"\n  [paper Table 1 quoted-format check] normalized CRPS, MDN on Energy, CKME-recalibrated: {r['mean']:.3f}+/-{r['std']:.3f}  vs 1.000 uncalibrated")
P(f"\n=== O(n log n) characteristic EDK (re-run at larger n) ===")
P(f"max rel err SORT vs BRUTE (150 cases): {A['max_rel_err']:.2e} (<1e-10)")
for n, t in zip(A["sort_sizes"], A["sort_times_s"]): P(f"   n={n:7d}  {t*1e3:8.3f} ms")
P(f"SORT slope={A['sort_slope']:.3f} (0.9-1.3) ; BRUTE slope={A['brute_slope']:.3f} (1.8-2.1)")

R = {}
R["A_edk_exact_lt_1e10"] = A["max_rel_err"] < 1e-10
R["A_onlogn_slopes"] = (0.9 <= A["sort_slope"] <= 1.3) and (1.8 <= A["brute_slope"] <= 2.1)
R["C1_ckme_beats_raw_cond_ge8"] = beat_raw >= 8
R["C1_ckme_lowest_suite_ace"] = stats["ace"]["ckme"][0] < min(stats["ace"][m][0] for m in ["raw","kuleshov","song"])
R["C1_ckme_lowest_suite_condece"] = stats["condece"]["ckme"][0] < min(stats["condece"][m][0] for m in ["raw","kuleshov","song"])
R["C1_ckme_wins_ace_ge7"] = wc["ace"][0] >= 7
R["C2_ckme_lowest_suite_crps_lt1"] = (stats["crps"]["ckme"][0] < min(stats["crps"][m][0] for m in ["kuleshov","song"])) and (stats["crps"]["ckme"][0] < 1.0)
R["C2_ckme_improves_crps_ge8"] = crps_imp >= 8
R["C2_california_paired_decisive"] = paired["california/ace"]["z"] >= 3.0   # marquee dataset decisive
claim1 = R["C1_ckme_beats_raw_cond_ge8"] and R["C1_ckme_lowest_suite_ace"] and R["C1_ckme_lowest_suite_condece"] and R["C1_ckme_wins_ace_ge7"]
claim2 = R["A_edk_exact_lt_1e10"] and R["A_onlogn_slopes"] and R["C2_ckme_lowest_suite_crps_lt1"] and R["C2_ckme_improves_crps_ge8"] and R["C2_california_paired_decisive"]
P("\n=== Predeclared rules ===")
for k, v in R.items(): P(f"  {k:<38}: {v}")
P(f"CLAIM1 verified: {claim1}  |  CLAIM2 verified: {claim2}")

out = dict(datasets={ds: dict(n=ntot[ds], features=feats[ds]) for ds in DS}, models=MD, model_names=MDNAME, methods=METH,
           lam=cells[("california","gbm")]["lam"],
           suite_stats={mt: {m: dict(mean=stats[mt][m][0], lo=stats[mt][m][1], hi=stats[mt][m][2]) for m in METH} for mt in MET},
           per_dataset=perds, per_cell={f"{ds}/{md}": cm[(ds, md)] for (ds, md) in cm},
           table1_normalized_crps=table1,
           paired=paired, best_prior={mt: bestp(mt) for mt in MET},
           wins={k: list(v) for k, v in wc.items()}, beat_raw_ace=int(beat_raw), crps_improve=int(crps_imp),
           paired_suite_ace_z=zc, part_a_edk_onlogn=A, rules={k: bool(v) for k, v in R.items()},
           claim1_verified=bool(claim1), claim2_verified=bool(claim2),
           numpy=np.__version__, scipy=__import__("scipy").__version__)
(HERE / "results_real.json").write_text(json.dumps(out, indent=2))
(HERE / "_combine2_out.txt").write_text("\n".join(lines))
P("\nwrote results_real.json")
