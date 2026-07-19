#!/usr/bin/env python3
"""
c1fix_aggregate.py -- aggregate cells_c1fix/<dataset>__<model>.json produced by
c1fix_run.py, print per-cell tables, evaluate the predeclared rules U1/U2
(defined in c1fix_run.py header, identical to the judged verdict's rules), and
write results_c1fix.json + results_c1fix.csv.
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CELLS = HERE / "cells_c1fix"
DATASETS = ["california", "wine_red", "concrete", "energy", "diabetes"]
MODELS = ["gbm", "rf"]
METHODS = ["raw", "kuleshov", "song", "ckme"]
LABEL = {"raw": "raw", "kuleshov": "Kuleshov'18", "song": "Song'19", "ckme": "CKME"}


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


def main():
    cells = {}
    for ds in DATASETS:
        for mo in MODELS:
            f = CELLS / f"{ds}__{mo}.json"
            if not f.exists():
                raise SystemExit(f"missing cell {f.name} -- run c1fix_run.py first")
            cells[(ds, mo)] = json.loads(f.read_text())

    agg = {}
    for key, c in cells.items():
        agg[key] = {"n_seeds": c["n_seeds"], "sizes": c["sizes_tr_ca_te"],
                    "lam": [round(v, 6) for v in c["ckme_lambda"]]}
        for m in METHODS:
            ps = c["per_seed"][m]
            agg[key][m] = {"crps_norm_mean": ms(ps["crps_norm"])[0],
                           "crps_norm_sd": ms(ps["crps_norm"])[1],
                           "skce_mean": ms(ps["skce"])[0],
                           "skce_sd": ms(ps["skce"])[1],
                           "stat_mean": ms(ps["stat"])[0],
                           "accept_frac": float(np.mean(ps["accept"]))}

    n_cells = len(agg)
    print(f"=== C1 FIX RUN: verdict-mandated suite, {n_cells} dataset x model cells "
          f"(GBM/RF), faithful CKME (official ReCalibration.jl port) ===")
    print("\n-- SKCE auto-calibration test (AsymptoticSKCETest, alpha=5%): "
          "mean unbiased SKCE | acceptance --")
    hdr = f"{'cell (n_test)':<22}" + "".join(f"{LABEL[m]:>20}" for m in METHODS)
    print(hdr)
    for ds in DATASETS:
        for mo in MODELS:
            a = agg[(ds, mo)]
            row = "".join(f"   {a[m]['skce_mean']:9.2e}|{a[m]['accept_frac']:4.2f}"
                          for m in METHODS)
            print(f"{ds+'/'+mo+' ('+str(a['sizes'][2])+')':<22}{row}")

    print("\n-- normalised CRPS (raw = 1.000) --")
    print(f"{'cell':<22}" + "".join(f"{LABEL[m]:>20}" for m in METHODS))
    for ds in DATASETS:
        for mo in MODELS:
            a = agg[(ds, mo)]
            row = "".join(f"    {a[m]['crps_norm_mean']:9.3f}+-{a[m]['crps_norm_sd']:.3f}"
                          for m in METHODS)
            print(f"{ds+'/'+mo:<22}{row}")

    acc_mean = {m: float(np.mean([agg[k][m]["accept_frac"] for k in agg]))
                for m in METHODS}
    skce_better = sum(agg[k]["ckme"]["skce_mean"] < agg[k]["raw"]["skce_mean"]
                      for k in agg)
    stat_better = sum(agg[k]["ckme"]["stat_mean"] < agg[k]["raw"]["stat_mean"]
                      for k in agg)
    crps_better = sum(agg[k]["ckme"]["crps_norm_mean"] < 1.0 for k in agg)
    per_seed_wins = 0
    per_seed_tot = 0
    for k in agg:
        c = cells[k]
        r = np.asarray(c["per_seed"]["raw"]["skce"])
        s = np.asarray(c["per_seed"]["ckme"]["skce"])
        per_seed_wins += int((s < r).sum())
        per_seed_tot += r.size
    print("\n-- Suite means --")
    print("  acceptance fraction:", {LABEL[m]: round(acc_mean[m], 3) for m in METHODS})
    print(f"  CKME unbiased SKCE < raw on {skce_better}/{n_cells} cells "
          f"(U2 metric); per-seed {per_seed_wins}/{per_seed_tot}")
    print(f"  CKME official test statistic < raw on {stat_better}/{n_cells} cells "
          f"(auxiliary)")
    print(f"  CKME CRPS_norm < 1 on {crps_better}/{n_cells} cells")

    U1 = all(acc_mean["ckme"] >= acc_mean[m] for m in ["raw", "kuleshov", "song"])
    U2 = skce_better >= int(np.ceil(0.7 * n_cells))
    print("\n=== Predeclared rules (identical to judged U1/U2) ===")
    print(f"  U1 CKME highest suite-mean acceptance      : {U1}  "
          f"(CKME {acc_mean['ckme']:.3f} vs raw {acc_mean['raw']:.3f}, "
          f"Kuleshov {acc_mean['kuleshov']:.3f}, Song {acc_mean['song']:.3f})")
    print(f"  U2 CKME lowers mean SKCE vs raw >=70% cells: {U2}  "
          f"({skce_better}/{n_cells}; official-stat auxiliary {stat_better}/{n_cells})")
    verdict = "verified" if (U1 and U2) else ("partial" if (U1 or U2) else "failed")
    print(f"VERDICT (C1 fix suite): {verdict}")

    res = {"datasets": DATASETS, "models": MODELS, "methods": METHODS,
           "cells": {f"{ds}__{mo}": agg[(ds, mo)] for ds in DATASETS for mo in MODELS},
           "suite": {"accept_mean": acc_mean,
                     "ckme_skce_better_cells": int(skce_better),
                     "ckme_skce_better_seeds": [int(per_seed_wins), int(per_seed_tot)],
                     "ckme_stat_better_cells": int(stat_better),
                     "ckme_crps_lt1_cells": int(crps_better), "n_cells": n_cells},
           "rules": {"U1_ckme_highest_acceptance": bool(U1),
                     "U2_ckme_skce_better_70pct": bool(U2)},
           "verdict": verdict}
    (HERE / "results_c1fix.json").write_text(json.dumps(res, indent=2))
    with open(HERE / "results_c1fix.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "method", "n_seeds", "n_test",
                    "skce_mean", "skce_sd", "stat_mean", "accept_frac",
                    "crps_norm_mean", "crps_norm_sd"])
        for ds in DATASETS:
            for mo in MODELS:
                a = agg[(ds, mo)]
                for m in METHODS:
                    w.writerow([ds, mo, m, a["n_seeds"], a["sizes"][2],
                                a[m]["skce_mean"], a[m]["skce_sd"], a[m]["stat_mean"],
                                a[m]["accept_frac"], round(a[m]["crps_norm_mean"], 6),
                                round(a[m]["crps_norm_sd"], 6)])
    print("wrote results_c1fix.json, results_c1fix.csv")


if __name__ == "__main__":
    main()
