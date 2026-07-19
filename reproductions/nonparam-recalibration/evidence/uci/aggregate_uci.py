#!/usr/bin/env python3
r"""
aggregate_uci.py -- aggregate cells/<dataset>__<model>.json produced by
uci_repro.py, print the claim tables, evaluate the PREDECLARED rules
(U1/U2/U3, see uci_repro.py header), and write results_uci.json +
results_uci.csv.

PAPER_CRPS below are the paper's own Table 1 numbers ("CRPS scores relative
to the base model", mean +/- sd over its 20 predefined splits), extracted
from the arXiv HTML (arxiv.org/html/2602.13362v1, Table 1) for the 5 datasets
x 4 model families run here.  They are used ONLY as comparison targets
(evidence-gate item 5) -- never as results.  GPBETA is NA for DRF/BNN in the
paper (its official implementation requires Gaussian base predictions).
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CELLS = HERE / "cells"
DATASETS = ["yacht", "bostonHousing", "energy", "concrete", "wine-quality-red"]
MODELS = ["gdn", "mdn", "bnn", "drf"]
METHODS = ["none", "pit", "beta", "ckme"]

# Table 1 (paper): normalised CRPS mean+/-sd -- (CKME, PIT, GPBETA); None(T)=1.
PAPER_CRPS = {
    ("yacht", "gdn"):            {"ckme": (1.276, 0.537), "pit": (0.997, 0.077), "gpbeta": (0.905, 0.198)},
    ("yacht", "mdn"):            {"ckme": (1.326, 0.591), "pit": (1.058, 0.130), "gpbeta": (0.870, 0.131)},
    ("yacht", "bnn"):            {"ckme": (0.889, 0.341), "pit": (1.097, 0.142), "gpbeta": None},
    ("yacht", "drf"):            {"ckme": (0.714, 0.252), "pit": (0.933, 0.089), "gpbeta": None},
    ("bostonHousing", "gdn"):    {"ckme": (1.150, 0.170), "pit": (1.000, 0.033), "gpbeta": (0.995, 0.096)},
    ("bostonHousing", "mdn"):    {"ckme": (1.175, 0.149), "pit": (1.007, 0.030), "gpbeta": (1.027, 0.064)},
    ("bostonHousing", "bnn"):    {"ckme": (0.934, 0.068), "pit": (1.004, 0.028), "gpbeta": None},
    ("bostonHousing", "drf"):    {"ckme": (0.927, 0.061), "pit": (1.002, 0.023), "gpbeta": None},
    ("energy", "gdn"):           {"ckme": (0.767, 0.224), "pit": (0.981, 0.048), "gpbeta": (0.984, 0.048)},
    ("energy", "mdn"):           {"ckme": (0.594, 0.178), "pit": (1.019, 0.046), "gpbeta": (1.153, 0.071)},
    ("energy", "bnn"):           {"ckme": (0.693, 0.211), "pit": (1.008, 0.043), "gpbeta": None},
    ("energy", "drf"):           {"ckme": (0.719, 0.198), "pit": (0.998, 0.039), "gpbeta": None},
    ("concrete", "gdn"):         {"ckme": (1.045, 0.047), "pit": (0.999, 0.014), "gpbeta": (1.000, 0.021)},
    ("concrete", "mdn"):         {"ckme": (1.032, 0.038), "pit": (1.003, 0.018), "gpbeta": (1.038, 0.028)},
    ("concrete", "bnn"):         {"ckme": (0.928, 0.042), "pit": (1.001, 0.015), "gpbeta": None},
    ("concrete", "drf"):         {"ckme": (0.915, 0.034), "pit": (1.000, 0.009), "gpbeta": None},
    ("wine-quality-red", "gdn"): {"ckme": (0.901, 0.026), "pit": (0.999, 0.009), "gpbeta": (1.000, 0.011)},
    ("wine-quality-red", "mdn"): {"ckme": (1.060, 0.042), "pit": (1.002, 0.007), "gpbeta": (1.014, 0.021)},
    ("wine-quality-red", "bnn"): {"ckme": (0.899, 0.022), "pit": (1.000, 0.006), "gpbeta": None},
    ("wine-quality-red", "drf"): {"ckme": (0.902, 0.019), "pit": (1.000, 0.005), "gpbeta": None},
}
SHORT = {"yacht": "yacht", "bostonHousing": "housing", "energy": "energy",
         "concrete": "concrete", "wine-quality-red": "wine"}


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def main():
    cells = {}
    for ds in DATASETS:
        for mo in MODELS:
            f = CELLS / f"{ds}__{mo}.json"
            if not f.exists():
                raise SystemExit(f"missing cell {f.name} -- run uci_repro.py first")
            cells[(ds, mo)] = json.loads(f.read_text())

    agg = {}
    for key, c in cells.items():
        agg[key] = {}
        for m in METHODS:
            ps = c["per_split"][m]
            cn_m, cn_s = ms(ps["crps_norm"])
            sk_m, sk_s = ms(ps["skce"])
            st_m, st_s = ms(ps["stat"])
            acc = float(np.mean(ps["accept"]))
            agg[key][m] = {"crps_norm_mean": cn_m, "crps_norm_sd": cn_s,
                           "skce_mean": sk_m, "skce_sd": sk_s,
                           "stat_mean": st_m, "stat_sd": st_s,
                           "accept_frac": acc,
                           "crps_raw_units_mean": ms(ps["crps"])[0]}

    n_splits = cells[(DATASETS[0], MODELS[0])]["n_splits"]
    print(f"=== UCI benchmark (paper protocol, {n_splits} predefined splits/cell, "
          f"{len(cells)} dataset x model cells) ===")
    print("\n-- Normalised CRPS (this run vs paper Table 1; base model None(T) = 1.000) --")
    hdr = f"{'cell':<16}{'PIT ours':>10}{'PIT paper':>12}{'CKME ours':>11}{'CKME paper':>13}  agree2sd"
    print(hdr)
    agree_cnt, agree_tot = 0, 0
    for ds in DATASETS:
        for mo in MODELS:
            a = agg[(ds, mo)]
            pp = PAPER_CRPS[(ds, mo)]
            pm, psd = pp["ckme"]
            om, osd = a["ckme"]["crps_norm_mean"], a["ckme"]["crps_norm_sd"]
            ok = abs(om - pm) <= 2.0 * (osd + psd)
            agree_cnt += ok; agree_tot += 1
            a["ckme"]["paper_crps"] = pp["ckme"]; a["ckme"]["agree_2sd"] = bool(ok)
            a["pit"]["paper_crps"] = pp["pit"]
            print(f"{SHORT[ds]+'/'+mo:<16}"
                  f"{a['pit']['crps_norm_mean']:>7.3f}±{a['pit']['crps_norm_sd']:.2f}"
                  f"{pp['pit'][0]:>8.3f}±{pp['pit'][1]:.2f}"
                  f"{om:>8.3f}±{osd:.2f}"
                  f"{pm:>9.3f}±{psd:.2f}"
                  f"{'  YES' if ok else '   no'}")

    print("\n-- SKCE auto-calibration test (CalibrationTests.jl AsymptoticSKCETest, official) --")
    print("   `stat` = official test statistic (run_SKCE_test.jl output); acc at alpha=5%")
    print(f"{'cell':<16}" + "".join(f"{m:>19}" for m in METHODS) + "   (stat | acc)")
    for ds in DATASETS:
        for mo in MODELS:
            a = agg[(ds, mo)]
            row = "".join(f"  {a[m]['stat_mean']:10.2e}|{a[m]['accept_frac']:4.2f}"
                          for m in METHODS)
            print(f"{SHORT[ds]+'/'+mo:<16}{row}")
    print("\n-- SKCE_uq unbiased estimate (auxiliary; discretisation-sensitive) --")
    print(f"{'cell':<16}" + "".join(f"{m:>12}" for m in METHODS))
    for ds in DATASETS:
        for mo in MODELS:
            a = agg[(ds, mo)]
            row = "".join(f"  {a[m]['skce_mean']:10.2e}" for m in METHODS)
            print(f"{SHORT[ds]+'/'+mo:<16}{row}")

    acc_mean = {m: float(np.mean([agg[k][m]["accept_frac"] for k in agg])) for m in METHODS}
    stat_better = sum(agg[k]["ckme"]["stat_mean"] < agg[k]["none"]["stat_mean"] for k in agg)
    skce_better = sum(agg[k]["ckme"]["skce_mean"] < agg[k]["none"]["skce_mean"] for k in agg)
    crps_better = sum(agg[k]["ckme"]["crps_norm_mean"] < 1.0 for k in agg)
    print("\n-- Suite means --")
    print("  acceptance fraction:", {m: round(acc_mean[m], 3) for m in METHODS})
    print(f"  CKME official SKCE-test statistic < raw on {stat_better}/{len(agg)} cells "
          f"(U2 metric; run_SKCE_test.jl `stat` column)")
    print(f"  CKME unbiased SKCE_uq < raw on {skce_better}/{len(agg)} cells (auxiliary); "
          f"CKME CRPS_norm < 1 on {crps_better}/{len(agg)} cells")

    em = agg[("energy", "mdn")]["ckme"]
    em_pm, em_psd = PAPER_CRPS[("energy", "mdn")]["ckme"]
    em_ok = abs(em["crps_norm_mean"] - em_pm) <= 2.0 * (em["crps_norm_sd"] + em_psd)
    print(f"  Registered Energy-MDN example: ours {em['crps_norm_mean']:.3f}"
          f"±{em['crps_norm_sd']:.3f} vs paper {em_pm}±{em_psd} -> "
          f"{'agree' if em_ok else 'DISAGREE'} (2sd rule)")

    U1 = (acc_mean["ckme"] >= acc_mean["none"] and acc_mean["ckme"] >= acc_mean["pit"]
          and acc_mean["ckme"] >= acc_mean["beta"])
    U2 = stat_better >= int(np.ceil(0.7 * len(agg)))
    U3 = agree_cnt >= int(np.ceil(0.7 * agree_tot))
    print("\n=== Predeclared rules ===")
    print(f"  U1 CKME highest mean acceptance (Fig.1)     : {U1}")
    print(f"  U2 CKME lowers mean SKCE-test stat vs raw >=70% cells : {U2}  "
          f"({stat_better}/{len(agg)}; auxiliary SKCE_uq {skce_better}/{len(agg)})")
    print(f"  U3 Table-1 CRPS agreement (2sd) >=70% cells : {U3}  ({agree_cnt}/{agree_tot})")
    verdict = "verified" if (U1 and U2 and U3) else ("partial" if (U1 and U2) else "not-verified")
    print(f"VERDICT: {verdict}")

    res = {"n_splits": n_splits, "datasets": DATASETS, "models": MODELS,
           "methods": METHODS,
           "cells": {f"{ds}__{mo}": agg[(ds, mo)] for ds in DATASETS for mo in MODELS},
           "suite": {"accept_mean": acc_mean, "ckme_stat_better_cells": int(stat_better),
                     "ckme_skce_better_cells": int(skce_better),
                     "ckme_crps_lt1_cells": int(crps_better), "n_cells": len(agg),
                     "table1_agree_cells": int(agree_cnt)},
           "energy_mdn_registered_example": {
               "ours_mean": em["crps_norm_mean"], "ours_sd": em["crps_norm_sd"],
               "paper_mean": em_pm, "paper_sd": em_psd, "agree_2sd": bool(em_ok)},
           "rules": {"U1_ckme_highest_acceptance": bool(U1),
                     "U2_ckme_skce_better_70pct": bool(U2),
                     "U3_table1_crps_agree_70pct": bool(U3)},
           "verdict": verdict,
           "paper_targets_source": "arxiv.org/html/2602.13362v1 Table 1 / Figure 1"}
    (HERE / "results_uci.json").write_text(json.dumps(res, indent=2))

    with open(HERE / "results_uci.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "method", "crps_norm_mean", "crps_norm_sd",
                    "skce_mean", "skce_sd", "stat_mean", "stat_sd", "accept_frac",
                    "paper_crps_mean", "paper_crps_sd", "agree_2sd"])
        for ds in DATASETS:
            for mo in MODELS:
                for m in METHODS:
                    a = agg[(ds, mo)][m]
                    pt = PAPER_CRPS[(ds, mo)].get({"beta": "gpbeta"}.get(m, m))
                    w.writerow([ds, mo, m,
                                round(a["crps_norm_mean"], 6), round(a["crps_norm_sd"], 6),
                                a["skce_mean"], a["skce_sd"],
                                a["stat_mean"], a["stat_sd"], a["accept_frac"],
                                pt[0] if pt else "", pt[1] if pt else "",
                                a.get("agree_2sd", "")])
    print("wrote results_uci.json, results_uci.csv")


if __name__ == "__main__":
    main()
