#!/usr/bin/env python3
"""
c1fix_ablate.py [chunk <budget_s>] -- tuning-artifact ablation for the C1 fix
run: shows the U1/U2 outcome is NOT a bandwidth/regularisation artifact.

On three representative cells (wine_red/gbm, energy/rf, concrete/gbm; first
3 seeds each) CKME is re-run with
  * kernel bandwidths = c x median heuristic, c in {0.25, 0.5, 1, 2, 4}
    (both the distribution kernel and the observation kernel are scaled, as
    both use the median heuristic in the official code), CV lambda; and
  * lambda fixed to {1e-4, 1e-2, 1} at c = 1 (bypassing the CV).
For every variant the same AsymptoticSKCETest is run; reported: mean unbiased
SKCE, mean p-value, acceptance, mean CRPS_norm.  Resumable via _ablate/*.json.
Writes results_ablate.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c1fix_run import (CFG, METHODS, ckme_recalibrate, crps_from_cdf,  # noqa: E402
                       emp_cdf_on_grid, fit_predict_gbm, fit_predict_rf,
                       gbm_cdf, rf_cdf, load_dataset, make_grid,
                       masses_from_cdf, skce_test, cdf_at_points_gauss)

ACACHE = HERE / "_ablate"
ACACHE.mkdir(exist_ok=True)
CELLS3 = [("wine_red", "gbm"), ("energy", "rf"), ("concrete", "gbm")]
N_SEEDS = 3
BW = [0.25, 0.5, 1.0, 2.0, 4.0]
LAMS = [1e-4, 1e-2, 1.0]


def seed_data(ds, model, si):
    cfg = CFG[ds]
    X, y = load_dataset(ds)
    rng = np.random.default_rng(7000 + 101 * si)
    pool = rng.permutation(len(y))[:cfg["nsub"]]
    ntr, nca, nte = cfg["ntr"], cfg["ncal"], cfg["ntest"]
    itr, ica, ite = (pool[:ntr], pool[ntr:ntr + nca], pool[ntr + nca:ntr + nca + nte])
    xm, xs = X[itr].mean(0), X[itr].std(0) + 1e-8
    ym, ys = y[itr].mean(), y[itr].std() + 1e-8
    Xtr, Xca, Xte = [(X[i] - xm) / xs for i in (itr, ica, ite)]
    ytr, yca, yte = [(y[i] - ym) / ys for i in (itr, ica, ite)]
    g = make_grid(np.concatenate([ytr, yca, yte]))
    if model == "gbm":
        (muc, sdc), (mut, sdt) = fit_predict_gbm(Xtr, ytr, [Xca, Xte], cfg["gbt"], si)
        F_ca, F_te = gbm_cdf(muc, sdc, g), gbm_cdf(mut, sdt, g)
    else:
        Pc, Pt = fit_predict_rf(Xtr, ytr, [Xca, Xte], cfg["rft"], si)
        F_ca, F_te = rf_cdf(Pc, g), rf_cdf(Pt, g)
    return F_ca, F_te, yca, yte, g


def one(ds, model, si, tag, bw, lam_over):
    f = ACACHE / f"{ds}__{model}__{si}__{tag}.json"
    if f.exists():
        return json.loads(f.read_text())
    t0 = time.perf_counter()
    F_ca, F_te, yca, yte, g = seed_data(ds, model, si)
    if tag == "rawbase":
        F = np.maximum.accumulate(np.clip(F_te, 0, 1), axis=1)
        lam = 0.0
    else:
        W_ck, lam, _ = ckme_recalibrate(F_ca, F_te, yca, g, bw_scale=bw,
                                        lam_override=lam_over)
        F = np.maximum.accumulate(np.clip(emp_cdf_on_grid(yca, W_ck, g), 0, 1), axis=1)
    crps = float(crps_from_cdf(F, yte, g).mean())
    Fb = np.maximum.accumulate(np.clip(F_te, 0, 1), axis=1)
    crps_raw = float(crps_from_cdf(Fb, yte, g).mean())
    rng2 = np.random.default_rng(50_000 + 97 * si)
    est, stat, p = skce_test(masses_from_cdf(F), F, yte, g, rng2)
    out = {"skce": est, "stat": stat, "pval": p, "accept": bool(p >= 0.05),
           "crps_norm": crps / (crps_raw + 1e-300), "lam": float(lam),
           "elapsed_s": time.perf_counter() - t0}
    f.write_text(json.dumps(out))
    print(f"  {ds}/{model} s{si} {tag}: skce={est:.3e} p={p:.3f} "
          f"({out['elapsed_s']:.1f}s)", flush=True)
    return out


def tasks():
    for ds, model in CELLS3:
        for si in range(N_SEEDS):
            yield ds, model, si, "rawbase", 1.0, None
            for c in BW:
                yield ds, model, si, f"bw{c}", c, None
            for lv in LAMS:
                yield ds, model, si, f"lam{lv}", 1.0, lv


def main():
    budget = None
    if len(sys.argv) > 1 and sys.argv[1] == "chunk":
        budget = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    t0 = time.perf_counter()
    done = True
    for ds, model, si, tag, c, lv in tasks():
        f = ACACHE / f"{ds}__{model}__{si}__{tag}.json"
        if f.exists():
            continue
        if budget is not None and time.perf_counter() - t0 > budget:
            print("BUDGET reached", flush=True)
            done = False
            break
        one(ds, model, si, tag, c, lv)
    if not done:
        sys.exit(0)
    # aggregate
    res = {}
    print("\n=== Ablation: CKME SKCE/acceptance vs bandwidth scale and lambda ===")
    for ds, model in CELLS3:
        res[f"{ds}__{model}"] = {}
        print(f"-- {ds}/{model} (mean over {N_SEEDS} seeds; raw baseline first) --")
        for tag in (["rawbase"] + [f"bw{c}" for c in BW] + [f"lam{lv}" for lv in LAMS]):
            rows = [json.loads((ACACHE / f"{ds}__{model}__{si}__{tag}.json").read_text())
                    for si in range(N_SEEDS)]
            e = {k: float(np.mean([r[k] for r in rows]))
                 for k in ["skce", "pval", "accept", "crps_norm", "lam"]}
            res[f"{ds}__{model}"][tag] = e
            print(f"  {tag:<9} skce={e['skce']:9.3e}  p={e['pval']:.3f}  "
                  f"acc={e['accept']:.2f}  crps_n={e['crps_norm']:.3f}  "
                  f"lam={e['lam']:.4g}")
    (HERE / "results_ablate.json").write_text(json.dumps(res, indent=2))
    print("wrote results_ablate.json")


if __name__ == "__main__":
    main()
