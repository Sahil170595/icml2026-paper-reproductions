#!/usr/bin/env python3
r"""
power_analysis.py prep <seed> | chunk <budget_s> | aggregate

CLINCHER EXPERIMENT for the C1 fix (explains, not just asserts, the U1
failure).  Root cause 2 on the C1 page states that the paper's own UCI
protocol test sizes (31-160) sit inside a no-power regime for the SKCE
auto-calibration test -- this script MEASURES that directly instead of
arguing it.

Design: reuse the flagship california/gbm cell (the one cell where CKME's
own recalibrated output IS ever accepted, and where CKME cuts SKCE 15.2x).
For 3 seeds, train the GBM base model once (ntr=3000, ncal=800) and build
raw / Kuleshov / CKME predictive CDFs on a held-out TEST POOL of 2000 points
(disjoint from train+calib).  The lambda-CV step of the faithful CKME
(imported verbatim from c1fix_run.py, same file, same functions) depends
only on the calibration block, so CKME weights for the whole pool are
computed once per seed and then SUBSAMPLED -- exactly consistent with the
per-seed CKME already reported on the C1 page, just re-used at many test
sizes.  For each of test sizes {30,50,100,200,350,600,1000} (spanning the
paper's own small-sample regime up to the fix-suite's largest cell) and 6
random subsamples per size per seed, we re-run the SAME verbatim-ported
SKCE AsymptoticSKCETest (bootstrap null, boot=150 here for speed; c1fix
main run uses 1000 -- boot count only affects Monte-Carlo p-value noise,
not the kernel/statistic definition, and is documented as a deviation)
and record the accept/reject decision per method.

Deterministic: split RNG 7000+101*seed (same scheme as c1fix_run.py),
subsample RNG 555000 + 1000*seed + 10*size_idx + rep, SKCE-bootstrap RNG
777000 + ... .  OMP_NUM_THREADS=1.  Resumable: per-seed cache in
_pa_cache/, per-seed results in _pa_parts/, aggregate in results_power.json.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import warnings

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1fix_run import (
    load_dataset, fit_predict_gbm, gbm_cdf, cdf_at_points_gauss, make_grid,
    ckme_recalibrate, emp_cdf_on_grid, masses_from_cdf, skce_test, recal_pit,
)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_pa_cache"
PARTS = HERE / "_pa_parts"
CACHE.mkdir(exist_ok=True)
PARTS.mkdir(exist_ok=True)

DATASET = "california"
MODEL_GBT = 100
NTR, NCAL, POOL = 3000, 800, 2000
SEEDS = 3
SIZES = [30, 50, 100, 200, 350, 600, 1000]
REPEATS = 6
BOOT = 150
METHODS = ["raw", "kuleshov", "ckme"]


def prep_seed(si):
    cache_f = CACHE / f"seed{si}.npz"
    if cache_f.exists():
        print(f"cache seed {si} exists", flush=True)
        return
    t0 = time.perf_counter()
    X, y = load_dataset(DATASET)
    rng = np.random.default_rng(7000 + 101 * si)   # same split RNG as c1fix_run.py
    pool_idx = rng.permutation(len(y))[:NTR + NCAL + POOL]
    itr = pool_idx[:NTR]
    ica = pool_idx[NTR:NTR + NCAL]
    ipo = pool_idx[NTR + NCAL:NTR + NCAL + POOL]
    xm, xs = X[itr].mean(0), X[itr].std(0) + 1e-8
    ym, ys = y[itr].mean(), y[itr].std() + 1e-8
    Xtr, Xca, Xpo = [(X[i] - xm) / xs for i in (itr, ica, ipo)]
    ytr, yca, ypo = [(y[i] - ym) / ys for i in (itr, ica, ipo)]
    g = make_grid(np.concatenate([ytr, yca, ypo]))

    (muc, sdc), (mup, sdp) = fit_predict_gbm(Xtr, ytr, [Xca, Xpo], MODEL_GBT, si)
    F_ca, F_po = gbm_cdf(muc, sdc, g), gbm_cdf(mup, sdp, g)
    Z_cal = cdf_at_points_gauss(muc, sdc, yca)

    pit_map = recal_pit(Z_cal)
    F_kul_po = np.clip(pit_map(F_po), 0.0, 1.0)

    W_ck, lam, meds = ckme_recalibrate(F_ca, F_po, yca, g)
    F_ck_po = emp_cdf_on_grid(yca, W_ck, g)

    def fix(F):
        F = np.clip(F, 0.0, 1.0)
        return np.maximum.accumulate(F, axis=1)

    np.savez_compressed(
        cache_f, g=g, y_pool=ypo,
        F_raw=fix(F_po), F_kuleshov=fix(F_kul_po), F_ckme=fix(F_ck_po),
        lam=lam,
    )
    print(f"PREP seed {si} done lam={lam:.4g} ({time.perf_counter()-t0:.1f}s)", flush=True)


def chunk(budget):
    t0 = time.perf_counter()
    for si in range(SEEDS):
        if not (CACHE / f"seed{si}.npz").exists():
            if time.perf_counter() - t0 > budget:
                print("BUDGET reached (prep)", flush=True)
                return
            prep_seed(si)

    part_f_of = lambda si: PARTS / f"seed{si}.json"
    for si in range(SEEDS):
        part_f = part_f_of(si)
        recs = json.loads(part_f.read_text()) if part_f.exists() else []
        done = {(r["size"], r["rep"], r["method"]) for r in recs}
        cache = np.load(CACHE / f"seed{si}.npz")
        g, y_pool = cache["g"], cache["y_pool"]
        F_by_method = {m: cache[f"F_{m}"] for m in METHODS}
        n_pool = y_pool.size
        changed = False
        for szi, size in enumerate(SIZES):
            for rep in range(REPEATS):
                sub_rng = np.random.default_rng(555_000 + 1000 * si + 10 * szi + rep)
                idx = sub_rng.choice(n_pool, size=size, replace=False)
                y_sub = y_pool[idx]
                for m in METHODS:
                    key = (size, rep, m)
                    if key in done:
                        continue
                    if time.perf_counter() - t0 > budget:
                        print("BUDGET reached (skce)", flush=True)
                        if changed:
                            part_f.write_text(json.dumps(recs))
                        return
                    F_sub = F_by_method[m][idx]
                    Wg = masses_from_cdf(F_sub)
                    rng2 = np.random.default_rng(777_000 + 1000 * si + 10 * szi + rep + hash(m) % 97)
                    est, stat, p = skce_test(Wg, F_sub, y_sub, g, rng2, boot=BOOT)
                    recs.append({"seed": si, "size": size, "rep": rep, "method": m,
                                 "skce": est, "stat": stat, "pval": p,
                                 "accept": bool(p >= 0.05)})
                    done.add(key)
                    changed = True
        if changed:
            part_f.write_text(json.dumps(recs))
            print(f"seed {si}: {len(recs)}/{len(SIZES)*REPEATS*len(METHODS)} records", flush=True)
    total = sum(len(json.loads(part_f_of(si).read_text())) if part_f_of(si).exists() else 0
                for si in range(SEEDS))
    want = SEEDS * len(SIZES) * REPEATS * len(METHODS)
    print(f"TOTAL RECORDS: {total}/{want}", flush=True)


def aggregate():
    all_recs = []
    for si in range(SEEDS):
        f = PARTS / f"seed{si}.json"
        if not f.exists():
            print(f"MISSING seed {si}", flush=True)
            return
        all_recs.extend(json.loads(f.read_text()))
    want = SEEDS * len(SIZES) * REPEATS * len(METHODS)
    if len(all_recs) != want:
        print(f"INCOMPLETE: {len(all_recs)}/{want} records -- rerun chunk", flush=True)
        return
    table = {}
    for size in SIZES:
        table[size] = {}
        for m in METHODS:
            rows = [r for r in all_recs if r["size"] == size and r["method"] == m]
            acc = float(np.mean([r["accept"] for r in rows]))
            skce = float(np.mean([r["skce"] for r in rows]))
            table[size][m] = {"accept_frac": acc, "skce_mean": skce, "n": len(rows)}
    out = {"dataset": DATASET, "model": "gbm", "seeds": SEEDS, "sizes": SIZES,
           "repeats": REPEATS, "boot": BOOT, "methods": METHODS, "table": table}
    (HERE / "results_power.json").write_text(json.dumps(out, indent=2))
    print("=== POWER ANALYSIS: acceptance fraction (alpha=5%) vs test-set size, california/gbm ===")
    print("(paper's own protocol tests at 31-160 points; fix suite mandated up to 1000)")
    hdr = "n_test".rjust(8) + "".join(m.rjust(14) for m in METHODS)
    print(hdr)
    for size in SIZES:
        row = str(size).rjust(8)
        for m in METHODS:
            row += f"{table[size][m]['accept_frac']:.3f}".rjust(14)
        print(row)
    print("\nmean SKCE (should fall with size = kernel MMD estimate converging, not power):")
    print(hdr)
    for size in SIZES:
        row = str(size).rjust(8)
        for m in METHODS:
            row += f"{table[size][m]['skce_mean']:.2e}".rjust(14)
        print(row)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prep":
        prep_seed(int(sys.argv[2]))
    elif cmd == "chunk":
        chunk(float(sys.argv[2]) if len(sys.argv) > 2 else 30.0)
    elif cmd == "aggregate":
        aggregate()
    else:
        print(__doc__)
