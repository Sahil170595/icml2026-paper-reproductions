#!/usr/bin/env python3
r"""
c1fix_run.py [dataset] [model] | chunk <budget_s>   -- resumable; parts in _parts/,
cells in cells_c1fix/<dataset>__<model>.json.

CLAIM-1 FIX RUN: the judge scored C1 INCONCLUSIVE because the predeclared rules
U1/U2 failed (CKME suite acceptance 0.435 < raw 0.55; CKME lowered mean SKCE on
5/20 cells).  Diagnosis (each item evidenced in the C1 page):
  R1  The judged Study-A run used an ad-hoc exact conditional-MC null for the
      SKCE test instead of the paper's actual CalibrationTests.jl
      AsymptoticSKCETest bootstrap null; the MC null is far more powerful
      against the n_cal-atom empirical output of CKME (paper Eq. 22) and
      inverted the Figure-1 ranking.  Fixed (verbatim port); archived run in
      ../uci/cells_mcnull.
  R2  The judged suite was the 5 SMALLEST UCI datasets (test sizes 31-160) --
      exactly the "extremely small sample sizes" the paper EXCLUDES from its
      Figure-1 claim ("there was generally sufficient evidence to reject ...
      across most datasets (excluding those with extremely small sample
      sizes)").  There the raw models are accepted ~0.69 of the time (nothing
      to correct) while CKME pays a mechanical discreteness cost.
  R3  The supporting real-data study used a Nadaraya-Watson stand-in for CKME
      (conditioned on the predicted mean, fixed blend 0.2) -- NOT the paper's
      algorithm.
THIS RUN executes the verdict-mandated 5-real-dataset x 2-model suite
(california, wine_red, concrete, energy, diabetes x GBM/RF; test sets up to
1500 points, so the SKCE test has real power) with a FAITHFUL implementation
of the paper's CKME, verified line-by-line against the official repository
(github.com/adamgnuj/recalibration_experiment @ 12b4a203, files fetched
2026-07-18):
  * kernel over predictive distributions (paper Eq. 19+23, EDK):
    D_ij = sqrt(max(ED_ij,0)) with ED the pairwise energy-distance matrix from
    the MAE identity (ReCalibration.jl `_di`); median heuristic on the
    CALIBRATION block (`median_distance(...[.!tm,.!tm])`); k = Laplace(0,
    median).pdf(D)  (`Q_kernel_distr = repr.kernel * median_preds;
    pdf(Q_kernel_distr, ...)`).
  * observation kernel: Laplace(0, median|y_i-y_j|).pdf (ReCalibration.jl
    `kernelmatrix(MH::MedianHeuristic, calib_obs)`).
  * lambda: 5-fold CV masks (`_draw_cv_masks`, seed 42), per-fold eigendecomp,
    embedding loss with per-column euclidean simplex projection
    (lambda_cross_validation.jl `_error_per_cv`), Brent scalar optimisation on
    [0, 10000] (`optimize(l -> ..., 0.0, 10_000)`).
  * beta = (K_cc + lambda*n_c*I)^-1 k_c(Q) (run_recalibration `_rK \ ...`),
    then per-column euclidean simplex projection; recalibrated prediction =
    sum_i w_i delta_{y_cal_i}  (paper Eq. 20-22).
  * SKCE auto-calibration test: SKCETest.jl EmpiricalSKCETest tensor kernel
    (Kq on the raw energy-distance matrix -- NOT its sqrt -- with its own
    median heuristic, exactly as the official file does) handed to
    CalibrationTests.jl v0.6.3 AsymptoticSKCETest: statistic =
    n/(n-1)*SKCE_uq - SKCE_b, p by bootstrap_ccdf (1000 iters), accept iff
    p >= 0.05.
Baselines: Kuleshov 2018 (marginal PIT/ecdf map) and Song 2019 (Beta
distribution-calibration family, MLE on calibration PITs), as in the judged
runs.  Base models (real, trained per seed): GBM = heteroscedastic Gradient
Boosting -> N(mu(x), sigma(x)^2); RF = kernel-smoothed Random-Forest
predictive (DRF-style).  Deviations kept from the judged bundle: predictive
CDFs handled on a 512-point grid (D4); numpy RNG bit-streams differ from
Julia's (D5/D6).

PREDECLARED RULES (same U1/U2 as the judged verdict, fixed before this run):
  U1: CKME has the highest suite-mean SKCE-test acceptance fraction
      (alpha=5%) of {raw, Kuleshov'18, Song'19, CKME} over the 10 cells.
  U2: CKME lowers the mean unbiased SKCE estimate (SKCE_uq, the quantity the
      paper reports) vs raw on >= 70% of cells (>=7/10).  The official test
      statistic version is also reported as auxiliary.
Every number is measured from executed code; nothing is copied from the paper.
Deterministic seeds; OMP_NUM_THREADS=1.
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
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
CELLS = HERE / "cells_c1fix"
CELLS.mkdir(exist_ok=True)
PARTS = HERE / "_parts"
PARTS.mkdir(exist_ok=True)

GRID_T = 512
SKCE_BOOT = 1000
DATASETS = ["diabetes", "energy", "concrete", "wine_red", "california"]
MODELS = ["gbm", "rf"]
CFG = {
    # california reduced vs Study B (ntr 5000->3000, ncal 1500->800, ntest
    # 1500->1000, seeds 8->6) so one seed fits the <45 s per-call execution
    # budget of this environment; still by far the largest-n cell of the suite.
    "california": dict(nsub=8000, ntr=3000, ncal=800, ntest=1000, seeds=6, gbt=100, rft=80),
    "wine_red":   dict(nsub=1599, ntr=999,  ncal=300,  ntest=300,  seeds=10, gbt=200, rft=180),
    "concrete":   dict(nsub=1030, ntr=630,  ncal=200,  ntest=200,  seeds=10, gbt=200, rft=180),
    "energy":     dict(nsub=768,  ntr=468,  ncal=150,  ntest=150,  seeds=10, gbt=200, rft=180),
    "diabetes":   dict(nsub=442,  ntr=250,  ncal=96,   ntest=96,   seeds=10, gbt=200, rft=180),
}


# --------------------------- data (same sources as judged Study B) ----------
def load_dataset(name):
    from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
    if name == "california":
        d = fetch_california_housing()
        return np.asarray(d.data, float), np.asarray(d.target, float)
    if name == "diabetes":
        d = load_diabetes()
        return np.asarray(d.data, float), np.asarray(d.target, float)
    if name == "concrete":
        d = fetch_openml("Concrete_Data", version=1, as_frame=True)
        c = list(d.frame.columns)
        return d.frame[c[:-1]].to_numpy(float), d.frame[c[-1]].to_numpy(float)
    if name == "energy":
        d = fetch_openml("energy-efficiency", version=1, as_frame=True)
        y = d.target.to_numpy() if hasattr(d.target, "to_numpy") else d.target
        return d.data.select_dtypes("number").to_numpy(float), np.asarray(y, float)
    if name == "wine_red":
        d = fetch_openml("wine-quality-red", version=1, as_frame=True)
        return d.data.select_dtypes("number").to_numpy(float), np.asarray(d.target, float)
    raise ValueError(name)


# --------------------------- base models (as judged Study B) ----------------
def fit_predict_gbm(Xtr, ytr, Xq_list, trees, seed):
    from sklearn.ensemble import GradientBoostingRegressor
    mean = GradientBoostingRegressor(n_estimators=trees, max_depth=3, learning_rate=0.05,
                                     random_state=seed, subsample=1.0).fit(Xtr, ytr)
    res = ytr - mean.predict(Xtr)
    logv = GradientBoostingRegressor(n_estimators=max(trees - 50, 100), max_depth=3,
                                     learning_rate=0.05, random_state=seed + 1
                                     ).fit(Xtr, np.log(res ** 2 + 1e-6))
    return [(mean.predict(Xq), np.maximum(np.sqrt(np.exp(logv.predict(Xq))), 1e-2))
            for Xq in Xq_list]


def fit_predict_rf(Xtr, ytr, Xq_list, trees, seed):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=1,
                               min_samples_leaf=3).fit(Xtr, ytr)
    return [np.stack([e.predict(Xq) for e in rf.estimators_], axis=1) for Xq in Xq_list]


def gbm_cdf(mu, sd, g):
    return np.clip(ndtr((g[None, :] - mu[:, None]) / sd[:, None]), 0.0, 1.0)


def rf_cdf(P, g, y=None):
    """Kernel-smoothed RF predictive CDF on the grid; if y given also return
    the exact PIT values F_i(y_i)."""
    T = P.shape[1]
    sd = 0.9 * P.std(axis=1) * T ** (-1.0 / 5.0)
    sd = np.maximum(sd, 0.05 * (np.median(np.abs(P - np.median(P, axis=1, keepdims=True)),
                                          axis=1) + 1e-6) + 1e-3)
    F = np.zeros((P.shape[0], g.size))
    chunk = 64
    for a in range(0, P.shape[0], chunk):
        z = (g[None, None, :] - P[a:a + chunk, :, None]) / sd[a:a + chunk, None, None]
        F[a:a + chunk] = ndtr(z).mean(axis=1)
    if y is None:
        return np.clip(F, 0.0, 1.0)
    Z = ndtr((y[:, None] - P) / sd[:, None]).mean(axis=1)
    return np.clip(F, 0.0, 1.0), np.clip(Z, 0.0, 1.0)


# --------------------------- grid helpers (verbatim from ../uci) ------------
def make_grid(y_all):
    lo, hi = float(np.min(y_all)), float(np.max(y_all))
    span = hi - lo + 1e-12
    return np.linspace(lo - 0.75 * span, hi + 0.75 * span, GRID_T)


def masses_from_cdf(F):
    Wg = np.diff(F, axis=1, prepend=0.0)
    Wg = np.maximum(Wg, 0.0)
    s = Wg.sum(1, keepdims=True)
    return Wg / np.maximum(s, 1e-300)


def pairwise_mae_from_cdf(F, h):
    u = F.sum(1) * h
    M = (F @ F.T) * h
    return u[:, None] + u[None, :] - 2.0 * M


def crps_from_cdf(F, y, g):
    h = g[1] - g[0]
    step = (y[:, None] <= g[None, :]).astype(float)
    return ((F - step) ** 2).sum(1) * h


def cdf_at_points_gauss(mu, sd, y):
    return np.clip(ndtr((y - mu) / sd), 0.0, 1.0)


def emp_cdf_on_grid(atoms, W, g):
    order = np.argsort(atoms)
    aa, Ws = atoms[order], W[:, order]
    cs = np.cumsum(Ws, axis=1)
    idx = np.searchsorted(aa, g, side="right")
    F = np.concatenate([np.zeros((W.shape[0], 1)), cs], axis=1)[:, idx]
    return np.clip(F, 0.0, 1.0)


# --------------------------- kernels & CKME (faithful; see header) ----------
def laplace_pdf(d, b):
    return np.exp(-np.abs(d) / b) / (2.0 * b)


def median_uppertri(D):
    iu = np.triu_indices(D.shape[0], k=1)
    return float(np.median(D[iu]))


def simplex_proj_cols(V):
    n = V.shape[0]
    M = -np.sort(-V, axis=0)
    cs = np.cumsum(M, axis=0)
    R = M - (cs - 1.0) / np.arange(1, n + 1)[:, None]
    rho = np.maximum((R > 0).cumsum(0).max(0), 1)
    theta = (cs[rho - 1, np.arange(V.shape[1])] - 1.0) / rho
    return np.maximum(V - theta[None, :], 0.0)


def recal_pit(Z_cal):
    zs = np.sort(Z_cal)
    return lambda F: np.searchsorted(zs, F, side="right") / zs.size


def recal_beta(Z_cal):
    z = np.clip(Z_cal, 1e-4, 1 - 1e-4)
    a, b, _, _ = beta_dist.fit(z, floc=0, fscale=1)
    return (lambda F: beta_dist.cdf(np.clip(F, 0.0, 1.0), a, b)), (float(a), float(b))


def ckme_recalibrate(F_cal, F_test, y_cal, g, bw_scale=1.0, lam_override=None):
    """Official CKME (ReCalibration.jl + lambda_cross_validation.jl); returns
    (weights over y_cal atoms per test point, lambda, median bandwidths).
    bw_scale/lam_override are only used by the ablation (default = paper)."""
    h = g[1] - g[0]
    n_c = F_cal.shape[0]
    F_ct = np.vstack([F_cal, F_test])
    mae = pairwise_mae_from_cdf(F_ct, h)
    ed = mae - 0.5 * (np.diag(mae)[:, None] + np.diag(mae)[None, :])
    dist = np.sqrt(np.maximum(ed, 0.0))
    dist = 0.5 * (dist + dist.T)
    med_q = max(median_uppertri(dist[:n_c, :n_c]), 1e-12) * bw_scale
    Kq = laplace_pdf(dist, med_q)

    D_obs = np.abs(y_cal[:, None] - y_cal[None, :])
    med_o = max(median_uppertri(D_obs), 1e-12) * bw_scale
    L = laplace_pdf(D_obs, med_o)

    rng = np.random.default_rng(42)          # cv_mask_seed = 42 (official)
    n_cv = 5
    ms = np.concatenate([np.full(int(np.ceil(n_c / n_cv)), i + 1) for i in range(n_cv)])
    rng.shuffle(ms)
    masks = [(ms != i + 1)[:n_c] for i in range(n_cv)]
    Kcc = Kq[:n_c, :n_c]
    if lam_override is None:
        eigs = []
        for m in masks:
            vals, vecs = np.linalg.eigh(Kcc[np.ix_(m, m)])
            eigs.append((vals, vecs, m))

        def cv_err(lam):
            tot = 0.0
            for vals, vecs, m in eigs:
                nm = int(m.sum())
                B = vecs @ ((vecs.T @ Kcc[np.ix_(m, ~m)]) / (vals + lam * nm)[:, None])
                B = simplex_proj_cols(B)
                Lmm_B = L[np.ix_(m, m)] @ B
                obs_self = np.trace(L[np.ix_(~m, ~m)])
                cross = float(np.sum(B * L[np.ix_(m, ~m)]))
                pred_self = float(np.sum(B * Lmm_B))
                tot += (obs_self + pred_self - 2.0 * cross) / (~m).sum()
            return tot / n_cv

        lam = float(minimize_scalar(cv_err, bounds=(0.0, 10_000.0), method="bounded").x)
    else:
        lam = float(lam_override)
    B = np.linalg.solve(Kcc + lam * n_c * np.eye(n_c), Kq[:n_c, n_c:])
    B = simplex_proj_cols(B)
    return B.T, lam, (med_q, med_o)


# --------------------------- SKCE test (verbatim from ../uci) ---------------
def skce_test(Wg, F, y_test, g, rng, boot=SKCE_BOOT):
    h = g[1] - g[0]
    n = y_test.size
    D_obs = np.abs(y_test[:, None] - y_test[None, :])
    med_o = max(median_uppertri(D_obs), 1e-12)
    Ly = laplace_pdf(D_obs, med_o)

    mae = pairwise_mae_from_cdf(F, h)
    ed = mae - 0.5 * (np.diag(mae)[:, None] + np.diag(mae)[None, :])
    ed = 0.5 * (ed + ed.T)
    med_q = max(abs(median_uppertri(ed)), 1e-12)
    Kq = laplace_pdf(ed, med_q)

    Kg = laplace_pdf(g[:, None] - g[None, :], med_o)
    Lqy = Wg @ laplace_pdf(g[:, None] - y_test[None, :], med_o)
    A = Wg @ Kg
    Lqq = A @ Wg.T

    H = Kq * (Ly - Lqy - Lqy.T + Lqq)
    H = 0.5 * (H + H.T)
    hsum = float(H.sum())
    dsum = float(np.trace(H))
    est = (hsum - dsum) / (n * (n - 1))
    skce_b = hsum / (n * n)
    statistic = n / (n - 1) * est - skce_b
    alpha = n / (n - 1)
    C = rng.multinomial(n, np.full(n, 1.0 / n), size=boot).astype(np.float64)
    rows = H.sum(1)
    diagH = np.diag(H).copy()
    M = C @ H
    quad = np.einsum("bi,bi->b", M, C)
    T = (alpha * (quad - C @ diagH) - 2.0 * (C @ rows)) / (n * n)
    pval = float(np.mean(T >= statistic))
    return est, statistic, pval


# --------------------------- per-seed driver --------------------------------
METHODS = ["raw", "kuleshov", "song", "ckme"]


def run_seed(ds, model, si):
    part_f = PARTS / f"{ds}__{model}__{si}.json"
    if part_f.exists():
        return json.loads(part_f.read_text())
    t0 = time.perf_counter()
    cfg = CFG[ds]
    X, y = load_dataset(ds)
    rng = np.random.default_rng(7000 + 101 * si)     # same split RNG as Study B
    pool = rng.permutation(len(y))[:cfg["nsub"]]
    ntr, nca, nte = cfg["ntr"], cfg["ncal"], cfg["ntest"]
    itr = pool[:ntr]
    ica = pool[ntr:ntr + nca]
    ite = pool[ntr + nca:ntr + nca + nte]
    xm, xs = X[itr].mean(0), X[itr].std(0) + 1e-8
    ym, ys = y[itr].mean(), y[itr].std() + 1e-8
    Xtr, Xca, Xte = [(X[i] - xm) / xs for i in (itr, ica, ite)]
    ytr, yca, yte = [(y[i] - ym) / ys for i in (itr, ica, ite)]
    g = make_grid(np.concatenate([ytr, yca, yte]))

    if model == "gbm":
        (muc, sdc), (mut, sdt) = fit_predict_gbm(Xtr, ytr, [Xca, Xte], cfg["gbt"], si)
        F_ca, F_te = gbm_cdf(muc, sdc, g), gbm_cdf(mut, sdt, g)
        Z_cal = cdf_at_points_gauss(muc, sdc, yca)
    elif model == "rf":
        Pc, Pt = fit_predict_rf(Xtr, ytr, [Xca, Xte], cfg["rft"], si)
        (F_ca, Z_cal), F_te = rf_cdf(Pc, g, yca), rf_cdf(Pt, g)
    else:
        raise ValueError(model)

    pit_map = recal_pit(Z_cal)
    beta_map, ab = recal_beta(Z_cal)
    W_ck, lam, meds = ckme_recalibrate(F_ca, F_te, yca, g)
    F_ck = emp_cdf_on_grid(yca, W_ck, g)

    F_methods = {"raw": F_te, "kuleshov": pit_map(F_te), "song": beta_map(F_te),
                 "ckme": F_ck}
    res = {"sizes": [int(ntr), int(nca), int(nte)], "lam": lam,
           "med_q": meds[0], "med_o": meds[1], "beta_ab": list(ab), "methods": {}}
    crps_base = None
    for m in METHODS:
        F = np.clip(F_methods[m], 0.0, 1.0)
        F = np.maximum.accumulate(F, axis=1)
        crps = float(crps_from_cdf(F, yte, g).mean())
        if m == "raw":
            crps_base = crps
        Wg = masses_from_cdf(F)
        rng2 = np.random.default_rng(30_000 + 97 * si + 7 * METHODS.index(m))
        est, stat, p = skce_test(Wg, F, yte, g, rng2)
        res["methods"][m] = {"crps": crps, "crps_norm": crps / (crps_base + 1e-300),
                             "skce": est, "stat": stat, "pval": p,
                             "accept": bool(p >= 0.05)}
    res["elapsed_s"] = time.perf_counter() - t0
    tmp = part_f.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.rename(part_f)
    print(f"  part {ds}/{model} seed {si}: lam={lam:.4g} "
          f"pval[raw,ckme]=[{res['methods']['raw']['pval']:.3f},"
          f"{res['methods']['ckme']['pval']:.3f}] "
          f"skce[raw,ckme]=[{res['methods']['raw']['skce']:.3e},"
          f"{res['methods']['ckme']['skce']:.3e}] "
          f"({res['elapsed_s']:.1f}s)", flush=True)
    return res


def assemble_cell(ds, model):
    n_seeds = CFG[ds]["seeds"]
    parts = []
    for si in range(n_seeds):
        f = PARTS / f"{ds}__{model}__{si}.json"
        if not f.exists():
            return False
        parts.append(json.loads(f.read_text()))
    per_seed = {m: {k: [pt["methods"][m][k] for pt in parts]
                    for k in ["crps", "crps_norm", "skce", "stat", "pval", "accept"]}
                for m in METHODS}
    import scipy
    import sklearn
    out = {"dataset": ds, "model": model, "n_seeds": n_seeds, "methods": METHODS,
           "sizes_tr_ca_te": parts[0]["sizes"], "per_seed": per_seed,
           "ckme_lambda": [pt["lam"] for pt in parts],
           "med_q": [pt["med_q"] for pt in parts],
           "beta_ab": [pt["beta_ab"] for pt in parts],
           "grid_T": GRID_T, "skce_boot": SKCE_BOOT,
           "elapsed_s": sum(pt["elapsed_s"] for pt in parts),
           "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                   "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                   "threads": os.environ["OMP_NUM_THREADS"]}}
    (CELLS / f"{ds}__{model}.json").write_text(json.dumps(out, indent=2))
    print(f"ASSEMBLED {ds}/{model} ({out['elapsed_s']:.1f}s total)", flush=True)
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chunk":
        budget = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
        tasks = [(d, m, si) for d in DATASETS for m in MODELS
                 for si in range(CFG[d]["seeds"])]
        t0 = time.perf_counter()
        for d, m, si in tasks:
            if (CELLS / f"{d}__{m}.json").exists():
                continue
            if (PARTS / f"{d}__{m}__{si}.json").exists():
                continue
            if time.perf_counter() - t0 > budget:
                print("BUDGET reached", flush=True)
                break
            run_seed(d, m, si)
        else:
            print("ALL PARTS DONE", flush=True)
        for d in DATASETS:
            for m in MODELS:
                if not (CELLS / f"{d}__{m}.json").exists():
                    try:
                        assemble_cell(d, m)
                    except Exception:
                        pass
        missing = [(d, m) for d in DATASETS for m in MODELS
                   if not (CELLS / f"{d}__{m}.json").exists()]
        print("MISSING CELLS:", len(missing), missing, flush=True)
        sys.exit(0)
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    key2 = sys.argv[2] if len(sys.argv) > 2 else ""
    for d in DATASETS:
        for m in MODELS:
            if key not in ("", d, m) or key2 not in ("", d, m):
                continue
            if (CELLS / f"{d}__{m}.json").exists():
                print(f"skip {d}/{m}", flush=True)
                continue
            print(f"RUN {d}/{m}", flush=True)
            for si in range(CFG[d]["seeds"]):
                run_seed(d, m, si)
            assemble_cell(d, m)
    print("PRESENT:", sorted(p.name for p in CELLS.glob("*.json")), flush=True)
