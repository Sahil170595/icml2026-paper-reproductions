#!/usr/bin/env python3
r"""
repro_claim1.py
Target claim 1: "Nonparametric re-calibration corrects calibration error without
restrictive parametric assumptions via conditional kernel mean embeddings (CKME)."

Paper: "Nonparametric Distribution Regression Re-calibration", ICML 2026,
arXiv 2602.13362 (OpenReview fTl7NXYtAB).

DESIGN (deterministic CPU, numpy/scipy, fixed seeds)
----------------------------------------------------
Simulate miscalibrated distributional forecasts for a heteroscedastic
regression problem  Y | X = x , then apply four post-hoc recalibrators:
  none   : raw (miscalibrated) forecast.
  margin : marginal PIT recalibration = empirical-CDF map R(u)=P_hat(Z<=u)
           (Kuleshov et al. 2018 style; a PRIOR method).
  param  : parametric recalibration = single global Gaussian variance scale s,
           forecast N(mu,(s*sigma)^2), s fit to minimise PIT error
           (a restrictive PARAMETRIC assumption; Song et al. 2019 family).
  ckme   : the paper's mechanism = CONDITIONAL kernel mean embedding recalib.
           Nadaraya-Watson weights beta_i(x)=K((x-x_i)/h)/sum_j K(...) are the
           normalised CKME weights (Eq.11 with a Gaussian kernel), giving a
           conditional empirical CDF of the PIT:
             G_x(u) = sum_i beta_i(x) 1{ Z_i <= u }.

Forecast CDF F_x(y)=Phi((y-mu_hat)/sigma_hat); PIT Z=F_x(Y).
Forecast is PIT-calibrated iff Z~Uniform(0,1) (marginal, weak);
AUTO-calibrated iff Z~Uniform WITHIN every conditioning region of X (strong).

METRICS
  pit_ece(z) : mean_p | P_hat(Z<=p) - p |  over a grid of levels p (marginal).
  cce(z,x)   : mean over K equal-frequency X-bins of pit_ece within the bin
               = auto-calibration proxy (conditional calibration error).
  map_err    : L1 gap between the estimated CKME conditional recalibration map
               G_x0 and the CLOSED-FORM true map G*_x0 (no test-sampling floor,
               exposes the genuine nonparametric convergence rate).

SCENARIOS: calibrated (control), overconf (global), cancel (over/under on the
two halves of X so marginal error cancels), biashetero (opposite bias on the
halves), shape (skewed truth vs a Gaussian forecast -> a variance scale can
never fix the shape).

PREDECLARED ACCEPTANCE RULES (fixed before running)
  C1 correction    : every miscalibrated scenario, CKME cuts CCE vs 'none'
                     by >= 2x   (cce_ckme <= 0.5 * cce_none).
  C2 nonparametric : on shape-mismatch AND cancellation, CKME beats the
                     parametric variance-scaler  (cce_ckme < cce_param).
  C3 marginal-insuf: on error-cancellation the marginal PIT map barely helps
                     (cce_margin >= 0.7*cce_none) while CKME cuts it
                     (cce_ckme <= 0.5*cce_none)  -> conditioning needed.
  C4 consistency   : recalibration error decays with n.
                     (a) marginal empirical-CDF map -> DKW sqrt(n) rate:
                         log-log slope in [-0.65,-0.35];
                     (b) CKME conditional map error -> clear nonparametric
                         decay: log-log slope <= -0.20.
  C5 control       : on an already-calibrated forecaster CKME does no real harm
                     (cce_ckme <= 1.5*cce_none and both < 0.03).
VERDICT = 'verified' iff C1,C2,C4 all hold (C3,C5 supporting); else 'toy'.

CPU-only, single thread, self-contained.
"""
import json, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri  # normal CDF and its inverse (vectorised)

OUT = Path(__file__).with_name("results.json")
GRID = np.linspace(0.0, 1.0, 101)
UGRID = np.linspace(0.01, 0.99, 99)   # for map-error (avoid +-inf at 0/1)
K_BINS = 10


def pit_ece(z):
    z = np.sort(np.asarray(z, float))
    cov = np.searchsorted(z, GRID, side="right") / z.size
    return float(np.mean(np.abs(cov - GRID)))


def cce(z, x, K=K_BINS):
    order = np.argsort(x)
    vals = [pit_ece(z[b]) for b in np.array_split(order, K)]
    return float(np.mean(vals))


def recal_marginal(z_cal, z_test):
    zc = np.sort(z_cal)
    return np.searchsorted(zc, z_test, side="right") / zc.size


def recal_param(y_c, mu_c, sd_c, y_t, mu_t, sd_t):
    best_e, best_s = np.inf, 1.0
    for s in np.linspace(0.30, 3.0, 55):
        e = pit_ece(ndtr((y_c - mu_c) / (s * sd_c)))
        if e < best_e:
            best_e, best_s = e, s
    return ndtr((y_t - mu_t) / (best_s * sd_t))


def recal_ckme(x_cal, z_cal, x_test, z_test, h, chunk=512):
    x_cal = np.asarray(x_cal, float); z_cal = np.asarray(z_cal, float)
    out = np.empty(x_test.size, float)
    for a in range(0, x_test.size, chunk):
        xb = x_test[a:a + chunk]; zb = z_test[a:a + chunk]
        d = (xb[:, None] - x_cal[None, :]) / h
        W = np.exp(-0.5 * d * d)
        W /= (W.sum(axis=1, keepdims=True) + 1e-300)
        ind = (z_cal[None, :] <= zb[:, None])
        out[a:a + chunk] = np.einsum("ij,ij->i", W, ind)
    return out


def ckme_map(x_cal, z_cal, x0, ugrid, h):
    """Estimated conditional recalibration map G_x0(u) at a query point x0."""
    w = np.exp(-0.5 * ((x0 - x_cal) / h) ** 2)
    w /= (w.sum() + 1e-300)
    return (w[None, :] * (z_cal[None, :] <= ugrid[:, None])).sum(axis=1)


def bandwidth(n, x):
    return 0.9 * np.std(x) * n ** (-1.0 / 5.0)


def gen(rng, n, scenario):
    x = rng.uniform(0.0, 1.0, n)
    mu = np.sin(2 * np.pi * x)
    sd_true = 0.30 + 0.20 * x
    if scenario == "shape":
        # skewed truth (standardised exponential, mean0/var1, skew>0); a Gaussian
        # forecast with the correct mean and variance is only SHAPE-wrong, which
        # no global variance scale can repair.
        y = mu + sd_true * (rng.standard_exponential(n) - 1.0)
    else:
        y = mu + sd_true * rng.standard_normal(n)
    if scenario == "calibrated":
        mu_hat, sd_hat = mu.copy(), sd_true.copy()
    elif scenario == "overconf":
        mu_hat, sd_hat = mu.copy(), 0.5 * sd_true
    elif scenario == "cancel":
        sd_hat = np.where(x < 0.5, 0.5 * sd_true, 1.9 * sd_true)
        mu_hat = mu.copy()
    elif scenario == "biashetero":
        mu_hat = mu + np.where(x < 0.5, 0.45, -0.45)
        sd_hat = sd_true.copy()
    elif scenario == "shape":
        mu_hat, sd_hat = mu.copy(), sd_true.copy()
    else:
        raise ValueError(scenario)
    return x, y, mu_hat, sd_hat


def pit_of(y, mu, sd):
    return ndtr((y - mu) / sd)


def true_cancel_map(x0, u):
    """Closed-form conditional recalibration map for the 'cancel' scenario.
    x<0.5: sd_hat=0.5 sd_true -> Z=Phi(2 eta) -> G*(u)=Phi(Phi^{-1}(u)/2).
    x>0.5: sd_hat=1.9 sd_true -> Z=Phi(eta/1.9) -> G*(u)=Phi(1.9 Phi^{-1}(u))."""
    zi = ndtri(u)
    return ndtr(zi / 2.0) if x0 < 0.5 else ndtr(1.9 * zi)


def head_to_head(n_cal=2500, n_test=2000, seeds=5):
    scenarios = ["calibrated", "overconf", "cancel", "biashetero", "shape"]
    methods = ["none", "margin", "param", "ckme"]
    acc = {sc: {m: [] for m in methods} for sc in scenarios}
    marg = {sc: {m: [] for m in methods} for sc in scenarios}
    for sc in scenarios:
        for s in range(seeds):
            rng = np.random.default_rng(1000 + 7 * s)
            xc, yc, muc, sdc = gen(rng, n_cal, sc)
            xt, yt, mut, sdt = gen(rng, n_test, sc)
            zc = pit_of(yc, muc, sdc); zt = pit_of(yt, mut, sdt)
            h = bandwidth(n_cal, xc)
            rec = {
                "none":   zt,
                "margin": recal_marginal(zc, zt),
                "param":  recal_param(yc, muc, sdc, yt, mut, sdt),
                "ckme":   recal_ckme(xc, zc, xt, zt, h),
            }
            for m in methods:
                acc[sc][m].append(cce(rec[m], xt))
                marg[sc][m].append(pit_ece(rec[m]))
    cce_mean = {sc: {m: float(np.mean(acc[sc][m])) for m in methods} for sc in scenarios}
    marg_mean = {sc: {m: float(np.mean(marg[sc][m])) for m in methods} for sc in scenarios}
    return cce_mean, marg_mean, scenarios, methods


def rate_marginal(n_grid, n_eval=150000, seeds=5):
    errs = []
    for n in n_grid:
        e = []
        for s in range(seeds):
            rng = np.random.default_rng(5000 + 11 * s + n)
            xc, yc, muc, sdc = gen(rng, n, "overconf")
            xt, yt, mut, sdt = gen(rng, n_eval, "overconf")
            zc = pit_of(yc, muc, sdc); zt = pit_of(yt, mut, sdt)
            e.append(pit_ece(recal_marginal(zc, zt)))
        errs.append(float(np.mean(e)))
    slope = float(np.polyfit(np.log(n_grid), np.log(errs), 1)[0])
    return list(map(float, errs)), slope


def rate_ckme_map(n_grid, seeds=6):
    """L1 error of the estimated CKME conditional recalibration map vs the
    closed-form true map at query points x0=0.25 and 0.75 (cancel scenario).
    No test-sampling floor -> reveals the genuine nonparametric rate."""
    x0s = [0.25, 0.75]
    gstar = {x0: true_cancel_map(x0, UGRID) for x0 in x0s}
    errs = []
    for n in n_grid:
        e = []
        for s in range(seeds):
            rng = np.random.default_rng(9000 + 13 * s + n)
            xc, yc, muc, sdc = gen(rng, n, "cancel")
            zc = pit_of(yc, muc, sdc)
            h = bandwidth(n, xc)
            for x0 in x0s:
                g = ckme_map(xc, zc, x0, UGRID, h)
                e.append(float(np.mean(np.abs(g - gstar[x0]))))
        errs.append(float(np.mean(e)))
    slope = float(np.polyfit(np.log(n_grid), np.log(errs), 1)[0])
    return list(map(float, errs)), slope


def main():
    t0 = time.perf_counter()
    print("=== Claim 1: nonparametric CKME recalibration corrects calibration error ===\n")

    cce_m, marg_m, scenarios, methods = head_to_head()
    print("Conditional calibration error (auto-cal proxy), mean over 5 seeds:")
    print(f"{'scenario':<12}" + "".join(f"{m:>10}" for m in methods))
    for sc in scenarios:
        print(f"{sc:<12}" + "".join(f"{cce_m[sc][m]:>10.4f}" for m in methods))
    print("\nMarginal PIT error (same runs):")
    print(f"{'scenario':<12}" + "".join(f"{m:>10}" for m in methods))
    for sc in scenarios:
        print(f"{sc:<12}" + "".join(f"{marg_m[sc][m]:>10.4f}" for m in methods))

    n_grid_m = [250, 500, 1000, 2000, 4000]
    err_m, slope_m = rate_marginal(n_grid_m)
    n_grid_c = [500, 1000, 2000, 4000, 8000, 16000]
    err_c, slope_c = rate_ckme_map(n_grid_c)
    print("\n=== Consistency / rate ===")
    print("(a) marginal empirical-CDF recal, marginal PIT error vs n:")
    for n, e in zip(n_grid_m, err_m):
        print(f"    n={n:6d}  err={e:.5f}")
    print(f"    log-log slope = {slope_m:.3f}  (DKW target -0.5, band [-0.65,-0.35])")
    print("(b) CKME conditional map error vs true map vs n (cancel):")
    for n, e in zip(n_grid_c, err_c):
        print(f"    n={n:6d}  map_err={e:.5f}")
    print(f"    log-log slope = {slope_c:.3f}  (nonparametric ~ n^-2/5, target <= -0.20)")

    mis = ["overconf", "cancel", "biashetero", "shape"]
    C1 = all(cce_m[sc]["ckme"] <= 0.5 * cce_m[sc]["none"] for sc in mis)
    C2 = all(cce_m[sc]["ckme"] < cce_m[sc]["param"] for sc in ["shape", "cancel"])
    C3 = (cce_m["cancel"]["margin"] >= 0.7 * cce_m["cancel"]["none"] and
          cce_m["cancel"]["ckme"] <= 0.5 * cce_m["cancel"]["none"])
    C4a = (-0.65 <= slope_m <= -0.35)
    C4b = (slope_c <= -0.20)
    C4 = C4a and C4b
    C5 = (cce_m["calibrated"]["ckme"] <= 1.5 * cce_m["calibrated"]["none"] and
          cce_m["calibrated"]["ckme"] < 0.03 and cce_m["calibrated"]["none"] < 0.03)
    verified = bool(C1 and C2 and C4)
    print("\n=== Predeclared rules ===")
    for name, val in [("C1 correction>=2x", C1), ("C2 beats parametric", C2),
                      ("C3 marginal insufficient", C3),
                      ("C4a marginal DKW rate", C4a), ("C4b CKME map rate", C4b),
                      ("C5 control (no harm)", C5)]:
        print(f"  {name:<26}: {val}")
    print(f"\nVERDICT: {'verified' if verified else 'toy'}")
    dt = time.perf_counter() - t0
    print(f"elapsed {dt:.1f}s | numpy {np.__version__}")

    res = {
        "claim": "Nonparametric CKME re-calibration corrects calibration error "
                 "without restrictive parametric assumptions",
        "cce_mean": cce_m, "marginal_pit_error_mean": marg_m,
        "scenarios": scenarios, "methods": methods,
        "rate_marginal": {"n": n_grid_m, "err": err_m, "loglog_slope": slope_m,
                          "target_band": [-0.65, -0.35], "law": "DKW sqrt(n)"},
        "rate_ckme_map": {"n": n_grid_c, "map_err": err_c, "loglog_slope": slope_c,
                          "target": "<= -0.20", "law": "nonparametric ~ n^-2/5"},
        "rules": {"C1_correction": bool(C1), "C2_beats_parametric": bool(C2),
                  "C3_marginal_insufficient": bool(C3),
                  "C4a_marginal_DKW": bool(C4a), "C4b_ckme_map_rate": bool(C4b),
                  "C5_control_no_harm": bool(C5)},
        "verdict": "verified" if verified else "toy",
        "config": {"n_cal_h2h": 2500, "n_test_h2h": 2000, "seeds": 5,
                   "K_bins": K_BINS, "kernel": "Gaussian NW (CKME Eq.11)"},
        "elapsed_s": dt, "numpy": np.__version__,
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
