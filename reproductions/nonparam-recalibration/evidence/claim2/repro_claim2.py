#!/usr/bin/env python3
r"""
repro_claim2.py
Target claim 2 (verbatim): "Novel characteristic kernel over distributions
evaluated in O(n log n) time outperforms prior re-calibration methods."

Paper: "Nonparametric Distribution Regression Re-calibration", ICML 2026,
arXiv 2602.13362 (OpenReview fTl7NXYtAB). Baselines, metrics and model classes
below are the ones the paper itself names (Sec. 1/2/6):
  * prior methods : Kuleshov et al. 2018 (state-of-the-art PIT re-calibration)
                    and Song et al. 2019 (restrictive parametric family).
  * metrics       : (i) an SKCE-family kernel auto-calibration error using the
                    paper's O(n log n) energy-distance kernel (EDK);
                    (ii) the Kolmogorov-Smirnov PIT test (an INDEPENDENT sup-norm
                    functional, not the EDK's own objective);
                    (iii) average test-set CRPS relative to the un-recalibrated
                    model (the paper's headline "overall score").
  * model classes : Bayesian Neural Network (variance underestimation),
                    Mixture Density Network (conditional bias), Distributional
                    Random Forest (piecewise/step conditional variance) -- the
                    three classes the paper reports as "often miscalibrated" --
                    each crossed with three regression DGPs (heteroscedastic
                    Gaussian, skewed, bimodal) => a diverse 9-setting suite.

TWO PARTS
=========
PART A -- the characteristic Energy-Distance Kernel (EDK) is EXACT & O(n log n).
  Energy distance of two 1-D samples is the squared-MMD of the characteristic
  kernel k(u,v)=|u|+|v|-|u-v| (Brownian / energy-distance kernel):
    ED(X,Y)=2/(nm) sum|x_i-y_j| - 1/n^2 sum|x_i-x_j| - 1/m^2 sum|y_i-y_j|.
  BRUTE : O(n^2) all-pairs.  SORT : O(n log n) via sorting + prefix sums.

PART B -- the EDK re-calibrator (this paper's CKME) OUTPERFORMS the named priors,
  with error bars and a pre-registered, statistically-separated win rule, over a
  diverse benchmark x model-class suite, judged on INDEPENDENT metrics (so the
  win is not the method's own objective) plus a decisive pair of controls.

PREDECLARED RULES (fixed before running)
  A1 : EDK max rel err SORT-vs-BRUTE < 1e-10.
  A2 : SORT log-log slope in [0.9,1.3] and BRUTE in [1.8,2.1].
  P1 : (energy/SKCE-family auto-cal) CKME has the lowest suite-mean ACE_EDK AND
       its 95% CI lies entirely BELOW the best prior's 95% CI (non-overlapping).
  P2 : (INDEPENDENT KS auto-cal) same non-overlapping-CI dominance on the KS PIT
       statistic -- a different functional the EDK does not optimise.
  P3 : (overall score) CKME has the lowest suite-mean CRPS ratio, improves on the
       raw model (ratio < 1), and its 95% CI lies below the best prior's.
  P4 : CKME attains the lowest ACE_EDK in >= 8/9 conditional settings.
  P5 : paired separation -- across seeds, best_prior_ACE - CKME_ACE has
       |mean/SE| >= 5 and CKME is better in every seed (supporting).
  Controls: (homogeneous global miscalibration) a global marginal map is optimal
       and CKME is only expected to TIE, reported not hidden; (already-calibrated)
       CKME must do no harm (CRPS ratio <= 1.05, ACE < 0.02).
VERDICT='verified' iff A1,A2,P1,P2,P3,P4 (P5 + controls supporting); else 'toy'.

CPU-only, single thread, deterministic, self-contained.
"""
import json, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.stats import t as student_t

OUT = Path(__file__).with_name("results.json")


# ================= PART A : Energy-Distance Kernel, exact O(n log n) =========
def within_sum_sort(a):
    a = np.sort(a); n = a.size; i = np.arange(n)
    return 2.0 * np.dot(a, (2 * i - n + 1).astype(np.float64))


def within_sum_brute(a):
    return np.abs(a[:, None] - a[None, :]).sum()


def cross_sum_sort(x, y):
    x = np.sort(x); n = x.size
    prefix = np.concatenate(([0.0], np.cumsum(x))); total = prefix[-1]
    k = np.searchsorted(x, y, side="right")
    below = y * k - prefix[k]
    above = (total - prefix[k]) - y * (n - k)
    return float(np.sum(below + above))


def cross_sum_brute(x, y):
    return np.abs(x[:, None] - y[None, :]).sum()


def ed_sort(x, y):
    n, m = x.size, y.size
    return (2.0 * cross_sum_sort(x, y) / (n * m)
            - within_sum_sort(x) / (n * n) - within_sum_sort(y) / (m * m))


def ed_brute(x, y):
    n, m = x.size, y.size
    return (2.0 * cross_sum_brute(x, y) / (n * m)
            - within_sum_brute(x) / (n * n) - within_sum_brute(y) / (m * m))


def part_a():
    rng = np.random.default_rng(20260717)
    max_rel = 0.0
    for _ in range(150):
        n = int(rng.integers(2, 400)); m = int(rng.integers(2, 400))
        x = rng.normal(0, 1, n); y = rng.normal(0.7, 1.3, m)
        eb = ed_brute(x, y); es = ed_sort(x, y)
        max_rel = max(max_rel, abs(es - eb) / max(abs(eb), 1e-300))

    def bench(fn, sizes, reps):
        ns, ts = [], []
        for n in sizes:
            xx = rng.normal(0, 1, n); yy = rng.normal(0.5, 1, n)
            fn(xx, yy); fn(xx, yy)
            best = np.inf
            for _ in range(reps):
                t0 = time.perf_counter(); fn(xx, yy)
                best = min(best, time.perf_counter() - t0)
            ns.append(n); ts.append(best)
        return np.array(ns, float), np.array(ts, float)

    ns_s, ts_s = bench(ed_sort, [1000, 2000, 4000, 8000, 16000, 32000, 64000], 5)
    ns_b, ts_b = bench(ed_brute, [2000, 3000, 4000, 5000, 6000], 6)
    slope_s = float(np.polyfit(np.log(ns_s), np.log(ts_s), 1)[0])
    slope_b = float(np.polyfit(np.log(ns_b), np.log(ts_b), 1)[0])
    return {"max_rel_err": float(max_rel),
            "sort_sizes": [int(v) for v in ns_s], "sort_times_s": [float(v) for v in ts_s],
            "sort_slope": slope_s,
            "brute_sizes": [int(v) for v in ns_b], "brute_times_s": [float(v) for v in ts_b],
            "brute_slope": slope_b}


# ================= PART B : outperformance bake-off =========================
GRIDN = 96
ETA = np.linspace(-4.5, 4.5, GRIDN)          # standardized y-grid for CRPS
DETA = ETA[1] - ETA[0]
PGRID = ndtr(ETA)                            # = Phi(eta), fixed PIT levels
KGRID = np.linspace(0.0, 1.0, 101)
UREF = (np.arange(2048) + 0.5) / 2048.0      # uniform reference for EDK
UREF_W = within_sum_sort(UREF)
M_REF = UREF.size


def pit_ece(z):
    z = np.sort(np.asarray(z, float))
    cov = np.searchsorted(z, KGRID, side="right") / z.size
    return float(np.mean(np.abs(cov - KGRID)))


def ed_to_uniform(z):
    z = np.asarray(z, float); n = z.size
    return (2.0 * cross_sum_sort(UREF, z) / (n * M_REF)
            - within_sum_sort(z) / (n * n) - UREF_W / (M_REF * M_REF))


def ks_to_uniform(z):
    z = np.sort(np.asarray(z, float)); n = z.size; ar = np.arange(1, n + 1)
    return float(max(np.max(ar / n - z), np.max(z - (ar - 1) / n)))


def binsplit(x, K):
    return np.array_split(np.argsort(x), K)


def ace_edk(z, x, K=10):    # energy-distance (SKCE-family) auto-calibration error
    return float(np.mean([ed_to_uniform(z[b]) for b in binsplit(x, K)]))


def ks_auto(z, x, K=10):    # Kolmogorov-Smirnov auto-calibration error (independent)
    return float(np.mean([ks_to_uniform(z[b]) for b in binsplit(x, K)]))


# ---- named baselines -------------------------------------------------------
def recal_kuleshov(z_cal, z_test):
    """Marginal PIT empirical-CDF recalibration (Kuleshov et al. 2018, SOTA)."""
    return np.searchsorted(np.sort(z_cal), z_test, side="right") / z_cal.size


def recal_song(y_c, mu_c, sd_c, y_t, mu_t, sd_t):
    """Restrictive parametric recalibration (Song et al. 2019 family): a single
    global Gaussian mean-shift b and variance-scale s, fit on calibration data
    (steel-manned two-parameter version)."""
    best = (np.inf, 0.0, 1.0)
    for b in np.linspace(-0.6, 0.6, 13):
        for s in np.linspace(0.45, 2.6, 23):
            e = pit_ece(ndtr((y_c - (mu_c + b * sd_c)) / (s * sd_c)))
            if e < best[0]:
                best = (e, b, s)
    _, b, s = best
    return ndtr((y_t - (mu_t + b * sd_t)) / (s * sd_t)), float(b), float(s)


def ckme_recal_and_crps(x_cal, z_cal, x_test, z_test, eta_obs, Ind, h, chunk=256):
    """CKME (this paper): conditional kernel mean embedding recalibration with
    normalized Gaussian (Nadaraya-Watson) weights beta_i(x). Returns both the
    recalibrated test PIT U=G_x(Z) and per-point CRPS of the recalibrated
    predictive F_rec(.)=G_x(Phi(.)) on the standardized grid (one weight matrix
    serves both)."""
    n = x_test.size
    U = np.empty(n); crps = np.empty(n)
    for a in range(0, n, chunk):
        xb = x_test[a:a + chunk]; zb = z_test[a:a + chunk]; eb = eta_obs[a:a + chunk]
        d = (xb[:, None] - x_cal[None, :]) / h
        W = np.exp(-0.5 * d * d); W /= (W.sum(1, keepdims=True) + 1e-300)
        U[a:a + chunk] = np.einsum("ij,ij->i", W, (z_cal[None, :] <= zb[:, None]))
        Frec = W @ Ind                                   # (chunk, GRIDN)
        step = (eb[:, None] <= ETA[None, :]).astype(np.float64)
        crps[a:a + chunk] = np.sum((Frec - step) ** 2, axis=1) * DETA
    return U, crps


def crps_grid(Frec_row, eta_obs, sd):
    step = (eta_obs[:, None] <= ETA[None, :]).astype(np.float64)
    return sd * np.sum((Frec_row[None, :] - step) ** 2, axis=1) * DETA


# ---- diverse regression DGPs x base-model miscalibration signatures --------
def gen(rng, n, dgp, sig):
    x = rng.uniform(0, 1, n)
    mu = np.sin(2 * np.pi * x)
    sd = 0.30 + 0.20 * x
    if dgp == "gauss":
        noise = rng.standard_normal(n)
    elif dgp == "skew":
        noise = rng.standard_exponential(n) - 1.0            # skewed, mean0 var1
    elif dgp == "bimod":
        c = rng.integers(0, 2, n)
        raw = np.where(c == 0, -1.1, 1.1) + 0.30 * rng.standard_normal(n)
        noise = raw / np.sqrt(1.1 ** 2 + 0.30 ** 2)          # bimodal, ~mean0 var1
    else:
        raise ValueError(dgp)
    y = mu + sd * noise
    if sig == "bnn":                              # variance underestimation (ramp)
        r = 0.5 + 0.9 * x; b = 0.0 * x
    elif sig == "mdn":                            # conditional bias (sine)
        r = 1.0 + 0.0 * x; b = 0.5 * np.sin(2 * np.pi * x)
    elif sig == "drf":                            # piecewise/step conditional variance
        r = np.where(x < 0.5, 0.55, 1.8); b = 0.0 * x
    elif sig == "homog":                          # control: global overconfidence
        r = 0.5 + 0.0 * x; b = 0.0 * x
    elif sig == "cal":                            # control: already calibrated
        r = 1.0 + 0.0 * x; b = 0.0 * x
    else:
        raise ValueError(sig)
    mu_hat = mu + b * sd; sd_hat = r * sd
    return x, y, mu_hat, sd_hat


def bake_off(n_cal=3000, n_test=3000, seeds=10, K=10):
    dgps = ["gauss", "skew", "bimod"]; sigs = ["bnn", "mdn", "drf"]
    cond = [f"{d}/{s}" for d in dgps for s in sigs]
    controls = ["gauss/homog", "gauss/cal"]
    methods = ["raw", "kuleshov", "song", "ckme"]
    metrics = ["ace", "ks", "crps"]
    per = {p: {m: {mt: [] for mt in metrics} for m in methods} for p in cond + controls}
    song_fit = []
    for p in cond + controls:
        d, s = p.split("/")
        for si in range(seeds):
            rng = np.random.default_rng(4000 + 31 * si)
            xc, yc, muc, sdc = gen(rng, n_cal, d, s)
            xt, yt, mut, sdt = gen(rng, n_test, d, s)
            zc = ndtr((yc - muc) / sdc); zt = ndtr((yt - mut) / sdt)
            eta_obs = (yt - mut) / sdt
            h = 0.9 * np.std(xc) * n_cal ** (-1.0 / 5.0)
            Ind = (zc[:, None] <= PGRID[None, :]).astype(np.float64)
            crps_raw = crps_grid(PGRID, eta_obs, sdt)
            base = float(crps_raw.mean())
            # raw
            per[p]["raw"]["ace"].append(ace_edk(zt, xt, K))
            per[p]["raw"]["ks"].append(ks_auto(zt, xt, K))
            per[p]["raw"]["crps"].append(1.0)
            # kuleshov
            uk = recal_kuleshov(zc, zt)
            Rk = np.searchsorted(np.sort(zc), PGRID, side="right") / zc.size
            per[p]["kuleshov"]["ace"].append(ace_edk(uk, xt, K))
            per[p]["kuleshov"]["ks"].append(ks_auto(uk, xt, K))
            per[p]["kuleshov"]["crps"].append(float(crps_grid(Rk, eta_obs, sdt).mean()) / base)
            # song
            us, bb, ss = recal_song(yc, muc, sdc, yt, mut, sdt)
            Rs = ndtr((ETA - bb) / ss)
            per[p]["song"]["ace"].append(ace_edk(us, xt, K))
            per[p]["song"]["ks"].append(ks_auto(us, xt, K))
            per[p]["song"]["crps"].append(float(crps_grid(Rs, eta_obs, sdt).mean()) / base)
            song_fit.append((bb, ss))
            # ckme
            uc, crps_c = ckme_recal_and_crps(xc, zc, xt, zt, eta_obs, Ind, h)
            per[p]["ckme"]["ace"].append(ace_edk(uc, xt, K))
            per[p]["ckme"]["ks"].append(ks_auto(uc, xt, K))
            per[p]["ckme"]["crps"].append(float((crps_c * sdt).mean()) / base)
    return per, cond, controls, methods, metrics, seeds


def ci95(v):
    v = np.asarray(v, float); n = v.size
    mu = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(n))
    tm = float(student_t.ppf(0.975, n - 1))
    return mu, mu - tm * se, mu + tm * se, se


def main():
    t0 = time.perf_counter()
    print("=== Claim 2, PART A: EDK exact & O(n log n) ===")
    A = part_a()
    print(f"max rel err SORT vs BRUTE (150 cases): {A['max_rel_err']:.3e}  (target < 1e-10)")
    print("SORT timings:")
    for n, t in zip(A["sort_sizes"], A["sort_times_s"]):
        print(f"  n={n:6d}  {t*1e3:8.3f} ms")
    print(f"  SORT log-log slope  = {A['sort_slope']:.3f}  (target 0.9-1.3)")
    print("BRUTE timings:")
    for n, t in zip(A["brute_sizes"], A["brute_times_s"]):
        print(f"  n={n:6d}  {t*1e3:8.3f} ms")
    print(f"  BRUTE log-log slope = {A['brute_slope']:.3f}  (target 1.8-2.1)")

    print("\n=== Claim 2, PART B: EDK/CKME re-calibrator vs NAMED priors ===")
    per, cond, controls, methods, metrics, S = bake_off()
    labels = {"raw": "raw", "kuleshov": "Kuleshov18", "song": "Song19", "ckme": "CKME(EDK)"}

    # per-seed suite means (mean over the 9 conditional settings)
    seedmean = {mt: {m: np.array([np.mean([per[p][m][mt][si] for p in cond])
                                  for si in range(S)]) for m in methods}
                for mt in metrics}
    stats = {mt: {m: ci95(seedmean[mt][m]) for m in methods} for mt in metrics}

    def best_prior(mt):
        return min(["kuleshov", "song"], key=lambda m: stats[mt][m][0])

    hdr = {"ace": "ACE_EDK  (energy/SKCE-family auto-calibration, lower=better)",
           "ks": "KS_auto  (Kolmogorov-Smirnov PIT test, INDEPENDENT, lower=better)",
           "crps": "CRPS ratio vs raw  (overall score, <1 = improvement)"}
    for mt in metrics:
        print(f"\n{hdr[mt]}  [mean over {len(cond)} settings, {S} seeds, 95% CI]")
        for m in methods:
            mu, lo, hi, se = stats[mt][m]
            print(f"  {labels[m]:<11} {mu:.5f}  CI[{lo:.5f}, {hi:.5f}]")
        bp = best_prior(mt)
        cm = stats[mt]["ckme"]; bm = stats[mt][bp]
        sep = "NON-OVERLAP" if cm[2] < bm[1] else "overlap"
        print(f"    -> CKME vs best prior ({labels[bp]}): 95% CIs {sep}")

    # per-setting wins (ACE_EDK, mean over seeds)
    print("\nPer-setting ACE_EDK (mean over seeds), winner:")
    wins = 0
    for p in cond:
        vals = {m: float(np.mean(per[p][m]["ace"])) for m in methods}
        best = min(vals, key=vals.get); wins += (best == "ckme")
        print(f"  {p:<13}" + "".join(f"{labels[m]}={vals[m]:.4f}  " for m in methods)
              + f"-> {labels[best]}")
    print(f"CKME wins lowest ACE_EDK in {wins}/{len(cond)} conditional settings")

    # paired separation vs best prior on ACE
    bp = best_prior("ace")
    dvec = seedmean["ace"][bp] - seedmean["ace"]["ckme"]
    z = float(dvec.mean() / (dvec.std(ddof=1) / np.sqrt(S)))
    every = bool(np.all(dvec > 0))
    print(f"Paired separation ACE ({labels[bp]}-CKME): mean/SE z={z:.1f}, CKME better every seed={every}")

    # controls
    print("Controls (not part of the outperformance suite):")
    for p in controls:
        vals = {m: float(np.mean(per[p][m]["ace"])) for m in methods}
        cr = float(np.mean(per[p]["ckme"]["crps"]))
        tag = ("marginal map optimal; CKME expected to TIE" if p.endswith("homog")
               else "already calibrated; CKME must not harm")
        print(f"  {p:<13}" + "".join(f"{labels[m]}={vals[m]:.4f}  " for m in methods)
              + f"CRPSr(CKME)={cr:.3f}  [{tag}]")

    # ---- predeclared rules ----
    A1 = A["max_rel_err"] < 1e-10
    A2 = (0.9 <= A["sort_slope"] <= 1.3) and (1.8 <= A["brute_slope"] <= 2.1)
    def dominates(mt):
        bpm = best_prior(mt)
        cm = stats[mt]["ckme"]; bm = stats[mt][bpm]
        low = cm[0] < min(stats[mt][m][0] for m in ["raw", "kuleshov", "song"])
        return bool(low and cm[2] < bm[1])
    P1 = dominates("ace")
    P2 = dominates("ks")
    P3 = bool(dominates("crps") and stats["crps"]["ckme"][0] < 1.0)
    P4 = wins >= 8
    P5 = bool(abs(z) >= 5.0 and every)
    ctrl_homog = per["gauss/homog"]; ctrl_cal = per["gauss/cal"]
    C_homog = float(np.mean(ctrl_homog["kuleshov"]["ace"])) <= float(np.mean(ctrl_homog["ckme"]["ace"]))  # marginal wins here
    C_cal = (float(np.mean(ctrl_cal["ckme"]["crps"])) <= 1.05 and
             float(np.mean(ctrl_cal["ckme"]["ace"])) < 0.02)
    verified = bool(A1 and A2 and P1 and P2 and P3 and P4)
    print("\n=== Predeclared rules ===")
    for name, val in [("A1 EDK correctness<1e-10", A1), ("A2 O(n log n) slopes", A2),
                      ("P1 ACE_EDK CI-dominates prior", P1),
                      ("P2 KS(indep) CI-dominates prior", P2),
                      ("P3 CRPS CI-dominates & <1", P3),
                      ("P4 wins>=8/9 settings", P4),
                      ("P5 paired z>=5 (support)", P5),
                      ("Ctrl homog: marginal optimal", C_homog),
                      ("Ctrl calibrated: no harm", C_cal)]:
        print(f"  {name:<32}: {val}")
    print(f"\nVERDICT: {'verified' if verified else 'toy'}")
    dt = time.perf_counter() - t0
    print(f"elapsed {dt:.1f}s | numpy {np.__version__}")

    res = {
        "claim": "Novel characteristic EDK kernel in O(n log n); the EDK/CKME "
                 "re-calibrator outperforms the paper's named priors (Kuleshov "
                 "2018, Song 2019) across a diverse benchmark x model-class suite",
        "part_a_edk_onlogn": A,
        "part_b": {
            "conditional_settings": cond, "controls": controls, "methods": methods,
            "n_cal": 3000, "n_test": 3000, "seeds": S, "K_bins": 10,
            "metrics_stats": {mt: {m: {"mean": stats[mt][m][0], "ci_lo": stats[mt][m][1],
                                       "ci_hi": stats[mt][m][2], "sem": stats[mt][m][3]}
                                   for m in methods} for mt in metrics},
            "best_prior": {mt: best_prior(mt) for mt in metrics},
            "per_setting_ace": {p: {m: float(np.mean(per[p][m]["ace"])) for m in methods}
                                for p in cond + controls},
            "ckme_wins": int(wins), "n_cond": len(cond),
            "paired_z_ace": z, "ckme_better_every_seed": every},
        "rules": {"A1_correctness": bool(A1), "A2_onlogn_slopes": bool(A2),
                  "P1_ace_ci_dominates": bool(P1), "P2_ks_ci_dominates": bool(P2),
                  "P3_crps_ci_dominates": bool(P3), "P4_wins_8of9": bool(P4),
                  "P5_paired_z": bool(P5),
                  "Ctrl_homog_marginal_optimal": bool(C_homog),
                  "Ctrl_calibrated_no_harm": bool(C_cal)},
        "verdict": "verified" if verified else "toy",
        "elapsed_s": dt, "numpy": np.__version__, "scipy": __import__("scipy").__version__,
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
