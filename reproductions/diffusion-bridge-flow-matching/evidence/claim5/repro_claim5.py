"""
Claim 5 - Flow Matching (FM) degrades more steeply than Diffusion Bridge (DB)
when training data size is reduced (Fig 3b / Fig 5 / Tables 5-7 of
arXiv 2509.24531 / OpenReview aIFgQusnPy).  Mechanism: Remark 4.3 / Appendix A.3
- finite empirical measures violate Brenier absolute continuity, so FM's linear
(t,1-t) OT interpolation "loses validity" at small n; DB's drift/reference process
is data-size robust.

(a) FACTUAL: paper FID vs training data size (Box128 inpainting, Tables 5+6).
(b) TOY MECHANISM (exact, CPU, SCALED UP v2): Gaussian OT N(0,I)->N(m,Sigma) in R^d.
    Estimate the TRUE OT displacement midpoint g*(x)=1/2 x + 1/2 T(x) from n training
    pairs two ways:
      FM  = nonparametric EMPIRICAL OT-plan plug-in (discrete OT via linear
            assignment + 1-NN barycentric map)  -> curse-of-dim rate ~ n^{-1/d};
      DB  = parametric reference/drift plug-in (Gaussian moments -> matrix sqrt)
            -> rate ~ n^{-1/2}, robust to small n.
    Measure held-out E||g_hat(x) - g*(x)|| vs n. FM must degrade more steeply as n
    shrinks (larger error, shallower/curse-limited recovery), DB stays low.

v2 scale-up: n grid widened to 25..1000 (was 50..1000; low end extended 50->25 to match
the real-data experiment's floor), seeds increased 3 -> 8 for a
proper 95% CI (t-distribution) on the per-seed log-log degradation slope, and the
FM-vs-DB slope difference is now tested for statistical separation (paired per-seed
95% CI + one-sided Wilcoxon signed-rank across the 8 seeds). This is the SAME exact
closed-form mechanism as v1 (Remark 4.3's absolute-continuity/curse-of-dimensionality
argument), just with more data points on the degradation curve and enough seeds to
claim statistical, not just qualitative, separation of the two rates.
"""
import json, numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as tdist, wilcoxon

# ---------- (a) paper FID vs data size, Box128 inpainting ----------
data_size = np.array([500, 1000, 5000, 27000], float)
FID_FM = np.array([45.81, 37.23, 17.87, 17.84])
FID_DB = np.array([11.34,  9.43,  8.59,  7.71])
deg_FM = FID_FM[0]/FID_FM[-1]      # 500 vs 27000
deg_DB = FID_DB[0]/FID_DB[-1]
inc_FM = FID_FM[0]-FID_FM[-1]
inc_DB = FID_DB[0]-FID_DB[-1]
slope_FM = float(np.polyfit(np.log10(data_size), np.log10(FID_FM), 1)[0])
slope_DB = float(np.polyfit(np.log10(data_size), np.log10(FID_DB), 1)[0])

# ---------- (b) toy OT-map estimation vs n (SCALED UP) ----------
def sqrtm_sym(S):
    w, V = np.linalg.eigh(0.5*(S+S.T))
    w = np.clip(w, 1e-9, None)
    return (V*np.sqrt(w)) @ V.T

def ci95(arr):
    arr = np.asarray(arr, float)
    n = len(arr)
    if n < 2:
        return float(arr.mean()), 0.0
    m = arr.mean(); se = arr.std(ddof=1) / np.sqrt(n)
    hw = tdist.ppf(0.975, df=n - 1) * se
    return float(m), float(hw)

def run_toy(d=6, ns=(25, 50, 100, 200, 500, 1000), ntest=2000, seeds=range(8)):
    rng0 = np.random.default_rng(7)
    Q = rng0.standard_normal((d, d)); Sigma = Q @ Q.T / d + 0.5 * np.eye(d)
    m = rng0.standard_normal(d) * 1.2
    A = sqrtm_sym(Sigma)
    def g_true(X): return 0.5 * X + 0.5 * (m + X @ A.T)
    Xtest = rng0.standard_normal((ntest, d))
    Gte = g_true(Xtest)
    errFM = {n: [] for n in ns}; errDB = {n: [] for n in ns}
    for sd in seeds:
        rng = np.random.default_rng(100 + sd)
        for n in ns:
            Xs = rng.standard_normal((n, d))
            Ys = m + (rng.standard_normal((n, d))) @ A.T
            C = ((Xs[:, None, :] - Ys[None, :, :]) ** 2).sum(-1)
            ri, ci = linear_sum_assignment(C)
            Ymap = Ys[ci]
            d2 = ((Xtest[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
            nn = np.argmin(d2, axis=1)
            Tn = Ymap[nn]
            gFM = 0.5 * Xtest + 0.5 * Tn
            errFM[n].append(float(np.mean(np.linalg.norm(gFM - Gte, axis=1))))
            mhat = Ys.mean(0); Shat = np.cov(Ys.T, bias=True)
            Ahat = sqrtm_sym(Shat)
            gDB = 0.5 * Xtest + 0.5 * (mhat + Xtest @ Ahat.T)
            errDB[n].append(float(np.mean(np.linalg.norm(gDB - Gte, axis=1))))
    eFM = {n: float(np.mean(v)) for n, v in errFM.items()}
    eDB = {n: float(np.mean(v)) for n, v in errDB.items()}
    ns_a = np.array(ns, float)
    sFM = float(np.polyfit(np.log10(ns_a), np.log10([eFM[n] for n in ns]), 1)[0])
    sDB = float(np.polyfit(np.log10(ns_a), np.log10([eDB[n] for n in ns]), 1)[0])
    # per-seed slopes -> CI + paired separation test (the "degradation-RATE" statistic)
    seeds = list(seeds)
    fm_slopes, db_slopes = [], []
    for si in range(len(seeds)):
        fm_e = np.array([errFM[n][si] for n in ns])
        db_e = np.array([errDB[n][si] for n in ns])
        fm_slopes.append(float(np.polyfit(np.log10(ns_a), np.log10(fm_e), 1)[0]))
        db_slopes.append(float(np.polyfit(np.log10(ns_a), np.log10(db_e), 1)[0]))
    fm_slopes = np.array(fm_slopes); db_slopes = np.array(db_slopes)
    diffs = fm_slopes - db_slopes  # want < 0: FM slope MORE NEGATIVE (steeper) than DB slope
    fm_m, fm_hw = ci95(fm_slopes)
    db_m, db_hw = ci95(db_slopes)
    diff_m, diff_hw = ci95(diffs)
    separated = (diff_m + diff_hw) < 0
    wstat, wp = wilcoxon(diffs, alternative="less")
    return dict(d=d, ns=list(ns), eFM=eFM, eDB=eDB, sFM=sFM, sDB=sDB,
                fm_slope_mean=fm_m, fm_slope_ci95_hw=fm_hw,
                db_slope_mean=db_m, db_slope_ci95_hw=db_hw,
                slope_diff_mean=diff_m, slope_diff_ci95_hw=diff_hw,
                slope_separated_95ci=bool(separated),
                wilcoxon_stat=float(wstat), wilcoxon_p_onesided=float(wp),
                n_seeds=len(seeds), fm_slopes=fm_slopes.tolist(), db_slopes=db_slopes.tolist())

toy = run_toy()
d, ns, eFM, eDB, sFM, sDB = toy["d"], toy["ns"], toy["eFM"], toy["eDB"], toy["sFM"], toy["sDB"]

print("="*80)
print("Claim 5  -  FM degrades more steeply than DB as training data shrinks")
print("arXiv 2509.24531 / OpenReview aIFgQusnPy")
print("="*80)
print("(a) PAPER FID vs training data size (Box128 inpainting, Tables 5+6):")
print(f"{'data':>8} {'FID_FM':>8} {'FID_DB':>8}")
for i in range(len(data_size)):
    print(f"{data_size[i]:8.0f} {FID_FM[i]:8.2f} {FID_DB[i]:8.2f}")
print(f"  degradation 500 vs 27000:  FM x{deg_FM:.2f} (+{inc_FM:.1f} FID)   "
      f"DB x{deg_DB:.2f} (+{inc_DB:.1f} FID)")
print(f"  FM absolute degradation is {inc_FM/inc_DB:.1f}x that of DB")
print(f"  loglog slope FID vs data size:  FM {slope_FM:.3f}  DB {slope_DB:.3f}  "
      f"(FM steeper: {slope_FM < slope_DB})")
print()
print(f"(b) TOY OT-map estimation (SCALED v2), N(0,I)->N(m,Sigma) in R^{d}; "
      f"n = {ns}; {toy['n_seeds']} seeds; held-out ||g_hat-g*||:")
print(f"{'n':>7} {'FM err':>10} {'DB err':>10} {'FM/DB':>8}")
for n in ns:
    print(f"{n:7d} {eFM[n]:10.4f} {eDB[n]:10.4f} {eFM[n]/eDB[n]:8.2f}")
print(f"  loglog slope of MEAN error vs n:  FM {sFM:.3f}   DB {sDB:.3f}")
print(f"  per-seed slope (95% CI, {toy['n_seeds']} seeds): "
      f"FM {toy['fm_slope_mean']:.3f}+/-{toy['fm_slope_ci95_hw']:.3f}   "
      f"DB {toy['db_slope_mean']:.3f}+/-{toy['db_slope_ci95_hw']:.3f}")
print(f"  paired slope diff (FM-DB): {toy['slope_diff_mean']:.3f}+/-{toy['slope_diff_ci95_hw']:.3f}"
      f"  -> 95% CI upper bound {toy['slope_diff_mean']+toy['slope_diff_ci95_hw']:.3f}"
      f" {'< 0 (STATISTICALLY SEPARATED)' if toy['slope_separated_95ci'] else '>= 0 (not separated)'}")
print(f"  one-sided Wilcoxon signed-rank (FM slope < DB slope): "
      f"W={toy['wilcoxon_stat']:.2f}, p={toy['wilcoxon_p_onesided']:.2e}")
print(f"  FM error at n={ns[0]} is {eFM[ns[0]]/eDB[ns[0]]:.1f}x DB; DB's parametric")
print(f"    moment-plug-in estimator converges at the classical n^-1/2 rate while FM's")
print(f"    nonparametric empirical-OT-map estimator is curse-of-dimensionality limited.")
print()
paper_ok = (slope_FM < slope_DB) and (inc_FM > 3*inc_DB)
toy_ok = (eFM[ns[0]] > 1.5*eDB[ns[0]]) and (eFM[ns[0]] > eFM[ns[-1]])
print(f"VERDICT paper numbers show FM steeper degradation: {paper_ok}")
print(f"VERDICT toy mechanism (FM worse & curse-limited at small n): {toy_ok}")
print(f"VERDICT toy degradation-RATE statistically separated (95% CI, {toy['n_seeds']} seeds): "
      f"{toy['slope_separated_95ci']}")
print("SCOPE: (a) paper's reported FID (not CPU-trainable); (b) exact toy of the")
print("       Brenier/absolute-continuity mechanism (Remark 4.3), not a CelebA run.")
print("="*80)

with open("results.json","w") as f:
    json.dump(dict(paper=dict(data_size=data_size.tolist(),
                     FID_FM=FID_FM.tolist(), FID_DB=FID_DB.tolist(),
                     deg_ratio_FM=deg_FM, deg_ratio_DB=deg_DB,
                     abs_increase_FM=inc_FM, abs_increase_DB=inc_DB,
                     slope_FM=slope_FM, slope_DB=slope_DB, FM_steeper=bool(paper_ok)),
                   toy=dict(d=d, ns=ns, err_FM=eFM, err_DB=eDB,
                     slope_FM=sFM, slope_DB=sDB, toy_ok=bool(toy_ok),
                     n_seeds=toy['n_seeds'],
                     fm_slope_mean=toy['fm_slope_mean'], fm_slope_ci95_hw=toy['fm_slope_ci95_hw'],
                     db_slope_mean=toy['db_slope_mean'], db_slope_ci95_hw=toy['db_slope_ci95_hw'],
                     slope_diff_mean=toy['slope_diff_mean'], slope_diff_ci95_hw=toy['slope_diff_ci95_hw'],
                     slope_separated_95ci=toy['slope_separated_95ci'],
                     wilcoxon_stat=toy['wilcoxon_stat'], wilcoxon_p_onesided=toy['wilcoxon_p_onesided']),
                   verdict=bool(paper_ok and toy_ok and toy['slope_separated_95ci'])), f, indent=2)
print("wrote results.json")
