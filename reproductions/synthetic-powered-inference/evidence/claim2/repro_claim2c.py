"""
Claim 2 (application C) - GESPI CONFIDENCE-INTERVAL COVERAGE + WIDTH.
Decisive verification of the paper's CORE distribution-free guarantee (Theorem 3.2) on the
third error-control task in the paper's taxonomy (Table 1): predictive inference / MISCOVERAGE
control, i.e. confidence intervals - the "deterministic-V" case the paper flags explicitly
(the parameter of interest is V; footnote in Sec 3.3). Ground truth theta* is KNOWN, so we can
measure coverage exactly.

Why this is decisive (not a toy proxy): coverage and CI width are the textbook validity/efficiency
metrics for interval estimation; the DGP has a known target; and the base intervals are EXACT
Gaussian z-intervals, so OnlyReal coverage is exactly 1-alpha and any miscoverage of a naive method
is unambiguously the distribution-shift bias - not asymptotic slop. The distribution-free property
(holds for ANY synthetic law Q) is exercised by an adversarially shifted synthetic set.

Setup (paper's canonical simulation knobs, App. C):  real n=50 draws ~ N(theta*, 1), theta*=0;
synthetic N=500 draws ~ N(mu_s, 1).  alpha=5%, eps=2%, distribution-free bound = alpha+eps = 7%
miscoverage (=> coverage >= 1-(alpha+eps) = 93%).  matched: mu_s=0 (P=Q).  adversarial: mu_s=0.5.

Named methods (paper Sec 4): OnlyReal = z-interval on real only (base, exactly 1-alpha valid);
OnlySynth = z-interval on synthetic only (paper's named method WITHOUT guarantees);
NaivePooled = z-interval pooling real+synthetic as if one sample (the naive-pooling control, no
guardrail); GESPI = paper's Eq.2 confidence-set rule
   C_GESPI = C_real(alpha)  INTERSECT  ( C_pool(alpha)  UNION  C_real(alpha+eps) ).
theta* in C_GESPI  <=>  in C_real(alpha) AND ( in C_pool(alpha) OR in C_real(alpha+eps) ).
Guarantee (Thm 3.2 / 3.3): miscoverage <= alpha + min{eps, c*d_TV(P,Q)} <= alpha+eps for ANY Q,
and GESPI is sandwiched between the real alpha and alpha+eps intervals (=> never wider than OnlyReal).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, numpy as np
from scipy.stats import norm
from pathlib import Path

sigma, theta = 1.0, 0.0
n, N, alpha, eps = 50, 500, 0.05, 0.02
bound = alpha + eps                      # 0.07 distribution-free miscoverage bound
z_a  = norm.ppf(1 - alpha / 2)           # 1.95996
z_ae = norm.ppf(1 - (alpha + eps) / 2)   # 1.81191
se_r, se_p, se_s = sigma/np.sqrt(n), sigma/np.sqrt(n+N), sigma/np.sqrt(N)
T, SEED = 200000, 31415926
NAMES = ["OnlyReal", "OnlyReal_ae", "NaivePooled", "OnlySynth", "GESPI"]

def run(mu_s, rng):
    xr = rng.normal(theta, se_r, T)      # real sample mean ~ N(theta*, 1/n)
    xs = rng.normal(mu_s,  se_s, T)      # synthetic sample mean ~ N(mu_s, 1/N)
    xp = (n*xr + N*xs) / (n + N)          # pooled mean ~ N(N*mu_s/(n+N), 1/(n+N))
    lo_ra,  hi_ra  = xr - z_a*se_r,  xr + z_a*se_r     # real  @ alpha
    lo_rae, hi_rae = xr - z_ae*se_r, xr + z_ae*se_r    # real  @ alpha+eps (guardrail)
    lo_pa,  hi_pa  = xp - z_a*se_p,  xp + z_a*se_p     # pool  @ alpha
    lo_sa,  hi_sa  = xs - z_a*se_s,  xs + z_a*se_s     # synth @ alpha
    in_ra  = (lo_ra <= theta) & (theta <= hi_ra)
    in_rae = (lo_rae <= theta) & (theta <= hi_rae)
    in_pa  = (lo_pa <= theta) & (theta <= hi_pa)
    in_sa  = (lo_sa <= theta) & (theta <= hi_sa)
    in_g   = in_ra & (in_pa | in_rae)                 # GESPI Eq.2 coverage
    w_ra, w_rae, w_pa, w_sa = hi_ra-lo_ra, hi_rae-lo_rae, hi_pa-lo_pa, hi_sa-lo_sa
    # GESPI set = C_real(a+e) UNION (C_real(a) INTERSECT C_pool(a)); report total length
    inlo, inhi = np.maximum(lo_ra, lo_pa), np.minimum(hi_ra, hi_pa)
    inner_empty = inhi < inlo
    inw = np.where(inner_empty, 0.0, inhi - inlo)
    ov_lo, ov_hi = np.maximum(lo_rae, inlo), np.minimum(hi_rae, inhi)
    ov = np.where((~inner_empty) & (ov_hi > ov_lo), ov_hi - ov_lo, 0.0)
    w_g = w_rae + inw - ov
    def cov_stat(ind):
        c = float(ind.mean()); return c, float(np.sqrt(c*(1-c)/T))
    def wid_stat(w):
        return float(w.mean()), float(w.std()/np.sqrt(T))
    out = {}
    for nm, ind, w in [("OnlyReal",in_ra,w_ra), ("OnlyReal_ae",in_rae,w_rae),
                       ("NaivePooled",in_pa,w_pa), ("OnlySynth",in_sa,w_sa), ("GESPI",in_g,w_g)]:
        c, cse = cov_stat(ind); wm, wse = wid_stat(w)
        out[nm] = dict(coverage=c, coverage_se=cse, miscoverage=1-c,
                       width=wm, width_se=wse)
    return out

rng = np.random.default_rng(SEED)
print(f"GESPI Claim 2C | confidence-interval coverage+width | n={n} N={N} alpha={alpha} eps={eps} "
      f"bound(miscov)={bound} reps={T} seed={SEED}")
print(f"exact Gaussian z-intervals: OnlyReal coverage is exactly 1-alpha; theta*={theta} known.")
regimes = [(0.0, "matched(P=Q)"), (0.2, "mild-shift"), (0.5, "ADVERSARIAL")]
results = {}
for mu_s, tag in regimes:
    r = run(mu_s, rng); results[tag] = r
    print(f"\n== {tag}  mu_s={mu_s}  (nominal coverage 1-a={1-alpha}, guardrail 1-(a+e)={round(1-bound,2)}) ==")
    print(f"{'method':>12} {'coverage':>9} {'+-SE':>7} {'miscov':>7} {'<=a+e?':>7} {'width':>7} {'vsOnlyReal':>10}")
    wor = r["OnlyReal"]["width"]
    for nm in NAMES:
        d = r[nm]
        redu = 100*(1 - d["width"]/wor)
        print(f"{nm:>12} {d['coverage']:9.4f} {d['coverage_se']:7.4f} {d['miscoverage']:7.4f} "
              f"{str(d['miscoverage']<=bound+1e-9):>7} {d['width']:7.4f} {redu:9.2f}%")

# ---- pre-registered acceptance checks (fixed before the run) ----
print("\n--- Checks (pre-registered) ---")
mat, adv = results["matched(P=Q)"], results["ADVERSARIAL"]
mild = results["mild-shift"]
# 1) distribution-free validity: GESPI miscoverage <= alpha+eps in EVERY regime
gespi_df_valid = all(results[t]["GESPI"]["miscoverage"] <= bound + 2*results[t]["GESPI"]["coverage_se"]
                     for _, t in regimes)
# 2) informative: GESPI coverage close to nominal 1-alpha (miscoverage ~ alpha)
gespi_informative = mat["GESPI"]["coverage"] >= (1 - alpha) - 0.01
# 3) OnlyReal exact 1-alpha
onlyreal_exact = all(abs(results[t]["OnlyReal"]["coverage"] - (1-alpha)) <= 0.004 for _, t in regimes)
# 4) efficiency: GESPI strictly narrower than OnlyReal, statistically separated
gw, ow = mat["GESPI"]["width"], mat["OnlyReal"]["width"]
gwse, owse = mat["GESPI"]["width_se"], mat["OnlyReal"]["width_se"]
dse = float(np.hypot(gwse, owse)); gap = ow - gw
eff_sep = gap > 5*max(dse, 1e-12)
red_pct = 100*(1 - gw/ow)
# 5) decisive control: naive methods VALID when matched but VIOLATE (coverage crashes) under shift
naive_valid_matched = (mat["NaivePooled"]["coverage"] >= 0.94) and (mat["OnlySynth"]["coverage"] >= 0.94)
naive_breaks_adv = (adv["NaivePooled"]["coverage"] < 0.90) and (adv["OnlySynth"]["coverage"] < 0.90)
print(f"[validity] GESPI miscoverage <= alpha+eps={bound} in all regimes: {gespi_df_valid} "
      f"(matched={mat['GESPI']['miscoverage']:.4f}, mild={mild['GESPI']['miscoverage']:.4f}, adv={adv['GESPI']['miscoverage']:.4f})")
print(f"[validity] GESPI coverage ~ nominal 1-alpha when informative: {gespi_informative} "
      f"(matched={mat['GESPI']['coverage']:.4f})")
print(f"[validity] OnlyReal coverage exactly 1-alpha={1-alpha} (base is valid): {onlyreal_exact}")
print(f"[efficiency] GESPI CI narrower than OnlyReal, {red_pct:.2f}% reduction, gap {gap:.4f} = {gap/dse:.0f} SE: {eff_sep}")
print(f"[control] naive methods valid when matched (cov>=0.94): {naive_valid_matched} "
      f"(pool={mat['NaivePooled']['coverage']:.4f}, synth={mat['OnlySynth']['coverage']:.4f})")
print(f"[control] naive methods VIOLATE under shift (cov<0.90): {naive_breaks_adv} "
      f"(pool={adv['NaivePooled']['coverage']:.4f}, synth={adv['OnlySynth']['coverage']:.4f})")
PASS = bool(gespi_df_valid and gespi_informative and onlyreal_exact and eff_sep
            and naive_valid_matched and naive_breaks_adv)
print(f"\nVERDICT PASS = {PASS}")

out = dict(experiment="claim2c_confidence_interval_coverage", paper="arXiv 2509.20345 GESPI (miscoverage task)",
           n=n, N=N, alpha=alpha, eps=eps, bound_miscoverage=bound, sigma=sigma, theta_star=theta,
           reps=T, seed=SEED, results=results,
           checks=dict(gespi_distribution_free_valid=gespi_df_valid, gespi_informative_near_nominal=bool(gespi_informative),
                       onlyreal_exact_1_minus_alpha=bool(onlyreal_exact), efficiency_separated=bool(eff_sep),
                       gespi_width_reduction_pct=red_pct, naive_valid_when_matched=bool(naive_valid_matched),
                       naive_violates_under_shift=bool(naive_breaks_adv), PASS=PASS))
Path(__file__).with_name("results_2c.json").write_text(json.dumps(out, indent=2))
print("wrote results_2c.json")
