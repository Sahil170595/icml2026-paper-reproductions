"""
Claim 2 (application A) - GESPI conformal RISK CONTROL (protein-structure task, Section 4.1).
Decisive CPU verification of the paper's distribution-free RISK-control guarantee (Thm 3.2) on a
controlled residue-level DGP with KNOWN error law (so realized risk is measured against ground truth).
AlphaFold2 inference is not CPU-runnable, so the literal CASP-14 pLDDT outputs are out of scope; the
GUARANTEE the claim rests on - valid risk control + less abstention, distribution-free - is what we test.

Setup mirrors 4.1: each residue has a confidence score c (like pLDDT) and a binary error indicator
e (err > 3 Angstrom). Abstain on low-confidence residues (c < lambda). Risk(lambda) = fraction of
ACCEPTED residues that are errors, controlled <= alpha via conformal risk control (Angelopoulos 2024):
  lambda_hat = inf{ lambda : (n/(n+1)) Rhat(lambda) + 1/(n+1) <= alpha }  ->  Rhat(lambda) <= alpha - (1-alpha)/n.
Paper knobs matched: real proteins n=10, synthetic N=1000, eps=5%, alpha in {5,10,15}%.

Named methods (paper Sec 4) + the naive-pooling control:
  OnlyReal    = lambda from real calibration only        (base; paper: conservatively abstains a lot)
  OnlySynth   = lambda from synthetic calibration only    (paper's named method WITHOUT risk guarantees)
  NaivePooled = lambda from pooled real+synthetic, no guardrail (the naive-pooling control)
  GESPI       = lambda_GESPI = min(lambda_real_alpha, max(lambda_pool_alpha, lambda_real_(alpha+eps)))  (Eq.2)
Guarantee (Thm 3.2): GESPI risk <= alpha + min{eps, c*d_TV(P,Q)} <= alpha+eps for ANY synthetic Q.
DGP: real error law p(e=1|c)=sigmoid(-1.5 c), c~N(0,1).
  informative synth: same conditional law, mild covariate shift c~N(0.1,1.05).
  adversarial synth: OVER-optimistic p(e=1|c)=sigmoid(-1.5 c - 1.3) (looks cleaner) -> naive UNDER-controls.
Every risk/abstention is a Monte-Carlo mean over T calibration trials with reported MC standard error.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, numpy as np
from pathlib import Path

n, N = 10, 1000
eps = 0.05
alphas = [0.05, 0.10, 0.15]
M = 50                      # residues per real protein  -> Rr = n*M real residues
Rr = n * M
Rs = 5000                  # synthetic residues per trial (subsample of N=1000 proteins)
T, SEED = 3000, 20260717
Rtest = 200000             # fixed large test pool (real law) for exact risk/abstention evaluation
a_slope = 1.5

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def draw(rng, shape, shift=0.0, scale=1.0, adv_bias=0.0):
    c = rng.normal(shift, scale, size=shape)
    p = sigmoid(-a_slope * c - adv_bias)
    e = (rng.random(shape) < p).astype(np.float64)
    return c, e

def thresholds(c, e, n_examples, target_levels):
    """Vectorized conformal-risk-control threshold(s). c,e: (T, R). level -> (T,) threshold."""
    Tn, R = c.shape
    order = np.argsort(-c, axis=1)                      # descending confidence
    c_sorted = np.take_along_axis(c, order, axis=1)
    e_sorted = np.take_along_axis(e, order, axis=1)
    Sk = np.cumsum(e_sorted, axis=1)                    # errors among top-k
    out = {}
    for a in target_levels:
        t = a - (1.0 - a) / n_examples                  # allowed Rhat
        cap = t * R
        k = (Sk <= cap).sum(axis=1)                     # number accepted (top-k)
        lam = np.empty(Tn)
        abstain_all = k <= 0
        accept_all = k >= R
        mid = ~abstain_all & ~accept_all
        lam[abstain_all] = np.inf
        lam[accept_all] = -np.inf
        idx = k[mid]
        lam[mid] = 0.5 * (c_sorted[mid, idx - 1] + c_sorted[mid, idx])
        out[round(a, 3)] = lam
    return out

rng = np.random.default_rng(SEED)
tc, te = draw(rng, (Rtest,))                            # fixed test pool from the REAL error law
tc_sorted = np.sort(tc)
o = np.argsort(tc); te_by_c = te[o]
suffix_err = np.concatenate([np.cumsum(te_by_c[::-1])[::-1], [0.0]])   # sum e for c>=tc_sorted[j]
def eval_thr(lam):
    lam = np.asarray(lam, float)
    pos = np.searchsorted(tc_sorted, lam, side="left")
    risk = suffix_err[pos] / Rtest
    abstain = pos / Rtest
    risk = np.where(np.isposinf(lam), 0.0, risk)
    abstain = np.where(np.isposinf(lam), 1.0, abstain)
    risk = np.where(np.isneginf(lam), te.mean(), risk)
    abstain = np.where(np.isneginf(lam), 0.0, abstain)
    return risk, abstain

def mc(x):  # Monte-Carlo mean and standard error of the mean over T trials
    return float(x.mean()), float(x.std() / np.sqrt(len(x)))

results = {}
print(f"GESPI Claim 2A | conformal RISK control | n={n} N={N} eps={eps} M={M} T={T} Rtest={Rtest} seed={SEED}")
print("(risk/abstention are Monte-Carlo means over T trials +- MC standard error)")
for regime, kw in [("informative", dict(shift=0.1, scale=1.05, adv_bias=0.0)),
                   ("adversarial", dict(shift=0.0, scale=1.0, adv_bias=1.3))]:
    rc, re_ = draw(rng, (T, Rr))                                   # real calib
    sc, se_ = draw(rng, (T, Rs), **kw)                             # synth calib
    pc = np.concatenate([rc, sc], axis=1); pe = np.concatenate([re_, se_], axis=1)
    lev = sorted(set(alphas + [a + eps for a in alphas]))
    lam_real  = thresholds(rc, re_, n, lev)
    lam_synth = thresholds(sc, se_, N, alphas)                     # OnlySynth (no RNG use -> stream unchanged)
    lam_pool  = thresholds(pc, pe, n + N, alphas)
    print(f"\n== regime: {regime} ==")
    print(f"{'alpha':>5} {'method':>11} {'risk':>7} {'+-SE':>6} {'<=a?':>5} {'<=a+e?':>6} {'abstain%':>8} {'+-SE':>5} {'absRedVsReal%':>13}")
    results[regime] = {}
    for a in alphas:
        la = lam_real[round(a, 3)]; lae = lam_real[round(a + eps, 3)]
        lp = lam_pool[round(a, 3)]; ls = lam_synth[round(a, 3)]
        lg = np.minimum(la, np.maximum(lp, lae))
        r_or, ab_or = eval_thr(la)                                 # OnlyReal reference (for paired reduction)
        row = {}
        methods = [("OnlyReal", la), ("OnlySynth", ls), ("NaivePooled", lp), ("GESPI", lg)]
        for name, lam in methods:
            risk, ab = eval_thr(lam)
            mrisk, mrisk_se = mc(risk); mab, mab_se = mc(ab * 100)
            # paired abstention reduction vs OnlyReal, with MC SE of the paired difference
            dred = (ab_or - ab)                                    # >0 => this method abstains less
            mdred, sdred = mc(dred * 100)
            redpct = float((1 - ab.mean() / max(ab_or.mean(), 1e-12)) * 100)
            row[name] = dict(risk=mrisk, risk_se=mrisk_se, abstain_pct=mab, abstain_se=mab_se,
                             valid_le_alpha=bool(mrisk <= a + 3e-3),
                             guard_le_alpha_eps=bool(mrisk <= a + eps + 3e-3),
                             abstain_reduction_vs_onlyreal_pct=redpct,
                             abstain_reduction_pp=mdred, abstain_reduction_pp_se=sdred)
            tag_red = f"{redpct:6.1f} ({mdred/max(sdred,1e-9):4.0f}SE)" if name == "GESPI" else f"{redpct:6.1f}"
            print(f"{a:5.2f} {name:>11} {mrisk:7.4f} {mrisk_se:6.4f} {str(mrisk<=a+3e-3):>5} "
                  f"{str(mrisk<=a+eps+3e-3):>6} {mab:8.1f} {mab_se:5.2f} {tag_red:>13}")
        results[regime][f"{a:.2f}"] = row

# ---- pre-registered acceptance checks (fixed before the run) ----
print("\n--- Checks (pre-registered) ---")
inf = results["informative"]; adv = results["adversarial"]
onlyreal_valid  = all(inf[f"{a:.2f}"]["OnlyReal"]["valid_le_alpha"] for a in alphas)
gespi_valid_inf = all(inf[f"{a:.2f}"]["GESPI"]["valid_le_alpha"] for a in alphas)
gespi_guard_adv = all(adv[f"{a:.2f}"]["GESPI"]["guard_le_alpha_eps"] for a in alphas)
# efficiency: GESPI abstains statistically-significantly less than OnlyReal (paired, >5 SE) at every alpha
eff_sep = all(inf[f"{a:.2f}"]["GESPI"]["abstain_reduction_pp"] >
              5 * inf[f"{a:.2f}"]["GESPI"]["abstain_reduction_pp_se"] for a in alphas)
red = [round(inf[f"{a:.2f}"]["GESPI"]["abstain_reduction_vs_onlyreal_pct"], 1) for a in alphas]
# decisive controls: BOTH naive methods (paper's OnlySynth AND naive pooling) VIOLATE risk under adversarial synth
naive_pool_breaks  = any(adv[f"{a:.2f}"]["NaivePooled"]["risk"] > a + 0.01 for a in alphas)
only_synth_breaks  = any(adv[f"{a:.2f}"]["OnlySynth"]["risk"]  > a + 0.01 for a in alphas)
print(f"[validity] OnlyReal risk <= alpha (informative): {onlyreal_valid}")
print(f"[validity] GESPI risk <= alpha (informative): {gespi_valid_inf}")
print(f"[validity] GESPI risk <= alpha+eps guardrail (adversarial): {gespi_guard_adv}")
print(f"[efficiency] GESPI abstains LESS than OnlyReal at every alpha (>5 SE), reductions%={red}: {eff_sep}")
print(f"[control] NaivePooled VIOLATES risk (> alpha) under adversarial synth: {naive_pool_breaks}")
print(f"[control] OnlySynth (paper baseline) VIOLATES risk (> alpha) under adversarial synth: {only_synth_breaks}")
PASS = bool(onlyreal_valid and gespi_valid_inf and gespi_guard_adv and eff_sep and naive_pool_breaks and only_synth_breaks)
print(f"\nVERDICT PASS = {PASS}")

out = dict(experiment="claim2a_conformal_risk_control", paper="arXiv 2509.20345 GESPI Sec 4.1",
           n=n, N=N, eps=eps, alphas=alphas, M=M, Rs=Rs, T=T, Rtest=Rtest, seed=SEED, results=results,
           checks=dict(onlyreal_valid=bool(onlyreal_valid), gespi_valid_informative=bool(gespi_valid_inf),
                       gespi_guardrail_adversarial=bool(gespi_guard_adv), efficiency_separated=bool(eff_sep),
                       abstain_reductions_pct=red, naive_pooled_breaks=bool(naive_pool_breaks),
                       only_synth_breaks=bool(only_synth_breaks), PASS=PASS))
Path(__file__).with_name("results_2a.json").write_text(json.dumps(out, indent=2))
print("wrote results_2a.json")
