"""
Claim 1 reproduction - GESPI safely enhances sample efficiency (Type I control + power gain).
Paper: "Statistical Inference Leveraging Synthetic Data with Distribution-Free Guarantees"
       (GESPI), arXiv 2509.20345, OpenReview sxLncu2Fhx. Appendix C.1 + Section 3.2 / Theorem 3.2.

Exact paper setup (Appendix C.1, "Hypothesis testing with simulated data"):
  real data      D_n  ~ Binomial(n=50,  rho)
  synthetic data D_N  ~ Binomial(N=500, rho_synt)   (may differ arbitrarily from real)
  one-sided test  H0: rho = 0.5   vs   H1: rho > 0.5
  RANDOMIZED binomial test (paper's exact wording), target alpha = 5%, GESPI eps = 2%.
  Guardrail bound = alpha + eps = 0.07 (Theorem 3.2, distribution-free, no assumption on synth).

GESPI decision rule (Eq. 3, Section 3.2):
  phi_GESPI = phi_{n,alpha}  OR  ( phi_{n,N,alpha}  AND  phi_{n,alpha+eps} )
  phi_{n,alpha}     : randomized test on real  D_n            at level alpha
  phi_{n,alpha+eps} : randomized test on real  D_n            at level alpha+eps  (shared U -> nested)
  phi_{n,N,alpha}   : randomized test on pooled D_n u D_N     at level alpha       (independent U)
  OnlyReal = phi_{n,alpha}. OnlySynth = randomized test on synthetic only (shown to be invalid).

Figure-6 regimes reproduced:
  (a) real ALT, synth NULL   rho=0.60 rho_s=0.50  -> GESPI power ~ OnlyReal (no harm)
  (b) both ALT               rho=0.60 rho_s=0.55  -> GESPI power > OnlyReal (efficiency gain)
  (c) both NULL              rho=0.50 rho_s=0.50  -> Type I ~ alpha
  (d) real NULL, synth ALT   rho=0.50 rho_s=0.70  -> ADVERSARIAL: Type I <= alpha+eps (guardrail)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, numpy as np
from scipy.stats import binom
from pathlib import Path

n, N, p0 = 50, 500, 0.5
alpha, eps = 0.05, 0.02
bound = alpha + eps
REPS, SEED = 40000, 12345

def rand_pvalue(k, m, p0=0.5, rng=None, U=None):
    # randomized upper-tail p-value: P(X>k) + U*P(X=k); ~Uniform(0,1) exactly under H0.
    if U is None:
        U = rng.random(size=np.shape(k))
    return binom.sf(k, m, p0) + U * binom.pmf(k, m, p0)

def run(rho, rho_s, rng):
    k_n = rng.binomial(n, rho, REPS)
    k_N = rng.binomial(N, rho_s, REPS)
    k_pool = k_n + k_N
    U_real = rng.random(REPS)          # shared across alpha and alpha+eps -> nested tests
    U_pool = rng.random(REPS)
    U_syn  = rng.random(REPS)
    p_real = rand_pvalue(k_n, n, p0, U=U_real)
    p_pool = rand_pvalue(k_pool, n + N, p0, U=U_pool)
    p_syn  = rand_pvalue(k_N, N, p0, U=U_syn)
    phi_real_a   = p_real <= alpha
    phi_real_ae  = p_real <= (alpha + eps)
    phi_pool_a   = p_pool <= alpha
    phi_gespi    = phi_real_a | (phi_pool_a & phi_real_ae)
    phi_onlyreal = phi_real_a
    phi_onlysyn  = p_syn <= alpha
    return (phi_gespi.mean(), phi_onlyreal.mean(), phi_onlysyn.mean(),
            float((phi_gespi >= phi_onlyreal).all()))

def se(p, m=REPS): return float(np.sqrt(p * (1 - p) / m))

rng = np.random.default_rng(SEED)
regimes = [
    ("a", 0.60, 0.50, "power",  "real ALT, synth NULL"),
    ("b", 0.60, 0.55, "power",  "both ALT"),
    ("c", 0.50, 0.50, "TypeI",  "both NULL"),
    ("d", 0.50, 0.70, "TypeI",  "real NULL, ADVERSARIAL synth ALT"),
]
print(f"GESPI Claim 1 | randomized binomial test | n={n} N={N} alpha={alpha} eps={eps} bound={bound} reps={REPS} seed={SEED}")
print(f"{'reg':3} {'rho':>4} {'rho_s':>5} {'metric':>6} {'GESPI':>7} {'+-SE':>6} {'OnlyReal':>8} {'OnlySyn':>7} {'G>=OR':>6}  desc")
res = {}
for rid, rho, rs, metric, desc in regimes:
    g, o, s, dom = run(rho, rs, rng)
    res[rid] = dict(rho=rho, rho_synt=rs, metric=metric, desc=desc,
                    gespi=g, gespi_se=se(g), onlyreal=o, onlyreal_se=se(o),
                    onlysynth=s, gespi_dominates_onlyreal=dom)
    print(f"{rid:3} {rho:4.2f} {rs:5.2f} {metric:>6} {g:7.4f} {se(g):6.4f} {o:8.4f} {s:7.4f} {str(bool(dom)):>6}  {desc}")

# ---- Figure 6 sweep over rho_synt ----
sweep_synt = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70]
sweep = {"alt_rho0.60": {}, "null_rho0.50": {}}
print("\nFigure-6 sweep over rho_synt:")
for rho, key in [(0.60, "alt_rho0.60"), (0.50, "null_rho0.50")]:
    lab = "POWER" if rho > 0.5 else "TypeI"
    print(f"  rho={rho} ({lab}):  rho_s ->  GESPI / OnlyReal")
    for rs in sweep_synt:
        g, o, s, dom = run(rho, rs, rng)
        sweep[key][f"{rs:.2f}"] = dict(gespi=g, onlyreal=o, onlysynth=s)
        print(f"    {rs:.2f}: {g:.4f} / {o:.4f}   (OnlySynth {s:.4f})")

# ---- acceptance checks ----
typeI_c, typeI_d = res["c"]["gespi"], res["d"]["gespi"]
pa_g, pa_o = res["b"]["gespi"], res["b"]["onlyreal"]
diff = pa_g - pa_o
diff_se = float(np.sqrt(res["b"]["gespi_se"]**2 + res["b"]["onlyreal_se"]**2))
all_null_bounded = all(res[r]["gespi"] <= bound for r in ["c", "d"])
benign_near_alpha = abs(typeI_c - alpha) <= 0.01
power_gain_b = diff > 2 * diff_se
never_loses = all(res[r]["gespi_dominates_onlyreal"] for r in res)
onlysyn_invalid_d = res["d"]["onlysynth"] > bound  # OnlySynth breaks under adversarial null
PASS = all_null_bounded and benign_near_alpha and power_gain_b and never_loses
print("\n--- Checks ---")
print(f"[validity] Type I <= bound {bound} in null regimes c,d: {all_null_bounded} "
      f"(c={typeI_c:.4f}, d={typeI_d:.4f} adversarial)")
print(f"[validity] benign-null (c) Type I ~ alpha={alpha}: {benign_near_alpha} (c={typeI_c:.4f})")
print(f"[efficiency] power gain (b): GESPI={pa_g:.4f} > OnlyReal={pa_o:.4f}, "
      f"diff={diff:.4f} > 2SE={2*diff_se:.4f}? {power_gain_b}")
print(f"[no-harm] GESPI power >= OnlyReal in EVERY regime: {never_loses}")
print(f"[context] OnlySynth INVALID under adversarial null (d): {res['d']['onlysynth']:.4f} > {bound}? {onlysyn_invalid_d}")
print(f"\nVERDICT PASS = {PASS}")

out = dict(experiment="claim1_binomial_typeI_power", paper="arXiv 2509.20345 GESPI",
           n=n, N=N, alpha=alpha, eps=eps, bound=bound, reps=REPS, seed=SEED,
           test="randomized one-sided binomial", regimes=res, sweep=sweep,
           checks=dict(all_null_typeI_le_bound=bool(all_null_bounded),
                       benign_null_near_alpha=bool(benign_near_alpha),
                       power_gain_b_gt_2se=bool(power_gain_b),
                       diff_b=float(diff), diff_b_2se=float(2*diff_se),
                       gespi_never_loses_power=bool(never_loses),
                       onlysynth_invalid_adversarial=bool(onlysyn_invalid_d),
                       PASS=bool(PASS)))
Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=2))
print("wrote results.json")
