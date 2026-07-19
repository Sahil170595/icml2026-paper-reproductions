"""
Claim 2 (application B) - GESPI one-sided win-rate HYPOTHESIS TEST (large-reasoning-model task, Sec 4.2).
Decisive CPU verification of the paper's distribution-free TYPE-I guarantee (Thm 3.2) for the Type-I
error-control task, on a Bernoulli win-rate DGP matched to the paper's exact knobs. Running the actual
LLMs is not CPU-feasible; the GUARANTEE the claim rests on - valid Type-I control + higher power with
limited labels, distribution-free - is what we test.

Paper knobs (Sec 4.2 / Fig 4): real n=15 AIME25 problems, synthetic N=100 OlympiadBench problems,
  H0: p=0.5 (A no better than B) vs H1: p>0.5, alpha=5%, eps=2%, bound=alpha+eps=0.07, randomized
  binomial test. Type I is estimated by SHUFFLING responses (the paper's design that forces the null).

Two comparisons bracket the regime (a stronger and a weaker true signal). NOTE: the win-rate values
below are illustrative DGP settings that reproduce the qualitative Fig-4 behavior; they are NOT the
paper's measured per-model win rates (those require running the real LLMs).

Named methods (paper Sec 4) + the naive-pooling control:
  OnlyReal  = randomized test on real only, phi_{n,alpha}                  (base)
  OnlySynth = randomized test on synthetic only, phi_{N,alpha}             (paper's named method, NO guarantee)
  GESPI     = phi_{n,alpha} OR (phi_{n,N,alpha} AND phi_{n,alpha+eps})     (Eq.3)
Type-I and power are Monte-Carlo means over REPS reps with Wald standard errors.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, numpy as np
from scipy.stats import binom
from pathlib import Path

n, N = 15, 100
p0, alpha, eps = 0.5, 0.05, 0.02
bound = alpha + eps
REPS, SEED = 40000, 424242

def rand_p(k, m, U):  # randomized upper-tail p-value ~ Uniform(0,1) under H0 -> exact level
    return binom.sf(k, m, p0) + U * binom.pmf(k, m, p0)

def run(p_real, p_synt, rng, rng_s):
    k_n = rng.binomial(n, p_real, REPS)
    k_N = rng.binomial(N, p_synt, REPS)
    Ur = rng.random(REPS); Up = rng.random(REPS)      # main stream (unchanged order)
    Us = rng_s.random(REPS)                            # independent stream for OnlySynth randomization
    pr = rand_p(k_n, n, Ur)                            # shared U for real @ alpha and alpha+eps (nested)
    pp = rand_p(k_n + k_N, n + N, Up)
    ps = rand_p(k_N, N, Us)
    phi_ra  = pr <= alpha
    phi_rae = pr <= (alpha + eps)
    phi_pa  = pp <= alpha
    gespi   = phi_ra | (phi_pa & phi_rae)
    onlyreal = phi_ra
    onlysynth = ps <= alpha
    return (float(gespi.mean()), float(onlyreal.mean()), float(onlysynth.mean()),
            float((gespi >= onlyreal).all()))

def se(p, m=REPS): return float(np.sqrt(p * (1 - p) / m))

rng   = np.random.default_rng(SEED)
rng_s = np.random.default_rng(SEED + 777)              # independent side stream (keeps main numbers fixed)
comps = [
    ("Comp1 (strong signal)", 0.70, 0.67),
    ("Comp2 (weaker signal)", 0.63, 0.60),
]
print(f"GESPI Claim 2B | LLM win-rate test | n={n} N={N} alpha={alpha} eps={eps} bound={bound} reps={REPS} seed={SEED}")
res = {}
print("\n[POWER] real & synthetic under the alternative (p>0.5):")
print(f"{'comparison':24} {'p_real':>6} {'p_syn':>6} {'GESPI':>7} {'OnlyReal':>8} {'gain':>7} {'OnlySynth':>9} {'G>=OR':>6}")
for label, pr, ps_ in comps:
    g, o, s, dom = run(pr, ps_, rng, rng_s)
    res.setdefault(label, {})["power"] = dict(p_real=pr, p_synt=ps_, gespi=g, gespi_se=se(g),
                                              onlyreal=o, onlyreal_se=se(o), onlysynth=s, dominates=dom)
    print(f"{label:24} {pr:6.2f} {ps_:6.2f} {g:7.4f} {o:8.4f} {g-o:7.4f} {s:9.4f} {str(bool(dom)):>6}")

print("\n[TYPE I] shuffled responses -> real null (p=0.5):")
print(f"{'synthetic setting':30} {'p_real':>6} {'p_syn':>6} {'GESPI':>7} {'+-SE':>6} {'OnlyReal':>8} {'OnlySynth':>9} {'G<=a+e?':>7}")
typeI = {}
for tag, ps_ in [("benign shuffle (p_syn=0.50)", 0.50), ("ADVERSARIAL (p_syn=0.65)", 0.65)]:
    g, o, s, dom = run(0.50, ps_, rng, rng_s)
    typeI[tag] = dict(p_real=0.50, p_synt=ps_, gespi=g, gespi_se=se(g), onlyreal=o, onlyreal_se=se(o),
                      onlysynth=s, within_bound=bool(g <= bound))
    print(f"{tag:30} {0.50:6.2f} {ps_:6.2f} {g:7.4f} {se(g):6.4f} {o:8.4f} {s:9.4f} {str(bool(g<=bound)):>7}")
res["typeI"] = typeI

# ---- pre-registered acceptance checks ----
print("\n--- Checks (pre-registered) ---")
power_gain = all(res[l]["power"]["gespi"] - res[l]["power"]["onlyreal"] >
                 2 * np.hypot(res[l]["power"]["gespi_se"], res[l]["power"]["onlyreal_se"])
                 for l, _, _ in comps)
never_loses = all(res[l]["power"]["dominates"] for l, _, _ in comps)
above_alpha = all(res[l]["power"]["gespi"] > alpha for l, _, _ in comps)
typeI_ok = all(v["within_bound"] for v in typeI.values())
benign_near_alpha = abs(typeI["benign shuffle (p_syn=0.50)"]["gespi"] - alpha) <= 0.01
only_synth_breaks = typeI["ADVERSARIAL (p_syn=0.65)"]["onlysynth"] > bound   # paper's named baseline VIOLATES
print(f"[efficiency] GESPI power > OnlyReal power (>2SE) in BOTH comparisons: {power_gain}")
for l, _, _ in comps:
    pw = res[l]["power"]; print(f"    {l}: GESPI={pw['gespi']:.4f} OnlyReal={pw['onlyreal']:.4f} "
                                f"gain={pw['gespi']-pw['onlyreal']:.4f} (2SE={2*np.hypot(pw['gespi_se'],pw['onlyreal_se']):.4f})")
print(f"[no-harm] GESPI power >= OnlyReal always: {never_loses}")
print(f"[validity] GESPI Type I <= bound {bound} incl adversarial: {typeI_ok} "
      f"(benign={typeI['benign shuffle (p_syn=0.50)']['gespi']:.4f}, adv={typeI['ADVERSARIAL (p_syn=0.65)']['gespi']:.4f})")
print(f"[validity] benign shuffle Type I ~ alpha={alpha}: {benign_near_alpha}")
print(f"[control] OnlySynth (paper baseline) VIOLATES Type I under adversarial shuffle: {only_synth_breaks} "
      f"(OnlySynth Type I={typeI['ADVERSARIAL (p_syn=0.65)']['onlysynth']:.4f} > {bound})")
PASS = bool(power_gain and never_loses and above_alpha and typeI_ok and benign_near_alpha and only_synth_breaks)
print(f"\nVERDICT PASS = {PASS}")

out = dict(experiment="claim2b_llm_winrate_test", paper="arXiv 2509.20345 GESPI Sec 4.2",
           n=n, N=N, alpha=alpha, eps=eps, bound=bound, reps=REPS, seed=SEED, results=res,
           checks=dict(power_gain_gt_2se=bool(power_gain), gespi_never_loses=bool(never_loses),
                       detects_A_beats_B=bool(above_alpha), typeI_within_bound=bool(typeI_ok),
                       benign_near_alpha=bool(benign_near_alpha), only_synth_breaks=bool(only_synth_breaks),
                       PASS=PASS))
Path(__file__).with_name("results_2b.json").write_text(json.dumps(out, indent=2))
print("wrote results_2b.json")
