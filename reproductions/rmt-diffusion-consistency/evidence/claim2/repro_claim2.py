"""
Independent NumPy/scipy reproduction -- CLAIM 2
Paper: "A Random Matrix (Theory) Perspective on the Consistency of Diffusion
Models", OpenReview iPjuUQbkfl / arXiv 2602.02908.

CLAIM 2 (Figure 2, Proposition 4.1). Finite-sample covariance effects renormalize
the effective noise scale in the EXPECTED linear denoiser through a self-consistent
map sigma^2 -> kappa(sigma^2), causing OVER-SHRINKAGE of low-variance directions.

Optimal linear denoiser with EMPIRICAL covariance Sigma_hat (mean known, mu=0):
        D*(x;sigma) = Sigma_hat (Sigma_hat + sigma^2 I)^{-1} x .
Paper's deterministic-equivalence (DE) relation (Eq. DE / Prop 4.1):
        Sigma_hat (Sigma_hat + lambda I)^{-1}  ~=  Sigma (Sigma + kappa(lambda) I)^{-1}
so E[ u_k^T Sigma_hat(Sigma_hat+sigma^2 I)^{-1} u_k ]  ->  lam_k/(lam_k+kappa(sigma^2)),
i.e. the population per-mode gain but with sigma^2 RENORMALIZED to kappa(sigma^2)>=sigma^2.
Self-consistent equation (Eq. 4, tr = NORMALIZED trace, tr[I]=1):
        kappa(lambda) - lambda = gamma * kappa(lambda) * tr[Sigma (Sigma + kappa I)^{-1}],
        gamma = d / n .

TEST / ACCEPTANCE RULE.
 (A) per-mode gain: max_k |emp_gain_k - RMT_gain_k| <= 0.02  AND this max is
     << max_k |emp_gain_k - naive_gain_k| (naive = lam/(lam+sigma^2), i.e. no
     finite-sample renormalization).  => the empirical denoiser follows the
     kappa-RENORMALIZED law, not the naive one.
 (B) over-shrinkage: for the lowest-variance mode, RMT_gain < naive_gain
     (strict shrinkage below the population gain), and it grows with gamma.
 (C) d->large: at fixed gamma, RMS_k|emp_gain - RMT_gain| DECREASES as d grows
     (the DE becomes exact in the high-dimensional limit).
FALSIFICATION: empirical gain matches the naive sigma^2 law (no renormalization),
or kappa(sigma^2)=sigma^2, or the d->large error does not shrink.
"""
import json, time
import numpy as np
from scipy.optimize import brentq

t0 = time.time()

def make_spectrum(d, alpha=1.0):
    lam = (np.arange(1, d + 1.0)) ** (-alpha)
    return lam / lam.mean()                      # normalized so mean eigenvalue = 1

def solve_kappa(sig2, gamma, lam):
    h = lambda k: np.mean(lam / (lam + k))       # tr[Sigma (Sigma+kI)^{-1}], normalized
    g = lambda k: k - sig2 - gamma * k * h(k)
    hi = sig2 + gamma * lam.mean() + 1.0         # kappa <= lambda + gamma*tr(Sigma)
    return brentq(g, sig2 * (1 + 1e-12) + 1e-15, hi, xtol=1e-13, rtol=1e-13)

def emp_gain(d, n, sig2, lam, T, seed):
    """E[ diag( Sigma_hat (Sigma_hat+sig2 I)^{-1} ) ] over T dataset draws (Sigma diagonal
       => population eigenvectors are the coordinate axes)."""
    rng = np.random.default_rng(seed)
    s12 = np.sqrt(lam); acc = np.zeros(d)
    I = np.eye(d)
    for _ in range(T):
        X = rng.standard_normal((n, d)) * s12            # rows ~ N(0, Sigma)
        Sh = (X.T @ X) / n
        acc += np.diag(Sh @ np.linalg.inv(Sh + sig2 * I))
    return acc / T

print("=" * 78)
print("CLAIM 2  noise renormalization sigma^2 -> kappa(sigma^2) + overshrinkage")
print("paper: arXiv 2602.02908 / iPjuUQbkfl  (Fig 2, Prop 4.1) -- independent NumPy")
print("=" * 78)

# ---- headline config: paper Fig 2C uses n=1000, gamma ~= 3.1 ; we match gamma ----
d, n, alpha = 256, 80, 1.0
gamma = d / n
lam = make_spectrum(d, alpha)
sig2 = 0.05
T = 400
kap = solve_kappa(sig2, gamma, lam)
print(f"spectrum: power-law lam_j ~ j^-{alpha}, d={d}, mean(lam)=1, "
      f"lam_max={lam.max():.3f}, lam_min={lam.min():.4f}")
print(f"n={n}  gamma=d/n={gamma:.3f}  sigma^2={sig2}  T={T} dataset draws")
print(f"self-consistent renormalized noise kappa(sigma^2) = {kap:.5f}  "
      f"(kappa/sigma^2 = {kap/sig2:.2f}x)")
print()

g_emp = emp_gain(d, n, sig2, lam, T, seed=0)
g_rmt = lam / (lam + kap)                # RMT (kappa-renormalized)  prediction
g_pop = lam / (lam + sig2)              # naive population (no finite-sample) gain
print("per-mode denoiser gain  u_k^T Sigma_hat(Sigma_hat+sig2 I)^-1 u_k :")
print(f"  {'mode':>5} {'lam_k':>9} {'emp':>9} {'RMT(kappa)':>11} {'naive(sig2)':>12} "
      f"{'|emp-RMT|':>10} {'|emp-naive|':>11}")
probe = [0, 1, 5, 20, 60, 150, 255]
rows = []
for k in probe:
    er = abs(g_emp[k] - g_rmt[k]); en = abs(g_emp[k] - g_pop[k])
    print(f"  {k:5d} {lam[k]:9.4f} {g_emp[k]:9.4f} {g_rmt[k]:11.4f} {g_pop[k]:12.4f} "
          f"{er:10.4f} {en:11.4f}")
    rows.append(dict(mode=k, lam=float(lam[k]), emp=float(g_emp[k]),
                     rmt=float(g_rmt[k]), naive=float(g_pop[k])))
max_err_rmt = float(np.abs(g_emp - g_rmt).max())
max_err_pop = float(np.abs(g_emp - g_pop).max())
print(f"\n  max_k |emp - RMT|   = {max_err_rmt:.4f}")
print(f"  max_k |emp - naive| = {max_err_pop:.4f}   "
      f"(naive law is {max_err_pop/max_err_rmt:.0f}x worse)")
A_ok = (max_err_rmt <= 0.02) and (max_err_pop > 5 * max_err_rmt)
print(f"  (A) emp follows kappa-renormalized law (not naive)? {A_ok}")

# ---- (B) over-shrinkage of the lowest-variance mode + kappa grows with gamma ----
klow = d - 1
oversh = g_pop[klow] - g_rmt[klow]
B_ok = oversh > 0
print(f"\n  (B) over-shrinkage of lowest-var mode k={klow} (lam={lam[klow]:.4f}): "
      f"naive gain {g_pop[klow]:.3f} -> RMT gain {g_rmt[klow]:.3f}  "
      f"(shrunk by {oversh:.3f}, {100*oversh/g_pop[klow]:.0f}%)  strict? {B_ok}")
print("  kappa(sigma^2) vs aspect ratio gamma  (more data -> less renormalization):")
kap_gamma = []
for gg in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
    kg = solve_kappa(sig2, gg, lam)
    kap_gamma.append(dict(gamma=gg, kappa=float(kg), ratio=float(kg/sig2)))
    print(f"      gamma={gg:5.2f}  kappa={kg:.5f}  kappa/sigma^2={kg/sig2:6.2f}x")
mono_gamma = all(kap_gamma[i]['kappa'] < kap_gamma[i+1]['kappa'] for i in range(len(kap_gamma)-1))
print(f"  kappa increases monotonically with gamma? {mono_gamma}")

# ---- (C) high-dimensional limit: RMS error shrinks as d grows (fixed gamma) ----
print(f"\n  (C) DE becomes exact as d->large (fixed gamma={gamma:.2f}, sigma^2={sig2}):")
dscan = []
for dd in [64, 128, 256, 512]:
    nn = int(round(dd / gamma)); lm = make_spectrum(dd, alpha)
    kp = solve_kappa(sig2, gamma, lm)
    ge = emp_gain(dd, nn, sig2, lm, T=200, seed=123)
    rms = float(np.sqrt(np.mean((ge - lm / (lm + kp)) ** 2)))
    dscan.append(dict(d=dd, n=nn, rms=rms))
    print(f"      d={dd:4d}  n={nn:4d}  RMS_k|emp-RMT gain| = {rms:.5f}")
C_ok = all(dscan[i]['rms'] > dscan[i+1]['rms'] for i in range(len(dscan)-1))
print(f"  RMS error strictly decreasing in d? {C_ok}")

verified = bool(A_ok and B_ok and mono_gamma and C_ok)
print("\n" + "=" * 78)
print(f"VERDICT claim2: (A) renormalized-law match={A_ok}  (B) overshrinkage={B_ok}  "
      f"kappa(gamma) monotone={mono_gamma}  (C) d->large convergence={C_ok}")
print(f"CLAIM 2 VERIFIED = {verified}")
print("=" * 78)

out = dict(
    paper="arXiv 2602.02908 / iPjuUQbkfl", claim=2,
    config=dict(d=d, n=n, gamma=gamma, alpha=alpha, sigma2=sig2, T=T),
    kappa=float(kap), kappa_over_sig2=float(kap / sig2),
    gains=rows, max_err_rmt=max_err_rmt, max_err_naive=max_err_pop,
    overshrink_lowmode=dict(mode=klow, lam=float(lam[klow]),
                            naive=float(g_pop[klow]), rmt=float(g_rmt[klow]),
                            shrink=float(oversh)),
    kappa_vs_gamma=kap_gamma, d_scaling=dscan,
    A_ok=bool(A_ok), B_ok=bool(B_ok), kappa_monotone=bool(mono_gamma),
    C_ok=bool(C_ok), verified=verified, runtime_s=round(time.time() - t0, 2))
with open("results.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote results.json  runtime=%.2fs" % (time.time() - t0))
