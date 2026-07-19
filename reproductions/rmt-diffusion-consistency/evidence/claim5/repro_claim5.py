"""
Independent NumPy/scipy reproduction -- CLAIM 5
Paper: "A Random Matrix (Theory) Perspective on the Consistency of Diffusion
Models", OpenReview iPjuUQbkfl / arXiv 2602.02908.

CLAIM 5 (Figure 5 / Section 6). UNet and DiT experiments validate the theory's
predictions about CONSISTENCY, OVER-SHRINKAGE, and EIGENMODE-DEPENDENT deviations
in the NON-MEMORIZATION regime.

CPU-faithful surrogate. Training UNet/DiT on FFHQ is out of CPU scope. We use a
genuinely NON-LINEAR denoiser -- the exact non-parametric Bayes (KDE) denoiser,
the exact minimiser of the DSM objective for a finite training set:
        D_kde(x;sigma) = sum_i softmax_i(-||x-x_i||^2 / (2 sigma^2)) x_i .
It is NOT the linear-Gaussian model (it can memorise), yet on Gaussian data it
converges to the population posterior mean as n grows. We show it reproduces the
paper's deep-net phenomenology beyond the linear theory:
  (A) CONSISTENCY / non-memorization transition,
  (B) convergence of generations to the Gaussian (linear-theory) predictor,
  (C) OVER-SHRINKAGE of low-variance eigenmodes (effective gain below the naive
      Wiener gain), eigenmode-dependent and decaying with dataset size -- exactly
      the theory's prediction (Prop 4.1 / Claim 2).

ACCEPTANCE RULE.
 (A) cross-split MSE of the NON-LINEAR denoiser decreases with n and drops BELOW the
     nearest-training-neighbour distance (more similar across splits than to training
     data -> non-memorization).
 (B) MSE(nonlinear denoiser, population linear/Gaussian Wiener denoiser) decreases
     monotonically with n.
 (C) over-shrinkage: the nonlinear denoiser's EFFECTIVE per-mode gain is below the
     naive Wiener gain lam_k/(lam_k+sigma^2) for the low-variance half (pull toward
     the mean), the shrinkage DECAYS with n, and the gain profile is eigenmode-ordered
     (Spearman with lam_k > 0.9 -> anisotropic / eigenmode-dependent).
FALSIFICATION: cross-split deviation never falls below nearest-neighbour (pure
memorization), generations do not approach the Gaussian predictor, or no over-
shrinkage (nonlinear gain >= naive) / gain not eigenmode-ordered.
"""
import json, time
import numpy as np
from scipy.optimize import brentq

t0 = time.time()

def make_spectrum(d, alpha=1.0):
    lam = (np.arange(1, d + 1.0)) ** (-alpha)
    return lam / lam.mean()

def solve_kappa(sig2, gamma, lam):
    h = lambda k: np.mean(lam / (lam + k))
    return brentq(lambda k: k - sig2 - gamma * k * h(k),
                  sig2 * (1 + 1e-12) + 1e-15, sig2 + gamma * lam.mean() + 1.0, xtol=1e-13)

def kde_denoise(Xq, Xtr, sig2):
    d2 = np.sum(Xq ** 2, 1)[:, None] + np.sum(Xtr ** 2, 1)[None, :] - 2.0 * Xq @ Xtr.T
    lg = -d2 / (2.0 * sig2); lg -= lg.max(1, keepdims=True)
    w = np.exp(lg); w /= w.sum(1, keepdims=True)
    return w @ Xtr

def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])

print("=" * 78)
print("CLAIM 5  non-linear denoiser validates consistency / overshrinkage /")
print("         eigenmode-dependence (Fig 5) -- CPU non-parametric surrogate for UNet/DiT")
print("=" * 78)

d, alpha, sig2 = 14, 1.0, 0.70
lam = make_spectrum(d, alpha); s12 = np.sqrt(lam)
Sig = np.diag(lam)
Wlin = Sig @ np.linalg.inv(Sig + sig2 * np.eye(d))
naive_gain = lam / (lam + sig2)
rng = np.random.default_rng(0); Q = 256
x0 = rng.standard_normal((Q, d)) * s12
Xq = x0 + np.sqrt(sig2) * rng.standard_normal((Q, d))
Dlin = Xq @ Wlin.T
print(f"d={d}  sigma^2={sig2}  {Q} held-out noisy queries  (spectrum lam_j~j^-{alpha})\n")

# ---------------- (A) consistency / non-memorization ; (B) -> Gaussian ----------------
print("NON-LINEAR (KDE) denoiser across dataset size n:")
print(f"  {'n':>5} {'cross-split':>12} {'->Gaussian':>11} {'nearest-NN':>11} {'cross/NN':>9}")
rows = []
for n in [16, 32, 64, 128, 256, 512, 1024, 2048]:
    rg = np.random.default_rng(1000 + n)
    X1 = rg.standard_normal((n, d)) * s12; X2 = rg.standard_normal((n, d)) * s12
    D1 = kde_denoise(Xq, X1, sig2); D2 = kde_denoise(Xq, X2, sig2)
    cross = float(np.mean(np.sum((D1 - D2) ** 2, 1)) / d)
    togauss = float(np.mean(np.sum((0.5 * (D1 + D2) - Dlin) ** 2, 1)) / d)
    nn = float(np.mean(np.min(np.sum((D1[:, None, :] - X1[None, :, :]) ** 2, 2), 1)) / d)
    rows.append(dict(n=n, cross=cross, togauss=togauss, nn=nn, ratio=cross / nn))
    print(f"  {n:5d} {cross:12.4f} {togauss:11.4f} {nn:11.4f} {cross/nn:9.3f}")
A_ok = (rows[-1]['ratio'] < 1.0) and (rows[-1]['cross'] < rows[0]['cross']) and (rows[0]['ratio'] > 1.0)
B_ok = all(rows[i]['togauss'] >= rows[i + 1]['togauss'] - 1e-6 for i in range(len(rows) - 1)) and rows[-1]['togauss'] < rows[0]['togauss']
print(f"\n  (A) non-memorization (cross-split MSE falls, crosses below nearest-NN)? {A_ok}")
print(f"      cross/NN: n=16 -> {rows[0]['ratio']:.2f} (memorization) ; n=2048 -> {rows[-1]['ratio']:.2f} (consistent)")
print(f"  (B) generations approach the Gaussian predictor (MSE decreasing)? {B_ok}")
print(f"      MSE(nonlinear,Gaussian): n=16 -> {rows[0]['togauss']:.4f} ; n=2048 -> {rows[-1]['togauss']:.4f}")

# ---------------- (C) over-shrinkage (effective per-mode gain) ----------------
print("\n(C) OVER-SHRINKAGE: effective per-mode gain of the NON-LINEAR denoiser")
print("    g_k = <u_k,D(x)>.<u_k,x_clean> / |<u_k,x_clean>|^2, vs naive Wiener lam/(lam+sig^2):")
Qg = 4000; R = 6
overs = []
for n in [200, 1000]:
    gamma = d / n; kap = solve_kappa(sig2, gamma, lam)
    rmt_gain = lam / (lam + kap)
    gain = np.zeros(d)
    for r in range(R):
        rg = np.random.default_rng(3000 + r)
        c0 = rg.standard_normal((Qg, d)) * s12
        xq = c0 + np.sqrt(sig2) * rg.standard_normal((Qg, d))
        Xtr = rg.standard_normal((n, d)) * s12
        D = kde_denoise(xq, Xtr, sig2)
        gain += np.sum(D * c0, 0) / np.sum(c0 * c0, 0)
    gain /= R
    lowhalf = slice(d // 2, d)
    over = float(np.mean(naive_gain[lowhalf] - gain[lowhalf]))    # >0 = overshrinkage
    sp = spearman(gain, lam)
    overs.append(dict(n=n, gamma=gamma, kappa=float(kap), gain=[float(x) for x in gain],
                      overshrink_low=over, spearman_lam=sp))
    print(f"  n={n} gamma={gamma:.3f} kappa={kap:.3f}:")
    for k in [0, 4, 9, 13]:
        print(f"      mode {k:2d} lam={lam[k]:6.3f}: KDE gain={gain[k]:.3f}  "
              f"naive={naive_gain[k]:.3f}  rmt(kappa)={rmt_gain[k]:.3f}")
    print(f"      low-mode over-shrinkage (naive-KDE) = {over:+.3f} (>0 ?)  "
          f"gain eigenmode-ordered Spearman(g,lam)={sp:.3f}")
oversh_present = overs[0]['overshrink_low'] > 0 and overs[1]['overshrink_low'] > 0
oversh_decays = overs[1]['overshrink_low'] < overs[0]['overshrink_low']
eig_ordered = overs[0]['spearman_lam'] > 0.9 and overs[1]['spearman_lam'] > 0.9
C_ok = oversh_present and oversh_decays and eig_ordered
print(f"\n  over-shrinkage present at finite n? {oversh_present}  "
      f"decays with n ({overs[0]['overshrink_low']:.3f}->{overs[1]['overshrink_low']:.3f})? {oversh_decays}  "
      f"eigenmode-ordered? {eig_ordered}")
print(f"  (C) over-shrinkage / eigenmode-dependence verified? {C_ok}")

verified = bool(A_ok and B_ok and C_ok)
print("\n" + "=" * 78)
print(f"VERDICT claim5: consistency(A)={A_ok}  ->Gaussian(B)={B_ok}  overshrinkage(C)={C_ok}")
print(f"CLAIM 5 VERIFIED (nonlinear surrogate) = {verified}")
print("=" * 78)

out = dict(paper="arXiv 2602.02908 / iPjuUQbkfl", claim=5,
           surrogate="non-parametric Bayes (KDE) denoiser as CPU proxy for UNet/DiT",
           config=dict(d=d, alpha=alpha, sigma2=sig2, Q=Q, Qg=Qg, R=R),
           rows=rows, overshrink=overs,
           A_ok=bool(A_ok), B_ok=bool(B_ok), C_ok=bool(C_ok),
           oversh_present=bool(oversh_present), oversh_decays=bool(oversh_decays),
           eig_ordered=bool(eig_ordered),
           verified=verified, runtime_s=round(time.time() - t0, 2))
with open("results.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote results.json  runtime=%.2fs" % (time.time() - t0))
