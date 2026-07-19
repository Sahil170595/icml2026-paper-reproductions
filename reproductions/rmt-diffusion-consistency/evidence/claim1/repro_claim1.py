"""
Independent NumPy/scipy reproduction -- CLAIM 1
Paper: "A Random Matrix (Theory) Perspective on the Consistency of Diffusion
Models", OpenReview iPjuUQbkfl / arXiv 2602.02908.

CLAIM 1 (Figure 1). Diffusion models trained on NON-OVERLAPPING dataset splits
generate visually similar samples from the SAME seed, and the similarity is
predicted by a Gaussian LINEAR-THEORY baseline (Wiener filter). The paper's key
quantitative signatures (Fig 1B): (i) generated images are MORE similar across
splits than to their nearest training neighbour (rules out memorization); (ii)
both splits track the linear/Gaussian predictor; (iii) samples closer to the
Gaussian solution are more consistent (positive correlation).

We reproduce the paper's OWN linear-theory baseline: the closed-form linear
diffusion "sampling map" (paper Eq. 3, sigma->0 limit; here mu=0):
        x(z) = Sigma_hat^{1/2} (Sigma_hat + sigmaT^2 I)^{-1/2} (sigmaT * z),   z ~ N(0,I).
Two disjoint splits give Sigma_hat_1, Sigma_hat_2; the population/Gaussian theory
uses Sigma. Same noise seed z is fed to both.

ACCEPTANCE RULE.
 (A) consistency vs memorization: cross-split MSE  <  nearest-training-neighbour
     distance (ratio < 1) in the non-memorization regime, and the ratio DECREASES
     with dataset size n.
 (B) linear-theory tracks: split->population(Gaussian) MSE decreases with n and is
     <= cross-split MSE (both splits collapse onto the shared Gaussian predictor).
 (C) decay: cross-split MSE ~ 1/n (log-log slope in [-1.2,-0.8]).
 (D) positive consistency-vs-Gaussian correlation (Fig 1B, paper r=0.244>0): per-seed
     Pearson r between distance-to-Gaussian-predictor and cross-split distance > 0.
FALSIFICATION: cross-split MSE >= nearest-neighbour distance (memorization), or
cross-split MSE does not decay with n, or splits diverge from the Gaussian predictor.
"""
import json, time
import numpy as np

t0 = time.time()

def make_spectrum(d, alpha=1.0):
    lam = (np.arange(1, d + 1.0)) ** (-alpha)
    return lam / lam.mean()

def sample_map(Sh, Zsig, sigmaT):
    """x(z) = Sh^{1/2} (Sh+sigmaT^2 I)^{-1/2} (sigmaT z), rows of Zsig are seeds z."""
    w, V = np.linalg.eigh(Sh); w = np.clip(w, 0.0, None)
    gain = np.sqrt(w) / np.sqrt(w + sigmaT ** 2)     # per-eigenmode
    A = (V * gain) @ V.T
    return (sigmaT * Zsig) @ A.T

print("=" * 78)
print("CLAIM 1  linear diffusion consistency across non-overlapping splits (Fig 1)")
print("paper: arXiv 2602.02908 / iPjuUQbkfl -- independent NumPy")
print("=" * 78)

d, alpha, sigmaT, nseed = 128, 1.0, 80.0, 512
lam = make_spectrum(d, alpha)
Sig = np.diag(lam)
rng = np.random.default_rng(0)
Z = rng.standard_normal((nseed, d))                  # 512 SHARED initial-noise seeds
# population (Gaussian-theory) sampling map, using the true covariance
x_pop = sample_map(Sig, Z, sigmaT)
scale = float(np.mean(np.sum(x_pop ** 2, 1)) / d)    # typical per-dim sample energy
print(f"spectrum lam_j ~ j^-{alpha}, d={d}, sigmaT={sigmaT}, {nseed} shared seeds")
print(f"typical generated-sample energy per dim (Gaussian predictor) = {scale:.4f}")
print(f"cross-seed baseline (different seeds, same model) MSE/dim = "
      f"{np.mean(np.sum((x_pop[:-1]-x_pop[1:])**2,1))/d:.4f}  (unrelated-sample scale)\n")

print(f"  {'n':>5} {'cross-split':>12} {'split->pop':>11} {'nearest-NN':>11} "
      f"{'ratio x/NN':>11} {'corr r':>8}")
rows = []
for n in [16, 32, 64, 128, 256, 512, 1024]:
    rg = np.random.default_rng(1000 + n)
    s12 = np.sqrt(lam)
    X1 = rg.standard_normal((n, d)) * s12
    X2 = rg.standard_normal((n, d)) * s12            # disjoint split
    Xf = np.vstack([X1, X2])
    Sh1 = (X1.T @ X1) / n; Sh2 = (X2.T @ X2) / n; Shf = (Xf.T @ Xf) / (2 * n)
    x1 = sample_map(Sh1, Z, sigmaT)
    x2 = sample_map(Sh2, Z, sigmaT)
    xf = sample_map(Shf, Z, sigmaT)
    cross = float(np.mean(np.sum((x1 - x2) ** 2, 1)) / d)
    topop = float(np.mean(np.sum((x1 - x_pop) ** 2, 1)) / d)
    # nearest training-neighbour distance for split-1 generated samples
    d1 = x1[:, None, :] - X1[None, :, :]
    nn = float(np.mean(np.min(np.sum(d1 ** 2, 2), 1)) / d)
    # per-seed correlation: distance-to-Gaussian-predictor  vs  cross-split distance
    a = np.sqrt(np.sum((x1 - x_pop) ** 2, 1)); b = np.sqrt(np.sum((x1 - x2) ** 2, 1))
    r = float(np.corrcoef(a, b)[0, 1])
    ratio = cross / nn
    print(f"  {n:5d} {cross:12.4f} {topop:11.4f} {nn:11.4f} {ratio:11.3f} {r:8.3f}")
    rows.append(dict(n=n, cross=cross, topop=topop, nn=nn, ratio=ratio, corr=r))

# ---- acceptance checks ----
import numpy as _np
ns = _np.array([r['n'] for r in rows], float)
cr = _np.array([r['cross'] for r in rows], float)
slope = float(_np.polyfit(_np.log10(ns), _np.log10(cr), 1)[0])
big = [r for r in rows if r['n'] >= 64]                 # non-memorization regime
A_ok = all(r['ratio'] < 1.0 for r in big) and rows[-1]['ratio'] < rows[0]['ratio']
B_ok = all(r['topop'] <= r['cross'] + 1e-9 for r in rows) and rows[-1]['topop'] < rows[0]['topop']
C_ok = -1.2 <= slope <= -0.8
D_ok = all(r['corr'] > 0 for r in rows)
verified = bool(A_ok and B_ok and C_ok and D_ok)
print(f"\n  (A) cross-split < nearest-NN (non-memorization) & ratio falls: {A_ok}")
print(f"      ratio cross/NN: n=64 -> {big[0]['ratio']:.3f},  n=1024 -> {rows[-1]['ratio']:.3f}")
print(f"  (B) splits track Gaussian predictor (split->pop <= cross, decays): {B_ok}")
print(f"  (C) cross-split MSE ~ 1/n : log-log slope = {slope:.3f} in [-1.2,-0.8]? {C_ok}")
print(f"  (D) consistency-vs-Gaussian correlation r>0 (paper r=0.244): "
      f"min r={min(r['corr'] for r in rows):.3f}  all positive? {D_ok}")
print("\n" + "=" * 78)
print(f"CLAIM 1 VERIFIED = {verified}   (A={A_ok} B={B_ok} C={C_ok} D={D_ok})")
print("=" * 78)

out = dict(paper="arXiv 2602.02908 / iPjuUQbkfl", claim=1,
           config=dict(d=d, alpha=alpha, sigmaT=sigmaT, nseed=nseed),
           sample_energy=scale, rows=rows, cross_slope_vs_n=slope,
           A_ok=bool(A_ok), B_ok=bool(B_ok), C_ok=bool(C_ok), D_ok=bool(D_ok),
           verified=verified, runtime_s=round(time.time() - t0, 2))
with open("results.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote results.json  runtime=%.2fs" % (time.time() - t0))
