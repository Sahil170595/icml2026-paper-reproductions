"""
Independent NumPy/scipy reproduction -- CLAIM 3
Paper: "A Random Matrix (Theory) Perspective on the Consistency of Diffusion
Models", OpenReview iPjuUQbkfl / arXiv 2602.02908.

CLAIM 3 (Result 4.2 / Proposition 4.2). The denoiser-variance theory predicts
ANISOTROPIC, LOCATION-DEPENDENT cross-split deviations that DECAY with dataset
size. Proposition 4.2 deterministic-equivalent (mean known, mu=0):

  Var_{Sigma_hat}[ v^T D*_{Sigma_hat}(x;sigma) ]
      ~=  [ kappa^2 / (n - df2(kappa)) ] * Diamond(v,kappa,Sigma) * Diamond(x,kappa,Sigma)
                     \___ 1/n scaling __/   \__ anisotropy __/     \_ inhomogeneity _/
  Diamond(u,kappa,Sigma) := u^T (Sigma+kappa I)^{-2} Sigma u = sum_j u_j^2 lam_j/(lam_j+kappa)^2
  df2(kappa) := Tr[Sigma^2 (Sigma+kappa I)^{-2}] (UN-normalized); kappa=kappa(sigma^2), Eq. 4.

ACCEPTANCE RULE.
 (A) anisotropy: probe v=u_k over modes -> empirical/predicted variance ratio in
     [0.85,1.15] for every mode; variance SPANS >5x across modes; Pearson(log emp,
     log pred) > 0.99.
 (B) inhomogeneity: input x over eigen-directions -> ratio in [0.85,1.15]; spans >3x.
 (C) decay with dataset size: (C1) the RMT formula predicts the measured variance at
     every n (ratio in [0.9,1.1]) and the variance decreases monotonically with n;
     (C2) in the large-n limit (kappa -> sigma^2 converged) the predicted variance
     follows the paper's GLOBAL 1/n scaling, log-log slope in [-1.1,-0.9].
FALSIFICATION: variance isotropic, location-independent, or not decaying with n.
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
                  sig2 * (1 + 1e-12) + 1e-15, sig2 + gamma * lam.mean() + 1.0,
                  xtol=1e-13, rtol=1e-13)

def diamond(u, kap, lam):
    return float(np.sum(u ** 2 * lam / (lam + kap) ** 2))

print("=" * 78)
print("CLAIM 3  denoiser variance: anisotropy + inhomogeneity + decay (Result 4.2)")
print("paper: arXiv 2602.02908 / iPjuUQbkfl  (Prop 4.2) -- independent NumPy")
print("=" * 78)

d, n, alpha, sig2 = 120, 100, 1.0, 0.10
gamma = d / n
lam = make_spectrum(d, alpha)
kap = solve_kappa(sig2, gamma, lam)
df2 = float(np.sum(lam ** 2 / (lam + kap) ** 2))
pref = kap ** 2 / (n - df2)
rng0 = np.random.default_rng(7)
x_fixed = rng0.standard_normal(d) * np.sqrt(lam)
print(f"d={d} n={n} gamma={gamma:.3f} sigma^2={sig2}  kappa={kap:.4f}  "
      f"df2(kappa)={df2:.3f}  n-df2={n-df2:.3f}  prefactor={pref:.4e}\n")

# ---------- (A) anisotropy : vectorised over all probe modes (x fixed) ----------
T = 6000
s12 = np.sqrt(lam); Isig = sig2 * np.eye(d)
W = np.empty((T, d)); rng = np.random.default_rng(101)
for t in range(T):
    X = rng.standard_normal((n, d)) * s12; Sh = (X.T @ X) / n
    W[t] = Sh @ np.linalg.solve(Sh + Isig, x_fixed)
emp_mode = W.var(axis=0)
dx = diamond(x_fixed, kap, lam)
pred_mode = pref * (lam / (lam + kap) ** 2) * dx
print("(A) ANISOTROPY -- probe v=u_k over eigenmodes (x fixed):")
print(f"    {'mode':>5} {'lam_k':>9} {'emp Var':>12} {'pred Var':>12} {'ratio':>7}")
aniso = []
for k in [0, 3, 10, 30, 70, 119]:
    aniso.append(dict(mode=k, lam=float(lam[k]), emp=float(emp_mode[k]),
                      pred=float(pred_mode[k]), ratio=float(emp_mode[k] / pred_mode[k])))
    print(f"    {k:5d} {lam[k]:9.4f} {emp_mode[k]:12.4e} {pred_mode[k]:12.4e} "
          f"{emp_mode[k]/pred_mode[k]:7.3f}")
span = float(emp_mode.max() / emp_mode.min())
rlog = float(np.corrcoef(np.log(emp_mode), np.log(pred_mode))[0, 1])
A_ok = all(0.85 <= a['ratio'] <= 1.15 for a in aniso) and span > 5 and rlog > 0.99
print(f"    variance span across all {d} modes = {span:.1f}x ; "
      f"Pearson(log emp,log pred)={rlog:.4f}")
print(f"    (A) anisotropy predicted? {A_ok}\n")

# ---------- (B) inhomogeneity : vectorised over input locations (v=u_5 fixed) ----------
jlist = [0, 3, 10, 30, 70, 119]
Xin = np.zeros((d, len(jlist)))
for c, j in enumerate(jlist):
    Xin[j, c] = np.sqrt(lam[j]) * np.sqrt(d)
vidx = 5
Wb = np.empty((T, len(jlist))); rng = np.random.default_rng(202)
for t in range(T):
    X = rng.standard_normal((n, d)) * s12; Sh = (X.T @ X) / n
    Wb[t] = (Sh @ np.linalg.solve(Sh + Isig, Xin))[vidx]
emp_x = Wb.var(axis=0)
dv = diamond(np.eye(d)[vidx], kap, lam)
print("(B) INHOMOGENEITY -- input x aligned with u_j (v=u_5 fixed):")
print(f"    {'x~u_j':>6} {'lam_j':>9} {'emp Var':>12} {'pred Var':>12} {'ratio':>7}")
inhom = []
for c, j in enumerate(jlist):
    pv = pref * dv * diamond(Xin[:, c], kap, lam)
    inhom.append(dict(jmode=j, lam=float(lam[j]), emp=float(emp_x[c]), pred=float(pv),
                      ratio=float(emp_x[c] / pv)))
    print(f"    {j:6d} {lam[j]:9.4f} {emp_x[c]:12.4e} {pv:12.4e} {emp_x[c]/pv:7.3f}")
span_x = float(emp_x.max() / emp_x.min())
B_ok = all(0.85 <= r['ratio'] <= 1.15 for r in inhom) and span_x > 3
print(f"    location-dependent variance span = {span_x:.1f}x")
print(f"    (B) inhomogeneity predicted? {B_ok}\n")

# ---------- (C1) decay with n : MC-validate the formula's n-dependence ----------
print("(C1) DECAY WITH DATASET SIZE -- formula tracks measured Var (v=u_10, x fixed):")
print(f"    {'n':>5} {'gamma':>7} {'kappa':>8} {'emp Var':>12} {'pred Var':>12} {'ratio':>7}")
v10 = np.eye(d)[10]; decay = []
for nn in [50, 100, 200, 400, 800]:
    gg = d / nn; kp = solve_kappa(sig2, gg, lam)
    df2n = float(np.sum(lam ** 2 / (lam + kp) ** 2)); pf = kp ** 2 / (nn - df2n)
    Tn = 4000; vals = np.empty(Tn); rr = np.random.default_rng(900 + nn)
    for t in range(Tn):
        X = rr.standard_normal((nn, d)) * s12; Sh = (X.T @ X) / nn
        vals[t] = (Sh @ np.linalg.solve(Sh + Isig, x_fixed))[10]
    dxn = diamond(x_fixed, kp, lam)      # inhomogeneity factor at the PER-n kappa
    ev = float(vals.var()); pv = pf * diamond(v10, kp, lam) * dxn
    decay.append(dict(n=nn, gamma=gg, kappa=float(kp), emp=ev, pred=pv, ratio=ev / pv))
    print(f"    {nn:5d} {gg:7.3f} {kp:8.4f} {ev:12.4e} {pv:12.4e} {ev/pv:7.3f}")
monotone = all(decay[i]['emp'] > decay[i + 1]['emp'] for i in range(len(decay) - 1))
C1_ok = all(0.9 <= r['ratio'] <= 1.1 for r in decay) and monotone
print(f"    formula matches at every n & Var decreases monotonically? {C1_ok}")

# ---------- (C2) asymptotic global 1/n scaling (predicted, kappa converged) ----------
print("\n(C2) ASYMPTOTIC 1/n scaling from the validated formula (large n, kappa->sigma^2):")
big_n = [2000, 8000, 32000, 128000]; predbig = []
for nn in big_n:
    gg = d / nn; kp = solve_kappa(sig2, gg, lam)
    df2n = float(np.sum(lam ** 2 / (lam + kp) ** 2))
    pv = (kp ** 2 / (nn - df2n)) * diamond(v10, kp, lam) * diamond(x_fixed, kp, lam)
    predbig.append(pv)
    print(f"    n={nn:7d}  kappa={kp:.5f}  predicted Var={pv:.4e}")
slope_big = float(np.polyfit(np.log10(big_n), np.log10(predbig), 1)[0])
C2_ok = -1.1 <= slope_big <= -0.9
print(f"    asymptotic log-log slope = {slope_big:.3f} in [-1.1,-0.9] (global 1/n)? {C2_ok}")

C_ok = C1_ok and C2_ok
verified = bool(A_ok and B_ok and C_ok)
print("\n" + "=" * 78)
print(f"VERDICT claim3: anisotropy(A)={A_ok}  inhomogeneity(B)={B_ok}  "
      f"decay(C1 formula={C1_ok}, C2 1/n={C2_ok})")
print(f"CLAIM 3 VERIFIED = {verified}")
print("=" * 78)

out = dict(paper="arXiv 2602.02908 / iPjuUQbkfl", claim=3,
           config=dict(d=d, n=n, gamma=gamma, alpha=alpha, sigma2=sig2, T=T),
           kappa=float(kap), df2=df2, prefactor=float(pref),
           anisotropy=aniso, aniso_span=span, aniso_logcorr=rlog,
           inhomogeneity=inhom, inhom_span=span_x,
           decay=decay, decay_monotone=bool(monotone),
           asymptotic=dict(n=big_n, pred=[float(x) for x in predbig], slope=slope_big),
           A_ok=bool(A_ok), B_ok=bool(B_ok), C1_ok=bool(C1_ok), C2_ok=bool(C2_ok),
           C_ok=bool(C_ok), verified=verified, runtime_s=round(time.time() - t0, 2))
with open("results.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote results.json  runtime=%.2fs" % (time.time() - t0))
