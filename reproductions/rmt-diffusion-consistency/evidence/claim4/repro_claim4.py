"""
Independent NumPy/scipy reproduction -- CLAIM 4
Paper: "A Random Matrix (Theory) Perspective on the Consistency of Diffusion
Models", OpenReview iPjuUQbkfl / arXiv 2602.02908.

CLAIM 4 (Results 5.1 & 5.2). The sampling-map analysis gives deterministic-
equivalence formulas for the EXPECTATION and VARIANCE over full diffusion
trajectories. Linear PF-ODE sampling map (Eq.3, sigma->0, mu=0, sigma_T->inf):
        x(xbar) = Sigma_hat^{1/2} xbar,   xbar ~ N(0, I).
Needs deterministic equivalence for FRACTIONAL matrix powers, from the integral
representation  A^{1/2} = (2/pi) int_0^inf A (A+u^2 I)^{-1} du  and the resolvent
DE  Sigma_hat (Sigma_hat+u^2 I)^{-1} ~= Sigma (Sigma+kappa(u^2) I)^{-1}.

Prop 5.1 (expectation, over-shrinkage to the mean):
   E[ u_k^T Sigma_hat^{1/2} u_k ] ~= (2/pi) int_0^inf lam_k/(lam_k+kappa(u^2)) du .
Prop 5.2 (variance of the generated sample):
   Var[ v^T Sigma_hat^{1/2} xbar ]
     ~= (4/pi^2) int int  [kappa kappa'/(n - df2(kappa,kappa'))]
              * Pent(v;kappa,kappa') * Pent(xbar;kappa,kappa')  du dv ,
   Pent(a;k,k') := sum_j a_j^2 lam_j/((lam_j+k)(lam_j+k')),
   df2(k,k') := sum_j lam_j^2/((lam_j+k)(lam_j+k'))  (UN-normalized), k=kappa(u^2), k'=kappa(v^2).

ACCEPTANCE RULE.
 (5.1) MEDIAN rel-err(emp, frac-DE) over modes < 3%; OVER-SHRINKAGE holds
       (frac-DE < sqrt(lam_k) and emp tracks DE, not naive); max rel-err DECREASES
       as d grows (DE exact in the high-dim limit).
 (5.2) empirical/DE ratio in [0.85,1.15] for every probe; variance anisotropic (>3x).
FALSIFICATION: the fractional-power DE integrals fail to track measurement, the
d->large error does not shrink, or the naive population sqrt is used.
"""
import json, time
import numpy as np
from scipy.optimize import brentq

t0 = time.time()

def make_spectrum(d):
    # bounded, spike-free anisotropic spectrum (log-spaced in [0.25,4.0], mean 1). The
    # fractional-power DE is a spectral-BULK statement, so we avoid a single dominant
    # outlier eigenvalue (BBP spike) whose O(1) finite-d edge fluctuation is outside the
    # bulk DE. The power-law/natural-image spectrum (with a spike) is carried by Claims 2,3.
    lam = np.geomspace(0.25, 4.0, d)
    return lam / lam.mean()

def solve_kappa(lmbda, gamma, lam):
    h = lambda k: np.mean(lam / (lam + k))
    lo = max(lmbda, 0.0) * (1 + 1e-12) + 1e-15
    return brentq(lambda k: k - lmbda - gamma * k * h(k),
                  lo, lmbda + gamma * lam.mean() + 1.0, xtol=1e-13, rtol=1e-13)

def tmap_nodes(NU=300):
    tt = np.linspace(0.0, 0.99985, NU)      # t in [0,1) -> u in [0,inf): captures lam/u^2 tail
    return tt / (1.0 - tt)

def E_sqrt_diag_emp(d, n, lam, T, seed):
    rng = np.random.default_rng(seed); s12 = np.sqrt(lam); acc = np.zeros(d)
    for _ in range(T):
        X = rng.standard_normal((n, d)) * s12; Sh = (X.T @ X) / n
        w, V = np.linalg.eigh(Sh); w = np.clip(w, 0.0, None)
        acc += np.einsum('ij,j,ij->i', V, np.sqrt(w), V)
    return acc / T

def E_sqrt_diag_rmt(d, gamma, lam, u):
    kap_u = np.array([solve_kappa(uu * uu, gamma, lam) for uu in u])
    pred = np.array([(2.0 / np.pi) * np.trapezoid(lam[k] / (lam[k] + kap_u), u)
                     for k in range(d)])
    return pred, kap_u

print("=" * 78)
print("CLAIM 4  sampling-map deterministic equivalence (Results 5.1 & 5.2)")
print("fractional-power DE:  x = Sigma_hat^{1/2} xbar   -- independent NumPy")
print("=" * 78)

d, n = 140, 110
gamma = d / n
lam = make_spectrum(d)
u = tmap_nodes(300)
print(f"d={d} n={n} gamma={gamma:.3f}  bounded spectrum [0.25,4.0] (mean 1), "
      f"lam_max={lam.max():.2f}")
print(f"quadrature: u in [0,{u[-1]:.0f}] via t-map, {len(u)} nodes\n")

# ---------------- Prop 5.1 : expectation / over-shrinkage ----------------
T1 = 1000
E_emp = E_sqrt_diag_emp(d, n, lam, T1, seed=11)
E_rmt, kap_u = E_sqrt_diag_rmt(d, gamma, lam, u)
E_pop = np.sqrt(lam)
print("Prop 5.1  E[ u_k^T Sigma_hat^{1/2} u_k ]  (over-shrinkage to the mean):")
print(f"    {'mode':>5} {'lam_k':>9} {'emp':>9} {'frac-DE':>9} {'naive sqrt':>10} {'rel-err':>9}")
p51 = []
for k in [0, 10, 30, 60, 100, 139]:
    re = abs(E_emp[k] - E_rmt[k]) / E_emp[k]
    print(f"    {k:5d} {lam[k]:9.4f} {E_emp[k]:9.4f} {E_rmt[k]:9.4f} {E_pop[k]:10.4f} {re:9.4f}")
    p51.append(dict(mode=k, lam=float(lam[k]), emp=float(E_emp[k]),
                    rmt=float(E_rmt[k]), naive=float(E_pop[k]), relerr=float(re)))
rel_all = np.abs(E_emp - E_rmt) / np.maximum(E_emp, 1e-12)
med_rel = float(np.median(rel_all)); max_rel = float(rel_all.max())
low = slice(0, d // 2)
oversh = bool(np.all(E_rmt[low] < E_pop[low]) and
              np.mean(np.abs(E_emp[low] - E_rmt[low])) < np.mean(np.abs(E_emp[low] - E_pop[low])))
print(f"    median rel-err={med_rel:.4f}  max rel-err={max_rel:.4f}")
print(f"    over-shrinkage (frac-DE<sqrt(lam) & emp tracks DE not naive), low modes? {oversh}")
print("    d->large convergence of the fractional-power DE (fixed gamma):")
dscan = []
for dd in [70, 140, 280]:
    nn = int(round(dd / gamma)); lm = make_spectrum(dd)
    ee = E_sqrt_diag_emp(dd, nn, lm, T=300, seed=77)
    er, _ = E_sqrt_diag_rmt(dd, gamma, lm, u)
    mr = float((np.abs(ee - er) / np.maximum(ee, 1e-12)).max())
    dscan.append(dict(d=dd, n=nn, max_rel=mr))
    print(f"        d={dd:4d} n={nn:4d}  max rel-err = {mr:.4f}")
d_conv = all(dscan[i]['max_rel'] > dscan[i + 1]['max_rel'] for i in range(len(dscan) - 1))
P51_ok = (med_rel < 0.03) and oversh and d_conv
print(f"    max rel-err decreasing in d? {d_conv}")
print(f"    (5.1) fractional-power DE expectation verified? {P51_ok}\n")

# ---------------- Prop 5.2 : variance / double-integral DE ----------------
xbar = np.random.default_rng(22).standard_normal(d)
T2 = 3000; s12 = np.sqrt(lam)
probes = [0, 10, 30, 60, 100, 139]
Yk = np.empty((T2, d)); rng = np.random.default_rng(33)
for t in range(T2):
    X = rng.standard_normal((n, d)) * s12; Sh = (X.T @ X) / n
    w, V = np.linalg.eigh(Sh); w = np.clip(w, 0.0, None)
    Yk[t] = ((V * np.sqrt(w)) @ V.T) @ xbar
emp_var = Yk.var(axis=0)
K = kap_u
R = lam[None, :] / (lam[None, :] + K[:, None])
axb = (xbar ** 2) / lam
DF2 = R @ R.T
Wmm = np.outer(K, K) / (n - DF2)
pent_x = (R * axb[None, :]) @ R.T
print("Prop 5.2  Var[ v^T Sigma_hat^{1/2} xbar ]  (double-integral fractional DE):")
print(f"    {'mode':>5} {'lam_k':>9} {'emp Var':>12} {'DE Var':>12} {'ratio':>7}")
p52 = []
for k in probes:
    pent_v = np.outer(R[:, k], R[:, k]) / lam[k]
    integ = Wmm * pent_v * pent_x
    val = (4.0 / np.pi ** 2) * np.trapezoid(np.trapezoid(integ, u, axis=1), u)
    ratio = emp_var[k] / val
    print(f"    {k:5d} {lam[k]:9.4f} {emp_var[k]:12.4e} {val:12.4e} {ratio:7.3f}")
    p52.append(dict(mode=k, lam=float(lam[k]), emp=float(emp_var[k]),
                    de=float(val), ratio=float(ratio)))
span52 = float(max(p['emp'] for p in p52) / min(p['emp'] for p in p52))
P52_ok = all(0.85 <= p['ratio'] <= 1.15 for p in p52) and span52 > 3
print(f"    variance span across probes = {span52:.1f}x (anisotropic)")
print(f"    (5.2) double-integral fractional DE verified? {P52_ok}\n")

verified = bool(P51_ok and P52_ok)
print("=" * 78)
print(f"VERDICT claim4: Prop5.1 expectation={P51_ok}  Prop5.2 variance={P52_ok}")
print(f"CLAIM 4 VERIFIED = {verified}")
print("=" * 78)

out = dict(paper="arXiv 2602.02908 / iPjuUQbkfl", claim=4,
           config=dict(d=d, n=n, gamma=gamma, T1=T1, T2=T2, NU=len(u), umax=float(u[-1])),
           prop51=p51, prop51_med_rel=med_rel, prop51_max_rel=max_rel,
           overshrink=oversh, d_scan=dscan, d_converges=bool(d_conv),
           prop52=p52, prop52_span=span52,
           P51_ok=bool(P51_ok), P52_ok=bool(P52_ok),
           verified=verified, runtime_s=round(time.time() - t0, 2))
with open("results.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote results.json  runtime=%.2fs" % (time.time() - t0))
