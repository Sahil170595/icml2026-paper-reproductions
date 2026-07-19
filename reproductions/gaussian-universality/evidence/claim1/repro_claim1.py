"""
Claim 1 (Theorem 4.3 / Sec. 5) reproduction, arXiv 2604.03146:
"...an asymptotic min-max characterization of key statistics, enabling
approximation of the mean mu_thetahat and covariance C_thetahat of the ERM
estimator thetahat ... under general non-Gaussian data."

Ridge ERM: thetahat = argmin (1/n) sum_i (1/2)(y_i - x_i^T th)^2 + (lam/2)||th||^2
        => thetahat = (X^T X/n + lam I)^{-1} X^T y / n   (paper Eq.1 normalization)

The paper's min-max characterization specialized to ridge + isotropic C_x
coincides with the RMT deterministic equivalent (paper App. D.3 / Cor. 5.1).
We compute the deterministic-equivalent (DE) prediction of:
    m*      = <theta_star, mu_thetahat>      (MEAN summary of thetahat)
    alpha*^2 = tr(C_thetahat C_x) = tr(C_thetahat)   (COVARIANCE summary; C_x=I)
    risk*   = E||thetahat - theta_star||^2   (performance metric)
and test PERFORMANCE UNIVERSALITY: the SAME DE prediction holds when the
non-Gaussian iid design (Rademacher, centered-Exponential) replaces Gaussian.
Deterministic, CPU-only, fixed seeds.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, hashlib
from pathlib import Path
import numpy as np

t_start = time.time()
OUT = Path(__file__).with_name("results.json")

# ---------------- setup ----------------
p, n = 200, 400
gamma = p / n                 # 0.5
lam = 0.50                    # ridge penalty (Eq.1 normalization)
sigma2 = 0.25                 # label-noise variance
sigma = np.sqrt(sigma2)
C_x_is_identity = True        # isotropic design; each law has iid entries var 1

rng_ts = np.random.default_rng(42)
theta_star = rng_ts.standard_normal(p)
theta_star /= np.linalg.norm(theta_star)   # unit-norm signal

# ---------- analytic deterministic-equivalent (min-max char., isotropic) ----------
# Qbar = (lam I + C_x/(1+delta))^{-1};  delta = (1/n) tr(C_x Qbar).
# C_x = I  =>  scalar q0 = 1/(1/(1+delta)+lam),  delta = gamma*q0.
def solve_delta(gam, l):
    d = 1.0
    for _ in range(100000):
        q0 = 1.0 / (1.0 / (1.0 + d) + l)
        nd = gam * q0
        if abs(nd - d) < 1e-15:
            d = nd; break
        d = nd
    return d

def g_of(l):
    d = solve_delta(gamma, l)
    return 1.0 / (1.0 / (1.0 + d) + l)      # g = (1/p) tr E[R],  R=(S+lI)^{-1}

g = g_of(lam)
dl = lam * 1e-6
h = -(g_of(lam + dl) - g_of(lam - dl)) / (2 * dl)   # h = (1/p) tr E[R^2] = -dg/dlam

m_pred = 1.0 - lam * g
q_pred = 1.0 - 2 * lam * g + lam**2 * h + sigma2 * gamma * (g - lam * h)  # E||thetahat||^2
alpha2_pred = q_pred - m_pred**2                                          # tr(C_thetahat)
alpha2_pred_cf = lam**2 * (h - g**2) + sigma2 * gamma * (g - lam * h)     # closed-form cross-check
risk_pred = q_pred - 2 * m_pred + 1.0                                     # E||thetahat-theta*||^2

# ---- independent numeric validation of the DE formula on ONE large Gaussian design ----
rngV = np.random.default_rng(7)
nV, pV = 2000, 1000  # gamma=0.5
Xv = rngV.standard_normal((nV, pV))
Sv = Xv.T @ Xv / nV
Rv = np.linalg.inv(Sv + lam * np.eye(pV))
g_direct = np.trace(Rv) / pV
h_direct = np.trace(Rv @ Rv) / pV
print(f"[DE-check] analytic g={g:.6f} h={h:.6f} | direct(p={pV}) g={g_direct:.6f} h={h_direct:.6f}")
print(f"[DE-check] alpha2_pred={alpha2_pred:.6e} closed-form={alpha2_pred_cf:.6e}")
print(f"[theory] m*={m_pred:.6f}  alpha*^2=tr(C_th)={alpha2_pred:.6e}  risk*={risk_pred:.6f}")

# ---------------- feature laws (all iid, mean 0, var 1, C_x = I) ----------------
def gen(kind, size, rng):
    if kind == "gaussian":
        return rng.standard_normal(size)
    if kind == "rademacher":                       # +/-1  (excess kurt -2)
        return rng.integers(0, 2, size).astype(np.float64) * 2.0 - 1.0
    if kind == "exponential":                      # Exp(1)-1 (skew 2, excess kurt 6)
        return rng.standard_exponential(size) - 1.0
    raise ValueError(kind)

def run_law(kind, M, seed):
    rng = np.random.default_rng(seed)
    Th = np.empty((M, p))
    m_draw = np.empty(M); risk_draw = np.empty(M)
    for i in range(M):
        X = gen(kind, (n, p), rng)
        y = X @ theta_star + sigma * rng.standard_normal(n)
        A = X.T @ X / n + lam * np.eye(p)
        b = X.T @ y / n
        th = np.linalg.solve(A, b)
        Th[i] = th
        m_draw[i] = theta_star @ th
        risk_draw[i] = th @ th - 2 * (theta_star @ th) + 1.0
    mu = Th.mean(0)
    m_emp = float(mu @ theta_star)
    q_emp = float((Th * Th).sum(1).mean())
    alpha2_emp = float(Th.var(axis=0, ddof=1).sum())      # tr(sample cov), unbiased
    risk_emp = float(risk_draw.mean())
    m_se = float(m_draw.std(ddof=1) / np.sqrt(M))
    risk_se = float(risk_draw.std(ddof=1) / np.sqrt(M))
    return dict(law=kind, M=M, m_emp=m_emp, m_se=m_se, q_emp=q_emp,
                alpha2_emp=alpha2_emp, risk_emp=risk_emp, risk_se=risk_se)

M = 2000
laws = [("gaussian", 1001), ("rademacher", 1002), ("exponential", 1003)]
rows = []
print("\n[PartA] Performance/mean-cov universality: DE theory vs empirical ridge")
print(f"  {'law':11s} {'m_emp':>9s} {'m*':>9s} {'gap%':>6s} | "
      f"{'a2_emp':>9s} {'a2*':>9s} {'gap%':>6s} | {'risk_emp':>9s} {'risk*':>9s} {'gap%':>6s}")
for kind, sd in laws:
    r = run_law(kind, M, sd)
    r["m_gap_rel"] = abs(r["m_emp"] - m_pred) / abs(m_pred)
    r["alpha2_gap_rel"] = abs(r["alpha2_emp"] - alpha2_pred) / abs(alpha2_pred)
    r["risk_gap_rel"] = abs(r["risk_emp"] - risk_pred) / abs(risk_pred)
    r["m_within3se"] = bool(abs(r["m_emp"] - m_pred) < 3 * r["m_se"])
    rows.append(r)
    print(f"  {kind:11s} {r['m_emp']:9.5f} {m_pred:9.5f} {100*r['m_gap_rel']:6.2f} | "
          f"{r['alpha2_emp']:9.5f} {alpha2_pred:9.5f} {100*r['alpha2_gap_rel']:6.2f} | "
          f"{r['risk_emp']:9.5f} {risk_pred:9.5f} {100*r['risk_gap_rel']:6.2f}")

# cross-law universality: max pairwise relative spread of risk across the 3 laws
risks = np.array([r["risk_emp"] for r in rows])
alpha2s = np.array([r["alpha2_emp"] for r in rows])
risk_spread = float((risks.max() - risks.min()) / risks.mean())
alpha2_spread = float((alpha2s.max() - alpha2s.min()) / alpha2s.mean())
print(f"  cross-law risk spread = {100*risk_spread:.2f}%   alpha^2 spread = {100*alpha2_spread:.2f}%")

max_m_gap = max(r["m_gap_rel"] for r in rows)
max_a2_gap = max(r["alpha2_gap_rel"] for r in rows)
max_risk_gap = max(r["risk_gap_rel"] for r in rows)
TOL = 0.04
verdict_A = bool(max_m_gap < TOL and max_a2_gap < TOL and max_risk_gap < TOL)
print(f"\n[Verdict A] max gaps: m={100*max_m_gap:.2f}% alpha2={100*max_a2_gap:.2f}% "
      f"risk={100*max_risk_gap:.2f}%  (tol {100*TOL:.0f}%) -> universality_holds={verdict_A}")

results = dict(
    claim="Min-max characterization approximates mean mu_thetahat and covariance "
          "C_thetahat of high-dim ERM under non-Gaussian data (Thm 4.3 / Sec 5).",
    setup=dict(p=p, n=n, gamma=gamma, lam=lam, sigma2=sigma2, M=M,
               C_x="identity", design_laws=[k for k, _ in laws]),
    deterministic_equivalent=dict(g=float(g), h=float(h), g_direct=float(g_direct),
                                  h_direct=float(h_direct), m_pred=float(m_pred),
                                  alpha2_pred=float(alpha2_pred),
                                  alpha2_pred_cf=float(alpha2_pred_cf),
                                  q_pred=float(q_pred), risk_pred=float(risk_pred)),
    per_law=rows,
    cross_law=dict(risk_spread_rel=risk_spread, alpha2_spread_rel=alpha2_spread),
    max_rel_gap=dict(m=float(max_m_gap), alpha2=float(max_a2_gap), risk=float(max_risk_gap)),
    tol=TOL,
    verdict_universality_holds=verdict_A,
    runtime_s=None,
)
results["runtime_s"] = round(time.time() - t_start, 2)
OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\n[done] wrote {OUT.name}  runtime={results['runtime_s']}s")
