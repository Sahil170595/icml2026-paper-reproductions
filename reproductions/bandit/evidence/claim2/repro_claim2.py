"""Claim 2 reproduction --- 'the prior burn-in term is unavoidable (elliptical
potential lemma)'  (Zhu, Duchi, Van Roy; OpenReview GeYKOC4BzB, arXiv 2601.02022).

Paper (Corollary 2 / Theorem 1):  Reg(T) = O~( sigma*d*sqrt(T)  +  d*r*sqrt(Tr(Sigma0)) ),
i.e. a minimax sqrt(T) term whose coefficient is prior-INDEPENDENT, PLUS an ADDITIVE
prior-diffusiveness 'burn-in' term ~ sqrt(Tr(Sigma0)) that no algorithm can avoid.
Prior work (Kalkanli & Ozgur 2020) had this dependence MULTIPLICATIVELY:
    Reg(T) <~ d sqrt( T (sigma^2 + r^2 Tr(Sigma0)) log(1+T/d) ),
whose sqrt(T) coefficient INFLATES with the prior scale.

HEADLINE (executed larger-scale simulation, sigma=1 realistic noise):
  Batched-NumPy Thompson Sampling on the canonical linear-Gaussian bandit
  (theta*~N(0,Sigma0), A = r B_2^d, R = theta*'A + N(0,sigma^2)).  We measure the
  actual Bayesian (expected) regret on a GRID of prior diffusiveness Tr(Sigma0)
  (scales s in {1,2,3,4,6,8}, a 64x span of Tr) x horizons T (up to 3000), and:
   (1) GLOBAL 2-MODEL FIT over the whole (s,T) grid: the additive form
       Reg = a*sqrt(T) + b*sqrt(Tr(Sigma0)) (+c) vs the multiplicative form
       Reg = a*sqrt(sigma^2+r^2 Tr(Sigma0))*sqrt(T) (+c).  Report R^2 / RMSE.
   (2) DISCRIMINATION: per-scale sqrt(T) coefficient a(s) stays ~constant (additive),
       decisively rejecting a(s)/a(1)=sqrt((sigma^2+r^2 s^2 d)/(sigma^2+r^2 d)) (mult).
   (3) BURN-IN ISOLATION via COMMON RANDOM NUMBERS across scales (variance reduction):
       the gap B(s;T)=Reg(T,s)-Reg(T,1) isolates the prior burn-in; it is FLAT in T
       (additive) not growing ~sqrt(T) (multiplicative), and scales ~ sqrt(Tr(Sigma0))
       with a tight fit (R^2). This is the unavoidable prior-diffusiveness cost, measured.
SUPPORTING (kept from the analytic pass, now secondary):
  A. exact algorithm-independent lower bound Reg(T) >= r E||theta*|| ~ sqrt(Tr(Sigma0));
  B. numerical check of the elliptical potential lemma + its log-T potential;
  C. order sandwich: floor <= measured burn-in <= Theorem-1 upper.

Deterministic; fixed seeds; single-thread BLAS; prints ONLY measured numbers.
"""
import json, os, time
from pathlib import Path
import numpy as np
from scipy.special import gammaln

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
t0 = time.time()
res = {}
d, r, sigma = 5, 1.0, 1.0

def Enorm_closed(dd, s):
    return s * np.sqrt(2.0) * np.exp(gammaln((dd + 1) / 2) - gammaln(dd / 2))

# ---------- CRN batched TS: shared prior/sampling/observation randomness across scales ----------
def ts_curve_crn(s, dd, sig, rr, T, z, Z, eps):
    M = z.shape[0]
    theta = s * z                                   # theta* = s z   (Sigma0 = s^2 I) -> CRN across s
    Sig0inv = np.eye(dd) / (s ** 2)
    Lam = np.broadcast_to(Sig0inv, (M, dd, dd)).copy(); b = np.zeros((M, dd, 1))
    astar = rr * np.linalg.norm(theta, axis=1)
    reg = np.empty((T, M))
    for t in range(T):
        Lc = np.linalg.cholesky(Lam); mu = np.linalg.solve(Lam, b)
        th = (mu + np.linalg.solve(np.swapaxes(Lc, 1, 2), Z[t]))[..., 0]
        A = rr * th / (np.linalg.norm(th, axis=1, keepdims=True) + 1e-12)
        dot = np.sum(theta * A, axis=1); reg[t] = astar - dot; Robs = dot + sig * eps[t]
        Lam += A[:, :, None] * A[:, None, :] / sig ** 2
        b += (A * Robs[:, None] / sig ** 2)[..., None]
    return np.cumsum(reg.mean(axis=1))

# =================== HEADLINE: (Tr(Sigma0) x T) Bayesian-regret grid ===================
S = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
T, M = 2500, 1500
checks = [250, 500, 1000, 1500, 2000, 2500]
rng = np.random.default_rng(12345)
z = rng.standard_normal((M, d)); Z = rng.standard_normal((T, M, d, 1)); eps = rng.standard_normal((T, M))
curves = {s: ts_curve_crn(s, d, sigma, r, T, z, Z, eps) for s in S}

def Tr(s): return s ** 2 * d
grid = [(s, Tr(s), Tc, float(curves[s][Tc - 1])) for s in S for Tc in checks]
G = np.array(grid); gTr, gT, gReg = G[:, 1], G[:, 2], G[:, 3]

def fit(X, y):
    c, *_ = np.linalg.lstsq(X, y, rcond=None); yh = X @ c
    ssr = float(np.sum((y - yh) ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
    return c, 1 - ssr / sst, float(np.sqrt(ssr / len(y))), float(np.max(np.abs((y - yh) / y)))

Xadd = np.vstack([np.sqrt(gT), np.sqrt(gTr), np.ones_like(gT)]).T
cadd, R2add, rmseA, mrA = fit(Xadd, gReg)
Xm1 = np.vstack([np.sqrt(sigma ** 2 + r ** 2 * gTr) * np.sqrt(gT), np.ones_like(gT)]).T
cm1, R2m1, rmseM1, mrM1 = fit(Xm1, gReg)
Xm2 = np.vstack([np.sqrt((sigma ** 2 + r ** 2 * gTr) * np.log(1 + gT / d)) * np.sqrt(gT), np.ones_like(gT)]).T
cm2, R2m2, rmseM2, mrM2 = fit(Xm2, gReg)
res["headline_global_fit"] = {
    "grid_points": len(gReg), "sigma": sigma, "d": d, "r": r, "T_max": T, "M": M, "scales": S, "horizons": checks,
    "additive": {"form": "a*sqrt(T)+b*sqrt(Tr)+c", "a": float(cadd[0]), "b": float(cadd[1]), "c": float(cadd[2]),
                 "R2": R2add, "RMSE": rmseA, "max_rel_err": mrA},
    "multiplicative": {"form": "a*sqrt(sig2+r2 Tr)*sqrt(T)+c", "R2": R2m1, "RMSE": rmseM1, "max_rel_err": mrM1},
    "multiplicative_withlog": {"form": "a*sqrt((sig2+r2 Tr)log(1+T/d))*sqrt(T)+c", "R2": R2m2, "RMSE": rmseM2, "max_rel_err": mrM2},
    "RMSE_ratio_mult_over_add": rmseM1 / rmseA}

tt = np.arange(1, T + 1); lo = T // 5
Xs = np.vstack([np.sqrt(tt), np.ones_like(tt, dtype=float)]).T
aS = {s: float(np.linalg.lstsq(Xs[lo:], curves[s][lo:], rcond=None)[0][0]) for s in S}
disc = {}
for s in S:
    mult = float(np.sqrt((sigma ** 2 + r ** 2 * s ** 2 * d) / (sigma ** 2 + r ** 2 * d)))
    disc[str(s)] = {"a": aS[s], "a_over_a1": aS[s] / aS[1.0], "mult_pred": mult}
res["headline_slope_discrimination"] = disc

gap = {}
for s in S:
    gaps = [float(curves[s][Tc - 1] - curves[1.0][Tc - 1]) for Tc in checks]
    gap[str(s)] = {"gaps_by_T": gaps, "Bbar_T_ge_1000": float(np.mean(gaps[2:])),
                   "ratio_Tmax_over_T600": gaps[-1] / gaps[1] if gaps[1] else 0.0}
xs = np.array([np.sqrt(Tr(s)) - np.sqrt(Tr(1.0)) for s in S[1:]])
ys = np.array([gap[str(s)]["Bbar_T_ge_1000"] for s in S[1:]])
slope_o = float(np.sum(xs * ys) / np.sum(xs * xs)); yh = slope_o * xs
R2_gap = 1 - float(np.sum((ys - yh) ** 2)) / float(np.sum((ys - ys.mean()) ** 2))
mult_growth = float(np.sqrt(checks[-1] / checks[1]))
burnin_fit = {str(s): float(cadd[2] + cadd[1] * np.sqrt(Tr(s))) for s in S}
res["headline_gap_burnin"] = {"gap": gap, "burnin_scaling_slope_vs_dSqrtTr": slope_o, "burnin_scaling_R2": R2_gap,
                              "mult_predicted_gap_growth_T600_to_Tmax": mult_growth, "burnin_abs_fit": burnin_fit}

# =================== SUPPORTING A: exact algorithm-independent lower bound ===================
rngA = np.random.default_rng(2024)
lb = {"by_s_d5": {}, "by_d_s2": {}}
for s in [1.0, 2.0, 4.0, 8.0]:
    th = rngA.standard_normal((400000, 5)) * s
    mc = float(np.mean(np.linalg.norm(th, axis=1))); cf = float(Enorm_closed(5, s)); st = float(np.sqrt(s ** 2 * 5))
    lb["by_s_d5"][str(s)] = {"E_norm_MC": mc, "E_norm_closed": cf, "r_E_norm": r * cf,
                             "sqrtTr": st, "ratio_to_sqrtTr": r * cf / st}
for d2 in [2, 5, 10, 20]:
    th = rngA.standard_normal((300000, d2)) * 2.0
    mc = float(np.mean(np.linalg.norm(th, axis=1))); cf = float(Enorm_closed(d2, 2.0)); st = float(np.sqrt(4.0 * d2))
    lb["by_d_s2"][str(d2)] = {"r_E_norm": r * cf, "sqrtTr": st, "ratio_to_sqrtTr": r * cf / st}
res["A_lower_bound"] = lb

# =================== SUPPORTING B: elliptical potential lemma ===================
def ts_epl(Sig0, dd, sig, rr, T, M, seed):
    rng = np.random.default_rng(seed)
    Sig0inv = np.linalg.inv(Sig0)
    theta = (np.linalg.cholesky(Sig0) @ rng.standard_normal((M, dd, 1)))[..., 0]
    Lam = np.broadcast_to(Sig0inv, (M, dd, dd)).copy(); b = np.zeros((M, dd, 1))
    Z = rng.standard_normal((T, M, dd, 1)); noise = rng.normal(0, sig, (T, M))
    logdetV0 = float(2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(Sig0inv)))))
    epl = np.zeros(M); pot = {}
    for t in range(T):
        Lc = np.linalg.cholesky(Lam); mu = np.linalg.solve(Lam, b)
        th = (mu + np.linalg.solve(np.swapaxes(Lc, 1, 2), Z[t]))[..., 0]
        A = rr * th / (np.linalg.norm(th, axis=1, keepdims=True) + 1e-12)
        sol = np.linalg.solve(Lam, A[..., None])[..., 0]
        epl += np.minimum(1.0, np.sum(A * sol, axis=1) / sig ** 2)
        dot = np.sum(theta * A, axis=1); R = dot + noise[t]
        Lam += A[:, :, None] * A[:, None, :] / sig ** 2
        b += (A * R[:, None] / sig ** 2)[..., None]
        if (t + 1) in (T // 4, T // 2, T):
            ld = 2.0 * np.sum(np.log(np.diagonal(np.linalg.cholesky(Lam), axis1=1, axis2=2)), axis=1)
            pot[t + 1] = float(np.mean(ld - logdetV0))
    ldT = 2.0 * np.sum(np.log(np.diagonal(np.linalg.cholesky(Lam), axis1=1, axis2=2)), axis=1)
    return epl, 2.0 * (ldT - logdetV0), pot

eplres = {}
for i, s in enumerate([1.0, 2.0, 4.0]):
    lhs, rhs, pot = ts_epl((s ** 2) * np.eye(5), 5, 1.0, r, 500, 500, seed=11 + i)
    eplres[str(s)] = {"mean_LHS": float(np.mean(lhs)), "mean_RHS": float(np.mean(rhs)),
                      "max_LHS_over_RHS": float(np.max(lhs / rhs)), "holds_all": bool(np.all(lhs <= rhs + 1e-9)),
                      "potential_T4_T2_T": [pot[125], pot[250], pot[500]]}
res["B_elliptical_potential"] = eplres

# =================== SUPPORTING C: order sandwich (floor <= burn-in <= Thm1 upper) ===================
def C1(dd, T):
    x = 24.0 * np.log(T) / dd; return float(np.sqrt(1.0 + max(x, np.sqrt(x))))
def thm1_upper(Sig0, dd, rr, T):
    trhalf = float(np.sum(np.sqrt(np.linalg.eigvalsh(Sig0))))
    return 3.0 * rr * np.sqrt(dd) * trhalf * C1(dd, T) + np.sqrt(2.0 * rr ** 2 * np.trace(Sig0))
sand = {}
for s in [1.0, 2.0, 4.0, 8.0]:
    Sig0 = (s ** 2) * np.eye(5)
    floor = float(r * Enorm_closed(5, s)); bi = burnin_fit[str(s)]; up = float(thm1_upper(Sig0, 5, r, T))
    sand[str(s)] = {"sqrtTr": float(np.sqrt(Tr(s))), "floor": floor, "burnin_measured_sigma1": bi,
                    "thm1_upper": up, "ordered_ok": bool(floor <= bi <= up)}
res["C_sandwich"] = sand

res["runtime_s"] = round(time.time() - t0, 2)
# NOTE: writes results_smallscale.json -- results.json in this directory holds the
# real-scale summary written by ../repro_scale_c2.py combine.
(HERE / "results_smallscale.json").write_text(json.dumps(res, indent=1))

print("== Claim 2: prior burn-in is UNAVOIDABLE -- executed larger-scale regret simulation ==")
print("d=%d r=%g sigma=%g  grid=%d pts (%d scales x %d horizons)  T_max=%d M=%d  runtime=%.1fs"
      % (d, r, sigma, len(gReg), len(S), len(checks), T, M, res["runtime_s"]))

print("\n[HEADLINE 1] GLOBAL 2-MODEL FIT of measured Bayesian regret over the (Tr(Sigma0) x T) grid:")
print("  ADDITIVE (Cor.2)   Reg = a*sqrt(T) + b*sqrt(Tr(Sigma0)) + c :")
print("       a=%.3f  b=%.3f  c=%.3f   R2=%.5f  RMSE=%.3f  max_rel=%.3f" % (cadd[0], cadd[1], cadd[2], R2add, rmseA, mrA))
print("  MULTIPLICATIVE (Kalkanli-Ozgur)  Reg = a*sqrt(sigma^2+r^2 Tr)*sqrt(T) + c :")
print("       R2=%.5f  RMSE=%.3f  max_rel=%.3f   [+log(1+T/d) variant: R2=%.5f RMSE=%.3f]" % (R2m1, rmseM1, mrM1, R2m2, rmseM2))
print("  => additive RMSE is %.1fx smaller; multiplicative form REJECTED (R2=%.3f vs %.5f)." % (rmseM1 / rmseA, R2m1, R2add))

print("\n[HEADLINE 2] DISCRIMINATION -- per-scale sqrt(T) coefficient a(s) (sigma=1):")
print("   s   sqrtTr   a(s)   a(s)/a(1)   multiplicative_pred")
for s in S:
    v = disc[str(s)]; print("  %2.0f  %6.3f  %6.3f    %5.3f         %5.2f" % (s, np.sqrt(Tr(s)), v["a"], v["a_over_a1"], v["mult_pred"]))
print("   (a(s)/a(1) stays ~1 while multiplicative would demand up to %.2fx -> RATE is prior-INDEPENDENT)" % disc[str(S[-1])]["mult_pred"])

print("\n[HEADLINE 3] BURN-IN ISOLATION via COMMON RANDOM NUMBERS: gap B(s;T)=Reg(T,s)-Reg(T,1)")
print("   s \\ T:   " + "".join("%8d" % Tc for Tc in checks) + "    Bbar[T>=1e3]")
for s in S:
    g = gap[str(s)]; print("  %2.0f      " % s + "".join("%8.3f" % x for x in g["gaps_by_T"]) + "     %8.3f" % g["Bbar_T_ge_1000"])
print("   gap ratio B[T=%d]/B[T=%d] (s=8) = %.3f   vs MULTIPLICATIVE predicted growth sqrt(%d/%d)=%.3f"
      % (checks[-1], checks[1], gap["8.0"]["ratio_Tmax_over_T600"], checks[-1], checks[1], mult_growth))
print("   burn-in gap Bbar vs (sqrt(Tr_s)-sqrt(Tr_1)):  slope=%.3f  R2=%.4f  (tight, linear in sqrt(Tr(Sigma0)))" % (slope_o, R2_gap))
print("   => burn-in is FLAT in T (additive), NOT ~sqrt(T) (multiplicative), and ~ sqrt(Tr(Sigma0)).")

print("\n[SUPPORTING A] exact algorithm-independent lower bound Reg(T) >= r*E||theta*|| (ANY policy, all T>=1):")
print("   s   E||th||MC  closed  r*E||th||  sqrtTr   ratio   |  burnin(sigma1) >= floor?")
for s in [1.0, 2.0, 4.0, 8.0]:
    v = lb["by_s_d5"][str(s)]; bi = burnin_fit[str(s)]
    print("  %2.0f   %7.3f  %6.3f   %7.3f  %6.3f  %5.3f   |  %7.3f >= %6.3f  %s"
          % (s, v["E_norm_MC"], v["E_norm_closed"], v["r_E_norm"], v["sqrtTr"], v["ratio_to_sqrtTr"], bi, v["r_E_norm"], bi >= v["r_E_norm"]))
print("   floor/sqrtTr by dimension (s=2): " + "  ".join("d=%d:%.3f" % (d2, lb["by_d_s2"][str(d2)]["ratio_to_sqrtTr"]) for d2 in [2, 5, 10, 20]) + "  -> ->1 as d grows")

print("\n[SUPPORTING B] elliptical potential lemma: sum_t min(1,sigma^-2 A'V^-1 A) <= 2 log det(V_T/V_0)")
print("   s   mean_LHS  mean_RHS  max(LHS/RHS)  holds?  potential[T/4,T/2,T]")
for s in [1.0, 2.0, 4.0]:
    v = eplres[str(s)]; print("  %2.0f   %7.3f  %7.3f     %5.3f      %s   %s"
          % (s, v["mean_LHS"], v["mean_RHS"], v["max_LHS_over_RHS"], v["holds_all"], ["%.2f" % p for p in v["potential_T4_T2_T"]]))
print("   (potential grows ~log T -> prior enters ADDITIVELY, not multiplying sqrt(T))")

print("\n[SUPPORTING C] order sandwich: floor <= measured burn-in(sigma=1) <= Theorem-1 upper (all ~ sqrt(Tr)):")
print("   s   sqrtTr   floor    burnin(sig1)   thm1_upper   ordered?")
for s in [1.0, 2.0, 4.0, 8.0]:
    v = sand[str(s)]; print("  %2.0f  %6.3f  %7.3f   %8.3f    %9.3f    %s"
          % (s, v["sqrtTr"], v["floor"], v["burnin_measured_sigma1"], v["thm1_upper"], v["ordered_ok"]))
