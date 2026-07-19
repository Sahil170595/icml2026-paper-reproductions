"""Claim 1 reproduction --- 'Prior Diffusiveness and Regret in the Linear-Gaussian
Bandit' (Zhu, Duchi, Van Roy; OpenReview GeYKOC4BzB, arXiv 2601.02022).

Corollary 2:  Reg(T) = O~( sigma*d*sqrt(T)  +  d*r*sqrt(Tr(Sigma0)) ).
Sharp Theorem 1 burn-in term is  3 r sqrt(d) Tr(Sigma0^{1/2}) C1 + sqrt(2 r^2 Tr(Sigma0)),
and Cauchy-Schwarz Tr(Sigma0^{1/2}) <= sqrt(d Tr(Sigma0)) gives Corollary 2's dr sqrt(Tr(Sigma0)).
The prior 'burn-in' decouples ADDITIVELY from the minimax rate sigma*d*sqrt(T); only the
noise sigma scales sqrt(T).  Prior work (Kalkanli & Ozgur 2020) was MULTIPLICATIVE:
    Reg(T) <~ d sqrt( T (sigma^2 + r^2 Tr(Sigma0)) log(1+T/d) ).

Independent batched-NumPy Thompson Sampler, canonical model
(theta*~N(0,Sigma0), A=r B_2^d, R=theta*'A+N(0,sigma^2)). Deterministic; fixed seeds;
single-thread BLAS; prints ONLY measured numbers.

Tests:
 (a) sqrt(T) leading rate (log-log slope ~ 0.5);
 (b) DISCRIMINATION: leading sqrt(T) coeff a(s) ~ INDEPENDENT of prior scale s ==> additive,
     rejecting multiplicative a(s)/a(1)=sqrt((sigma^2+r^2 s^2 d)/(sigma^2+r^2 d));
 (c) burn-in ISOLATION (low-noise sigma=0.02): regret SATURATES (T-independent -> additive)
     and its magnitude scales ~linearly with Tr(Sigma0^{1/2}) (hence sqrt(Tr(Sigma0)));
 (d) noise control: leading coeff a(sigma) ~ proportional to sigma;
 (e) anisotropy control: burn-in tracks Tr(Sigma0^{1/2}), not the raw eigen-profile.
"""
import json, os, time
from pathlib import Path
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent

def ts_curve(Sig0, d, sigma, r, T, M, seed):
    rng = np.random.default_rng(seed)
    Sig0inv = np.linalg.inv(Sig0); L0 = np.linalg.cholesky(Sig0)
    theta = (L0 @ rng.standard_normal((M, d, 1)))[..., 0]
    Lam = np.broadcast_to(Sig0inv, (M, d, d)).copy(); b = np.zeros((M, d, 1))
    astar = r * np.linalg.norm(theta, axis=1)
    Z = rng.standard_normal((T, M, d, 1)); noise = rng.normal(0, sigma, (T, M))
    reg = np.empty((T, M))
    for t in range(T):
        Lc = np.linalg.cholesky(Lam); mu = np.linalg.solve(Lam, b)
        th = (mu + np.linalg.solve(np.swapaxes(Lc, 1, 2), Z[t]))[..., 0]
        A = r * th / (np.linalg.norm(th, axis=1, keepdims=True) + 1e-12)
        dot = np.sum(theta * A, axis=1); reg[t] = astar - dot; R = dot + noise[t]
        Lam += A[:, :, None] * A[:, None, :] / sigma**2
        b += (A * R[:, None] / sigma**2)[..., None]
    return np.cumsum(reg.mean(axis=1))

def fit_a_c(curve, T):
    tt = np.arange(1, T + 1)
    X = np.vstack([np.sqrt(tt), np.ones_like(tt, dtype=float)]).T
    a, c = np.linalg.lstsq(X[200:], curve[200:], rcond=None)[0]
    return float(a), float(c)

t0 = time.time()
d, r = 5, 1.0
res = {"config": {"d": d, "r": r}, }

# --- (a,b) sigma=1 sweep: sqrt(T) rate + additive-vs-multiplicative discrimination ---
sigma, T, M = 1.0, 2000, 1200
Sdisc = [1.0, 2.0, 4.0, 8.0]
prm = {}; curves = {}
for i, s in enumerate(Sdisc):
    c = ts_curve((s**2) * np.eye(d), d, sigma, r, T, M, seed=100 + i)
    curves[s] = c; a, cc = fit_a_c(c, T)
    prm[str(s)] = {"a_leading_sqrtT": a, "final_regret": float(c[-1])}
tt = np.arange(1, T + 1)
slope1 = float(np.polyfit(np.log(tt[100:]), np.log(curves[1.0][100:]), 1)[0])
a1 = prm["1.0"]["a_leading_sqrtT"]
disc = {}
for s in Sdisc:
    meas = prm[str(s)]["a_leading_sqrtT"] / a1
    mult = float(np.sqrt((sigma**2 + r**2 * s**2 * d) / (sigma**2 + r**2 * d)))
    disc[str(s)] = {"a": prm[str(s)]["a_leading_sqrtT"], "measured_ratio": float(meas),
                    "additive_pred": 1.0, "multiplicative_pred": mult}
res["sqrtT_loglog_slope_s1"] = slope1
res["discrimination_sigma1"] = disc

# --- (c) low-noise burn-in isolation: saturation (T-indep) + scaling vs Tr(Sigma0^{1/2}) ---
sig_lo, Tlo, Mlo = 0.02, 400, 2000
Sb = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
burn = {}; xs = []; ys = []
sat_shape = None
for i, s in enumerate(Sb):
    Sig0 = (s**2) * np.eye(d)
    c = ts_curve(Sig0, d, sig_lo, r, Tlo, Mlo, seed=700 + i)
    trhalf = float(np.sum(np.sqrt(np.linalg.eigvalsh(Sig0))))
    sqrtTr = float(np.sqrt(np.trace(Sig0)))
    burn[str(s)] = {"sqrtTr": sqrtTr, "Tr_Sigma0_half": trhalf, "reg_sat": float(c[-1]),
                    "reg_q": [float(c[Tlo // 4]), float(c[Tlo // 2]), float(c[-1])]}
    xs.append(trhalf); ys.append(float(c[-1]))
    if s == 4.0: sat_shape = burn[str(s)]["reg_q"]
xs = np.array(xs); ys = np.array(ys)
ll = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
A2 = np.vstack([xs, np.ones_like(xs)]).T
sl, ic = np.linalg.lstsq(A2, ys, rcond=None)[0]
yh = A2 @ np.array([sl, ic]); r2 = float(1 - np.sum((ys - yh)**2) / np.sum((ys - ys.mean())**2))
res["burnin_lownoise"] = {"sigma": sig_lo, "T": Tlo, "loglog_slope_vs_TrHalf": ll,
                          "linear_slope": float(sl), "linear_R2": r2,
                          "saturation_example_s4_q25_q50_q100": sat_shape, "per_s": burn}

# --- (d) noise control: leading coeff proportional to sigma (s=2) ---
noise_ctrl = {}
for j, sg in enumerate([0.5, 1.0, 2.0]):
    c = ts_curve(4.0 * np.eye(d), d, sg, r, 1200, 1000, seed=300 + j)
    a, _ = fit_a_c(c, 1200)
    noise_ctrl[str(sg)] = {"a": a, "a_over_sigma": float(a / sg)}
res["noise_control_s2"] = noise_ctrl

# --- (e) anisotropy control (equal Tr=20, different Tr(Sigma0^{1/2})) ---
aniso = {}
for k, (name, Sig0) in enumerate({"iso_4I": 4.0 * np.eye(d),
                                  "aniso_16_2_1_.5_.5": np.diag([16., 2., 1., .5, .5])}.items()):
    c = ts_curve(Sig0, d, sig_lo, r, Tlo, Mlo, seed=900 + k)
    aniso[name] = {"trace": float(np.trace(Sig0)),
                   "Tr_Sigma0_half": float(np.sum(np.sqrt(np.linalg.eigvalsh(Sig0)))),
                   "reg_sat": float(c[-1])}
res["anisotropy_control"] = aniso

res["runtime_s"] = round(time.time() - t0, 2)
# NOTE: writes results_smallscale.json -- results.json in this directory holds the
# real-scale summary written by ../repro_scale_c1.py combine.
(HERE / "results_smallscale.json").write_text(json.dumps(res, indent=1))

print("== Claim 1: additive prior burn-in in the linear-Gaussian bandit ==")
print("d=%d r=%g  runtime=%.1fs" % (d, r, res["runtime_s"]))
print("\n(a) sqrt(T) scaling (sigma=1): log-log slope (s=1) = %.3f   [theory 0.5]" % slope1)
print("\n(b) DISCRIMINATION -- leading sqrt(T) coeff a(s) vs prior scale s (sigma=1,d=5,r=1):")
print("  s    a(s)   a(s)/a(1)   additive_pred   multiplicative_pred")
for s in Sdisc:
    dd = disc[str(s)]
    print("%3.0f  %6.3f    %5.2f        %5.2f            %5.2f"
          % (s, dd["a"], dd["measured_ratio"], dd["additive_pred"], dd["multiplicative_pred"]))
print("\n(c) BURN-IN ISOLATION (sigma=0.02): regret saturates (T-independent) -> pure additive term")
print("    saturation (s=4): Reg[T/4]=%.2f  Reg[T/2]=%.2f  Reg[T]=%.2f" % tuple(sat_shape))
print("    scaling vs Tr(Sigma0^{1/2}):  log-log slope=%.3f [theory 1.0]  linear R2=%.4f" % (ll, r2))
print("  s   sqrtTr   Tr(Sig^.5)   Reg_sat   Reg/Tr(Sig^.5)")
for s in Sb:
    bb = burn[str(s)]
    print("%3.0f  %6.3f   %7.3f    %7.3f      %6.3f"
          % (s, bb["sqrtTr"], bb["Tr_Sigma0_half"], bb["reg_sat"], bb["reg_sat"] / bb["Tr_Sigma0_half"]))
print("\n(d) NOISE CONTROL (s=2): leading coeff a should be proportional to sigma")
for sg in [0.5, 1.0, 2.0]:
    print("   sigma=%.1f  a=%.3f  a/sigma=%.3f" % (sg, noise_ctrl[str(sg)]["a"], noise_ctrl[str(sg)]["a_over_sigma"]))
print("\n(e) ANISOTROPY CONTROL (both Tr(Sigma0)=20; burn-in tracks Tr(Sigma0^{1/2})):")
for name, v in aniso.items():
    print("   %-20s Tr=%.1f  Tr(Sig^.5)=%.2f  Reg_sat=%.2f" % (name, v["trace"], v["Tr_Sigma0_half"], v["reg_sat"]))
