"""
Claim 1 - Unifying SOC framework for Diffusion Bridge (DB) and Flow Matching (FM).
Paper: "Diffusion Bridge or Flow Matching? A Unifying Framework and Comparative
Analysis" (OpenReview aIFgQusnPy / arXiv 2509.24531), Section 4 + Proposition 4.1.

VERIFIABLE IDENTITY (the paper's central unification result):
  Proposition 4.1: under theta_t -> 0 and g_t = 1, the DB SOC problem (Eq 8)
  reduces EXACTLY to the FM SOC problem (Eq 9), and DB's optimal controller
  reduces to FM's.  With g_t = 1 the paper fixes theta_t = 1/(2 lambda^2)
  (from g_t^2 = 2 lambda^2 theta_t), so lambda^2 = 1/(2 theta).

  DB optimal controller (Eq 17, gamma->inf, g_t=1, constant theta):
      u_DB(t,x) = e^{-2 thbar} (x1 - x) / sig2,   thbar = theta(1-t),
                  sig2 = lambda^2 (1 - e^{-2 thbar}) = (1/(2theta))(1-e^{-2theta(1-t)})
      => u_DB(t,x) = 2 theta e^{-2theta(1-t)} (x1 - x) / (1 - e^{-2theta(1-t)}).
  FM optimal controller (Eq 10/17):
      u_FM(t,x) = (x1 - x)/(1 - t)   ==   x1 - x0   (along straight line x_t).

TESTS (all numeric, deterministic):
  T1  Controller reduction: max_t || u_DB(.;theta) - u_FM || -> 0 as theta -> 0,
      with first-order O(theta) rate (log-log slope of error vs theta ~ +1).
  T2  Drift vanishes: || theta (x1 - x_t) || -> 0 as theta -> 0 (DB dynamics
      dx=[theta(x1-x)+u]dt collapse to FM dynamics dx=u dt).
  T3  FM two-form identity (Eq 10): (x1 - x_t)/(1-t) == (x1 - x0) along the FM
      straight-line interpolant x_t=(1-t)x0+t x1, to machine precision.
  T4  Cost reduction (Prop 4.1 at the objective level): J_DB(theta) -> J_FM as
      theta -> 0, where J_FM = 1/2 ||x1-x0||^2 and J_DB is the exact min-energy
      LQ control cost theta/(e^{2theta}-1) ||x1-x0||^2 (derived + simulated).
"""
import json, numpy as np

rng = np.random.default_rng(20260717)
d = 16
x0 = rng.standard_normal(d)
x1 = rng.standard_normal(d) * 1.5 + 0.7      # generic distinct endpoint
disp = x1 - x0
disp2 = float(disp @ disp)                    # ||x1 - x0||^2

def u_DB(t, x, theta):
    a = 2.0 * theta * (1.0 - t)               # = 2 thbar
    return 2.0 * theta * np.exp(-a) * (x1 - x) / (1.0 - np.exp(-a))

def u_FM(t, x):
    return (x1 - x) / (1.0 - t)

# ---- T1: controller reduction over a t-grid, at a fixed reference trajectory ----
ts = np.linspace(0.02, 0.98, 25)
thetas = np.array([1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 1e-4, 1e-6])
maxerr = []
for th in thetas:
    e = 0.0
    for t in ts:
        xt = (1.0 - t) * x0 + t * x1          # common reference point
        e = max(e, float(np.linalg.norm(u_DB(t, xt, th) - u_FM(t, xt))))
    maxerr.append(e)
maxerr = np.array(maxerr)
# fit order: log(err) vs log(theta) over the small-theta regime
m = thetas <= 0.1
rate_slope = float(np.polyfit(np.log10(thetas[m]), np.log10(maxerr[m]), 1)[0])

# ---- T2: drift term magnitude along reference path, vs theta ----
drift_norm = []
for th in thetas:
    dn = 0.0
    for t in ts:
        xt = (1.0 - t) * x0 + t * x1
        dn = max(dn, float(np.linalg.norm(th * (x1 - xt))))
    drift_norm.append(dn)
drift_norm = np.array(drift_norm)

# ---- T3: FM two-form identity (x1-xt)/(1-t) == x1-x0 ----
t3err = 0.0
for t in ts:
    xt = (1.0 - t) * x0 + t * x1
    t3err = max(t3err, float(np.linalg.norm((x1 - xt) / (1.0 - t) - (x1 - x0))))

# ---- T4: cost reduction J_DB(theta) -> J_FM as theta->0 ----
def J_FM():
    return 0.5 * disp2
def J_DB_analytic(theta):
    # exact min-energy LQ control cost for dx=-theta x+theta x1+u, x0->x1
    return theta / (np.expm1(2.0 * theta)) * disp2
def J_DB_sim(theta, Nt=200000, tmax=1.0 - 1e-6):
    # simulate closed-loop ODE dx = theta(x1-x) + u_DB, integrate 1/2||u||^2
    tg = np.linspace(0.0, tmax, Nt + 1)
    dt = tg[1] - tg[0]
    x = x0.copy(); cost = 0.0
    for i in range(Nt):
        t = tg[i]
        u = u_DB(t, x, theta)
        cost += 0.5 * float(u @ u) * dt
        x = x + (theta * (x1 - x) + u) * dt      # explicit Euler
    land = float(np.linalg.norm(x - x1))         # endpoint pinning check
    return cost, land

Jfm = J_FM()
cost_rows = []
for th in [1.0, 0.5, 0.125, 0.03125, 1e-3]:
    Ja = J_DB_analytic(th)
    Js, land = J_DB_sim(th)
    cost_rows.append(dict(theta=th, lam2=1.0/(2*th), J_DB_analytic=Ja, J_DB_sim=Js,
                          J_FM=Jfm, ratio_DB_FM=Ja/Jfm, land_err=land,
                          DB_le_FM=bool(Ja <= Jfm + 1e-12)))

print("="*74)
print("Claim 1  -  Unifying SOC framework: DB -> FM as theta->0 (Prop 4.1)")
print("arXiv 2509.24531 / OpenReview aIFgQusnPy | independent numpy, CPU, deterministic")
print("="*74)
print(f"endpoints in R^{d}; ||x1-x0||^2 = {disp2:.6f}")
print()
print("T1 controller reduction  max_t ||u_DB(theta)-u_FM||  ->  0")
for th, e in zip(thetas, maxerr):
    print(f"   theta={th:9.1e}   max_t||u_DB-u_FM|| = {e:.6e}")
print(f"   fitted order (loglog slope of err vs theta, theta<=0.1) = {rate_slope:.4f}  (theory 1.0)")
print()
print(f"T2 drift term  max_t||theta(x1-xt)||  at theta=1e-3 : {drift_norm[thetas==1e-3][0]:.3e}"
      f"  at theta=1e-6 : {drift_norm[thetas==1e-6][0]:.3e}   (-> 0)")
print()
print(f"T3 FM two-form identity  max_t ||(x1-xt)/(1-t) - (x1-x0)|| = {t3err:.3e}  (machine zero)")
print()
print("T4 cost reduction  J_DB(theta) <= J_FM,  J_DB -> J_FM as theta->0")
print(f"   J_FM = 1/2||x1-x0||^2 = {Jfm:.6f}")
for r in cost_rows:
    print(f"   theta={r['theta']:8.5f}  J_DB(analytic)={r['J_DB_analytic']:.6f}  "
          f"J_DB(sim)={r['J_DB_sim']:.6f}  ratio DB/FM={r['ratio_DB_FM']:.6f}  "
          f"land_err={r['land_err']:.2e}  DB<=FM={r['DB_le_FM']}")
print()
verdict = (rate_slope > 0.85 and maxerr[-1] < 1e-4 and t3err < 1e-9
           and all(r['DB_le_FM'] for r in cost_rows)
           and abs(cost_rows[-1]['ratio_DB_FM'] - 1.0) < 1e-2)
print(f"VERDICT reduction-to-FM identity holds (Prop 4.1): {verdict}")
print("="*74)

out = dict(
    d=d, disp2=disp2,
    thetas=thetas.tolist(), maxerr_controller=maxerr.tolist(),
    controller_reduction_order=rate_slope,
    drift_at_theta_1em3=float(drift_norm[thetas==1e-3][0]),
    drift_at_theta_1em6=float(drift_norm[thetas==1e-6][0]),
    fm_twoform_identity_err=t3err,
    J_FM=Jfm, cost_rows=cost_rows,
    verdict_prop41=bool(verdict),
)
with open("results.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote results.json")
