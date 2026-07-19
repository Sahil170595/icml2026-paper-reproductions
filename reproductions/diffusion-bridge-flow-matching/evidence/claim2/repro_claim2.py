"""
Claim 2 - Theorem 4.2: the Diffusion Bridge SOC cost is <= the Flow Matching cost.
Paper: OpenReview aIFgQusnPy / arXiv 2509.24531, Theorem 4.2 (Appendix A.2).

TARGET:  J(u*_DB) <= J(u*_FM),   J(u) := integral_0^1 (1/2)||u_t||^2 dt,
with g_t = 1 (so theta_t = 1/(2 lambda^2), lambda^2 = 1/(2 theta)) and the
closed-form optimal controllers (Eq 17):
    u*_DB(t,x) = e^{-2 thbar}(x1-x)/sig2,  thbar=theta(1-t), sig2=lambda^2(1-e^{-2thbar})
    u*_FM(t,x) = (x1-x)/(1-t).

The paper's proof (Eq 18-20) reduces to a POINTWISE integrand-coefficient bound:
    c_DB(t) = 1/(lambda^4 (e^{(1-t)/lambda^2} - 1)^2)   <=   c_FM(t) = 1/(1-t)^2,
which is exactly the elementary inequality  e^x - 1 >= x  (x = (1-t)/lambda^2).

RULE (both must hold for VERIFIED):
  (A) coefficient inequality:  max_t c_DB(t;lambda)/c_FM(t) <= 1  for every lambda,
      approaching 1 as lambda->inf (theta->0) [equality / Prop 4.1 limit].
  (B) actual optimal-control costs:  J_DB <= J_FM  for every lambda, with
      J_FM = (1/2)||x1-x0||^2 and J_DB = theta/(e^{2theta}-1) ||x1-x0||^2
      (exact min-energy LQ cost; also verified by direct closed-loop simulation).
FALSIFICATION: any lambda with c_DB/c_FM > 1+1e-9, or J_DB > J_FM (strictly),
would falsify Theorem 4.2.
"""
import json, numpy as np

rng = np.random.default_rng(4242)
d = 32
x0 = rng.standard_normal(d)
x1 = rng.standard_normal(d) + 2.0
disp = x1 - x0
disp2 = float(disp @ disp)

# paper hyper-parameter: lambda^2 = 30^2/255^2  (steady variance level)
lam2_paper = (30.0 / 255.0) ** 2
lambdas = np.array([np.sqrt(lam2_paper), 0.25, 0.5, 1.0, 2.0, 4.0, 10.0])

def coeff_ratio_max(lam):
    """max_t c_DB(t;lam)/c_FM(t) over t in (0,1); should be <= 1."""
    lam2 = lam * lam
    t = np.linspace(1e-6, 1.0 - 1e-9, 200001)
    om = 1.0 - t
    cDB = 1.0 / (lam2 * lam2 * np.expm1(om / lam2) ** 2)
    cFM = 1.0 / om ** 2
    r = cDB / cFM
    return float(np.max(r)), float(r[0]), float(r[-1])   # max, at t~0, at t->1

def J_FM():
    return 0.5 * disp2

def J_DB_analytic(theta):
    return theta / np.expm1(2.0 * theta) * disp2

def u_DB(t, x, theta):
    a = 2.0 * theta * (1.0 - t)
    return 2.0 * theta * np.exp(-a) * (x1 - x) / (1.0 - np.exp(-a))

def J_DB_sim(theta, Nt=300000, tmax=1.0 - 1e-6):
    tg = np.linspace(0.0, tmax, Nt + 1); dt = tg[1] - tg[0]
    x = x0.copy(); cost = 0.0
    for i in range(Nt):
        u = u_DB(tg[i], x, theta)
        cost += 0.5 * float(u @ u) * dt
        x = x + (theta * (x1 - x) + u) * dt
    return cost, float(np.linalg.norm(x - x1))

Jfm = J_FM()
rows = []
for lam in lambdas:
    theta = 1.0 / (2.0 * lam * lam)
    rmax, r0, r1 = coeff_ratio_max(lam)
    Ja = J_DB_analytic(theta)
    rows.append(dict(lam=float(lam), lam2=float(lam*lam), theta=float(theta),
                     coeff_ratio_max=rmax, coeff_ratio_t0=r0, coeff_ratio_t1=r1,
                     J_DB=float(Ja), J_FM=float(Jfm), ratio_DB_FM=float(Ja/Jfm),
                     A_pass=bool(rmax <= 1.0 + 1e-9), B_pass=bool(Ja <= Jfm + 1e-12)))

# direct-simulation cross check on 3 lambdas (confirms closed-form controller cost)
sim_rows = []
for lam in [0.25, 1.0, 4.0]:
    theta = 1.0 / (2.0 * lam * lam)
    Js, land = J_DB_sim(theta)
    Ja = J_DB_analytic(theta)
    sim_rows.append(dict(lam=lam, theta=theta, J_DB_sim=Js, J_DB_analytic=Ja,
                         rel_err=abs(Js-Ja)/Ja, land_err=land))

print("="*74)
print("Claim 2  -  Theorem 4.2:  J(u*_DB) <= J(u*_FM)")
print("arXiv 2509.24531 / OpenReview aIFgQusnPy | independent numpy, CPU, deterministic")
print("="*74)
print(f"endpoints R^{d};  ||x1-x0||^2 = {disp2:.6f};  J_FM = 1/2||x1-x0||^2 = {Jfm:.6f}")
print(f"paper lambda^2 = 30^2/255^2 = {lam2_paper:.6e}  (theta = {1/(2*lam2_paper):.4f})")
print()
print("(A) pointwise coefficient inequality  c_DB(t)/c_FM(t) <= 1   [proof mechanism e^x-1>=x]")
print(f"{'lambda':>9} {'lambda^2':>11} {'theta':>10} {'max ratio':>12} {'ratio t->1':>12} {'A':>4}")
for r in rows:
    print(f"{r['lam']:9.4f} {r['lam2']:11.4e} {r['theta']:10.4f} "
          f"{r['coeff_ratio_max']:12.6e} {r['coeff_ratio_t1']:12.8f} {str(r['A_pass']):>4}")
print()
print("(B) actual optimal-control costs  J_DB = theta/(e^{2theta}-1)||x1-x0||^2  <=  J_FM")
print(f"{'lambda':>9} {'theta':>10} {'J_DB':>12} {'J_FM':>12} {'J_DB/J_FM':>12} {'B':>4}")
for r in rows:
    print(f"{r['lam']:9.4f} {r['theta']:10.4f} {r['J_DB']:12.6f} {r['J_FM']:12.6f} "
          f"{r['ratio_DB_FM']:12.8f} {str(r['B_pass']):>4}")
print()
print("closed-loop simulation cross-check (J_DB integrated from RK/Euler ODE, endpoint pinned):")
for s in sim_rows:
    print(f"   lambda={s['lam']:.2f} theta={s['theta']:.4f}  J_DB_sim={s['J_DB_sim']:.6f} "
          f"J_DB_analytic={s['J_DB_analytic']:.6f}  rel_err={s['rel_err']:.2e}  land_err={s['land_err']:.2e}")
print()
allA = all(r['A_pass'] for r in rows); allB = all(r['B_pass'] for r in rows)
# equality/limit: at lambda=10 (theta small) ratio ~ 1
lim_ok = abs(rows[-1]['ratio_DB_FM'] - 1.0) < 5e-3
# strict inequality at paper lambda
strict = rows[0]['ratio_DB_FM'] < 0.5
print(f"(A) all coeff ratios <= 1: {allA}")
print(f"(B) all J_DB <= J_FM     : {allB}")
print(f"limit lambda=10 -> J_DB/J_FM = {rows[-1]['ratio_DB_FM']:.6f} (->1, Prop 4.1): {lim_ok}")
print(f"strict at paper lambda   -> J_DB/J_FM = {rows[0]['ratio_DB_FM']:.3e}: {strict}")
print(f"VERDICT Theorem 4.2 (J_DB <= J_FM): {allA and allB}")
print("="*74)

with open("results.json", "w") as f:
    json.dump(dict(d=d, disp2=disp2, J_FM=Jfm, lam2_paper=lam2_paper,
                   rows=rows, sim_rows=sim_rows,
                   A_all_pass=bool(allA), B_all_pass=bool(allB),
                   limit_to_FM_ok=bool(lim_ok), verdict_thm42=bool(allA and allB)),
              f, indent=2)
print("wrote results.json")
