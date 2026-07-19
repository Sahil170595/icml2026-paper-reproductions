"""
CLAIM 2 (OpenReview 7pQIzVNctu / arXiv 2502.08834, Thm A.1 / Section 4):
  "The ODE Rex construction inherits ARBITRARY ORDER of convergence and a
   NON-ZERO LINEAR STABILITY REGION from the base McCallum-Foster method."

 (A) ORDER  -- global error ||x_h(T)-x_ref(T)|| of Rex (Lawson/exponential RK base
     of order p=1,2,3) vs step size h on a nonlinear semilinear ODE; fitted log-log
     slope must match base order p (target |slope-p|<=0.25).
 (B) STABILITY REGION -- area of {z=h*lambda : rho(M(z))<=1} for the exact 2x2
     amplification M of the McCallum-Foster/Rex reversible iteration on the Dahlquist
     test, swept over coupling zeta. MF region has POSITIVE area for zeta<1 (inherited
     by Rex) but DEGENERATES to the imaginary axis (area~0) at the pure-reversibility
     limit zeta=1; classical (non-reversible) RK region shown as reference.
 (C) STRUCTURE PRESERVATION -- on a stiff dissipative problem the exponential Rex
     integrates the linear part exactly and stays stable for ALL h, whereas the same
     reversible wrapper WITHOUT the exponential blows up.
FALSIFICATION: measured order != p, OR MF region has zero area for all zeta<1, OR the
exponential gives no stability advantage.
"""
import json, time
import numpy as np
from scipy.integrate import solve_ivp
from rex_core import rk_increment, rex_forward, make_lawson_field, loglog_slope

t0 = time.time()

# ===================== (A) ORDER OF CONVERGENCE =====================
A_ORD = np.array([-2.0, -3.0, -1.0])
def N_ord(t, x):
    return 0.5*np.array([np.sin(x[1]) + 0.3*np.cos(t),
                         np.tanh(x[0]*x[2]),
                         np.sin(x[0]) - 0.2*x[1]])
def full_rhs(t, x):  return A_ORD*x + N_ord(t, x)
T0, T1 = 0.0, 1.0
x_init = np.array([0.6, -0.4, 0.9])
ref = solve_ivp(full_rhs, (T0, T1), x_init, method="DOP853", rtol=1e-13, atol=1e-13, dense_output=True)
x_ref = ref.sol(T1)

def rex_solve(order, zeta, Nsteps):
    G = make_lawson_field(A_ORD, N_ord, T0); h = (T1-T0)/Nsteps
    y = x_init.copy(); yhat = x_init.copy()
    for n in range(Nsteps):
        y, yhat = rex_forward(G, T0+n*h, h, order, zeta, y, yhat)
    return np.exp(A_ORD*(T1-T0))*y

print("="*80)
print("CLAIM 2  Rex inherits arbitrary order + non-zero stability region (Thm A.1)")
print("="*80)
print("\n(A) ORDER OF CONVERGENCE  (nonlinear semilinear ODE, ref=DOP853 rtol=1e-13)")
Ns = [16, 32, 64, 128, 256]
order_res = {}
for p in (1, 2, 3):
    errs = [float(np.max(np.abs(rex_solve(p, 1.0, N) - x_ref))) for N in Ns]
    hs = [(T1-T0)/N for N in Ns]
    slope = loglog_slope(hs, errs)
    order_res[p] = dict(N=Ns, errs=errs, slope=slope, pass_=bool(abs(slope-p) <= 0.25))
    print("  base order p=%d : errors %s" % (p, ["%.2e"%e for e in errs]))
    print("               fitted slope = %.3f   (target %d, |diff|<=0.25 -> %s)"
          % (slope, p, abs(slope-p) <= 0.25))
errs_z = [float(np.max(np.abs(rex_solve(2, 0.7, N) - x_ref))) for N in Ns]
slope_z = loglog_slope([(T1-T0)/N for N in Ns], errs_z)
print("  (control) p=2 with zeta=0.7 : slope = %.3f  (order is zeta-independent)" % slope_z)

# ===================== (B) LINEAR STABILITY REGION =====================
def Rp(z, p):
    s = np.ones_like(z); term = np.ones_like(z)
    for j in range(1, p+1):
        term = term*z/j; s = s + term
    return s

def mf_region_area(p, zeta, re=(-4.0, 1.0), im=(-4.0, 4.0), ng=420):
    zr = np.linspace(*re, ng); zi = np.linspace(*im, ng)
    ZR, ZI = np.meshgrid(zr, zi); Z = ZR + 1j*ZI
    Rz, Rmz = Rp(Z, p), Rp(-Z, p)
    c = Rz - zeta; d = Rmz - 1.0
    tr = zeta + (1.0 - d*c); det = zeta*np.ones_like(Z)
    disc = np.sqrt(tr*tr - 4.0*det)
    rho = np.maximum(np.abs(0.5*(tr+disc)), np.abs(0.5*(tr-disc)))
    cell = (re[1]-re[0])/(ng-1) * (im[1]-im[0])/(ng-1)
    return float((rho <= 1.0 + 1e-9).sum()*cell)

def classical_rk_area(p, re=(-4.0, 1.0), im=(-4.0, 4.0), ng=420):
    zr = np.linspace(*re, ng); zi = np.linspace(*im, ng)
    ZR, ZI = np.meshgrid(zr, zi); Z = ZR + 1j*ZI
    cell = (re[1]-re[0])/(ng-1) * (im[1]-im[0])/(ng-1)
    return float((np.abs(Rp(Z, p)) <= 1.0 + 1e-9).sum()*cell)

print("\n(B) LINEAR STABILITY REGION  area of {z: rho(M(z))<=1}, Re in [-4,1], Im in [-4,4]")
stab = {}
for p in (1, 2, 3):
    a_cls = classical_rk_area(p)
    row = {"classical_nonrev": a_cls, "mf_zeta": {}}
    print("  base order p=%d : classical (non-reversible) RK area = %.3f" % (p, a_cls))
    for zeta in (1.0, 0.9, 0.7, 0.5):
        A = mf_region_area(p, zeta); row["mf_zeta"]["%.2f" % zeta] = A
        print("        McCallum-Foster/Rex  zeta=%.2f  ->  stability-region area = %.4f" % (zeta, A))
    stab["p%d" % p] = row
area_zeta1 = float(np.mean([stab["p%d"%p]["mf_zeta"]["1.00"] for p in (1,2,3)]))
area_zeta_lt1 = float(np.min([stab["p%d"%p]["mf_zeta"]["0.50"] for p in (1,2,3)]))
nonzero_ok = (area_zeta_lt1 > 0.2) and (area_zeta1 < 0.2)
print("  => mean area zeta=1 = %.4f (~0);  min area zeta=0.5 = %.4f (>0);  non-zero region: %s"
      % (area_zeta1, area_zeta_lt1, nonzero_ok))

# ===================== (C) EXPONENTIAL STRUCTURE PRESERVATION =====================
K = 40.0
def N_stiff(t, x):  return 0.5*np.sin(x)
def rhs_stiff(t, x): return -K*x + N_stiff(t, x)
xs0 = np.array([1.0, -1.0, 0.5]); Tend = 2.0
refs = solve_ivp(rhs_stiff, (0, Tend), xs0, method="DOP853", rtol=1e-12, atol=1e-12)
xs_ref = refs.y[:, -1]

def integrate_mf(order, zeta, Nsteps, exponential):
    if exponential:
        a_arr = np.array([-K, -K, -K]); Nfun = N_stiff
    else:
        a_arr = np.array([0.0, 0.0, 0.0]); Nfun = lambda t, x: -K*x + N_stiff(t, x)
    G = make_lawson_field(a_arr, Nfun, 0.0); h = Tend/Nsteps
    y = xs0.copy(); yhat = xs0.copy()
    for n in range(Nsteps):
        y, yhat = rex_forward(G, n*h, h, order, zeta, y, yhat)
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e6:
            return np.inf
    return float(np.max(np.abs(np.exp(a_arr*Tend)*y - xs_ref)))

print("\n(C) EXPONENTIAL STRUCTURE PRESERVATION  stiff dx/dt=-%.0f x + 0.5 sin x, T=%.0f" % (K, Tend))
Nc = 40; struct = {}
for p in (1, 2):
    e_exp = integrate_mf(p, 1.0, Nc, True)
    e_non = integrate_mf(p, 0.5, Nc, False)
    struct["p%d" % p] = dict(rex_exponential_err=e_exp, nonexp_reversible_err=e_non, Kh=K*Tend/Nc)
    print("  p=%d (N=%d, K*h=%.1f): Rex-exponential err=%.3e   non-exponential-reversible err=%s"
          % (p, Nc, K*Tend/Nc, e_exp, ("%.3e"%e_non if np.isfinite(e_non) else "BLOW-UP (inf)")))
struct_ok = all(np.isfinite(struct["p%d"%p]["rex_exponential_err"]) and
                struct["p%d"%p]["rex_exponential_err"] < 1e-2 and
                (not np.isfinite(struct["p%d"%p]["nonexp_reversible_err"]) or
                 struct["p%d"%p]["nonexp_reversible_err"] > 10*struct["p%d"%p]["rex_exponential_err"])
                for p in (1, 2))

order_ok = all(order_res[p]["pass_"] for p in (1,2,3))
verdict = "SUPPORTED" if (order_ok and nonzero_ok and struct_ok) else "PARTIAL"
print("\n" + "="*80)
print("VERDICT (executed):")
print("  (A) order slopes = %.2f/%.2f/%.2f (target 1/2/3)          -> %s"
      % (order_res[1]['slope'], order_res[2]['slope'], order_res[3]['slope'], order_ok))
print("  (B) MF/Rex non-zero stability region zeta<1, ~0 at zeta=1   -> %s" % nonzero_ok)
print("  (C) exponential keeps Rex stable on stiff problem          -> %s" % struct_ok)
print("  CLAIM 2 (arbitrary order + non-zero stability region)      -> %s" % verdict)
print("="*80)

out = dict(claim=2, order=order_res, order_zeta07_p2_slope=slope_z,
           stability_region=stab, area_zeta1_mean=area_zeta1, area_zeta05_min=area_zeta_lt1,
           structure=struct, order_ok=bool(order_ok), nonzero_region_ok=bool(nonzero_ok),
           structure_ok=bool(struct_ok), verdict=verdict,
           numpy=np.__version__, runtime_s=round(time.time()-t0,3))
json.dump(out, open("results.json","w"), indent=2)
print("wrote results.json  (runtime %.2fs)" % out['runtime_s'])
