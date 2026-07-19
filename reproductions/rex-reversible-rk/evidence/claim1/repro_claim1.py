"""
CLAIM 1 (OpenReview 7pQIzVNctu / arXiv 2502.08834, Section 3):
  "Rex converts explicit Runge-Kutta and stochastic Runge-Kutta schemes into
   algebraically reversible EXPONENTIAL solvers for diffusion ODEs and SDEs."

Rex = McCallum-Foster reversible construction (Eqs. 6-7) around a Lawson/
exponential RK base method, for BOTH the probability-flow ODE and the reverse-
time diffusion SDE of a diffusion model whose data law is a Gaussian mixture
(=> ANALYTIC, genuinely nonlinear score; no neural net needed).

ACCEPTANCE RULE (algebraic reversibility): integrate FORWARD N steps (data->noise,
"inversion") then apply the EXACT backward step N times (noise->data,
"reconstruction"); round-trip error ||x_rec - x_0||_inf must be ~ machine precision
(<= 1e-9) for explicit-RK base orders p=1,2,3 (ODE) and the stochastic-RK /
Euler-Maruyama base (SDE), INDEPENDENTLY of step size h (algebraic, not truncation-
limited), and >=1e6x below the error of a non-reversible exponential-(S)RK solver of
the SAME order (whose reconstruction error equals its O(h^p) truncation error and
never reaches machine precision). Reversibility uses the measure-preserving limit
zeta=1 (see Claim 2 for the zeta<1 stability region and the reversibility tradeoff).
FALSIFICATION: round-trip error O(||x_0||), or only as small as the truncation error.
"""
import json, time
import numpy as np
from rex_core import (rk_increment, rex_forward, rex_backward,
                      exprk_forward, exprk_backward, make_lawson_field)

t0 = time.time()
A_LIN = -1.0
def alpha(t):  return np.exp(-t)
def sig2(t):   return 1.0 - np.exp(-2.0*t)
G2 = 2.0
D = 6
W  = np.array([0.4, 0.35, 0.25])
MU = np.array([[ 1.5]*D, [-1.2]*D, [0.3]*D]) * np.array([1,-1,1,-1,1,-1])[:D]
V  = np.array([0.20, 0.35, 0.15])

def score(t, x):
    a = alpha(t); s2 = sig2(t)
    var = a*a*V + s2
    diff = x[None, :] - a*MU
    q = -0.5*np.sum(diff*diff, axis=1)/var - 0.5*D*np.log(2*np.pi*var) + np.log(W)
    q = q - q.max(); r = np.exp(q); r = r/r.sum()
    return np.sum(r[:, None]*(-diff/var[:, None]), axis=0)

def N_ode(t, x):  return -0.5*G2*score(t, x)
def N_sde(t, x):  return -G2*score(t, x)
TSTART, TEND = 0.05, 2.5
ZETA = 1.0

def rex_ode_roundtrip(order, zeta, Nsteps, x0):
    G = make_lawson_field(A_LIN, N_ode, TSTART); h = (TEND-TSTART)/Nsteps
    y = x0.copy(); yhat = x0.copy()
    for n in range(Nsteps):        y, yhat = rex_forward(G, TSTART+n*h, h, order, zeta, y, yhat)
    ynoise = y.copy()
    for n in range(Nsteps-1,-1,-1): y, yhat = rex_backward(G, TSTART+n*h, h, order, zeta, y, yhat)
    return float(np.max(np.abs(y-x0))), float(np.max(np.abs(ynoise)))

def exprk_ode_roundtrip(order, Nsteps, x0):
    G = make_lawson_field(A_LIN, N_ode, TSTART); h = (TEND-TSTART)/Nsteps
    y = x0.copy()
    for n in range(Nsteps):        y = exprk_forward(G, TSTART+n*h, h, order, y)
    for n in range(Nsteps-1,-1,-1): y = exprk_backward(G, TSTART+n*h, h, order, y)
    return float(np.max(np.abs(y-x0)))

rng = np.random.default_rng(0)
x0 = alpha(TSTART)*MU[0] + np.sqrt(sig2(TSTART))*rng.standard_normal(D)
NST = 50

print("="*80)
print("CLAIM 1  Rex = algebraically reversible exponential (S)RK solver for diffusion")
print("  VP schedule alpha=e^-t sigma^2=1-e^-2t, GMM data D=%d, zeta=%.1f" % (D, ZETA))
print("  round-trip = FORWARD %d steps (data->noise) then EXACT BACKWARD %d steps" % (NST,NST))
print("="*80)
print("\n[ODE] probability-flow ODE   ||x0||_inf=%.4f" % np.max(np.abs(x0)))
print("  %-6s %-22s %-26s %-10s" % ("order","Rex recon err (inf)","non-rev control recon","ratio"))
ode = {}
for p in (1,2,3):
    rex_err, xn = rex_ode_roundtrip(p, ZETA, NST, x0)
    ctl = exprk_ode_roundtrip(p, NST, x0)
    ratio = ctl/max(rex_err,1e-300)
    ode[p] = dict(rex_recon=rex_err, control_recon=ctl, ratio=ratio, x_noise_absmax=xn)
    print("  p=%-4d %-22.3e %-26.3e %.2e" % (p, rex_err, ctl, ratio))

print("\n[ODE] reconstruction error vs #steps N (order p=2): Rex is h-INDEPENDENT")
print("  %-8s %-20s %-20s" % ("N steps","Rex recon (inf)","control recon (inf)"))
hindep = {}
for N in (25, 50, 100, 200):
    re,_ = rex_ode_roundtrip(2, ZETA, N, x0); ce = exprk_ode_roundtrip(2, N, x0)
    hindep[N] = dict(rex=re, control=ce)
    print("  %-8d %-20.3e %-20.3e" % (N, re, ce))

def gfun(t):  return np.sqrt(G2)
SEED = 12345
def dW_of(n, h):
    r = np.random.default_rng(SEED*1_000_003 + n)
    return np.sqrt(abs(h))*r.standard_normal(D)
def phi_sde(G, tn, y, h, dW):
    return rk_increment(G, tn, y, h, 1) + np.exp(-A_LIN*(tn-TSTART))*gfun(tn)*dW

def rex_sde_roundtrip(zeta, Nsteps, x0):
    G = make_lawson_field(A_LIN, N_sde, TSTART); h = (TEND-TSTART)/Nsteps
    y = x0.copy(); yhat = x0.copy()
    for n in range(Nsteps):
        tn = TSTART+n*h; dW = dW_of(n,h)
        P = phi_sde(G, tn, yhat, h, dW); y1 = zeta*y+(1-zeta)*yhat+P
        Pm = phi_sde(G, tn+h, y1, -h, dW); yhat = yhat-Pm; y = y1
    ynoise = y.copy()
    for n in range(Nsteps-1,-1,-1):
        tn = TSTART+n*h; dW = dW_of(n,h)
        Pm = phi_sde(G, tn+h, y, -h, dW); yhat = yhat+Pm
        P = phi_sde(G, tn, yhat, h, dW); y = (1/zeta)*y+(1-1/zeta)*yhat-(1/zeta)*P
    return float(np.max(np.abs(y-x0))), float(np.max(np.abs(ynoise)))

def em_sde_roundtrip(Nsteps, x0):
    G = make_lawson_field(A_LIN, N_sde, TSTART); h = (TEND-TSTART)/Nsteps
    y = x0.copy()
    for n in range(Nsteps): y = y + phi_sde(G, TSTART+n*h, y, h, dW_of(n,h))
    for n in range(Nsteps-1,-1,-1): y = y + phi_sde(G, TSTART+(n+1)*h, y, -h, dW_of(n,h))
    return float(np.max(np.abs(y-x0)))

print("\n[SDE] reverse-time diffusion SDE (Euler-Maruyama base), zeta=1.0:")
sde = {}
re, xn = rex_sde_roundtrip(1.0, NST, x0)
sde['rex'] = dict(rex_recon=re, x_noise_absmax=xn)
ctl = em_sde_roundtrip(NST, x0); sde['control_em'] = ctl
print("  Rex-SDE recon err (inf) = %.3e   (||x_noise||_inf=%.3f)" % (re, xn))
print("  non-reversible Euler-Maruyama control recon err (inf) = %.3e  (ratio %.2e)"
      % (ctl, ctl/max(re,1e-300)))
print("  Brownian increments regenerated from step index n -> NO full-path storage")

TOL = 1e-9
rex_all = [ode[p]['rex_recon'] for p in (1,2,3)] + [sde['rex']['rex_recon']]
rev_ok = max(rex_all) <= TOL
sep_ok = (min(ode[p]['ratio'] for p in (1,2,3)) >= 1e6) and \
         (sde['control_em']/max(sde['rex']['rex_recon'],1e-300) >= 1e6)
verdict = "SUPPORTED" if (rev_ok and sep_ok) else "NOT SUPPORTED"
print("\n" + "="*80)
print("VERDICT (executed):")
print("  Rex round-trip ~ machine precision (<=1e-9) ODE p=1,2,3 AND SDE : %s (max=%.2e)"
      % (rev_ok, max(rex_all)))
print("  Rex >= 1e6x more exact than same-order non-reversible control   : %s" % sep_ok)
print("  CLAIM 1 (reversible exponential (S)RK, diffusion ODE+SDE)       : %s" % verdict)
print("="*80)

out = dict(claim=1, schedule="VP alpha=e^-t sigma^2=1-e^-2t", D=D, Nsteps=NST, zeta=ZETA,
           tstart=TSTART, tend=TEND, x0_absmax=float(np.max(np.abs(x0))),
           ode=ode, ode_h_independence=hindep, sde=sde, tol=TOL,
           reversible_ok=bool(rev_ok), separation_ok=bool(sep_ok), verdict=verdict,
           numpy=np.__version__, runtime_s=round(time.time()-t0,3))
json.dump(out, open("results.json","w"), indent=2)
print("wrote results.json  (runtime %.2fs)" % out['runtime_s'])
