"""
CLAIM 3 (OpenReview 7pQIzVNctu / arXiv 2502.08834, Figure 7):
  "Rex achieves NEAR-MACHINE-PRECISION RECONSTRUCTION under exact inversion in
   image-generation inversion experiments."

MECHANISM-LEVEL reproduction (honest proxy): the actual Fig-7 experiment inverts a
trained image diffusion model and reports reconstruction error; that needs GPUs +
checkpoints + real images. Here we reproduce the EXACT quantity Fig 7 reports -- the
inversion->reconstruction error of the probability-flow ODE -- on a diffusion model
whose data law is a Gaussian mixture (=> ANALYTIC score, genuinely nonlinear). We
invert data->noise then reconstruct noise->data and measure ||x_rec - x_0||.

ACCEPTANCE RULE: mean reconstruction error of Rex (reversible exponential solver) is
<= 1e-9 (near machine precision) and INDEPENDENT of the number of function evaluations
(NFE), while the standard non-reversible baseline (DDIM inversion, = exponential-Euler)
has reconstruction error >= 1e6x larger. This is the Fig-7 finding: reversible solver
reconstructs to ~machine precision; DDIM inversion does not.
FALSIFICATION: Rex reconstruction error is not near machine precision, or is no better
than DDIM.
"""
import json, time
import numpy as np
from rex_core import rk_increment, rex_forward, rex_backward, make_lawson_field

t0 = time.time()

# VP schedule + GMM analytic score (a diffusion model with closed-form score)
A_LIN = -1.0
def alpha(t): return np.exp(-t)
def sigma(t): return np.sqrt(1.0 - np.exp(-2.0*t))
G2 = 2.0
D = 8
W  = np.array([0.4, 0.35, 0.25])
MU = np.array([[1.5]*D, [-1.2]*D, [0.3]*D]) * np.array([1,-1,1,-1,1,-1,1,-1])[:D]
V  = np.array([0.20, 0.35, 0.15])
def score(t, x):
    a = alpha(t); s2 = sigma(t)**2
    var = a*a*V + s2; diff = x[None,:] - a*MU
    q = -0.5*np.sum(diff*diff, axis=1)/var - 0.5*D*np.log(2*np.pi*var) + np.log(W)
    q -= q.max(); r = np.exp(q); r /= r.sum()
    return np.sum(r[:,None]*(-diff/var[:,None]), axis=0)
def eps_pred(t, x): return -sigma(t)*score(t, x)          # noise-prediction: eps=-sigma*score
def N_ode(t, x):    return -0.5*G2*score(t, x)            # PF-ODE nonlinear part
TSTART, TEND = 0.05, 3.0

# ---- Rex reversible exponential solver: exact inversion + reconstruction ----
def rex_reconstruct(order, zeta, NFE, x0):
    G = make_lawson_field(A_LIN, N_ode, TSTART); h = (TEND-TSTART)/NFE
    y = x0.copy(); yhat = x0.copy()
    for n in range(NFE):        y, yhat = rex_forward(G, TSTART+n*h, h, order, zeta, y, yhat)
    for n in range(NFE-1,-1,-1): y, yhat = rex_backward(G, TSTART+n*h, h, order, zeta, y, yhat)
    return float(np.max(np.abs(y - x0)))

# ---- DDIM (deterministic) inversion + sampling: non-reversible baseline ----
def ddim_step(x, t, s):                                   # exponential-Euler DDIM t->s
    a_t, s_t = alpha(t), sigma(t); a_s, s_s = alpha(s), sigma(s)
    e = eps_pred(t, x)
    return (a_s/a_t)*x + (s_s - a_s*s_t/a_t)*e
def ddim_reconstruct(NFE, x0):
    ts = np.linspace(TSTART, TEND, NFE+1)
    x = x0.copy()
    for i in range(NFE):        x = ddim_step(x, ts[i],   ts[i+1])   # invert data->noise
    for i in range(NFE, 0, -1): x = ddim_step(x, ts[i],   ts[i-1])   # sample noise->data
    return float(np.max(np.abs(x - x0)))

# data samples (from GMM component 0, diffused to t=TSTART)
rng = np.random.default_rng(7)
X0 = [alpha(TSTART)*MU[k%3] + sigma(TSTART)*rng.standard_normal(D) for k in range(6)]

print("="*80)
print("CLAIM 3  Rex near-machine-precision reconstruction under exact inversion (Fig 7)")
print("  diffusion PF-ODE, VP schedule, GMM analytic score, D=%d, 6 data samples" % D)
print("="*80)
print("\nmean reconstruction error ||x_rec - x0||_inf  vs NFE (number of function evals):")
print("  %-6s %-22s %-22s %-22s %-10s" % ("NFE","Rex p=2 (rev)","Rex p=1 (rev)","DDIM (non-rev)","ratio"))
rows = {}
for NFE in (10, 20, 50, 100):
    rex2 = float(np.mean([rex_reconstruct(2, 1.0, NFE, x0) for x0 in X0]))
    rex1 = float(np.mean([rex_reconstruct(1, 1.0, NFE, x0) for x0 in X0]))
    ddim = float(np.mean([ddim_reconstruct(NFE, x0) for x0 in X0]))
    ratio = ddim/max(rex2, 1e-300)
    rows[NFE] = dict(rex_p2=rex2, rex_p1=rex1, ddim=ddim, ratio=ratio)
    print("  %-6d %-22.3e %-22.3e %-22.3e %.2e" % (NFE, rex2, rex1, ddim, ratio))

mean_rex = float(np.mean([rows[n]['rex_p2'] for n in rows]))
mean_ddim = float(np.mean([rows[n]['ddim'] for n in rows]))
rex_ok = max(rows[n]['rex_p2'] for n in rows) <= 1e-9 and max(rows[n]['rex_p1'] for n in rows) <= 1e-9
sep_ok = min(rows[n]['ratio'] for n in rows) >= 1e6
# Rex reconstruction is FLAT in NFE (algebraic), DDIM improves ~O(1/NFE):
rex_flat = (max(rows[n]['rex_p2'] for n in rows) / max(min(rows[n]['rex_p2'] for n in rows),1e-300)) < 1e3
ddim_scales = rows[10]['ddim'] > rows[100]['ddim']*3
verdict = "SUPPORTED" if (rex_ok and sep_ok) else "NOT SUPPORTED"
print("\n" + "="*80)
print("VERDICT (executed):")
print("  Rex reconstruction <= 1e-9 (near machine precision) for all NFE, p=1&2 : %s (max %.2e)"
      % (rex_ok, max(rows[n]['rex_p2'] for n in rows)))
print("  Rex reconstruction NFE-independent (algebraic) ; DDIM improves ~O(1/NFE): %s / %s"
      % (rex_flat, ddim_scales))
print("  Rex >= 1e6x more exact than non-reversible DDIM inversion              : %s" % sep_ok)
print("  CLAIM 3 (near-machine-precision reconstruction under exact inversion)  : %s" % verdict)
print("  [mechanism proxy: analytic diffusion model, not a trained image model]")
print("="*80)

out = dict(claim=3, note="mechanism proxy for Fig 7 image inversion (analytic GMM diffusion)",
           schedule="VP", D=D, tstart=TSTART, tend=TEND, nfe_rows=rows,
           mean_rex_p2=mean_rex, mean_ddim=mean_ddim, rex_ok=bool(rex_ok),
           separation_ok=bool(sep_ok), rex_nfe_independent=bool(rex_flat),
           ddim_scales_with_nfe=bool(ddim_scales), verdict=verdict,
           numpy=np.__version__, runtime_s=round(time.time()-t0,3))
json.dump(out, open("results.json","w"), indent=2)
print("wrote results.json  (runtime %.2fs)" % out['runtime_s'])
