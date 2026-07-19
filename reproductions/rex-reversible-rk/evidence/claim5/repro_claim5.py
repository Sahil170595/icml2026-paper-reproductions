"""
CLAIM 5 (OpenReview 7pQIzVNctu / arXiv 2502.08834, Table 1):
  "Rex enables ACCURATE LIKELIHOOD-BASED BOLTZMANN SAMPLING on tri-alanine with
   flow models."

SCOPE (honest): the scored quantity is likelihood-based Boltzmann sampling of the
tri-alanine molecule with a TRAINED continuous normalizing flow -- that needs a
trained flow + molecular force field and is NOT reproducible on CPU. We reproduce the
SOLVER MECHANISM that makes likelihood-based Boltzmann sampling work: (i) accurate
continuous-change-of-variables log-likelihood along the flow ODE, and (ii) exact
reversibility so the density is bijectively consistent (unbiased importance weights).
The flow used is the diffusion probability-flow ODE of a Gaussian-mixture model, whose
marginal density is ANALYTIC -> exact ground-truth log-likelihood.

Three executed sub-tests:
 (A) LIKELIHOOD ACCURACY -- |log q_Rex(x) - log p_analytic(x)| for base order p=1,2,3;
     error decreases at the base order (accurate likelihood).
 (B) REVERSIBLE LIKELIHOOD CONSISTENCY -- augmented state (x, log-det) round-trips to
     machine precision => the change-of-variables is exactly consistent (unbiased).
 (C) BOLTZMANN IMPORTANCE SAMPLING -- self-normalized IS estimate of a target Boltzmann
     expectation using Rex log-likelihoods recovers the analytic target mean; a crude
     (1-step) likelihood biases it.
FALSIFICATION: Rex log-likelihood does not converge to the analytic value, or the
augmented round-trip is not machine-precision (biased likelihood).
"""
import json, time
import numpy as np
from scipy.special import logsumexp
from rex_core import rex_forward, rex_backward, make_lawson_field

t0 = time.time()
A_LIN = -1.0
def alpha(t): return np.exp(-t)
def sig2(t):  return 1.0 - np.exp(-2.0*t)
G2 = 2.0
D = 4
W  = np.array([0.5, 0.5])
MU = np.array([[1.2, -0.8, 0.5, -1.0], [-1.1, 0.9, -0.6, 0.7]])
V  = np.array([0.25, 0.30])

def logp_t(t, x):                                   # analytic diffused-GMM log-density at time t
    a = alpha(t); s2 = sig2(t); var = a*a*V + s2
    diff = x[None,:] - a*MU
    comp = -0.5*np.sum(diff*diff, axis=1)/var - 0.5*D*np.log(2*np.pi*var) + np.log(W)
    return float(logsumexp(comp))
def score(t, x):
    a = alpha(t); s2 = sig2(t); var = a*a*V + s2
    diff = x[None,:] - a*MU
    q = -0.5*np.sum(diff*diff, axis=1)/var - 0.5*D*np.log(2*np.pi*var) + np.log(W)
    q -= q.max(); r = np.exp(q); r /= r.sum()
    return np.sum(r[:,None]*(-diff/var[:,None]), axis=0)
def trace_dscore(t, x):                             # analytic tr(d score / d x)
    a = alpha(t); s2 = sig2(t); var = a*a*V + s2
    diff = x[None,:] - a*MU                          # (K,D)
    u = -diff/var[:,None]                            # (K,D) per-component score dirs
    q = -0.5*np.sum(diff*diff, axis=1)/var - 0.5*D*np.log(2*np.pi*var) + np.log(W)
    q -= q.max(); r = np.exp(q); r /= r.sum()
    sc = np.sum(r[:,None]*u, axis=0)                 # = score
    term1 = np.sum(r * np.sum(u*u, axis=1))          # sum_k r_k |u_k|^2
    term2 = float(sc @ sc)                           # |score|^2
    term3 = D * np.sum(r/var)                         # D sum_k r_k/var_k
    return term1 - term2 - term3
# PF-ODE F = a x + N ; divergence tr(dF/dx) = a*D - (g^2/2) tr(d score/dx)
def N_ode(t, x): return -0.5*G2*score(t, x)
def div_F(t, x): return A_LIN*D - 0.5*G2*trace_dscore(t, x)
TSTART, TEND = 0.05, 3.0

# ---- sanity: analytic trace vs finite differences ----
rngc = np.random.default_rng(1)
xc = alpha(0.4)*MU[0] + np.sqrt(sig2(0.4))*rngc.standard_normal(D)
h = 1e-6; num = 0.0
for i in range(D):
    e = np.zeros(D); e[i] = h
    num += (score(0.4, xc+e)[i] - score(0.4, xc-e)[i])/(2*h)
print("sanity: analytic tr(dscore/dx)=%.6f  finite-diff=%.6f  |diff|=%.2e"
      % (trace_dscore(0.4, xc), num, abs(trace_dscore(0.4, xc)-num)))

# ---- augmented flow (x, l) with l = accumulated divergence; a=0 for l-coord ----
Aaug = np.array([A_LIN]*D + [0.0])
def N_aug(t, X):
    x = X[:D]
    return np.concatenate([N_ode(t, x), [div_F(t, x)]])

def rex_loglik(order, zeta, steps, x):
    """log q(x) via Rex-integrated continuous change of variables (TSTART->TEND)."""
    G = make_lawson_field(Aaug, N_aug, TSTART); hh = (TEND-TSTART)/steps
    Y = np.concatenate([x, [0.0]]); Yh = Y.copy()
    for n in range(steps): Y, Yh = rex_forward(G, TSTART+n*hh, hh, order, zeta, Y, Yh)
    Xend = np.exp(Aaug*(TEND-TSTART))*Y
    xT, l = Xend[:D], Xend[D]                        # l = int_{TSTART}^{TEND} div dt
    return logp_t(TEND, xT) + l                      # log p_TSTART(x) = log p_TEND(xT) + int div

def rex_loglik_roundtrip(order, zeta, steps, x):
    G = make_lawson_field(Aaug, N_aug, TSTART); hh = (TEND-TSTART)/steps
    Y = np.concatenate([x, [0.0]]); Yh = Y.copy()
    for n in range(steps):        Y, Yh = rex_forward(G, TSTART+n*hh, hh, order, zeta, Y, Yh)
    for n in range(steps-1,-1,-1): Y, Yh = rex_backward(G, TSTART+n*hh, hh, order, zeta, Y, Yh)
    return float(np.max(np.abs(Y - np.concatenate([x, [0.0]]))))

print("="*82)
print("CLAIM 5  Rex likelihood-based Boltzmann sampling mechanism (Table 1) -- PROXY")
print("  flow = diffusion PF-ODE of a GMM (analytic density); NOT trained tri-alanine flow")
print("="*82)

# (A) likelihood accuracy vs analytic ground truth
rng = np.random.default_rng(5)
XS = [alpha(TSTART)*MU[k%2] + np.sqrt(sig2(TSTART))*rng.standard_normal(D) for k in range(20)]
truth = [logp_t(TSTART, x) for x in XS]
print("\n(A) LIKELIHOOD ACCURACY  mean |log q_Rex(x) - log p_analytic(x)| (nats), zeta=0.9")
likA = {}
for p in (1, 2, 3):
    errs = [abs(rex_loglik(p, 0.9, 40, x) - tr) for x, tr in zip(XS, truth)]
    likA[p] = float(np.mean(errs))
    print("  base order p=%d :  mean abs log-lik error = %.3e nats" % (p, likA[p]))
# order check: refine steps for p=2
e_coarse = np.mean([abs(rex_loglik(2, 0.9, 20, x) - tr) for x, tr in zip(XS, truth)])
e_fine   = np.mean([abs(rex_loglik(2, 0.9, 40, x) - tr) for x, tr in zip(XS, truth)])
rate2 = np.log2(e_coarse/max(e_fine,1e-300))
print("  order-2 refinement: err(20 steps)=%.3e -> err(40 steps)=%.3e  (halving rate ~%.2f, expect ~2)"
      % (e_coarse, e_fine, rate2))

# (B) reversible likelihood consistency
print("\n(B) REVERSIBLE LIKELIHOOD CONSISTENCY  augmented (x, log-det) round-trip error, zeta=1.0")
rterrs = {p: float(np.mean([rex_loglik_roundtrip(p, 1.0, 40, x) for x in XS])) for p in (1,2,3)}
for p in (1,2,3):
    print("  p=%d : mean round-trip error of (x, log-det) = %.3e  (machine precision => unbiased)" % (p, rterrs[p]))

# (C) Boltzmann importance sampling: target p = the flow model's OWN analytic density
#     (Boltzmann energy U(x) = -log p_analytic(x)); ideal weights are UNIFORM iff log q
#     is exact -> isolates the effect of likelihood accuracy on Boltzmann reweighting.
aS = alpha(TSTART)
true_2nd = float(np.sum(W * (aS*aS*np.sum(MU*MU, axis=1) + D*(aS*aS*V + sig2(TSTART)))))  # E_p[|x|^2]
def sample_q(zeta, steps, z):
    G = make_lawson_field(A_LIN, N_ode, TEND); hh = (TSTART-TEND)/steps
    y = z.copy(); yh = z.copy()
    for n in range(steps): y, yh = rex_forward(G, TEND+n*hh, hh, 3, zeta, y, yh)
    return np.exp(A_LIN*(TSTART-TEND))*y
def draw_samples(Ns=1000):
    r = np.random.default_rng(11); xs = []
    for _ in range(Ns):
        k = 0 if r.random() < 0.5 else 1
        z = alpha(TEND)*MU[k] + np.sqrt(alpha(TEND)**2*V[k] + sig2(TEND))*r.standard_normal(D)
        xs.append(sample_q(0.9, 24, z))
    return np.array(xs)
XSAMP = draw_samples(1000)
Utar = np.array([logp_t(TSTART, x) for x in XSAMP])      # -U = log p_analytic
def snis(order_logq, steps_logq):
    logq = np.array([rex_loglik(order_logq, 0.9, steps_logq, x) for x in XSAMP])
    logw = Utar - logq                                    # w = p_analytic / q_Rex
    logw -= logw.max(); w = np.exp(logw); w /= w.sum()
    est2nd = float(w @ np.sum(XSAMP*XSAMP, axis=1))
    ess = float(1.0/np.sum(w**2)/len(w))
    return est2nd, ess
est_acc, ess_acc = snis(3, 40)
est_crude, ess_crude = snis(1, 2)
err_acc = abs(est_acc - true_2nd); err_crude = abs(est_crude - true_2nd)
print("\n(C) BOLTZMANN IMPORTANCE SAMPLING  target p = model's own density (U=-log p_analytic)")
print("    observable E_p[|x|^2]; analytic truth = %.5f  (%d flow samples)" % (true_2nd, len(XSAMP)))
print("  accurate Rex log q (p3,40 steps): E_hat=%.5f  |err|=%.3e  ESS/N=%.3f" % (est_acc, err_acc, ess_acc))
print("  crude    log q (p1, 2 steps)    : E_hat=%.5f  |err|=%.3e  ESS/N=%.3f" % (est_crude, err_crude, ess_crude))

likA_ok = (likA[3] < likA[2] < likA[1]) and (1.7 <= rate2 <= 2.3) and (likA[3] < 1e-3)
rev_ok = max(rterrs.values()) <= 1e-9
boltz_ok = (err_acc < 0.05*true_2nd) and (ess_acc > 0.6) and (err_acc < err_crude) and (ess_acc > ess_crude)
verdict = "SUPPORTED (proxy)" if (likA_ok and rev_ok and boltz_ok) else "MIXED (proxy)"
print("\n" + "="*82)
print("VERDICT (executed proxy -- NOT trained tri-alanine Table 1):")
print("  (A) Rex log-likelihood inherits base order (halving rate %.2f~2) & p3<1e-3 : %s" % (rate2, likA_ok))
print("  (B) reversible (x,log-det) round-trip ~ machine precision (unbiased)      : %s (max %.2e)"
      % (rev_ok, max(rterrs.values())))
print("  (C) Boltzmann IS with accurate Rex likelihood: ESS/N=%.2f, |err|=%.2e     : %s"
      % (ess_acc, err_acc, boltz_ok))
print("  CLAIM 5 (accurate likelihood-based Boltzmann sampling mechanism)         : %s" % verdict)
print("  [trained tri-alanine flow / Table-1 numbers are OUT OF CPU SCOPE]")
print("="*82)

out = dict(claim=5, note="likelihood+reversibility proxy for Table 1 tri-alanine; trained flow out of CPU scope",
           D=D, tstart=TSTART, tend=TEND, likelihood_err=likA, order2_halving_rate=float(rate2),
           roundtrip_err=rterrs, boltzmann=dict(true_2nd_moment=true_2nd, est_accurate=est_acc,
           est_crude=est_crude, err_accurate=err_acc, err_crude=err_crude,
           ess_accurate=ess_acc, ess_crude=ess_crude),
           likelihood_ok=bool(likA_ok), reversible_ok=bool(rev_ok), boltzmann_ok=bool(boltz_ok),
           verdict=verdict, numpy=np.__version__, runtime_s=round(time.time()-t0,3))
json.dump(out, open("/sessions/keen-fervent-hamilton/mnt/icml-repro-pilot/submissions/rex-reversible-rk-pilot/.trackio/logbook/evidence-package/claim5/results.json","w"), indent=2)
print("wrote results.json  (runtime %.2fs)" % out['runtime_s'])
