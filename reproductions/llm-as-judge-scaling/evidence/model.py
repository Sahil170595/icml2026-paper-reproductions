"""Independent NumPy/SciPy reproduction of the tractable best-of-k model in
Halder & Pehlevan, "Demystifying LLM-as-a-Judge: Analytically Tractable
Model for Inference-Time Scaling" (arXiv:2512.19905, ICML 2026, OpenReview
ANVg7NnupP).  Official code: github.com/I-Halder/Demystifying-LLM-as-a-Judge-
Analytically-Tractable-Model-for-Inference-Time-Scaling @ 444b53c4 (BLR_zero_T.py
sha256 e00ee396b8445264..., BLR_non_zero_T.py sha256 d1164b306d45d0f2...).

THIS FILE DOES NOT IMPORT OR COPY THAT CODE.  It is an independent
implementation of the paper's stated setting (teacher/reward Bayesian linear
regression, best-of-k selection under a possibly-misspecified quadratic
reward) built from the paper's abstract/theorem statements, using a
genuinely-fitted finite Bayesian linear regression posterior (exact, not a
high-dimensional deterministic-equivalent approximation) plus an exact
order-statistics quadrature for the best-of-k selection step -- CPU-only,
deterministic, no LLMs.

---------------------------------------------------------------------------
Model
---------------------------------------------------------------------------
Teacher:   y = w_T . phi(x) + N(0, sigma^2),      phi(x) = x/sqrt(d), x~N(0,S^2 I_d)
Prior:     w ~ N(0, gamma^2 I_d)
Posterior: exact Bayesian linear regression on n training pairs (X,y):
             Lambda = (1/gamma^2) I + (1/sigma^2) X^T X
             w_post = (1/sigma^2) Lambda^{-1} X^T y
           At any test point x, the posterior PREDICTIVE is exactly Gaussian:
             m(x) = w_post . phi(x)                       (posterior mean)
             s(x)^2 = sigma^2 + phi(x)^T Lambda^{-1} phi(x) (predictive var)
Reward:    r(y,x) = -(y - y_R(x))^2,   y_R(x) = w_R . phi(x)
           w_R = w_T + eta*v  with v fixed, unit-norm-rescaled, orthogonal to
           w_T (eta=0 => reward IS the teacher; eta>0 => a genuine
           direction-changing misspecification, not just a rescaling).

Best-of-k (hard, temperature T=0) selection: draw k iid y_1..y_k ~ N(m(x),
s(x)^2) (the model's posterior-predictive "latent quality" distribution for
completions of prompt x) and keep the one maximizing r(y_i,x), i.e. nearest
to y_R(x).  Generalization error against the TEACHER:
   delta(x,k) = E[ (y_sel(x,k) - y_T(x))^2 ]
   delta(k)   = E_x[ delta(x,k) ]                (averaged over test prompts)

---------------------------------------------------------------------------
Exact order-statistics quadrature for delta(x,k)  (no Monte Carlo needed)
---------------------------------------------------------------------------
Standardize: b'(x) = (y_T(x)-m(x))/s(x), t'(x) = (y_R(x)-m(x))/s(x),
W_i = (y_i-m(x))/s(x) ~ N(0,1) iid.  Selection keeps W_sel = argmin_i|W_i-t'|.
Density of W_sel at w:  g(w) = k*phi(w)*S(|w-t'|)^{k-1},
  S(r) = 1 - [Phi(t'+r) - Phi(t'-r)]   (probability no other draw lands in
  the interval (t'-r, t'+r), i.e. the survival function of the "distance to
  nearest of the other k-1 draws").
A naive fixed grid in w becomes numerically unresolvable for large k because
g concentrates in a window of width ~1/k around t'.  Fix: substitute
tau = k*(w - t'), i.e. w = t' + tau/k.  The Jacobian (dw = dtau/k) exactly
cancels the leading k in g, giving
   delta(x,k) = integral over tau of  (w(tau)-b')^2 * phi(w(tau)) * S(|tau|/k)^{k-1}  dtau
which is well-conditioned on a SINGLE FIXED tau-grid for essentially any k
(the width of the resulting integrand in tau is O(1) regardless of k),
because S(|tau|/k)^{k-1} -> exp(-2*phi(t')*|tau|) as k->infinity for fixed
tau -- a fixed decay lengthscale in tau, not in w.  This is verified against
direct end-to-end Monte Carlo simulation in verification/verify_mc.py.

Caveat (disclosed in Limitations): this still requires the tau-grid to be
wide enough relative to 1/phi(t') for test points x with very large |t'(x)|
(rare, heavy reward-misalignment outliers); the reported k-ranges (up to a
few thousand) are well inside the regime where the fixed grid used here
(tau in [-80,80]) is accurate, cross-checked against Monte Carlo below.
"""
import numpy as np
from scipy.stats import norm

LOG2PI = np.log(2.0 * np.pi)


def fit_blr(n, d, S, sigma, gamma, w_T, rng):
    """Exact Bayesian linear regression fit on n simulated training pairs."""
    X = rng.normal(0.0, S, size=(n, d)) / np.sqrt(d)
    y = X @ w_T + sigma * rng.standard_normal(n)
    alpha = 1.0 / gamma**2
    beta = 1.0 / sigma**2
    Lam = alpha * np.eye(d) + beta * (X.T @ X)
    Lam_inv = np.linalg.inv(Lam)
    w_post = beta * (Lam_inv @ (X.T @ y))
    return w_post, Lam_inv


def test_stats(n_test, d, S, w_T, w_R, w_post, Lam_inv, sigma, rng):
    """Exact posterior-predictive mean/std, teacher value, reward target, for
    n_test iid random test prompts."""
    Xte = rng.normal(0.0, S, size=(n_test, d)) / np.sqrt(d)
    m = Xte @ w_post
    yT = Xte @ w_T
    yR = Xte @ w_R
    var = sigma**2 + np.einsum('ij,jk,ik->i', Xte, Lam_inv, Xte)
    s = np.sqrt(var)
    return m, s, yT, yR


def make_wR(w_T, eta, rng):
    """Reward weight vector w_R = w_T + eta*v, v a fixed unit-norm-rescaled
    direction orthogonal to w_T (||v||=||w_T||) so eta is a genuine,
    direction-changing misalignment magnitude (eta=0 => reward IS teacher)."""
    d = w_T.shape[0]
    v = rng.standard_normal(d)
    v = v - (v @ w_T) / (w_T @ w_T) * w_T
    v = v / np.linalg.norm(v) * np.linalg.norm(w_T)
    return w_T + eta * v, v


def delta_exact(bp, tp, s2, k_values, tau_grid):
    """Exact (quadrature) best-of-k generalization error delta(k), averaged
    over the test set, for every k in k_values.
    bp, tp, s2: arrays [N] = standardized teacher-bias, standardized
    reward-offset, predictive variance, one per test prompt.
    tau_grid: fixed 1-D array, the rescaled quadrature grid (see docstring).
    Returns: array [len(k_values)].
    """
    out = np.empty(len(k_values))
    for j, k in enumerate(k_values):
        r = np.abs(tau_grid) / k                                          # [G]
        w = tp[:, None] + (tau_grid / k)[None, :]                          # [N,G]
        logphi_w = -0.5 * w**2 - 0.5 * LOG2PI                              # [N,G]
        p = norm.cdf(tp[:, None] + r[None, :]) - norm.cdf(tp[:, None] - r[None, :])
        p = np.clip(p, 0.0, 1.0 - 1e-15)
        logS = np.log1p(-p)                                                # [N,G]
        f = np.exp(logphi_w + (k - 1) * logS)                              # [N,G]
        integrand = f * (w - bp[:, None])**2
        per_x = np.trapz(integrand, tau_grid, axis=1)                      # [N]
        out[j] = np.mean(s2 * per_x)
    return out


def delta_mc(m, s, yT, yR, k, n_trials, rng, batch=200):
    """Direct end-to-end Monte Carlo: draw k samples, hard-select by reward,
    measure squared error against the teacher. Independent of delta_exact
    (no quadrature, no order-statistics formula) -- used only as a
    cross-check in verification/verify_mc.py."""
    N = m.shape[0]
    total = np.zeros(N)
    for tstart in range(0, n_trials, batch):
        Tb = min(batch, n_trials - tstart)
        eps = rng.standard_normal((N, Tb, k))
        y = m[:, None, None] + s[:, None, None] * eps
        rew = -(y - yR[:, None, None])**2
        idx = np.argmax(rew, axis=2)
        y_sel = np.take_along_axis(y, idx[:, :, None], axis=2)[:, :, 0]
        err = (y_sel - yT[:, None])**2
        total += err.sum(axis=1)
    return total / n_trials


# ---------------------------------------------------------------------------
# Shared experiment configuration (used identically by all three claim
# scripts so the three regimes are directly comparable).
# ---------------------------------------------------------------------------
D = 60          # feature dimension
N_TRAIN = 300   # BLR training samples (alpha = d/n = 0.2)
S_STD = 1.0     # input feature std
SIGMA = 0.15    # teacher label noise std
GAMMA = 0.5     # weight prior std
N_TEST = 1200   # test prompts averaged over per delta(k)
TAU_GRID = np.linspace(-80.0, 80.0, 2001)

SEED_WT = 12345      # teacher weights + BLR training draw
SEED_TEST = 999      # test-prompt draw
SEED_MISALIGN = 42   # misalignment direction v


def build_fixed_posterior():
    rng = np.random.default_rng(SEED_WT)
    w_T = rng.standard_normal(D)
    w_post, Lam_inv = fit_blr(N_TRAIN, D, S_STD, SIGMA, GAMMA, w_T, rng)
    return w_T, w_post, Lam_inv


def eval_regime(eta, k_values, w_T=None, w_post=None, Lam_inv=None):
    """Convenience: build (or reuse) the fixed posterior, draw the fixed
    test set, apply misalignment eta, and return delta(k) plus diagnostics."""
    if w_T is None:
        w_T, w_post, Lam_inv = build_fixed_posterior()
    rng_w = np.random.default_rng(SEED_MISALIGN)
    w_R, v = make_wR(w_T, eta, rng_w)
    rng_te = np.random.default_rng(SEED_TEST)
    m, s, yT, yR = test_stats(N_TEST, D, S_STD, w_T, w_R, w_post, Lam_inv, SIGMA, rng_te)
    bp = (yT - m) / s
    tp = (yR - m) / s
    s2 = s**2
    d_of_k = delta_exact(bp, tp, s2, k_values, TAU_GRID)
    diag = {
        "eta": eta,
        "mean_teacher_bias2": float(np.mean((yT - m)**2)),
        "mean_reward_teacher_gap2": float(np.mean((yR - yT)**2)),
        "mean_s2": float(np.mean(s2)),
        "max_abs_tprime": float(np.max(np.abs(tp))),
    }
    return d_of_k, (m, s, yT, yR), diag
