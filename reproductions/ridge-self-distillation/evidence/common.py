"""
Shared closed-form machinery for reproducing:

  Dang, Patil, Rinaldo (2026). "Optimal Unconstrained Self-Distillation in
  Ridge Regression: Strict Improvements, Precise Asymptotics, and One-Shot
  Tuning." arXiv:2602.17565, OpenReview MdHcU4C4Rm.

This module implements only the estimators and exact-risk machinery of
Section 2 (structural, finite-sample, distribution-free) and Section 4
(one-shot GCV tuning) of the paper. It imports no code from any official
or third-party repository -- every formula below was transcribed directly
from the arXiv PDF (see docstrings for equation numbers) and cross-checked
internally by two independent solvers wherever the paper allows it
(direct optimum vs. derivative identity; closed-form GCV vs. brute-force
1-D minimization).

Ridge objective (paper Eq. 2):
    beta_hat(lambda) = argmin_b  ||y - X b||_2^2 / n  +  lambda ||b||_2^2
                      = (X^T X / n + lambda I)^{-1} X^T y / n

Pure-distilled (PD) predictor (Eq. 3 with xi=1): ridge fit of (X, yhat_lambda)
at the SAME penalty lambda, where yhat_lambda = X @ beta_hat(lambda).

Self-distilled (SD) predictor (Eq. 4, affine path):
    f_sd(x; lambda, xi) = (1 - xi) f_teacher(x) + xi f_pd(x)

Conditional squared prediction risk (Eq. 5) of a fixed linear predictor with
coefficient vector b, evaluated against the TRUE data-generating
(Sigma, beta, sigma^2) of Assumption A (test point independent of training
data, or an out-of-distribution test covariance Sigma_te / noise sigma2_te):
    R(b) = sigma2_te + (b - beta)^T Sigma_te (b - beta)
This is the *exact* population risk (an expectation over the test point
only, conditional on the fixed training draw D) -- no test-set Monte Carlo
is used anywhere in this reproduction; every risk number below is exact to
floating-point precision given (X, y) and the true (Sigma, beta, sigma2).
"""
import numpy as np


# --------------------------------------------------------------------------- #
#  Data generation (Assumption A: x = Sigma^{1/2} z, y = x^T beta + eps)
# --------------------------------------------------------------------------- #
def ar1_cov(p, rho):
    """AR(1) anisotropic covariance Sigma_ij = rho^|i-j|."""
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def aligned_signal(Sigma, r2, align_frac, top_frac, rng):
    """Deterministic signal beta with ||beta||^2 = r2, with `align_frac` of the
    signal energy placed on the top `top_frac` fraction of Sigma's eigenvectors
    (matches the paper's Figure 4 / Section F alignment construction)."""
    p = Sigma.shape[0]
    evals, evecs = np.linalg.eigh(Sigma)  # ascending
    k = max(1, int(round(top_frac * p)))
    top, rest = evecs[:, -k:], evecs[:, :-k]
    b_top = rng.standard_normal(k)
    b_top /= np.linalg.norm(b_top)
    beta = np.sqrt(align_frac) * (top @ b_top)
    if p - k > 0:
        b_rest = rng.standard_normal(p - k)
        b_rest /= np.linalg.norm(b_rest)
        beta = beta + np.sqrt(1 - align_frac) * (rest @ b_rest)
    beta *= np.sqrt(r2 / (beta @ beta))
    return beta


def make_linear_data(n, Sigma, beta, sigma, rng):
    """y = X beta + eps, X ~ N(0, Sigma) rows, eps ~ N(0, sigma^2)."""
    p = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n, p))
    X = Z @ L.T
    eps = sigma * rng.standard_normal(n)
    y = X @ beta + eps
    return X, y


# --------------------------------------------------------------------------- #
#  Teacher / pure-distilled ridge fits (Eq. 2-3)
# --------------------------------------------------------------------------- #
def ridge_beta(X, y, lam):
    n, p = X.shape
    A = X.T @ X / n + lam * np.eye(p)
    return np.linalg.solve(A, X.T @ y / n)


def teacher_pd_betas(X, y, lam):
    """Returns (beta_teacher, beta_pd, A) where A = X^T X/n + lambda I is the
    ridge normal-equations matrix (returned for reuse in the analytic
    derivative dbeta/dlambda = -A^{-1} beta_teacher)."""
    n, p = X.shape
    A = X.T @ X / n + lam * np.eye(p)
    beta_t = np.linalg.solve(A, X.T @ y / n)
    yhat_t = X @ beta_t
    beta_pd = np.linalg.solve(A, X.T @ yhat_t / n)
    return beta_t, beta_pd, A


# --------------------------------------------------------------------------- #
#  Exact population risk / structural quantities (Eq. 5-8)
# --------------------------------------------------------------------------- #
def risk_of(beta, beta_true, Sigma_te, sigma2_te):
    d = beta - beta_true
    return sigma2_te + d @ (Sigma_te @ d)


def structural_RCD(beta_t, beta_pd, beta_true, Sigma_te, sigma2_te):
    """R(lambda), C(lambda), D(lambda) from Eq. 6-7, evaluated exactly
    (population expectation over the test point only)."""
    d0 = beta_t - beta_true
    d_pd0 = beta_pd - beta_true
    dv = beta_pd - beta_t
    R = sigma2_te + d0 @ (Sigma_te @ d0)
    C = sigma2_te + d0 @ (Sigma_te @ d_pd0)
    D = dv @ (Sigma_te @ dv)
    return R, C, D


def optimal_sd_prop21(R, C, D):
    """Proposition 2.1 (Eq. 8): direct closed form via R, C, D."""
    xi = (R - C) / D
    Rsd = R - (R - C) ** 2 / D
    return xi, Rsd


def sd_risk_at(R, C, D, xi):
    """R_sd(lambda, xi) for ANY xi (not just the optimum); quadratic path
    used to score the plug-in / one-shot estimate's actual induced risk."""
    return R - 2.0 * xi * (R - C) + xi ** 2 * D


def R_prime_analytic(beta_t, A, Sigma_te, beta_true):
    """Analytic dR/dlambda using dbeta_t/dlambda = -A^{-1} beta_t (implicit
    differentiation of the ridge normal equations A(lambda) beta_t(lambda) =
    X^T y / n)."""
    dbeta = -np.linalg.solve(A, beta_t)
    d0 = beta_t - beta_true
    return 2.0 * d0 @ (Sigma_te @ dbeta)


def R_prime_findiff(X, y, lam, Sigma_te, beta_true, h=None):
    """Independent central-finite-difference cross-check of R'(lambda)."""
    if h is None:
        h = max(1e-6, lam * 1e-5)
    bt_p, _, _ = teacher_pd_betas(X, y, lam + h)
    bt_m, _, _ = teacher_pd_betas(X, y, lam - h)
    Rp = risk_of(bt_p, beta_true, Sigma_te, 0.0) + 0.0
    Rm = risk_of(bt_m, beta_true, Sigma_te, 0.0) + 0.0
    return (Rp - Rm) / (2 * h)


def optimal_sd_thm22(R, lam, Rprime, D):
    """Theorem 2.2 (Eq. 9): derivative / sign-rule form."""
    xi = -(lam / 2.0) * Rprime / D
    Rsd = R - (lam ** 2 / 4.0) * Rprime ** 2 / D
    return xi, Rsd


# --------------------------------------------------------------------------- #
#  One-shot GCV tuning (Section 4, Eq. 17-20)
# --------------------------------------------------------------------------- #
def hat_matrix(X, lam):
    """S_lambda = X (X^T X/n + lambda I)^{-1} X^T / n, the ridge smoother
    ("hat") matrix, so that yhat_lambda = S_lambda @ y. O(n^2 p + n^3);
    kept for small-n use / cross-checking against the SVD path below."""
    n, p = X.shape
    A = X.T @ X / n + lam * np.eye(p)
    XAinv = np.linalg.solve(A, X.T)  # p x n = A^{-1} X^T
    return X @ XAinv / n


def gcv_oneshot(X, y, lam):
    """One-shot GCV estimators (Eq. 17-19) of R(lambda), R_pd(lambda),
    C(lambda), and the resulting plug-in xi_hat*(lambda), Rsd_hat*(lambda).
    Uses ONLY the training data (X, y): no grid search over xi, no sample
    splitting, no student refit. df_pd = tr(S_lambda^2) because
    yhat_pd = S_lambda @ yhat_lambda = S_lambda^2 @ y as a function of the
    ORIGINAL response y (Efron's degrees-of-freedom trace definition,
    paper Section 4.1).

    Implemented via the economy SVD of X = U diag(s) V^T (n x r, r x r,
    p x r, r=min(n,p)) rather than forming the n x n hat matrix S_lambda
    explicitly: the hat matrix's eigenvalues along U are
    h_i = (s_i^2/n) / (s_i^2/n + lambda), so
        yhat_lambda    = U diag(h)   (U^T y)
        yhat_pd,lambda = U diag(h^2) (U^T y)      [= S_lambda^2 y]
        df_lambda = sum(h), df_pd,lambda = sum(h^2)
    This is O(n p min(n,p)) instead of O(n^3) and avoids ever materializing
    an n x n matrix, which matters at the n=3200 scale used below."""
    n = X.shape[0]
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    h = (s ** 2 / n) / (s ** 2 / n + lam)
    Uty = U.T @ y
    yhat_t = U @ (h * Uty)
    yhat_pd = U @ (h ** 2 * Uty)
    df_t = float(np.sum(h))
    df_pd = float(np.sum(h ** 2))
    r_t = (y - yhat_t) / (1 - df_t / n)
    r_pd = (y - yhat_pd) / (1 - df_pd / n)
    Rhat = float(r_t @ r_t) / n
    Rpdhat = float(r_pd @ r_pd) / n
    Chat = float(r_t @ r_pd) / n
    Dhat = Rhat + Rpdhat - 2 * Chat
    xi_hat = (Rhat - Chat) / Dhat
    Rsd_hat = Rhat - (Rhat - Chat) ** 2 / Dhat
    return dict(xi_hat=xi_hat, Rsd_hat=Rsd_hat, Rhat=Rhat, Rpdhat=Rpdhat,
                Chat=Chat, Dhat=Dhat, df_t=df_t, df_pd=df_pd)


# --------------------------------------------------------------------------- #
#  Isotropic asymptotic deterministic equivalents (Theorem 3.1 + Cor. 3.2,
#  isotropic specialization Sigma = I_p, beta ~ N(0, (r2/p) I_p))
# --------------------------------------------------------------------------- #
def isotropic_kappa(lam, gamma):
    """Closed-form positive root of the ridge companion equation
    kappa = lambda + gamma * kappa * tr(Sigma (Sigma+kappa I)^{-1})
    specialized to Sigma = I_p, i.e. kappa^2 + kappa(1-gamma-lambda) - lambda = 0."""
    b_coef = 1.0 - gamma - lam
    disc = b_coef ** 2 + 4.0 * lam
    return (-b_coef + np.sqrt(disc)) / 2.0


def isotropic_asymptotics(lam, gamma, r2, sigma2):
    """Deterministic equivalents R(lambda), C(lambda), R_pd(lambda) (Eq. 12-16
    of Theorem 3.1), specialized to isotropic Sigma=I_p and isotropic random
    signal beta~N(0,(r2/p)I_p), where q_k -> r2 / (1+kappa)^k a.s. and
    t_k = gamma / (1+kappa)^k. Returns dict with R, C, Rpd, D, xi, Rsd, kappa."""
    kappa = isotropic_kappa(lam, gamma)
    g = 1.0 + kappa
    t2, t3, t4 = gamma / g ** 2, gamma / g ** 3, gamma / g ** 4
    q2, q3, q4 = r2 / g ** 2, r2 / g ** 3, r2 / g ** 4
    b = 1.0 / (1.0 - t2)
    u2 = t2 * b
    u3 = t3 * b ** 3
    u4 = t4 * b ** 4 + 2 * t3 ** 2 * b ** 5
    E = kappa - b * lam + b ** 2 * kappa * lam * t3
    a2 = b * E ** 2 + b ** 4 * kappa ** 2 * lam ** 2 * t4 + b ** 5 * kappa ** 2 * lam ** 2 * t3 ** 2
    a3 = 2 * b ** 2 * kappa * lam * E
    a4 = b ** 3 * kappa ** 2 * lam ** 2

    R = kappa ** 2 * b * q2 + sigma2 * u2 + sigma2
    C = 2 * kappa ** 2 * b * q2 - (kappa * b * E * q2 + kappa ** 2 * b ** 2 * lam * q3) + sigma2 * (u2 - lam * u3) + sigma2
    Rpd = (4 * kappa ** 2 * b * q2
           - 2 * (2 * kappa * b * E * q2 + 2 * kappa ** 2 * b ** 2 * lam * q3)
           + (a2 * q2 + a3 * q3 + a4 * q4)
           + sigma2 * (u2 - 2 * lam * u3 + lam ** 2 * u4) + sigma2)
    D = R + Rpd - 2 * C
    xi = (R - C) / D
    Rsd = R - (R - C) ** 2 / D
    return dict(kappa=kappa, R=R, C=C, Rpd=Rpd, D=D, xi=xi, Rsd=Rsd)
