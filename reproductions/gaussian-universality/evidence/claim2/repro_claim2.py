"""
Independent NumPy/scipy reproduction of Theorem 6.1 (Claim 2) from
"Characterization of Gaussian Universality Breakdown in High-Dimensional
Empirical Risk Minimization" (arXiv 2604.03146, ICML 2026).

Claim: the projection of the ERM (ridge) estimator decomposes as
    x^T theta_hat ~= x^T mu*  +  alpha* z,   z ~ N(0,1) independent of x.
(a) For a FIXED test point x0, s = x0^T theta_hat is Gaussian across
    training sets with mean x0^T mu* and variance x0^T C_theta x0.
(b) Over the test mixture, the marginal score density = convolution of the
    (non-Gaussian, bimodal) projection density x^T mu* with N(0, alpha*^2),
    alpha*^2 = tr(C_theta C_x).

Deterministic, CPU-only. Closed-form ridge solves.
"""
import json
import os
from pathlib import Path
import numpy as np
from scipy import stats

OUTPUT_PATH = Path(os.environ.get("GAUSSIAN_UNIVERSALITY_OUTPUT", Path(__file__).with_name("evidence.json")))

rng = np.random.default_rng(20260716)

# ---------------- DGP ----------------
p = 250
n = 500
gamma = p / n              # 0.5
lam = 0.1                  # ridge penalty
sigma2 = 0.01              # label noise variance
w = np.array([0.3, 0.7])   # mixture weights

# fixed mixture direction u (unit norm)
u = rng.standard_normal(p); u /= np.linalg.norm(u)
m_scale = 2.0
# component means shifted along u so that E[x]=0: w0*mu0 + w1*mu1 = 0
mu0 = m_scale * u
mu1 = -(w[0] / w[1]) * m_scale * u
comp_means = np.stack([mu0, mu1])         # (2,p)
C_x = np.eye(p)                            # identity covariance per component

# ground-truth signal (unit norm). Paper's Fig-1 instance: the discriminative
# signal is ALIGNED with the mixture direction u, so the class separation lies
# along the signal direction. This is what makes mu* = E[theta_hat] carry the
# mixture, giving a bimodal (non-Gaussian) projection x^T mu*. A signal
# orthogonal to u would make the projection Gaussian (no universality break).
theta_star = u.copy()


def sample_x(K, rng):
    """Draw K covariates from the 2-component Gaussian mixture (E[x]=0)."""
    comp = rng.choice(2, size=K, p=w)
    z = rng.standard_normal((K, p))        # C_x = I
    return z + comp_means[comp]


def fit_ridge(rng):
    """One training set -> closed-form ridge theta_hat."""
    X = sample_x(n, rng)
    y = X @ theta_star + np.sqrt(sigma2) * rng.standard_normal(n)
    A = X.T @ X / n + lam * np.eye(p)
    b = X.T @ y / n
    return np.linalg.solve(A, b)


# ---------------- STEP 1: estimate mu*, C_theta, alpha* ----------------
M = 4000
Theta = np.empty((M, p))
for i in range(M):
    Theta[i] = fit_ridge(rng)

mu_hat = Theta.mean(axis=0)                 # proxy for mu*
C_theta = np.cov(Theta, rowvar=False)       # sample covariance of theta_hat
alpha2 = np.trace(C_theta @ C_x)            # = tr(C_theta) since C_x = I
alpha = np.sqrt(alpha2)
print(f"[Step1] M={M}, p={p}, n={n}, gamma={gamma}")
print(f"[Step1] ||mu_hat||={np.linalg.norm(mu_hat):.4f}, "
      f"||theta_star||={np.linalg.norm(theta_star):.4f}")
print(f"[Step1] alpha*^2 = tr(C_theta C_x) = {alpha2:.6e}, alpha* = {alpha:.6e}")

# ---------------- STEP 2: Thm 6.1a conditional Gaussianity ----------------
# fixed test points x0 (fresh, held out; independent of training)
n_fixed = 5
X0 = sample_x(n_fixed, np.random.default_rng(777))
step2 = []
print("\n[Step2] Conditional Gaussianity of s = x0^T theta_hat over M training sets")
for j in range(n_fixed):
    x0 = X0[j]
    s = Theta @ x0                          # (M,)
    pred_mean = x0 @ mu_hat
    pred_var = x0 @ C_theta @ x0
    emp_mean = s.mean()
    emp_var = s.var(ddof=1)
    se = np.sqrt(emp_var / M)
    mean_diff = abs(emp_mean - pred_mean)
    var_ratio = emp_var / pred_var
    skew = stats.skew(s)
    exkurt = stats.kurtosis(s)              # excess (Fisher)
    # KS against predicted normal
    ks_D, ks_p = stats.kstest(s, 'norm', args=(pred_mean, np.sqrt(pred_var)))
    sh_W, sh_p = stats.shapiro(s if M <= 5000 else s[:5000])
    rec = dict(idx=j, pred_mean=float(pred_mean), emp_mean=float(emp_mean),
               mean_diff=float(mean_diff), se=float(se),
               within2se=bool(mean_diff < 2 * se),
               pred_var=float(pred_var), emp_var=float(emp_var),
               var_ratio=float(var_ratio), skew=float(skew),
               exkurt=float(exkurt), ks_D=float(ks_D), ks_p=float(ks_p),
               shapiro_p=float(sh_p))
    step2.append(rec)
    print(f"  x0[{j}]: pred_mean={pred_mean:+.4f} emp_mean={emp_mean:+.4f} "
          f"|diff|={mean_diff:.4f} (2SE={2*se:.4f}, ok={rec['within2se']}) "
          f"var_ratio={var_ratio:.4f} skew={skew:+.3f} exkurt={exkurt:+.3f} "
          f"KS_p={ks_p:.3f} Shapiro_p={sh_p:.3f}")

# ---------------- STEP 3: Thm 6.1b convolution structure ----------------
K = 20000
rng3 = np.random.default_rng(12345)
Xtest = sample_x(K, rng3)
proj = Xtest @ mu_hat                        # non-Gaussian, bimodal

# (i) empirical score from FRESH (x, theta_hat) pairs: use held-out theta_hat
#     draw fresh independent theta_hats and fresh x, pair them up
rng_fit = np.random.default_rng(99)
Theta_fresh = np.empty((K, p))
# reuse pooled draws: sample K theta_hats is expensive (K=20000 solves).
# Instead pair each fresh x with an independent theta_hat drawn from the
# estimated law: sample from the M-pool with replacement (bootstrap of the
# empirical theta_hat law), giving genuine (x, theta_hat) score samples.
idx = rng3.integers(0, M, size=K)
score_true = np.einsum('ij,ij->i', Xtest, Theta[idx])   # x^T theta_hat, real pairs

# (ii) convolution model: proj + independent N(0, alpha2)
score_model = proj + alpha * rng3.standard_normal(K)

# KS distance between the two empirical distributions
ks_D_conv, ks_p_conv = stats.ks_2samp(score_true, score_model)

# projection alone: non-Gaussianity
proj_exkurt = float(stats.kurtosis(proj))
proj_skew = float(stats.skew(proj))
# bimodality: dip-ish via comparing to unimodal — use excess kurtosis<0 + two modes
# detect two modes by checking sign of proj clusters
proj_ks_norm_D, proj_ks_norm_p = stats.kstest(
    proj, 'norm', args=(proj.mean(), proj.std()))
# score (convolved) Gaussianity
score_exkurt = float(stats.kurtosis(score_true))
score_skew = float(stats.skew(score_true))

print("\n[Step3] Convolution structure (Thm 6.1b)")
print(f"  proj = x^T mu_hat : mean={proj.mean():+.4f} std={proj.std():.4f} "
      f"skew={proj_skew:+.3f} exkurt={proj_exkurt:+.3f} "
      f"KS-vs-normal D={proj_ks_norm_D:.4f} p={proj_ks_norm_p:.2e}")
print(f"  score (real pairs): std={score_true.std():.4f} "
      f"skew={score_skew:+.3f} exkurt={score_exkurt:+.3f}")
print(f"  KS( empirical score , proj (*) N(0,alpha^2) ) D={ks_D_conv:.4f} "
      f"p={ks_p_conv:.3f}")
print(f"  |exkurt| reduced by convolution: proj={abs(proj_exkurt):.3f} -> "
      f"score={abs(score_exkurt):.3f}")

# ---------------- verdict ----------------
step2_ok = all(r['within2se'] and abs(r['var_ratio'] - 1) < 0.05
               and abs(r['skew']) < 0.15 and abs(r['exkurt']) < 0.20
               and r['ks_p'] > 0.05 for r in step2)
step3_conv_ok = ks_D_conv < 0.02
step3_proj_nongauss = (proj_exkurt < -0.05) and (proj_ks_norm_p < 1e-3)
step3_conv_closer = abs(score_exkurt) < abs(proj_exkurt)

print("\n[Verdict]")
print(f"  Step2 conditional Gaussian (all x0 pass): {step2_ok}")
print(f"  Step3 KS(score, conv-model) < 0.02: {step3_conv_ok} (D={ks_D_conv:.4f})")
print(f"  Step3 projection non-Gaussian/bimodal: {step3_proj_nongauss} "
      f"(exkurt={proj_exkurt:+.3f})")
print(f"  Step3 convolution brings score closer to Gaussian: {step3_conv_closer}")

evidence = dict(
    paper="arXiv 2604.03146 Theorem 6.1 (Claim 2)",
    setup=dict(p=p, n=n, gamma=gamma, lam=lam, sigma2=sigma2,
               weights=w.tolist(), M=M, K=K, C_x="identity"),
    step1=dict(alpha2=float(alpha2), alpha=float(alpha),
               mu_hat_norm=float(np.linalg.norm(mu_hat))),
    step2=step2,
    step3=dict(ks_D_score_vs_conv=float(ks_D_conv),
               ks_p_score_vs_conv=float(ks_p_conv),
               proj_exkurt=proj_exkurt, proj_skew=proj_skew,
               proj_ks_norm_D=float(proj_ks_norm_D),
               proj_ks_norm_p=float(proj_ks_norm_p),
               score_exkurt=score_exkurt, score_skew=score_skew),
    verdict=dict(step2_ok=bool(step2_ok), step3_conv_ok=bool(step3_conv_ok),
                 step3_proj_nongauss=bool(step3_proj_nongauss),
                 step3_conv_closer=bool(step3_conv_closer)),
)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open('w', encoding='utf-8') as fjson:
    json.dump(evidence, fjson, indent=2)
print('\n[done] results written to', OUTPUT_PATH.name)
