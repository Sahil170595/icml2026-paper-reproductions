"""SGShift core implementation for concept-shift feature attribution.

Faithful reproduction of Lyu, Turcan & Wilder, "Explaining Concept Shift with
Interpretable Feature Attribution" (arXiv:2505.20634).

Method (Section 4 of the paper):
  - SGShift    : sparse (l1) GAM update term fit on top of a source base model,
                 with an offset given by the base model's logit on the target.
  - SGShift-A  : adds an absorption term (difference-in-difference) so that base
                 model misspecification is soaked up by a shared gamma, leaving
                 a target-specific delta that isolates genuine concept shift.
  - SGShift-K  : Model-X Gaussian knockoffs on the update design to control FDR.
  - SGShift-KA : knockoffs applied on top of the absorption-corrected offset.

We use a linear (K=1) GAM basis: one standardized column per original feature.
This is the linear special case of the paper's GAM update term and keeps the
knockoff construction standard (one column per feature). The semi-synthetic
concept shift we induce is additive & linear in the shifted features, matching
the paper's "additive transformations based on selected input features".

Everything is pure numpy/scipy/sklearn, CPU, single-threaded, deterministic.
"""
import numpy as np
from scipy.linalg import cholesky, eigh
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance

CLIP = 30.0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -CLIP, CLIP)))

def logit_from_proba(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

# ---------------------------------------------------------------------------
# FISTA solver: l1-penalized logistic regression with a fixed offset.
#   minimize  (1/n) sum[ log(1+exp(eta_i)) - y_i eta_i ] + lam * ||beta||_1
#   with eta = offset + B @ beta
# ---------------------------------------------------------------------------
def _specnorm(B, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(B.shape[1]); v /= np.linalg.norm(v) + 1e-12
    for _ in range(iters):
        u = B @ v
        v = B.T @ u
        nv = np.linalg.norm(v) + 1e-12
        v /= nv
    return np.sqrt(nv)

def fista_l1_logistic(B, y, offset, lam, max_iter=300, tol=1e-6, L=None, beta0=None,
                      pen_w=None):
    n, p = B.shape
    if L is None:
        L = 0.25 * (_specnorm(B) ** 2) / n + 1e-8
    if pen_w is None:
        pen_w = np.ones(p)
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    z = beta.copy()
    t = 1.0
    step = 1.0 / L
    for it in range(max_iter):
        eta = offset + B @ z
        grad = B.T @ (sigmoid(eta) - y) / n
        w = z - step * grad
        thr = step * lam * pen_w
        beta_new = np.sign(w) * np.maximum(np.abs(w) - thr, 0.0)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = beta_new + ((t - 1) / t_new) * (beta_new - beta)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
        t = t_new
    return beta

def lam_max(B, y, offset):
    n = B.shape[0]
    g = np.abs(B.T @ (y - sigmoid(offset))) / n
    return float(g.max())

def path_entry_scores(B, y, offset, n_lam=15, ratio=0.02):
    """Fit an l1 path; score each column by the largest lambda at which it is
    nonzero (entry lambda). Higher = enters earlier = more important."""
    lm = lam_max(B, y, offset)
    lams = lm * np.geomspace(1.0, ratio, n_lam)
    L = 0.25 * (_specnorm(B) ** 2) / B.shape[0] + 1e-8
    scores = np.zeros(B.shape[1])
    beta = np.zeros(B.shape[1])
    for lam in lams:
        beta = fista_l1_logistic(B, y, offset, lam, L=L, max_iter=150, beta0=beta)
        nz = np.abs(beta) > 1e-8
        newly = nz & (scores == 0)
        scores[newly] = lam
    # break ties among never-selected/equal by final magnitude
    scores = scores + 1e-9 * np.abs(beta)
    return scores

# ---------------------------------------------------------------------------
# Model-X Gaussian knockoffs (equicorrelated construction).
# ---------------------------------------------------------------------------
def sdp_s(R, n_iter=25):
    """Coordinate-ascent SDP for Model-X knockoffs on a correlation matrix R:
    maximize sum(s_j) s.t. 0 <= s_j <= 1 and 2R - diag(s) PSD. Each coordinate is
    set to its exact optimum (via a Schur complement) given the others, which
    keeps the knockoffs VALID (matching the true covariance) while maximizing
    power. Returns s in correlation scale (length p)."""
    p = R.shape[0]
    lam_min = max(eigh(R, eigvals_only=True).min(), 1e-8)
    s = np.minimum(1.0, 2.0 * lam_min) * np.ones(p)   # feasible equicorrelated start
    M = 2 * R - np.diag(s)
    eps = 1e-8
    for _ in range(n_iter):
        for j in range(p):
            idx = np.arange(p) != j
            Msub = M[np.ix_(idx, idx)]
            mj = M[idx, j]
            try:
                schur = mj @ np.linalg.solve(Msub + eps * np.eye(p - 1), mj)
            except np.linalg.LinAlgError:
                continue
            # need M[j,j] = 2 - s_j >= schur  -> s_j <= 2 - schur
            sj_max = min(1.0, max(0.0, 2.0 - schur - 1e-6))
            s[j] = sj_max
            M[j, j] = 2 - s[j]
    return np.clip(s, 0.0, 1.0)

def gaussian_knockoffs(X, seed=0):
    n, p = X.shape
    mu = X.mean(0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / n + 1e-6 * np.eye(p)
    d = np.diag(Sigma).copy(); d[d == 0] = 1.0
    Dh = np.diag(1.0 / np.sqrt(d))
    R = Dh @ Sigma @ Dh                      # correlation matrix
    s_corr = sdp_s(R)                        # valid SDP knockoff diagonal
    s = s_corr * d                           # scale back to feature variances
    D = np.diag(s)
    Sinv = np.linalg.inv(Sigma)
    M = Xc @ (np.eye(p) - Sinv @ D)          # conditional mean part
    V = 2 * D - D @ Sinv @ D                   # conditional covariance
    V = (V + V.T) / 2
    # ensure PSD
    ev, U = eigh(V)
    ev = np.clip(ev, 0, None)
    Vh = (U * np.sqrt(ev)) @ U.T
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((n, p)) @ Vh
    Xk = mu + M + E
    return Xk

def knockoff_threshold(W, q):
    """Standard knockoff+ threshold controlling FDR at level q."""
    ts = np.sort(np.abs(W[W != 0]))
    ts = np.concatenate([[0.0], ts])
    best = np.inf
    for t in ts:
        num = 1 + np.sum(W <= -t)
        den = max(1, np.sum(W >= t))
        if num / den <= q:
            best = t
            break
    return best

def path_abscoef(B, y, offset, n_lam=15, ratio=0.02):
    """l1 path; return per-column max |beta| across the path (LCD statistic)."""
    lm = lam_max(B, y, offset)
    lams = lm * np.geomspace(1.0, ratio, n_lam)
    L = 0.25 * (_specnorm(B) ** 2) / B.shape[0] + 1e-8
    mx = np.zeros(B.shape[1])
    beta = np.zeros(B.shape[1])
    for lam in lams:
        beta = fista_l1_logistic(B, y, offset, lam, L=L, max_iter=150, beta0=beta)
        mx = np.maximum(mx, np.abs(beta))
    return mx

def knockoff_stats(Z, y, offset, seed=0):
    """Return per-feature W = |beta_orig| - |beta_knockoff| via the lasso
    coefficient-difference (LCD) statistic on the augmented design [Z | Zk]."""
    p = Z.shape[1]
    Zk = gaussian_knockoffs(Z, seed=seed)
    A = np.hstack([Z, Zk])
    s = path_abscoef(A, y, offset)
    W = s[:p] - s[p:]
    return W

# ---------------------------------------------------------------------------
# Method entry points -> each returns a per-feature score (length p).
# ---------------------------------------------------------------------------
def m_sgshift(Zt, yt, off_t):
    return path_entry_scores(Zt, yt, off_t)

def _joint_absorb(Zs, ys, off_s, Zt, yt, off_t, ratio=4.0, n_lam=15, ratio_lam=0.02):
    """Joint difference-in-difference absorption (SGShift-A / Section 4.2).

    Stacks source and target. A shared term gamma acts on BOTH domains (absorbing
    base-model misspecification), while a target-only term delta captures the
    concept shift. delta is penalized `ratio`x more heavily than gamma
    (hierarchical regularization), so spurious shifts from poor model fit are
    soaked up by gamma rather than attributed to concept shift. Returns the full
    l1 path of (gamma, delta) and the per-feature delta entry-lambda score.
    """
    ns, nt = len(Zs), len(Zt)
    p = Zs.shape[1]
    y = np.concatenate([ys, yt])
    offset = np.concatenate([off_s, off_t])
    Zg = np.vstack([Zs, Zt])                       # gamma design (both domains)
    Zd = np.vstack([np.zeros_like(Zs), Zt])        # delta design (target only)
    B = np.hstack([Zg, Zd])                        # (ns+nt) x 2p
    pen = np.concatenate([np.ones(p), ratio * np.ones(p)])
    lm = lam_max(B, y, offset)
    lams = lm * np.geomspace(1.0, ratio_lam, n_lam)
    L = 0.25 * (_specnorm(B) ** 2) / (ns + nt) + 1e-8
    scores = np.zeros(p); beta = np.zeros(2 * p)
    for lam in lams:
        beta = fista_l1_logistic(B, y, offset, lam, L=L, max_iter=150, beta0=beta, pen_w=pen)
        delta_nz = np.abs(beta[p:]) > 1e-8
        newly = delta_nz & (scores == 0)
        scores[newly] = lam
    scores = scores + 1e-9 * np.abs(beta[p:])
    gamma = beta[:p]
    return scores, gamma

def _absorb_offset(Zs, ys, off_s, Zt, yt, off_t):
    """Return target offset corrected by the jointly-estimated shared gamma, for
    layering knockoffs on top (SGShift-KA)."""
    _, gamma = _joint_absorb(Zs, ys, off_s, Zt, yt, off_t)
    return off_t + Zt @ gamma, gamma

def m_sgshift_A(Zs, ys, off_s, Zt, yt, off_t):
    scores, _ = _joint_absorb(Zs, ys, off_s, Zt, yt, off_t)
    return scores

def m_sgshift_K(Zt, yt, off_t, n_ko=10, seed=0):
    """Derandomized knockoffs. Returns (freq_score, W_median)."""
    p = Zt.shape[1]
    Ws = np.zeros((n_ko, p))
    for b in range(n_ko):
        Ws[b] = knockoff_stats(Zt, yt, off_t, seed=seed * 131 + b)
    return Ws

def m_sgshift_KA(Zs, ys, off_s, Zt, yt, off_t, n_ko=10, seed=0):
    off2, _ = _absorb_offset(Zs, ys, off_s, Zt, yt, off_t)
    return m_sgshift_K(Zt, yt, off2, n_ko=n_ko, seed=seed)

def ko_score(Ws):
    """Continuous per-feature importance = median knockoff statistic over draws."""
    return np.median(Ws, axis=0)

def knockoff_select_derand(Ws, q, eta=0.5):
    """Derandomized selection: per-draw knockoff filter at q, aggregate freq."""
    n_ko, p = Ws.shape
    freq = np.zeros(p)
    for b in range(n_ko):
        W = Ws[b]
        thr = knockoff_threshold(W, q)
        sel = W >= thr
        freq += sel
    freq /= n_ko
    selected = freq >= eta
    return freq, selected

# ---------------------------------------------------------------------------
# Baselines (adapted from the paper's descriptions).
# ---------------------------------------------------------------------------
def _fit_logit(Z, y):
    m = LogisticRegression(max_iter=200, C=1.0)
    m.fit(Z, y)
    return m

def b_diff(Zs, ys, Zt, yt):
    ms = _fit_logit(Zs, ys); mt = _fit_logit(Zt, yt)
    d = mt.predict_proba(Zt)[:, 1] - ms.predict_proba(Zt)[:, 1]
    la = Lasso(alpha=1e-3, max_iter=2000)
    la.fit(Zt, d)
    return np.abs(la.coef_)

def b_whyshift(Zs, ys, Zt, yt):
    ms = _fit_logit(Zs, ys); mt = _fit_logit(Zt, yt)
    d = mt.predict_proba(Zt)[:, 1] - ms.predict_proba(Zt)[:, 1]
    tree = DecisionTreeRegressor(max_depth=6, random_state=0)
    tree.fit(Zt, d)
    return tree.feature_importances_.astype(float)

def b_shap(Zs, ys, Zt, yt, seed=0, sub=1500):
    ms = _fit_logit(Zs, ys); mt = _fit_logit(Zt, yt)
    rng = np.random.default_rng(seed)
    idx = rng.choice(Zt.shape[0], size=min(sub, Zt.shape[0]), replace=False)
    Zi = Zt[idx]; yi = yt[idx]
    pis = permutation_importance(ms, Zi, yi, n_repeats=3, random_state=seed)
    pit = permutation_importance(mt, Zi, yi, n_repeats=3, random_state=seed)
    return np.abs(pit.importances_mean - pis.importances_mean)

# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------
def auc_features(true_shift, score):
    if true_shift.sum() == 0 or true_shift.sum() == len(true_shift):
        return np.nan
    return float(roc_auc_score(true_shift, score))

def recall_at_fpr(true_shift, score, fpr_target=0.10):
    if true_shift.sum() == 0:
        return np.nan
    fpr, tpr, _ = roc_curve(true_shift, score)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0

def empirical_fdr_power(selected, true_shift):
    sel = selected.astype(bool); tru = true_shift.astype(bool)
    ns = sel.sum()
    fdr = float((sel & ~tru).sum()) / max(1, ns)
    power = float((sel & tru).sum()) / max(1, tru.sum())
    return fdr, power, int(ns)

# ---------------------------------------------------------------------------
# Model factory & semi-synthetic simulation.
# ---------------------------------------------------------------------------
def get_model(name, seed=0):
    if name == 'logit':
        return LogisticRegression(max_iter=300, C=1.0)
    if name == 'tree':
        return DecisionTreeClassifier(max_depth=6, random_state=seed)
    if name == 'gboost':
        return HistGradientBoostingClassifier(max_iter=80, max_depth=4,
                                               learning_rate=0.1, random_state=seed)
    raise ValueError(name)

def simulate_replicate(Z, dom, gen_name, base_name, n_shift=5, scale=(1.0, 2.2),
                       seed=0, fit_cap=15000):
    """Semi-synthetic concept-shift simulation preserving real covariates.

    Returns dict with source/target designs, relabeled outcomes, base offsets,
    and the ground-truth shifted-feature indicator.
    """
    rng = np.random.default_rng(seed)
    src_mask = dom == 0; tgt_mask = dom == 1
    Zs = Z[src_mask]; Zt = Z[tgt_mask]
    p = Z.shape[1]

    # 1) generator fit on real source split, then relabel source (clean labels)
    #    we synthesize a "real-like" source label via a random sparse linear rule
    #    so the generator has signal to learn; covariates stay real.
    w0 = rng.standard_normal(p) * 0.4
    # NONLINEAR source truth so that a gradient-boosting generator learns a
    # genuinely non-linear p(y|X); a linear (logistic) base model is then really
    # misspecified in the mismatched setting, which is what the absorption term
    # is designed to handle.
    iq = rng.choice(p, size=4, replace=False)
    nonlin = (1.2 * Zs[:, iq[0]] * Zs[:, iq[1]]
              + 0.8 * (Zs[:, iq[2]] ** 2 - 1.0)
              - 0.9 * np.abs(Zs[:, iq[3]]))
    p_src_true = sigmoid(Zs @ w0 + nonlin)
    y_src_seed = (rng.random(len(Zs)) < p_src_true).astype(float)
    gen = get_model(gen_name, seed=seed)
    gen.fit(Zs, y_src_seed)
    g_src = logit_from_proba(gen.predict_proba(Zs)[:, 1])
    g_tgt = logit_from_proba(gen.predict_proba(Zt)[:, 1])
    y_src = (rng.random(len(Zs)) < sigmoid(g_src)).astype(float)   # relabeled src

    # 2) induce sparse additive concept shift on the target
    shifted = np.zeros(p, dtype=int)
    idx = rng.choice(p, size=n_shift, replace=False)
    shifted[idx] = 1
    c = np.zeros(p)
    signs = rng.choice([-1, 1], size=n_shift)
    mags = rng.uniform(scale[0], scale[1], size=n_shift)
    c[idx] = signs * mags
    g_tgt_shift = g_tgt + Zt @ c
    y_tgt = (rng.random(len(Zt)) < sigmoid(g_tgt_shift)).astype(float)

    # 3) base model trained on relabeled source (matched or mismatched class)
    base = get_model(base_name, seed=seed + 1)
    base.fit(Zs, y_src)
    off_s = logit_from_proba(base.predict_proba(Zs)[:, 1])
    off_t = logit_from_proba(base.predict_proba(Zt)[:, 1])

    # Base & generator models are trained on the FULL real source/target. The
    # sparse update-term fit (the scored step) uses up to fit_cap target/source
    # rows -- the paper itself highlights SGShift's strong performance with
    # limited target samples (Fig 1). Full real sizes are recorded separately.
    def _cap(A, b, o, cap):
        if cap and len(A) > cap:
            ix = rng.choice(len(A), size=cap, replace=False)
            return A[ix], b[ix], o[ix]
        return A, b, o
    Zs2, ys2, off_s2 = _cap(Zs, y_src, off_s, fit_cap)
    Zt2, yt2, off_t2 = _cap(Zt, y_tgt, off_t, fit_cap)

    return dict(Zs=Zs2, ys=ys2, off_s=off_s2, Zt=Zt2, yt=yt2, off_t=off_t2,
                shifted=shifted, coef=c, n_src_full=len(Zs), n_tgt_full=len(Zt),
                n_fit=len(Zt2))
