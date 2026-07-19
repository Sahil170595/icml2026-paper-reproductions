"""
SGShift core: independent NumPy/scikit-learn reproduction of
"Explaining Concept Shift with Interpretable Feature Attribution" (SGShift),
arXiv 2505.20634 / OpenReview wpKA7G7Cqu.

Implements: sparse (l1) additive GAM shift model with an OFFSET from a base
source model (SGShift), the difference-in-difference absorption term (SGShift-A),
Model-X Gaussian knockoffs with derandomized stability selection and FDR/PFER
control (SGShift-K), the combined SGShift-KA, and the three adapted baselines
(Diff, WhyShift, SHAP). CPU-only, deterministic via numpy.random.default_rng
and fixed sklearn random_state. Single-thread (set OMP/OPENBLAS to 1).

Everything is written to be faithful to sections 3-5 of the paper:
  target logit under T:  g(E_T[y|X]) = f(X) + phi(X)^T delta,  delta sparse.
We use a per-feature additive basis phi (identity / linear-link GAM basis,
one basis function per feature) so that "shifted feature" == feature j with
delta_j != 0, which is exactly the quantity scored by AUC / recall.
"""
import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-9

# ----------------------------- basis --------------------------------------
def make_basis(M):
    """Per-feature additive basis. One linear basis function per feature:
    phi_j(x) = x_j (features are pre-standardized). Returns (Phi, groups)
    where groups[j] lists the design columns belonging to feature j."""
    Phi = M
    groups = [[j] for j in range(M.shape[1])]
    return Phi, groups

# ------------------------ Model-X Gaussian knockoffs ----------------------
class KnockoffSampler:
    """Equicorrelated Model-X Gaussian knockoffs (Candes et al. 2018).
    Precompute the conditional-mean map and noise Cholesky once; each draw
    only resamples the Gaussian noise (used for derandomization)."""
    def __init__(self, Z):
        n, p = Z.shape
        Sigma = (Z.T @ Z) / n + 1e-6 * np.eye(p)
        ev = np.linalg.eigvalsh(Sigma)
        lam_min = max(float(ev[0]), 1e-6)
        s = min(1.0, 2.0 * lam_min) * np.ones(p)
        D = np.diag(s)
        Sinv = np.linalg.inv(Sigma)
        self.M = np.eye(p) - Sinv @ D          # Ztilde_mean = Z @ M
        C = 2.0 * D - D @ Sinv @ D
        C = 0.5 * (C + C.T)
        cev, Q = np.linalg.eigh(C)
        cev = np.clip(cev, 0.0, None)
        self.L = Q @ np.diag(np.sqrt(cev))     # C = L L^T
        self.Z = Z
        self.mean = Z @ self.M
        self.p = p; self.n = n
    def draw(self, rng):
        noise = rng.standard_normal((self.n, self.p)) @ self.L.T
        return self.mean + noise

# ----------------------------- FISTA (l1) ---------------------------------
def _power_L(design, task, iters=15):
    m = design.shape[1]
    v = (np.ones(m) / np.sqrt(m)).astype(np.float32)
    lam = 1.0
    for _ in range(iters):
        w = design.T @ (design @ v)
        lam = float(np.linalg.norm(w))
        if lam < EPS:
            break
        v = w / lam
    return (0.25 if task == 'clf' else 1.0) * lam

def _resid(design, coef, y, offset, task):
    eta = offset + design @ coef
    if task == 'clf':
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30))) - y
    return eta - y

def fista(design, y, offset, pen, task, L, max_iter=80, coef0=None, tol=1e-6):
    """min_coef  loss(offset+design@coef) + sum_j pen_j |coef_j|  (float32 hot loop)."""
    m = design.shape[1]
    step = np.float32(1.0 / L)
    coef = np.zeros(m, np.float32) if coef0 is None else coef0.astype(np.float32)
    z = coef.copy(); t = 1.0
    thr = (step * pen).astype(np.float32)
    for _ in range(max_iter):
        g = design.T @ _resid(design, z, y, offset, task)
        w = z - step * g
        coef_new = np.sign(w) * np.maximum(np.abs(w) - thr, np.float32(0.0))
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = coef_new + np.float32((t - 1.0) / t_new) * (coef_new - coef)
        if np.max(np.abs(coef_new - coef)) < tol * (1.0 + np.max(np.abs(coef_new))):
            coef = coef_new; break
        coef = coef_new; t = t_new
    return coef

def _lam_max(design, y, offset, task, base_pen):
    r0 = _resid(design, np.zeros(design.shape[1], np.float32), y, offset, task)
    g0 = np.abs(design.T @ r0)
    return float(np.max(g0 / np.maximum(base_pen, EPS)))

def _path_area(design, y, offset, task, base_pen, n_lam=9, lo=0.02, max_iter=80):
    """l1 path (float32); return per-column integrated |coef| (area) and per-column
    entry lambda (largest lambda at which the column is nonzero)."""
    design = np.asarray(design, np.float32)
    y = np.asarray(y, np.float32); offset = np.asarray(offset, np.float32)
    base_pen = np.asarray(base_pen, np.float32)
    L = _power_L(design, task)
    lam_max = _lam_max(design, y, offset, task, base_pen)
    lams = lam_max * np.geomspace(1.0, lo, n_lam)
    m = design.shape[1]
    coef = np.zeros(m, np.float32); area = np.zeros(m); entry = np.zeros(m)
    for lam in lams:
        coef = fista(design, y, offset, lam * base_pen, task, L, max_iter=max_iter, coef0=coef)
        area += np.abs(coef)
        newly = (np.abs(coef) > 1e-8) & (entry == 0)
        entry[newly] = lam
    return area, entry

# ----------------------------- methods ------------------------------------
def sgshift(Phi_T, y_T, off_T, groups, task, n_lam=9):
    area, _ = _path_area(Phi_T, y_T, off_T, task, np.ones(Phi_T.shape[1]), n_lam=n_lam)
    return np.array([area[c].sum() for c in groups])

def sgshift_A(Phi_S, y_S, off_S, Phi_T, y_T, off_T, groups, task, rho=0.5, n_lam=9):
    nS, K = Phi_S.shape; nT = Phi_T.shape[0]
    design = np.zeros((nS + nT, 2 * K))
    design[:nS, :K] = Phi_S
    design[nS:, :K] = Phi_T
    design[nS:, K:] = Phi_T
    y = np.concatenate([y_S, y_T]); off = np.concatenate([off_S, off_T])
    base_pen = np.concatenate([rho * np.ones(K), np.ones(K)])  # lam_omega < lam_delta
    area, _ = _path_area(design, y, off, task, base_pen, n_lam=n_lam)
    d_area = area[K:]
    return np.array([d_area[c].sum() for c in groups])

def _knockoff_threshold(W, q):
    cand = np.sort(np.abs(W[np.abs(W) > 0]))
    for t in cand:
        num = 1 + np.sum(W <= -t); den = max(1, np.sum(W >= t))
        if num / den <= q:
            return t
    return np.inf

def _W_from_entry(entry, groups, K):
    """entry has 2K entries [real(K), knockoff(K)] at the basis level.
    Aggregate to feature level: W_k = max(Zr,Zk)*sign(Zr-Zk)."""
    er = entry[:K]; ek = entry[K:]
    W = np.zeros(len(groups))
    for gi, cols in enumerate(groups):
        Zr = er[cols].max(); Zk = ek[cols].max()
        W[gi] = max(Zr, Zk) * np.sign(Zr - Zk)
    return W

def sgshift_K(Phi_T, Z_T, y_T, off_T, groups, task, sampler, rng, B=4, qs=(0.1, 0.2, 0.3),
              pi=0.5, n_lam=9):
    K = Phi_T.shape[1]
    Wsum = np.zeros(len(groups))
    sel_count = {q: np.zeros(len(groups)) for q in qs}
    for b in range(B):
        Zt = sampler.draw(rng)
        Phit, _ = make_basis(Zt)
        design = np.concatenate([Phi_T, Phit], axis=1)
        _, entry = _path_area(design, y_T, off_T, task, np.ones(2 * K), n_lam=n_lam)
        W = _W_from_entry(entry, groups, K)
        Wsum += W
        for q in qs:
            tau = _knockoff_threshold(W, q)
            sel = (W >= tau) & (W > 0)
            sel_count[q][sel] += 1
    meanW = Wsum / B
    selections = {q: (sel_count[q] / B >= pi) for q in qs}
    return meanW, selections

def sgshift_KA(Phi_S, y_S, off_S, Phi_T, y_T, off_T, Z_T, groups, task, sampler, rng,
               B=4, qs=(0.1, 0.2, 0.3), pi=0.5, rho=0.5, n_lam=9):
    nS, K = Phi_S.shape; nT = Phi_T.shape[0]
    Wsum = np.zeros(len(groups))
    sel_count = {q: np.zeros(len(groups)) for q in qs}
    for b in range(B):
        Zt = sampler.draw(rng)
        Phit, _ = make_basis(Zt)
        # variables [omega(K), delta(K), delta_tilde(K)]
        design = np.zeros((nS + nT, 3 * K))
        design[:nS, :K] = Phi_S
        design[nS:, :K] = Phi_T
        design[nS:, K:2 * K] = Phi_T
        design[nS:, 2 * K:] = Phit
        y = np.concatenate([y_S, y_T]); off = np.concatenate([off_S, off_T])
        base_pen = np.concatenate([rho * np.ones(K), np.ones(2 * K)])
        _, entry = _path_area(design, y, off, task, base_pen, n_lam=n_lam)
        entry_dd = entry[K:]  # [delta(K), delta_tilde(K)]
        W = _W_from_entry(entry_dd, groups, K)
        Wsum += W
        for q in qs:
            tau = _knockoff_threshold(W, q)
            sel = (W >= tau) & (W > 0)
            sel_count[q][sel] += 1
    meanW = Wsum / B
    selections = {q: (sel_count[q] / B >= pi) for q in qs}
    return meanW, selections

# ----------------------------- baselines ----------------------------------
def _fit_base(Z, y, task, seed):
    if task == 'clf':
        m = LogisticRegression(max_iter=200, C=1.0)
    else:
        m = Ridge(alpha=1.0)
    m.fit(Z, y)
    return m

def _proba_or_pred(m, Z, task):
    if task == 'clf':
        p = m.predict_proba(Z)[:, 1]
        return np.clip(p, 1e-4, 1 - 1e-4)
    return m.predict(Z)

def baseline_diff(Z_S, y_S, Z_T, y_T, task, seed):
    mS = _fit_base(Z_S, y_S, task, seed); mT = _fit_base(Z_T, y_T, task, seed + 1)
    d = _proba_or_pred(mT, Z_T, task) - _proba_or_pred(mS, Z_T, task)
    lin = Lasso(alpha=0.01, max_iter=2000).fit(Z_T, d)
    return np.abs(lin.coef_)

def baseline_whyshift(Z_S, y_S, Z_T, y_T, task, seed):
    mS = _fit_base(Z_S, y_S, task, seed); mT = _fit_base(Z_T, y_T, task, seed + 1)
    d = _proba_or_pred(mT, Z_T, task) - _proba_or_pred(mS, Z_T, task)
    tree = DecisionTreeRegressor(max_depth=5, random_state=seed).fit(Z_T, d)
    return tree.feature_importances_.copy()

def baseline_shap(Z_S, y_S, Z_T, y_T, task, seed):
    # SHAP adapted (MBM+23): separate models, rank |change in mean-abs Shapley|.
    # Linear/logistic Shapley of feature j at x = coef_j*(x_j - E[x_j]); with
    # standardized features mean|SHAP_j| is proportional to |coef_j|.
    mS = _fit_base(Z_S, y_S, task, seed); mT = _fit_base(Z_T, y_T, task, seed + 1)
    cS = mS.coef_.ravel(); cT = mT.coef_.ravel()
    sS = np.abs(cS) * Z_S.std(0); sT = np.abs(cT) * Z_T.std(0)
    return np.abs(sT - sS)

# ----------------------------- metrics ------------------------------------
def auc(true01, imp):
    if len(np.unique(true01)) < 2:
        return float('nan')
    return float(roc_auc_score(true01, imp))

def recall_at_fpr(true01, imp, fpr_target=0.10):
    if len(np.unique(true01)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(true01, imp)
    return float(np.interp(fpr_target, fpr, tpr))

def fdp(true01, selected):
    s = int(selected.sum())
    if s == 0:
        return 0.0
    return float(np.sum(selected & (true01 == 0)) / s)

def power(true01, selected):
    a = int(true01.sum())
    if a == 0:
        return float('nan')
    return float(np.sum(selected & (true01 == 1)) / a)

# ----------------------------- simulation ---------------------------------
def relabel_and_shift(Zc, idx_S, idx_T, y_real, task, seed, a_shift, shift_scale, setting):
    """Semi-synthetic protocol (paper sec 5): fit a generator on the real source
    labels, relabel the source, induce a sparse additive concept shift on a known
    set A of features in the target, then fit a base model on the relabelled
    source. Returns everything needed to run the estimators plus ground truth A."""
    rng = default_rng(seed)
    p = Zc.shape[1]
    Z_S = Zc[idx_S]; Z_T = Zc[idx_T]
    yS_real = y_real[idx_S]
    # ---- generator (fit to REAL source labels) ----
    if setting == 'matched':
        gen = _fit_base(Z_S, yS_real, task, seed)
        f_gen = lambda Z: (np.log(_proba_or_pred(gen, Z, task) / (1 - _proba_or_pred(gen, Z, task)))
                           if task == 'clf' else gen.predict(Z))
    else:
        if task == 'clf':
            gen = HistGradientBoostingClassifier(max_iter=60, max_depth=4, random_state=seed,
                                                 learning_rate=0.15)
        else:
            gen = HistGradientBoostingRegressor(max_iter=60, max_depth=4, random_state=seed,
                                                learning_rate=0.15)
        gen.fit(Z_S, yS_real)
        f_gen = lambda Z: (np.log(_proba_or_pred(gen, Z, task) / (1 - _proba_or_pred(gen, Z, task)))
                           if task == 'clf' else gen.predict(Z))
    # ---- relabel source ----
    if task == 'clf':
        pS = 1.0 / (1.0 + np.exp(-np.clip(f_gen(Z_S), -30, 30)))
        y_S = (rng.random(len(pS)) < pS).astype(float)
    else:
        y_S = f_gen(Z_S) + rng.standard_normal(len(idx_S)) * 1.0
    # ---- induce sparse concept shift on target ----
    A = np.sort(rng.choice(p, size=a_shift, replace=False))
    signs = rng.choice([-1.0, 1.0], size=a_shift)
    mags = rng.uniform(0.8, 1.6, size=a_shift) * shift_scale
    delta_true = np.zeros(p); delta_true[A] = signs * mags
    shift_T = Z_T @ delta_true
    fT = f_gen(Z_T) + shift_T
    if task == 'clf':
        pT = 1.0 / (1.0 + np.exp(-np.clip(fT, -30, 30)))
        y_T = (rng.random(len(pT)) < pT).astype(float)
    else:
        y_T = fT + rng.standard_normal(len(idx_T)) * 1.0
    # ---- base model on relabelled source (offset provider) ----
    base = _fit_base(Z_S, y_S, task, seed + 7)   # base class = linear (mismatched vs GBM gen)
    if task == 'clf':
        off_S = base.decision_function(Z_S); off_T = base.decision_function(Z_T)
    else:
        off_S = base.predict(Z_S); off_T = base.predict(Z_T)
    true01 = np.zeros(p, dtype=int); true01[A] = 1
    return dict(Z_S=Z_S, Z_T=Z_T, y_S=y_S, y_T=y_T, off_S=off_S, off_T=off_T,
                true01=true01, A=A, delta_true=delta_true)

def run_replicate(Zc, idx_S, idx_T, y_real, task, seed, setting,
                  a_shift=5, shift_scale=1.5, B=5, qs=(0.1, 0.2, 0.3), pi=0.5, n_lam=9):
    d = relabel_and_shift(Zc, idx_S, idx_T, y_real, task, seed, a_shift, shift_scale, setting)
    Z_S, Z_T = d['Z_S'], d['Z_T']; y_S, y_T = d['y_S'], d['y_T']
    off_S, off_T = d['off_S'], d['off_T']; true01 = d['true01']
    Phi_S, groups = make_basis(Z_S); Phi_T, _ = make_basis(Z_T)
    rng = default_rng(seed + 1000)
    sampler = KnockoffSampler(Z_T)

    imp_sg = sgshift(Phi_T, y_T, off_T, groups, task, n_lam=n_lam)
    imp_a = sgshift_A(Phi_S, y_S, off_S, Phi_T, y_T, off_T, groups, task, n_lam=n_lam)
    meanW_k, sel_k = sgshift_K(Phi_T, Z_T, y_T, off_T, groups, task, sampler, rng, B=B, qs=qs, pi=pi, n_lam=n_lam)
    meanW_ka, sel_ka = sgshift_KA(Phi_S, y_S, off_S, Phi_T, y_T, off_T, Z_T, groups, task, sampler, rng, B=B, qs=qs, pi=pi, n_lam=n_lam)
    imp_diff = baseline_diff(Z_S, y_S, Z_T, y_T, task, seed)
    imp_why = baseline_whyshift(Z_S, y_S, Z_T, y_T, task, seed)
    imp_shap = baseline_shap(Z_S, y_S, Z_T, y_T, task, seed)

    out = {'seed': seed, 'setting': setting, 'a_shift': a_shift, 'p': int(Zc.shape[1]),
           'nS': int(len(idx_S)), 'nT': int(len(idx_T)), 'B': B, 'shift_scale': shift_scale}
    for name, imp in [('SGShift', imp_sg), ('SGShift-A', imp_a), ('SGShift-K', meanW_k),
                      ('SGShift-KA', meanW_ka), ('Diff', imp_diff), ('WhyShift', imp_why),
                      ('SHAP', imp_shap)]:
        out[name] = {'auc': auc(true01, imp), 'recall': recall_at_fpr(true01, imp)}
    for q in qs:
        out['SGShift-K'][f'fdp_q{q}'] = fdp(true01, sel_k[q])
        out['SGShift-K'][f'pow_q{q}'] = power(true01, sel_k[q])
        out['SGShift-K'][f'nsel_q{q}'] = int(sel_k[q].sum())
        out['SGShift-KA'][f'fdp_q{q}'] = fdp(true01, sel_ka[q])
        out['SGShift-KA'][f'pow_q{q}'] = power(true01, sel_ka[q])
        out['SGShift-KA'][f'nsel_q{q}'] = int(sel_ka[q].sum())
    return out
