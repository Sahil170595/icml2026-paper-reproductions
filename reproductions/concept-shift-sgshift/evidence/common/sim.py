"""Semi-synthetic concept-shift protocol (paper sec 5) on REAL covariates.
Main effect f = main_scale*(linear fit to real source labels) + quadratic main
effect (both domains). Sparse additive shift (linear+quadratic GAM components)
on a known set A in the target only. MATCHED base = GAM (logistic/ridge on the
[x, x^2-1] basis) captures f -> well-specified offset. MISMATCHED base = linear
(on x only) misses the quadratic main effect -> the 'poor model fit' regime the
absorption term targets. Estimators imported from the stable sgcore module."""
import numpy as np
from numpy.random import default_rng
import sgcore as sg

def _flogit(m, X, task):
    if task == 'clf':
        pr = np.clip(m.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4); return np.log(pr / (1 - pr))
    return m.predict(X)

def relabel_and_shift(Zc, iS, iT, y_real, task, seed, a_shift, shift_scale, setting,
                      q_main=0.6, main_scale=1.6):
    rng = default_rng(seed); p = Zc.shape[1]
    Z_S, Z_T = Zc[iS], Zc[iT]
    Phi_S, _ = sg.make_basis(Z_S); Phi_T, _ = sg.make_basis(Z_T)
    lin = sg._fit_base(Z_S, y_real[iS], task, seed)
    f_lin_S = _flogit(lin, Z_S, task); f_lin_T = _flogit(lin, Z_T, task)
    cq = rng.standard_normal(p) * q_main                    # quadratic main-effect coefs
    f_main_S = main_scale * f_lin_S + (Z_S ** 2 - 1) @ cq
    f_main_T = main_scale * f_lin_T + (Z_T ** 2 - 1) @ cq
    if task == 'clf':
        y_S = (rng.random(len(iS)) < 1 / (1 + np.exp(-np.clip(f_main_S, -30, 30)))).astype(float)
    else:
        y_S = f_main_S + rng.standard_normal(len(iS))
    A = np.sort(rng.choice(p, a_shift, replace=False))
    comps = rng.choice(['lin', 'quad', 'both'], a_shift, p=[0.2, 0.6, 0.2])
    db = np.zeros(2 * p)
    for k, j in enumerate(A):
        if comps[k] in ('lin', 'both'): db[2 * j] = rng.choice([-1., 1.]) * rng.uniform(0.8, 1.6) * shift_scale
        if comps[k] in ('quad', 'both'): db[2 * j + 1] = rng.choice([-1., 1.]) * rng.uniform(0.8, 1.6) * shift_scale
    fT = f_main_T + Phi_T @ db
    if task == 'clf':
        y_T = (rng.random(len(iT)) < 1 / (1 + np.exp(-np.clip(fT, -30, 30)))).astype(float)
    else:
        y_T = fT + rng.standard_normal(len(iT))
    if setting == 'matched':                                # GAM base captures quadratic main effect
        base = sg._fit_base(Phi_S, y_S, task, seed + 7)
        off_S = _flogit(base, Phi_S, task); off_T = _flogit(base, Phi_T, task)
    else:                                                   # linear base misses it (misspecified)
        base = sg._fit_base(Z_S, y_S, task, seed + 7)
        off_S = _flogit(base, Z_S, task); off_T = _flogit(base, Z_T, task)
    t01 = np.zeros(p, int); t01[A] = 1
    return dict(Z_S=Z_S, Z_T=Z_T, Phi_S=Phi_S, Phi_T=Phi_T, y_S=y_S, y_T=y_T,
                off_S=off_S, off_T=off_T, true01=t01)

def run_replicate(Zc, iS, iT, y_real, task, seed, setting, a_shift=6, shift_scale=0.35,
                  B=4, qs=(0.1, 0.2, 0.3), pi=0.5, n_lam=9, q_main=0.6, main_scale=1.6):
    d = relabel_and_shift(Zc, iS, iT, y_real, task, seed, a_shift, shift_scale, setting, q_main, main_scale)
    Z_S, Z_T, Phi_S, Phi_T = d['Z_S'], d['Z_T'], d['Phi_S'], d['Phi_T']
    y_S, y_T, off_S, off_T, t01 = d['y_S'], d['y_T'], d['off_S'], d['off_T'], d['true01']
    _, groups = sg.make_basis(Z_S)
    rng = default_rng(seed + 1000); samp = sg.KnockoffSampler(Z_T)
    imps = {'SGShift': sg.sgshift(Phi_T, y_T, off_T, groups, task, n_lam=n_lam),
            'SGShift-A': sg.sgshift_A(Phi_S, y_S, off_S, Phi_T, y_T, off_T, groups, task, n_lam=n_lam),
            'Diff': sg.baseline_diff(Z_S, y_S, Z_T, y_T, task, seed),
            'WhyShift': sg.baseline_whyshift(Z_S, y_S, Z_T, y_T, task, seed),
            'SHAP': sg.baseline_shap(Z_S, y_S, Z_T, y_T, task, seed)}
    mW_k, sel_k = sg.sgshift_K(Phi_T, Z_T, y_T, off_T, groups, task, samp, rng, B=B, qs=qs, pi=pi, n_lam=n_lam)
    mW_ka, sel_ka = sg.sgshift_KA(Phi_S, y_S, off_S, Phi_T, y_T, off_T, Z_T, groups, task, samp, rng, B=B, qs=qs, pi=pi, n_lam=n_lam)
    imps['SGShift-K'] = mW_k; imps['SGShift-KA'] = mW_ka
    out = {'seed': seed, 'setting': setting, 'a_shift': a_shift, 'p': int(Zc.shape[1]),
           'nS': int(len(iS)), 'nT': int(len(iT)), 'B': B, 'shift_scale': shift_scale}
    for nm, imp in imps.items():
        out[nm] = {'auc': sg.auc(t01, imp), 'recall': sg.recall_at_fpr(t01, imp)}
    for q in qs:
        for nm, sel in [('SGShift-K', sel_k), ('SGShift-KA', sel_ka)]:
            out[nm][f'fdp_q{q}'] = sg.fdp(t01, sel[q]); out[nm][f'pow_q{q}'] = sg.power(t01, sel[q]); out[nm][f'nsel_q{q}'] = int(sel[q].sum())
    return out
