"""Fast synthetic smoke test to validate SGShift core (no dataset download)."""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import sgshift as S

t0 = time.time()
rng = np.random.default_rng(0)
n, p = 4000, 20
# correlated real-like covariates
A = rng.standard_normal((p, p)) * 0.3 + np.eye(p)
Z = rng.standard_normal((n, p)) @ A
Z = (Z - Z.mean(0)) / Z.std(0)
dom = (rng.random(n) < 0.4).astype(int)

for gen, base, tag in [('logit', 'logit', 'matched'), ('gboost', 'logit', 'mismatched')]:
    aucs = {k: [] for k in ['SGShift', 'SGShift-A', 'SGShift-K', 'SGShift-KA', 'Diff', 'WhyShift', 'SHAP']}
    fdrs = []; pws = []
    for seed in range(4):
        d = S.simulate_replicate(Z, dom, gen, base, n_shift=7, seed=seed)
        tr = d['shifted']
        aucs['SGShift'].append(S.auc_features(tr, S.m_sgshift(d['Zt'], d['yt'], d['off_t'])))
        aucs['SGShift-A'].append(S.auc_features(tr, S.m_sgshift_A(d['Zs'], d['ys'], d['off_s'], d['Zt'], d['yt'], d['off_t'])))
        Wk = S.m_sgshift_K(d['Zt'], d['yt'], d['off_t'], n_ko=6, seed=seed)
        aucs['SGShift-K'].append(S.auc_features(tr, S.ko_score(Wk)))
        Wka = S.m_sgshift_KA(d['Zs'], d['ys'], d['off_s'], d['Zt'], d['yt'], d['off_t'], n_ko=6, seed=seed)
        aucs['SGShift-KA'].append(S.auc_features(tr, S.ko_score(Wka)))
        freqka, selka = S.knockoff_select_derand(Wka, q=0.2)
        fdr, pw, ns = S.empirical_fdr_power(selka, tr)
        fdrs.append(fdr); pws.append(pw)
        aucs['Diff'].append(S.auc_features(tr, S.b_diff(d['Zs'], d['ys'], d['Zt'], d['yt'])))
        aucs['WhyShift'].append(S.auc_features(tr, S.b_whyshift(d['Zs'], d['ys'], d['Zt'], d['yt'])))
        aucs['SHAP'].append(S.auc_features(tr, S.b_shap(d['Zs'], d['ys'], d['Zt'], d['yt'], seed=seed)))
    print(f"[{tag}] " + " ".join(f"{k}={np.nanmean(v):.3f}" for k, v in aucs.items()) +
          f" | KA_empFDR@0.2={np.mean(fdrs):.3f} KA_power@0.2={np.mean(pws):.3f}")
print("elapsed %.1fs" % (time.time() - t0))
