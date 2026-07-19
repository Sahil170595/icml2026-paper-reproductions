"""Cache all 3 real datasets to _cache/*.npz and probe SGShift on real diabetes.
Prints real numbers for calibration."""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.linalg import eigh
import data_prep as D
import sgshift as S

CACHE = os.path.join(os.path.dirname(__file__), '..', '_cache')
os.makedirs(CACHE, exist_ok=True)
t0 = time.time()

for name in ['diabetes', 'support2', 'adult']:
    X, y, dom, names = D.PREP[name]()
    Xs = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    np.savez_compressed(os.path.join(CACHE, name + '.npz'), X=Xs, y=y, dom=dom,
                        names=np.array(names, dtype=object))
    print(f"CACHED {name}: X={X.shape} feats={len(names)} "
          f"src(non-tgt)={int((dom==0).sum())} tgt={int((dom==1).sum())} "
          f"pos={y.mean():.4f} [{time.time()-t0:.0f}s]", flush=True)

# probe diabetes
Z, y, dom, names = D.load_npz(os.path.join(CACHE, 'diabetes.npz'))
Zt = Z[dom == 1]
R = np.corrcoef(Zt.T)
print("diabetes target corr lambda_min = %.3f  p=%d" % (max(eigh(R, eigvals_only=True).min(), 0), Z.shape[1]), flush=True)

for gen, base, tag in [('logit', 'logit', 'matched'), ('gboost', 'logit', 'mismatch')]:
    res = {k: [] for k in ['SG', 'SG-A', 'SG-K', 'SG-KA']}
    fdp1, pw1, fdp2, pw2 = [], [], [], []
    for seed in range(4):
        d = S.simulate_replicate(Z, dom, gen, base, n_shift=6, seed=seed)
        tr = d['shifted']
        res['SG'].append(S.auc_features(tr, S.m_sgshift(d['Zt'], d['yt'], d['off_t'])))
        res['SG-A'].append(S.auc_features(tr, S.m_sgshift_A(d['Zs'], d['ys'], d['off_s'], d['Zt'], d['yt'], d['off_t'])))
        Wk = S.m_sgshift_K(d['Zt'], d['yt'], d['off_t'], n_ko=5, seed=seed)
        res['SG-K'].append(S.auc_features(tr, S.ko_score(Wk)))
        Wka = S.m_sgshift_KA(d['Zs'], d['ys'], d['off_s'], d['Zt'], d['yt'], d['off_t'], n_ko=5, seed=seed)
        res['SG-KA'].append(S.auc_features(tr, S.ko_score(Wka)))
        # FDR at q=0.1 and 0.2 using KA single-draw filter
        off2, _ = S._absorb_offset(d['Zs'], d['ys'], d['off_s'], d['Zt'], d['off_t'])
        W = S.knockoff_stats(d['Zt'], d['yt'], off2, seed=seed)
        for q, F, P in [(0.1, fdp1, pw1), (0.2, fdp2, pw2)]:
            sel = W >= S.knockoff_threshold(W, q)
            f, pw, ns = S.empirical_fdr_power(sel, tr)
            F.append(f); P.append(pw)
    print(f"[{tag}] " + " ".join(f"{k}={np.nanmean(v):.3f}" for k, v in res.items()) +
          f" | FDP@.1={np.mean(fdp1):.3f} pow@.1={np.mean(pw1):.3f}"
          f" FDP@.2={np.mean(fdp2):.3f} pow@.2={np.mean(pw2):.3f} [{time.time()-t0:.0f}s]", flush=True)
print("elapsed %.1fs" % (time.time() - t0))
