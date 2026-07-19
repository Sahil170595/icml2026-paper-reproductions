"""Fast real-data validation: diabetes, small fit_cap, few seeds."""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, data_prep as D, sgshift as S

CACHE = os.path.join(os.path.dirname(__file__), '..', '_cache')
Z, y, dom, names = D.load_npz(os.path.join(CACHE, 'diabetes.npz'))
t0 = time.time()
for gen, base, tag in [('logit', 'logit', 'matched'), ('gboost', 'logit', 'mismatch')]:
    sg, sga, sgk, sgka, fdp, pw = [], [], [], [], [], []
    for seed in range(3):
        d = S.simulate_replicate(Z, dom, gen, base, n_shift=6, seed=seed, fit_cap=6000)
        tr = d['shifted']
        sg.append(S.auc_features(tr, S.m_sgshift(d['Zt'], d['yt'], d['off_t'])))
        sga.append(S.auc_features(tr, S.m_sgshift_A(d['Zs'], d['ys'], d['off_s'], d['Zt'], d['yt'], d['off_t'])))
        Wk = S.m_sgshift_K(d['Zt'], d['yt'], d['off_t'], n_ko=3, seed=seed)
        sgk.append(S.auc_features(tr, S.ko_score(Wk)))
        off2, _ = S._absorb_offset(d['Zs'], d['ys'], d['off_s'], d['Zt'], d['off_t'])
        Wka = S.m_sgshift_K(d['Zt'], d['yt'], off2, n_ko=3, seed=seed)
        sgka.append(S.auc_features(tr, S.ko_score(Wka)))
        sel = Wka[0] >= S.knockoff_threshold(Wka[0], 0.2)
        f, p, ns = S.empirical_fdr_power(sel, tr); fdp.append(f); pw.append(p)
    print(f"[{tag}] n_fit={d['n_fit']} SG={np.mean(sg):.3f} SG-A={np.mean(sga):.3f} "
          f"SG-K={np.mean(sgk):.3f} SG-KA={np.mean(sgka):.3f} "
          f"KA_FDP@.2={np.mean(fdp):.3f} KA_pow@.2={np.mean(pw):.3f} [{time.time()-t0:.0f}s]", flush=True)
print("elapsed %.1fs" % (time.time() - t0))
