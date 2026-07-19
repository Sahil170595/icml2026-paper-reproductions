"""Test FDR control fixes on real diabetes: Gaussian rank-transform of the design
before knockoffs, and derandomized selection."""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import norm, rankdata
import data_prep as D, sgshift as S

def gaussianize(Z):
    n = Z.shape[0]
    out = np.empty_like(Z)
    for j in range(Z.shape[1]):
        r = rankdata(Z[:, j], method='average')
        out[:, j] = norm.ppf((r - 0.5) / n)
    return out

CACHE = os.path.join(os.path.dirname(__file__), '..', '_cache')
Z, y, dom, names = D.load_npz(os.path.join(CACHE, 'diabetes.npz'))
t0 = time.time()
for gen, base, tag in [('logit', 'logit', 'matched'), ('gboost', 'logit', 'mismatch')]:
    raw1, raw2, g1, g2, dr1, dr2 = [], [], [], [], [], []
    pw = []
    for seed in range(4):
        d = S.simulate_replicate(Z, dom, gen, base, n_shift=6, seed=seed, fit_cap=6000)
        tr = d['shifted']
        off2, _ = S._absorb_offset(d['Zs'], d['ys'], d['off_s'], d['Zt'], d['off_t'])
        Zg = gaussianize(d['Zt'])
        # raw knockoffs, single draw, FDP@.1/.2
        Wr = S.knockoff_stats(d['Zt'], d['yt'], off2, seed=seed)
        raw1.append(S.empirical_fdr_power(Wr >= S.knockoff_threshold(Wr, 0.1), tr)[0])
        raw2.append(S.empirical_fdr_power(Wr >= S.knockoff_threshold(Wr, 0.2), tr)[0])
        # gaussianized knockoffs, single draw
        Wg = S.knockoff_stats(Zg, d['yt'], off2, seed=seed)
        g1.append(S.empirical_fdr_power(Wg >= S.knockoff_threshold(Wg, 0.1), tr)[0])
        fdp2, p2, ns = S.empirical_fdr_power(Wg >= S.knockoff_threshold(Wg, 0.2), tr)
        g2.append(fdp2); pw.append(p2)
        # gaussianized + derandomized (10 draws, eta=0.5) at q=.1/.2
        Ws = np.array([S.knockoff_stats(Zg, d['yt'], off2, seed=seed * 97 + b) for b in range(8)])
        _, sel1 = S.knockoff_select_derand(Ws, 0.1); dr1.append(S.empirical_fdr_power(sel1, tr)[0])
        _, sel2 = S.knockoff_select_derand(Ws, 0.2); dr2.append(S.empirical_fdr_power(sel2, tr)[0])
    print(f"[{tag}] raw FDP@.1={np.mean(raw1):.2f}@.2={np.mean(raw2):.2f} | "
          f"gauss FDP@.1={np.mean(g1):.2f}@.2={np.mean(g2):.2f} pow@.2={np.mean(pw):.2f} | "
          f"gauss+derand FDP@.1={np.mean(dr1):.2f}@.2={np.mean(dr2):.2f} [{time.time()-t0:.0f}s]", flush=True)
print("elapsed %.1fs" % (time.time() - t0))
