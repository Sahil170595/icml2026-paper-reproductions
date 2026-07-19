"""Knockoff FDR-control check on synthetic: matched -K, mismatched -K (no
absorption), and mismatched -KA (absorption). Reports AUC, power, empirical FDP
of the plain knockoff+ filter across seeds for target_lmin values."""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, sgshift as S

t0 = time.time()
rng = np.random.default_rng(1)
n, p = 4000, 24
A = rng.standard_normal((p, p)) * 0.35 + np.eye(p)
Z = rng.standard_normal((n, p)) @ A
Z = (Z - Z.mean(0)) / Z.std(0)
dom = (rng.random(n) < 0.4).astype(int)

def run(gen, base, absorb, tl, q, seeds=6):
    aucs, fdps, pows = [], [], []
    for seed in range(seeds):
        d = S.simulate_replicate(Z, dom, gen, base, n_shift=8, seed=seed)
        tr = d['shifted']
        if absorb:
            off2, _ = S._absorb_offset(d['Zs'], d['ys'], d['off_s'], d['Zt'], d['off_t'])
        else:
            off2 = d['off_t']
        W = S.knockoff_stats(d['Zt'], d['yt'], off2, seed=seed)
        thr = S.knockoff_threshold(W, q)
        sel = W >= thr
        fdp, pw, ns = S.empirical_fdr_power(sel, tr)
        aucs.append(S.auc_features(tr, W)); fdps.append(fdp); pows.append(pw)
    return np.nanmean(aucs), np.mean(fdps), np.mean(pows)

for tag, gen, base, absorb in [('matched-K', 'logit', 'logit', False),
                               ('mismatch-K', 'gboost', 'logit', False),
                               ('mismatch-KA', 'gboost', 'logit', True)]:
    for q in [0.1, 0.2]:
        a, f, pw = run(gen, base, absorb, None, q)
        print(f"{tag:12s} q={q}: AUC={a:.3f} empFDP={f:.3f} power={pw:.3f}")
print("elapsed %.1fs" % (time.time() - t0))
