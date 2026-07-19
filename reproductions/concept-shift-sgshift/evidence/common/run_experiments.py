"""Master experiment runner for the SGShift reproduction (STAGED / chunked).

Usage:
  python run_experiments.py init                 # truncate runs.jsonl
  python run_experiments.py <dataset> <config> <start> <count>   # append runs

For each real dataset and model setting (matched / mismatched) we run
semi-synthetic replicates with known ground-truth shifted features. Per replicate
we compute AUC and recall@FPR10% for SGShift / -A / -K / -KA and baselines
(Diff, WhyShift, SHAP), plus knockoff empirical FDR & power at q in {0.1, 0.2}
(single-draw knockoff+ filter and derandomized selection). Appends one JSON line
per (dataset, config, replicate) to _cache/runs.jsonl. All numbers are executed.
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import data_prep as D
import sgshift as S

CACHE = os.path.join(os.path.dirname(__file__), '..', '_cache')
DATASETS = ['diabetes', 'support2', 'adult']
GEN = {'matched': ('logit', 'logit'), 'mismatch': ('gboost', 'logit')}
FIT_CAP = int(os.environ.get('SG_FITCAP', '6000'))
N_SHIFT = 6
NKO_K, NKO_KA = 4, 6
RUNS = os.path.join(CACHE, 'runs.jsonl')

def one_run(Z, dom, cfgname, gen, base, seed):
    d = S.simulate_replicate(Z, dom, gen, base, n_shift=N_SHIFT, seed=seed, fit_cap=FIT_CAP)
    tr = d['shifted']
    Zs, ys, os_, Zt, yt, ot = d['Zs'], d['ys'], d['off_s'], d['Zt'], d['yt'], d['off_t']
    delta_scores, gamma = S._joint_absorb(Zs, ys, os_, Zt, yt, ot)   # joint absorption
    off2 = ot + Zt @ gamma
    sc = {}
    sc['SGShift'] = S.m_sgshift(Zt, yt, ot)
    sc['SGShift-A'] = delta_scores
    Wk = S.m_sgshift_K(Zt, yt, ot, n_ko=NKO_K, seed=seed)
    sc['SGShift-K'] = S.ko_score(Wk)
    Wka = S.m_sgshift_K(Zt, yt, off2, n_ko=NKO_KA, seed=seed)
    sc['SGShift-KA'] = S.ko_score(Wka)
    sc['Diff'] = S.b_diff(Zs, ys, Zt, yt)
    sc['WhyShift'] = S.b_whyshift(Zs, ys, Zt, yt)
    sc['SHAP'] = S.b_shap(Zs, ys, Zt, yt, seed=seed, sub=1000)
    auc = {k: S.auc_features(tr, v) for k, v in sc.items()}
    rec = {k: S.recall_at_fpr(tr, v) for k, v in sc.items()}
    ko = {}
    for q in (0.1, 0.2):
        sel = Wka[0] >= S.knockoff_threshold(Wka[0], q)
        f, p, ns = S.empirical_fdr_power(sel, tr)
        ko[f'fdp_single_{q}'] = f; ko[f'pow_single_{q}'] = p
        _, seld = S.knockoff_select_derand(Wka, q)
        fd, pd, nd = S.empirical_fdr_power(seld, tr)
        ko[f'fdp_derand_{q}'] = fd; ko[f'pow_derand_{q}'] = pd
    return dict(dataset=None, config=cfgname, gen=gen, base=base, seed=int(seed),
                n_src_full=int(d['n_src_full']), n_tgt_full=int(d['n_tgt_full']),
                n_fit=int(d['n_fit']), n_shift=N_SHIFT, n_feat=int(Z.shape[1]),
                auc=auc, recall=rec, ko=ko)

def main():
    if sys.argv[1] == 'init':
        open(RUNS, 'w').close()
        print('init runs.jsonl'); return
    ds, cfg, start, count = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    gen, base = GEN[cfg]
    Z, y, dom, names = D.load_npz(os.path.join(CACHE, ds + '.npz'))
    t0 = time.time()
    with open(RUNS, 'a') as out:
        for r in range(start, start + count):
            seed = 1000 * DATASETS.index(ds) + (0 if cfg == 'matched' else 500) + r
            rec = one_run(Z, dom, cfg, gen, base, seed)
            rec['dataset'] = ds
            out.write(json.dumps(rec) + '\n'); out.flush()
            print(f"{ds}/{cfg} rep{r} done [{time.time()-t0:.1f}s]", flush=True)
    print(f"CHUNK DONE {ds}/{cfg} {start}..{start+count-1} [{time.time()-t0:.1f}s]", flush=True)

if __name__ == '__main__':
    main()
