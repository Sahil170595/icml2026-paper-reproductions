"""Sample-efficiency experiment (Claim 2 fix: SGShift requires few target-domain
samples for effective shifted-feature detection).

Reuses sgshift.py's simulate_replicate() unchanged: base/generator models are
still trained on the full real source (as in the other claim pages), and the
concept-shift relabeling of the target domain is identical. The ONLY new knob
is an independent subsample of the TARGET design (Zt) down to n_target rows
before the SGShift / knockoff sparse fit AND before the baselines (Diff,
WhyShift, SHAP) refit their target-domain model -- source fit size is held
FIXED at SRC_CAP=6000 (matching the other claim pages) so the sweep isolates
target sample size, not source size.

STAGED for the 45s Bash cap: one invocation = one (dataset, n_target, seed)
cell, appended to _cache/sampleeff.jsonl. Deterministic: seeds are derived
only from (dataset index, n_target, replicate index) via numpy default_rng.

Usage:
  python sample_efficiency.py init
  python sample_efficiency.py <dataset> <n_target|full> <rep_start> <rep_count>
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import data_prep as D
import sgshift as S

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', '_cache')
OUT = os.path.join(CACHE, 'sampleeff.jsonl')
DATASETS = ['diabetes', 'support2', 'adult']
SRC_CAP = 6000          # fixed source fit size (matches other claim pages)
N_SHIFT = 6
NKO_K, NKO_KA = 4, 6
GEN, BASE = 'logit', 'logit'   # matched setting (mirrors reference sample-efficiency protocol)
SHAP_SUB = 800           # permutation-importance subsample cap (cost control)

def one_cell(ds, n_target_label, rep):
    di = DATASETS.index(ds)
    base_seed = 1000 * di + rep
    Z, y, dom, names = D.load_npz(os.path.join(CACHE, ds + '.npz'))
    d = S.simulate_replicate(Z, dom, GEN, BASE, n_shift=N_SHIFT, seed=base_seed, fit_cap=None)
    tr = d['shifted']
    Zs_full, ys_full, offs_full = d['Zs'], d['ys'], d['off_s']
    Zt_full, yt_full, offt_full = d['Zt'], d['yt'], d['off_t']
    n_tgt_full = len(Zt_full)
    n_target = n_tgt_full if n_target_label == 'full' else min(int(n_target_label), n_tgt_full)

    rng = np.random.default_rng(base_seed * 7919 + n_target + 3)
    if len(Zs_full) > SRC_CAP:
        ix_s = rng.choice(len(Zs_full), size=SRC_CAP, replace=False)
        Zs, ys, off_s = Zs_full[ix_s], ys_full[ix_s], offs_full[ix_s]
    else:
        Zs, ys, off_s = Zs_full, ys_full, offs_full
    if n_target < n_tgt_full:
        ix_t = rng.choice(n_tgt_full, size=n_target, replace=False)
        Zt, yt, off_t = Zt_full[ix_t], yt_full[ix_t], offt_full[ix_t]
    else:
        Zt, yt, off_t = Zt_full, yt_full, offt_full

    t0 = time.time()
    sc = {}
    sc['SGShift'] = S.m_sgshift(Zt, yt, off_t)
    delta_scores, gamma = S._joint_absorb(Zs, ys, off_s, Zt, yt, off_t)
    off2 = off_t + Zt @ gamma
    Wk = S.m_sgshift_K(Zt, yt, off_t, n_ko=NKO_K, seed=base_seed)
    sc['SGShift-K'] = S.ko_score(Wk)
    Wka = S.m_sgshift_K(Zt, yt, off2, n_ko=NKO_KA, seed=base_seed)
    sc['SGShift-KA'] = S.ko_score(Wka)
    # Baselines refit on the SAME subsampled target (this is the point of the
    # sweep: do baselines degrade faster than SGShift as n_target shrinks?).
    sc['Diff'] = S.b_diff(Zs, ys, Zt, yt)
    sc['WhyShift'] = S.b_whyshift(Zs, ys, Zt, yt)
    try:
        sc['SHAP'] = S.b_shap(Zs, ys, Zt, yt, seed=base_seed, sub=SHAP_SUB)
    except Exception:
        sc['SHAP'] = np.zeros(Zt.shape[1])
    auc = {k: S.auc_features(tr, v) for k, v in sc.items()}
    rec = {k: S.recall_at_fpr(tr, v) for k, v in sc.items()}
    ko = {}
    for q in (0.1, 0.2):
        _, seld = S.knockoff_select_derand(Wka, q)
        fdr, power, ns = S.empirical_fdr_power(seld, tr)
        ko[f'fdr_derand_{q}'] = fdr
        ko[f'power_derand_{q}'] = power
    dt = time.time() - t0
    return dict(dataset=ds, n_target_label=str(n_target_label), n_target=int(n_target),
                n_tgt_full=int(n_tgt_full), n_src=int(len(Zs)), rep=int(rep),
                seed=int(base_seed), n_shift=N_SHIFT, n_feat=int(Z.shape[1]),
                auc=auc, recall=rec, ko=ko, elapsed_s=round(dt, 3))

def main():
    os.makedirs(CACHE, exist_ok=True)
    if sys.argv[1] == 'init':
        open(OUT, 'w').close()
        print('init sampleeff.jsonl'); return
    ds, n_label, start, count = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    t0 = time.time()
    with open(OUT, 'a') as out:
        for r in range(start, start + count):
            rec = one_cell(ds, n_label, r)
            out.write(json.dumps(rec) + '\n'); out.flush()
            print(f"{ds} n_target={n_label} rep{r} auc(K)={rec['auc']['SGShift-K']:.3f} "
                  f"auc(KA)={rec['auc']['SGShift-KA']:.3f} auc(Diff)={rec['auc']['Diff']:.3f} "
                  f"[{time.time()-t0:.1f}s]", flush=True)
    print(f"CHUNK DONE {ds} n_target={n_label} rep{start}..{start+count-1} [{time.time()-t0:.1f}s]", flush=True)

if __name__ == '__main__':
    main()
