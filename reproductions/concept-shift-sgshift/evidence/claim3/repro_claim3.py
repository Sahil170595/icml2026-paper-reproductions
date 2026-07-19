"""Claim 3: SGShift variants substantially outperform baselines (Diff, WhyShift,
SHAP) at identifying shifted features.

For each dataset/config compares the best SGShift variant AUC to the best
baseline AUC. Writes results.json. Run common/run_experiments.py first."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import agg

rows = agg.load()
DS = ['diabetes', 'support2', 'adult']
SG = ['SGShift', 'SGShift-A', 'SGShift-K', 'SGShift-KA']
BL = ['Diff', 'WhyShift', 'SHAP']

print("CLAIM 3  SGShift > baselines (Diff / WhyShift / SHAP) at detecting shifted features")
print(f"{'dataset/config':20s} {'best SGShift':>13s} {'best baseline':>14s} {'gap(AUC)':>9s} {'SG wins':>8s}")
res = {}; gaps = []; wins = 0; total = 0
for ds in DS:
    for cfg in ['matched', 'mismatch']:
        sg = max(agg.agg_metric(rows, 'auc', m, ds, cfg)[0] for m in SG)
        bl = max(agg.agg_metric(rows, 'auc', m, ds, cfg)[0] for m in BL)
        gap = sg - bl
        gaps.append(gap); total += 1; wins += int(sg >= bl - 1e-9)
        res[f'{ds}/{cfg}'] = dict(best_sgshift=sg, best_baseline=bl, gap=gap)
        print(f"{ds+'/'+cfg:20s} {sg:.3f}        {bl:.3f}         {gap:+.3f}   {str(sg>=bl):>6s}")
mean_gap = sum(gaps) / len(gaps)
verdict = 'SUPPORTED' if wins == total and mean_gap > 0 else ('PARTIAL' if wins >= total - 1 else 'NOT SUPPORTED')
print(f"SGShift best-or-tied in {wins}/{total} cells; mean AUC gap = {mean_gap:+.3f}; VERDICT: {verdict}")
print("(paper: SGShift AUC typically 0.1-0.2 higher than baselines, recall 2-3x)")

json.dump(dict(claim='SGShift outperforms baselines Diff/WhyShift/SHAP',
               per_cell=res, cells_won=wins, cells_total=total, mean_auc_gap=mean_gap,
               verdict=verdict), open(os.path.join(HERE, 'results.json'), 'w'), indent=2)
print("wrote results.json")
