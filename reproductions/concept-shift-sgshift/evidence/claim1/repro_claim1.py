"""Claim 1: SGShift-KA (knockoffs + absorption) achieves AUC > 0.9 for
identifying shifted features.

Aggregates executed semi-synthetic runs (produced by common/run_experiments.py)
over all real datasets and both model settings. Prints a measured-vs-target
table and writes results.json. Run common/run_experiments.py first."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import agg

rows = agg.load()
DS = ['diabetes', 'support2', 'adult']
TARGET = 0.90

per = {}
allvals = []
for ds in DS:
    m, se, n = agg.agg_metric(rows, 'auc', 'SGShift-KA', ds)
    mk, sek, nk = agg.agg_metric(rows, 'auc', 'SGShift-K', ds)
    per[ds] = dict(ka_auc=m, ka_se=se, k_auc=mk, k_se=sek, n=n)
    allvals.append(m)
overall = sum(allvals) / len(allvals)

print("CLAIM 1  SGShift-KA AUC > 0.9 for identifying shifted features")
print(f"{'dataset':10s} {'SGShift-KA AUC':>18s} {'SGShift-K AUC':>16s} {'target':>8s} {'pass':>6s}")
for ds in DS:
    p = per[ds]
    print(f"{ds:10s} {p['ka_auc']:.3f} +/- {p['ka_se']:.3f}   {p['k_auc']:.3f} +/- {p['k_se']:.3f}"
          f"   >{TARGET:.2f}  {str(p['ka_auc'] > TARGET):>5s}")
print(f"{'MEAN':10s} {overall:.3f}")
n_pass = sum(1 for ds in DS if per[ds]['ka_auc'] > TARGET)
verdict = ('SUPPORTED' if overall > TARGET and n_pass >= 2 else 'NOT SUPPORTED')
print(f"overall mean AUC = {overall:.3f} ; datasets > 0.9: {n_pass}/3 ; VERDICT: {verdict}")
print("(paper Table 2: SGShift-KA diabetes 0.90/0.86, COVID 0.95, SUPPORT2 0.96)")

json.dump(dict(claim='SGShift-KA AUC > 0.9', target=TARGET, per_dataset=per,
               overall_mean_auc=overall, datasets_above_target=n_pass,
               verdict=verdict), open(os.path.join(HERE, 'results.json'), 'w'), indent=2)
print("wrote results.json")
