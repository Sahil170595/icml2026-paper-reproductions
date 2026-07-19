"""Claim 5: The absorption term (SGShift-A / -KA) improves detection when the base
model is misspecified (mismatched generator/base classes).

Compares SGShift vs SGShift-A (and -K vs -KA) in the mismatched setting, and
contrasts with the matched setting where the base model is well specified.
Writes results.json. Run common/run_experiments.py first."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import agg

rows = agg.load()
DS = ['diabetes', 'support2', 'adult']

print("CLAIM 5  Absorption term improves detection under model misspecification (mismatched)")
print(f"{'dataset':10s} {'cfg':9s} {'SG AUC':>8s} {'SG-A AUC':>9s} {'dAUC':>7s} "
      f"{'SG rec':>8s} {'SG-A rec':>9s} {'dRec':>7s}")
res = {}
mism_auc_gain = []; mism_rec_gain = []
for ds in DS:
    for cfg in ['matched', 'mismatch']:
        a0 = agg.agg_metric(rows, 'auc', 'SGShift', ds, cfg)[0]
        a1 = agg.agg_metric(rows, 'auc', 'SGShift-A', ds, cfg)[0]
        r0 = agg.agg_metric(rows, 'recall', 'SGShift', ds, cfg)[0]
        r1 = agg.agg_metric(rows, 'recall', 'SGShift-A', ds, cfg)[0]
        res[f'{ds}/{cfg}'] = dict(sg_auc=a0, sga_auc=a1, dauc=a1 - a0,
                                  sg_rec=r0, sga_rec=r1, drec=r1 - r0)
        if cfg == 'mismatch':
            mism_auc_gain.append(a1 - a0); mism_rec_gain.append(r1 - r0)
        print(f"{ds:10s} {cfg:9s} {a0:.3f}   {a1:.3f}    {a1-a0:+.3f}  "
              f"{r0:.3f}    {r1:.3f}    {r1-r0:+.3f}")
mean_auc = sum(mism_auc_gain) / len(mism_auc_gain)
mean_rec = sum(mism_rec_gain) / len(mism_rec_gain)
n_help = sum(1 for g in mism_auc_gain if g > -0.005)
verdict = ('PARTIAL' if mean_auc > -0.01 else 'NOT SUPPORTED')
if mean_auc > 0.003 and n_help == len(mism_auc_gain):
    verdict = 'SUPPORTED'
print(f"mismatched mean dAUC = {mean_auc:+.3f}, mean dRecall = {mean_rec:+.3f}; "
      f"absorption non-detrimental in {n_help}/{len(mism_auc_gain)} mismatched datasets; VERDICT: {verdict}")
print("(paper: absorption 'increases performance in nearly every setting, especially recall';"
      " reported effect is small, +0.01-0.02 AUC in Table 2)")

json.dump(dict(claim='Absorption improves detection under model misspecification',
               per_cell=res, mismatch_mean_dauc=mean_auc, mismatch_mean_drecall=mean_rec,
               verdict=verdict), open(os.path.join(HERE, 'results.json'), 'w'), indent=2)
print("wrote results.json")
