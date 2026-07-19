"""Claim 2: On the Diabetes Readmission dataset (real, ~73k samples, ~33 features,
split by ER vs non-ER admission), SGShift identifies concept shift.

Reports the real dataset sizes/split and SGShift-variant AUC/recall on diabetes.
Writes results.json. Run common/run_experiments.py first."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import agg

rows = agg.load()
ns, nt, nf, nfit, nsh = agg.sizes(rows, 'diabetes')
methods = ['SGShift', 'SGShift-A', 'SGShift-K', 'SGShift-KA', 'Diff', 'WhyShift', 'SHAP']
TARGET = 0.85   # paper diabetes SGShift-K/KA AUC ~0.90 (matched) / 0.84-0.86 (mismatch)

print("CLAIM 2  Diabetes Readmission (real, ER vs non-ER split): SGShift detects concept shift")
print(f"dataset: Diabetes 130-US 30-day readmission | non-ER(source)={ns} ER(target)={nt} "
      f"total={ns+nt} features={nf} | update-term fit sample n_fit={nfit} shifted={nsh}")
print(f"(paper: 73,615 samples, 33 features, ER vs non-ER split; source 49,213 / target 24,402)")
print(f"{'method':12s} {'AUC matched':>14s} {'AUC mismatch':>14s} {'Rec matched':>13s} {'Rec mismatch':>13s}")
res = {}
for mth in methods:
    am = agg.agg_metric(rows, 'auc', mth, 'diabetes', 'matched')
    ax = agg.agg_metric(rows, 'auc', mth, 'diabetes', 'mismatch')
    rm = agg.agg_metric(rows, 'recall', mth, 'diabetes', 'matched')
    rx = agg.agg_metric(rows, 'recall', mth, 'diabetes', 'mismatch')
    res[mth] = dict(auc_matched=am[0], auc_mismatch=ax[0], rec_matched=rm[0], rec_mismatch=rx[0])
    print(f"{mth:12s} {am[0]:.3f}+/-{am[1]:.3f}  {ax[0]:.3f}+/-{ax[1]:.3f}  "
          f"{rm[0]:.3f}       {rx[0]:.3f}")
best = max(res['SGShift-K']['auc_matched'], res['SGShift-KA']['auc_matched'],
          res['SGShift-K']['auc_mismatch'], res['SGShift-KA']['auc_mismatch'])
verdict = 'SUPPORTED' if best > TARGET else 'NOT SUPPORTED'
print(f"best SGShift-K/KA diabetes AUC = {best:.3f} (target > {TARGET}); VERDICT: {verdict}")

json.dump(dict(claim='Diabetes readmission ER vs non-ER: SGShift detects concept shift',
               dataset=dict(name='Diabetes130US-readmission', source_nonER=ns, target_ER=nt,
                            total=ns + nt, features=nf, n_fit=nfit, n_shift=nsh,
                            paper_ref='73615 x 33, source 49213 / target 24402'),
               target_auc=TARGET, best_knockoff_auc=best, methods=res, verdict=verdict),
          open(os.path.join(HERE, 'results.json'), 'w'), indent=2)
print("wrote results.json")
