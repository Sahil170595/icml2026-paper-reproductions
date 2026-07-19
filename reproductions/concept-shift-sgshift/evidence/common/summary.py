import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import agg
rows = agg.load()
print("total runs:", len(rows))
methods = ['SGShift', 'SGShift-A', 'SGShift-K', 'SGShift-KA', 'Diff', 'WhyShift', 'SHAP']
for ds in ['diabetes', 'support2', 'adult']:
    ns, nt, nf, nfit, nsh = agg.sizes(rows, ds)
    print(f"\n=== {ds}  src={ns} tgt={nt} feats={nf} n_fit={nfit} n_shift={nsh} ===")
    for cfg in ['matched', 'mismatch']:
        print(f" [{cfg}] AUC:", " ".join(f"{m}={agg.agg_metric(rows,'auc',m,ds,cfg)[0]:.3f}" for m in methods))
        print(f" [{cfg}] REC:", " ".join(f"{m}={agg.agg_metric(rows,'recall',m,ds,cfg)[0]:.3f}" for m in methods))
    for q in (0.1, 0.2):
        fs = agg.agg_ko(rows, f'fdp_single_{q}', ds)[0]; ps = agg.agg_ko(rows, f'pow_single_{q}', ds)[0]
        fd = agg.agg_ko(rows, f'fdp_derand_{q}', ds)[0]; pd = agg.agg_ko(rows, f'pow_derand_{q}', ds)[0]
        print(f"  KO q={q}: single FDP={fs:.3f} pow={ps:.3f} | derand FDP={fd:.3f} pow={pd:.3f}")
# overall KA auc
print("\nSGShift-KA AUC by dataset (matched+mismatch):",
      " ".join(f"{ds}={agg.agg_metric(rows,'auc','SGShift-KA',ds)[0]:.3f}+/-{agg.agg_metric(rows,'auc','SGShift-KA',ds)[1]:.3f}"
               for ds in ['diabetes','support2','adult']))
