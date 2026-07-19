import sys, os, json, numpy as np
ds = sys.argv[1]
path = os.path.join(os.path.dirname(__file__), '..', '_cache', f'{ds}_rows.jsonl')
rows = [json.loads(l) for l in open(path) if l.strip()]
M = ['SGShift','SGShift-A','SGShift-K','SGShift-KA','Diff','WhyShift','SHAP']
for setting in ['matched','mismatched']:
    rs=[r for r in rows if r['setting']==setting]
    if not rs: continue
    print(f'-- {ds} {setting} (n={len(rs)}) AUC/recall --')
    for m in M:
        au=[r[m]['auc'] for r in rs]; rc=[r[m]['recall'] for r in rs]
        print(f'   {m:11s} AUC {np.mean(au):.3f}  recall {np.mean(rc):.3f}')
