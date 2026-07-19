import sys, os, json, numpy as np
ds=sys.argv[1]; path=os.path.join(os.path.dirname(__file__),'..','_cache',f'{ds}_final.jsonl')
rows=[json.loads(l) for l in open(path) if l.strip()]
M=['SGShift','SGShift-A','SGShift-K','SGShift-KA','Diff','WhyShift','SHAP']
res={}
for setting in ['matched','mismatched']:
    rs=[r for r in rows if r['setting']==setting]
    if not rs: continue
    res[setting]={'n':len(rs)}
    print(f'-- {ds} {setting} (n={len(rs)}) --')
    for m in M:
        a=float(np.mean([r[m]['auc'] for r in rs])); rc=float(np.mean([r[m]['recall'] for r in rs]))
        res[setting][m]={'auc':round(a,3),'recall':round(rc,3)}
        print(f'   {m:11s} AUC {a:.3f}  recall {rc:.3f}')
    for m in ['SGShift-K','SGShift-KA']:
        for q in ['0.1','0.2','0.3']:
            fd=float(np.mean([r[m][f'fdp_q{q}'] for r in rs])); pw=float(np.mean([r[m][f'pow_q{q}'] for r in rs]))
            res[setting][m][f'fdp_q{q}']=round(fd,3); res[setting][m][f'pow_q{q}']=round(pw,3)
open(os.path.join(os.path.dirname(__file__),'..','_cache',f'{ds}_final_summary.json'),'w').write(json.dumps(res,indent=1))
