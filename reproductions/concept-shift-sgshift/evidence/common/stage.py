import sys, os, json, numpy as np, dprep, sim
ds = sys.argv[1]; seed0 = int(sys.argv[2]); nseed = int(sys.argv[3])
scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
Zc, dom, y, task = dprep.load(ds)
iS = np.where(dom == 0)[0]; iT = np.where(dom == 1)[0]
path = os.path.join(os.path.dirname(__file__), '..', '_cache', f'{ds}_rows.jsonl')
n = 0
with open(path, 'a') as fh:
    for seed in range(seed0, seed0 + nseed):
        for setting in ['matched', 'mismatched']:
            r = sim.run_replicate(Zc, iS, iT, y, task, seed=seed, setting=setting,
                                  a_shift=6, shift_scale=scale, B=3, n_lam=8)
            fh.write(json.dumps(r) + '\n'); fh.flush(); n += 1
print(f'{ds}: appended {n} rows to {os.path.basename(path)}; scale={scale}')
