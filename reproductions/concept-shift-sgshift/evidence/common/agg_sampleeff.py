"""Aggregate _cache/sampleeff.jsonl into a per-(dataset, n_target, method)
mean +- std AUC / FDR table for the Claim 2 sample-efficiency sweep."""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '..', '_cache')
IN = os.path.join(CACHE, 'sampleeff.jsonl')

METHODS = ['SGShift', 'SGShift-K', 'SGShift-KA', 'Diff', 'WhyShift', 'SHAP']
ORDER_N = {'diabetes': [100, 250, 500, 1000, 2500, 6000, 15000],
           'support2': [100, 250, 500, 1000, 2500, 4513]}

def load():
    rows = []
    with open(IN) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def agg(rows, ds, n_target, method):
    vals = [r['auc'][method] for r in rows
            if r['dataset'] == ds and r['n_target'] == n_target and method in r['auc']
            and r['auc'][method] == r['auc'][method]]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)

def main():
    rows = load()
    out = {}
    print(f"{'dataset':10s} {'n_target':9s} " + " ".join(f"{m:>16s}" for m in METHODS))
    for ds, ns in ORDER_N.items():
        out[ds] = {}
        for n in ns:
            cells = {}
            line = f"{ds:10s} {n:9d} "
            for m in METHODS:
                mean, std, cnt = agg(rows, ds, n, m)
                cells[m] = dict(mean=mean, std=std, n=cnt)
                s = f"{mean:.3f}+-{std:.3f}" if mean is not None else "  n/a  "
                line += f"{s:>16s} "
            print(line)
            out[ds][str(n)] = cells
    json.dump(out, open(os.path.join(CACHE, 'sampleeff_summary.json'), 'w'), indent=2)
    print("wrote _cache/sampleeff_summary.json")

if __name__ == '__main__':
    main()
