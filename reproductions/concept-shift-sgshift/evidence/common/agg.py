"""Aggregation helpers over _cache/runs.jsonl produced by run_experiments.py."""
import os, json, math

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_cache')

def load():
    path = os.path.join(CACHE, 'runs.jsonl')
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _stats(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return float('nan'), float('nan'), 0
    m = sum(vals) / n
    if n > 1:
        var = sum((v - m) ** 2 for v in vals) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = 0.0
    return m, se, n

def agg_metric(rows, group, method, dataset=None, config=None):
    """group in {'auc','recall'}; method is a key; optional dataset/config filter."""
    sel = [r for r in rows
           if (dataset is None or r['dataset'] == dataset)
           and (config is None or r['config'] == config)]
    return _stats([r[group][method] for r in sel])

def agg_ko(rows, field, dataset=None, config=None):
    sel = [r for r in rows
           if (dataset is None or r['dataset'] == dataset)
           and (config is None or r['config'] == config)]
    return _stats([r['ko'][field] for r in sel])

def sizes(rows, dataset):
    r = [x for x in rows if x['dataset'] == dataset][0]
    return r['n_src_full'], r['n_tgt_full'], r['n_feat'], r['n_fit'], r['n_shift']

def fmt(m, se):
    return f"{m:.3f} +/- {se:.3f}"
