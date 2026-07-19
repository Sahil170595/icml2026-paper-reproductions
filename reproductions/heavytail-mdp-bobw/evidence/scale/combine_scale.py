#!/usr/bin/env python3
"""Merge the scaled-MDP stage caches into results_scale.json + factor_scaling.csv.

Inputs (written by flagship_run.py / sweep_run.py into _cache/):
  flag_{adv,stoch}_<alpha>.json          -- flagship H=4,S=6,A=3 (|S|=19)
  sweep_{S,H,A}_{adv,stoch}_<alpha>.json -- factor sweeps at alpha=1.5
"""
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '_cache')
AL = ['1.3', '1.5', '2.0']
AXES = ['S', 'H', 'A']
SWEEP_ALPHA = '1.5'

flag = {r: {a: json.load(open(os.path.join(CACHE, 'flag_%s_%s.json' % (r, a))))
            for a in AL} for r in ('adv', 'stoch')}
sweep = {ax: {r: json.load(open(os.path.join(CACHE, 'sweep_%s_%s_%s.json' % (ax, r, SWEEP_ALPHA))))
              for r in ('adv', 'stoch')} for ax in AXES}

f0 = flag['adv']['1.3']
print('=== FLAGSHIP MDP: H=%d layers, S=%d/layer, A=%d  (|S|=%d decision states) ==='
      % (f0['H'], f0['S'], f0['A'], f0['n_states_total']))
print('\n--- adversarial Reg_T ~ T^{1/alpha} (worst-case gap; %d seeds) ---' % f0['nseeds'])
print('%5s %8s %9s %7s %9s %7s %6s' % ('alpha', '1/alpha', 'OM_slope', 'R2', 'UOB_slope', 'R2', 'match'))
for a in AL:
    d = flag['adv'][a]
    print('%5s %8.3f %9.3f %7.3f %9.3f %7.3f %3s/%s'
          % (a, d['target'], d['slope_om'], d['r2_slope_om'],
             d['slope_uob'], d['r2_slope_uob'], d['match_om'], d['match_uob']))
print('\n--- stochastic (fixed gap %.2f, Tmax=%d, %d seeds; window T>=%d) ---'
      % (flag['stoch']['1.3']['gap'], flag['stoch']['1.3']['tmax'],
         flag['stoch']['1.3']['nseeds'], flag['stoch']['1.3']['asy_t0']))
print('%5s %7s %7s %7s %7s %9s %9s %6s %6s' % ('alpha', 'OMwin', 'R2log', 'UOBwin', 'R2log2',
                                               'OMfull', 'UOBfull', 'OMfall', 'UOBfall'))
for a in AL:
    d = flag['stoch'][a]
    print('%5s %7.3f %7.3f %7.3f %7.3f %9.3f %9.3f %6s %6s'
          % (a, d['slope_om_win'], d['r2_om']['r2_log'], d['slope_uob_win'],
             d['r2_uob']['r2_log2'], d['slope_om_full'], d['slope_uob_full'],
             d['om_falls'], d['uob_falls']))

print('\n=== FACTOR SCALING (alpha=%s): empirical exponent b in Reg@Tend ~ factor^b ===' % SWEEP_ALPHA)
rows = []
for ax in AXES:
    for reg in ('adv', 'stoch'):
        d = sweep[ax][reg]
        print(' %s-axis %-5s values=%-14s fixed=%s  b_OM=%.3f (R2=%.3f)  b_UOB=%.3f (R2=%.3f)'
              % (ax, reg, d['values'], d['fixed'], d['exponent_om'], d['r2_exponent_om'],
                 d['exponent_uob'], d['r2_exponent_uob']))
        for r in d['runs']:
            for alg in ('om', 'uob'):
                rate = r['slope_%s' % alg] if reg == 'adv' else r['slope_%s_win' % alg]
                rows.append(dict(axis=ax, regime=reg, alpha=d['alpha'], factor=r['factor_value'],
                                 H=r['H'], S=r['S'], A=r['A'], n_states_total=r['n_states_total'],
                                 alg=alg.upper(), t_end=d['t_end'],
                                 reg_at_t_end=round(r['reg_%s' % alg][-1], 2),
                                 t_rate_slope=round(rate, 4),
                                 exponent=round(d['exponent_%s' % alg], 4),
                                 r2_exponent=round(d['r2_exponent_%s' % alg], 4)))

print('\n--- T-rate stability across scales (adv slope should track 1/alpha=%.3f) ---'
      % (1.0 / float(SWEEP_ALPHA)))
for ax in AXES:
    d = sweep[ax]['adv']
    so = [r['slope_om'] for r in d['runs']]; su = [r['slope_uob'] for r in d['runs']]
    print(' %s-axis: OM slope range [%.3f, %.3f]  UOB slope range [%.3f, %.3f]'
          % (ax, min(so), max(so), min(su), max(su)))

with open(os.path.join(HERE, 'factor_scaling.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

out = dict(flagship=dict(scale=dict(H=f0['H'], S=f0['S'], A=f0['A'],
                                    n_states_total=f0['n_states_total']),
                         adversarial={a: flag['adv'][a] for a in AL},
                         stochastic={a: flag['stoch'][a] for a in AL}),
           factor_sweeps={ax: {r: sweep[ax][r] for r in ('adv', 'stoch')} for ax in AXES},
           sweep_alpha=float(SWEEP_ALPHA))
json.dump(out, open(os.path.join(HERE, 'results_scale.json'), 'w'), indent=1)
print('\n[wrote results_scale.json bytes=%d, factor_scaling.csv rows=%d]'
      % (len(open(os.path.join(HERE, 'results_scale.json')).read()), len(rows)))
