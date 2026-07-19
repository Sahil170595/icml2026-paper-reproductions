#!/usr/bin/env python3
"""Factor sweeps for the poly(H,S,A) dependence of Theorems 4.1 / 5.1.

Sweeps ONE factor at a time (the other two fixed), runs BOTH algorithms
(HT-FTRL-OM known-P, HT-FTRL-UOB unknown-P) at every scale, and fits the
empirical polynomial exponent  Reg@Tend ~ factor^b  (log-log least squares,
with R^2).  The state-dependent base-loss range is NORMALIZED to [0, 0.75]
for every config (scale_lib._c_state), so the sweep changes ONLY the
combinatorial size -- never the loss scale.

Axes:  S in {3,6,12,24} at H=3,A=3   (|S| = 7,13,25,49)
       H in {2,3,4}     at S=6,A=3   (layers; |S| = 7,13,19)
       A in {3,6,9}     at H=3,S=6

Usage: python3 sweep_run.py {S|H|A} {adv|stoch} <alpha>
  -> _cache/sweep_{axis}_{regime}_<alpha>.json
"""
import os, sys, time
import scale_lib as L

AXES = {
    'S': dict(values=[3, 6, 12, 24], fixed=dict(H=3, A=3)),
    'H': dict(values=[2, 3, 4],      fixed=dict(S=6, A=3)),
    'A': dict(values=[3, 6, 9],      fixed=dict(H=3, S=6)),
}
ADV_H = [int(x) for x in os.environ.get('ADV_H', '1000,2000,4000,8000,16000,32000').split(',')]
ST_CK = [int(x) for x in os.environ.get('ST_CK', '1000,2000,4000,8000,16000,24000,32000,48000').split(',')]
ST_T0 = int(os.environ.get('ST_T0', '16000'))
NADV = int(os.environ.get('NADV', '6')); NST = int(os.environ.get('NST', '8'))


def main():
    axis = sys.argv[1]; regime = sys.argv[2]; alpha = float(sys.argv[3])
    ax = AXES[axis]; t0 = time.time()
    runs = []
    for v in ax['values']:
        cfg = dict(ax['fixed']); cfg[axis] = v
        tc = time.time()
        if regime == 'adv':
            ev = L.run_adv(alpha, cfg['H'], cfg['S'], cfg['A'], ADV_H,
                           nseeds=NADV, normalize_c=True)
        else:
            ev = L.run_stoch(alpha, cfg['H'], cfg['S'], cfg['A'], ST_CK[-1],
                             ST_CK, ST_T0, nseeds=NST, normalize_c=True)
        ev['factor_axis'] = axis; ev['factor_value'] = v
        ev['config_runtime_s'] = round(time.time() - tc, 1)
        runs.append(ev)
        if regime == 'adv':
            print('  [%s=%2d H=%d S=%d A=%d |S|=%2d] adv a=%s: OM %.1f @T=%d slope=%.3f | UOB %.1f slope=%.3f (%.1fs)'
                  % (axis, v, cfg['H'], cfg['S'], cfg['A'], ev['n_states_total'], alpha,
                     ev['reg_om'][-1], ADV_H[-1], ev['slope_om'],
                     ev['reg_uob'][-1], ev['slope_uob'], ev['config_runtime_s']))
        else:
            print('  [%s=%2d H=%d S=%d A=%d |S|=%2d] stoch a=%s: OM %.1f @T=%d win=%.3f | UOB %.1f win=%.3f (%.1fs)'
                  % (axis, v, cfg['H'], cfg['S'], cfg['A'], ev['n_states_total'], alpha,
                     ev['reg_om'][-1], ST_CK[-1], ev['slope_om_win'],
                     ev['reg_uob'][-1], ev['slope_uob_win'], ev['config_runtime_s']))
    vals = ax['values']
    b_om, r2_om = L.factor_fit(vals, [r['reg_om'][-1] for r in runs])
    b_uob, r2_uob = L.factor_fit(vals, [r['reg_uob'][-1] for r in runs])
    ev = dict(axis=axis, regime=regime, alpha=alpha, values=vals,
              fixed=ax['fixed'], runs=runs,
              reg_end_om=[r['reg_om'][-1] for r in runs],
              reg_end_uob=[r['reg_uob'][-1] for r in runs],
              exponent_om=b_om, r2_exponent_om=r2_om,
              exponent_uob=b_uob, r2_exponent_uob=r2_uob,
              t_end=ADV_H[-1] if regime == 'adv' else ST_CK[-1])
    L.finish(ev, t0, 'sweep_%s_%s_%s' % (axis, regime, str(alpha)))
    print(' SWEEP %s %s a=%s: Reg ~ %s^b  b_OM=%.3f (R2=%.3f)  b_UOB=%.3f (R2=%.3f)'
          % (axis, regime, alpha, axis, b_om, r2_om, b_uob, r2_uob))


if __name__ == '__main__':
    main()
