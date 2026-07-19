#!/usr/bin/env python3
"""FLAGSHIP scaled MDP: H=4 decision layers, S=6 states/layer (|S|=19 decision
states), A=3 actions -- the full claims (Thm 4.1 HT-FTRL-OM known-P, Thm 5.1
HT-FTRL-UOB unknown-P) in both BoBW regimes at a scale that is unambiguously a
multi-state MDP (2.7x the states of the base H=3/|S|=7 run, one more layer).

Usage: python3 flagship_run.py {adv|stoch} <alpha>
  -> _cache/flag_{adv|stoch}_<alpha>.json
Env overrides: H,S,A,C0,GST,NADV,NST,TMAX,ASY_T0,NBIS.
"""
import os, sys, time
import numpy as np
import scale_lib as L

H = int(os.environ.get('H', '4')); S = int(os.environ.get('S', '6')); A = int(os.environ.get('A', '3'))
C0 = float(os.environ.get('C0', '2.0')); GST = float(os.environ.get('GST', '0.9'))
NADV = int(os.environ.get('NADV', '6')); NST = int(os.environ.get('NST', '8'))
TMAX = int(os.environ.get('TMAX', '100000'))
ADV_H = [int(x) for x in os.environ.get('ADV_H', '1000,2000,4000,8000,16000,32000').split(',')]
CKPTS = [int(x) for x in os.environ.get('CKPTS', '1000,2000,4000,8000,16000,24000,32000,48000,64000,80000,100000').split(',')]
ASY_T0 = int(os.environ.get('ASY_T0', '32000'))


def main():
    regime = sys.argv[1]; alpha = float(sys.argv[2]); t0 = time.time()
    mom, var = L.m.empirical_moment(alpha)
    if regime == 'adv':
        ev = L.run_adv(alpha, H, S, A, ADV_H, nseeds=NADV, C0=C0)
    else:
        ev = L.run_stoch(alpha, H, S, A, TMAX, CKPTS, ASY_T0, nseeds=NST, gap=GST)
    ev['noise_moment'] = mom; ev['noise_var'] = var
    L.finish(ev, t0, 'flag_%s_%s' % (regime, str(alpha)))
    if regime == 'adv':
        print(' FLAGSHIP H=%d S=%d A=%d |S|=%d adv a=%s: OM/UOB slope=%.3f/%.3f tgt=%.3f match=%s/%s'
              % (H, S, A, ev['n_states_total'], alpha, ev['slope_om'], ev['slope_uob'],
                 ev['target'], ev['match_om'], ev['match_uob']))
    else:
        print(' FLAGSHIP H=%d S=%d A=%d |S|=%d stoch a=%s: OMwin=%.3f UOBwin=%.3f om_falls=%s uob_falls=%s R2log2(uob)=%.4f'
              % (H, S, A, ev['n_states_total'], alpha, ev['slope_om_win'], ev['slope_uob_win'],
                 ev['om_falls'], ev['uob_falls'], ev['r2_uob']['r2_log2']))


if __name__ == '__main__':
    main()
