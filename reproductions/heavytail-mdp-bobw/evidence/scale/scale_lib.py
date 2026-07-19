#!/usr/bin/env python3
"""scale_lib.py -- shared driver library for the SCALED reproduction of
"Best-of-Both-Worlds for Heavy-Tailed Markov Decision Processes"
(OpenReview j6gXeiPJ3z / arXiv 2602.01295).

Reuses the genuine layered-episodic-MDP engine `mdp_core.py` (Q-value backup
through the transition kernel, 1/alpha-Tsallis FTRL over occupancy measures,
skipping estimator, Upper Occupancy Bound for unknown P) from
`../mdp/mdp_core.py` -- the SAME engine that produced the base H=3/|S|=7/A=3
results -- and drives it at LARGER scales:

  * flagship_run.py : H=4 layers, S=6 states/layer (|S|=19 decision states),
                      A=3 -- both algorithms, both BoBW regimes.
  * sweep_run.py    : factor sweeps S in {3,6,12,24}, H in {2,3,4},
                      A in {3,6,9} to measure the empirical poly(H,S,A)
                      growth of the regret prefactor.

Seeding is identical to the base runs (adversarial seed=4000+T per horizon,
stochastic seed=20260717), deterministic `numpy.random.default_rng`.
"""
import os, sys, json, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'mdp'))
import mdp_core as m  # noqa: E402

NBIS = int(os.environ.get('NBIS', '5'))   # same bisection depth as base runs


# ---------------------------------------------------------------- fits ------
def r2(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    sr = np.sum((y - yh) ** 2); st = np.sum((y - y.mean()) ** 2)
    return float(1.0 - sr / st) if st > 0 else 0.0


def slope(T, reg):
    """log-log slope of reg vs T (least squares), with its R^2."""
    T = np.asarray(T, float); reg = np.asarray(reg, float); mk = reg > 0
    if mk.sum() < 2:
        return float('nan'), float('nan')
    co = np.polyfit(np.log(T[mk]), np.log(reg[mk]), 1)
    yh = np.polyval(co, np.log(T[mk]))
    return float(co[0]), r2(np.log(reg[mk]), yh)


def fits(T, reg):
    """R^2 of c0+c1*logT, c0+c1*logT+c2*log^2 T, c0+c1*sqrt(T) models."""
    T = np.asarray(T, float); reg = np.asarray(reg, float)
    lnT = np.log(T); sq = np.sqrt(T)
    Al = np.vstack([np.ones_like(lnT), lnT]).T
    A2 = np.vstack([np.ones_like(lnT), lnT, lnT ** 2]).T
    As = np.vstack([np.ones_like(sq), sq]).T
    cl, *_ = np.linalg.lstsq(Al, reg, rcond=None)
    c2, *_ = np.linalg.lstsq(A2, reg, rcond=None)
    cs, *_ = np.linalg.lstsq(As, reg, rcond=None)
    return dict(r2_log=r2(reg, Al @ cl), r2_log2=r2(reg, A2 @ c2),
                r2_sqrt=r2(reg, As @ cs))


def peak_fall(T, reg):
    T = np.asarray(T, float); reg = np.asarray(reg, float)
    rs = reg / np.sqrt(T); pk = int(np.argmax(rs))
    falls = bool(pk <= len(rs) - 2 and rs[-1] < rs[pk] and rs[-1] < rs[-2])
    return falls, pk, [float(v) for v in rs]


# ------------------------------------------------------------- drivers -----
def _c_state(S, normalize):
    """State-cost coefficient.  normalize=True keeps the state-dependent base
    loss range fixed at [0, 0.75] for EVERY S, so factor sweeps change ONLY the
    combinatorial size (H,S,A), never the loss scale.  normalize=False = the
    base runs' default 0.25 (loss range grows with S)."""
    if not normalize:
        return 0.25
    return 0.75 / max(S - 1, 1)


def run_adv(alpha, H, S, A, horizons, nseeds=6, C0=2.0, normalize_c=False):
    """Adversarial regime: per-horizon worst-case gap G=C0*T^{1/alpha-1};
    HT-FTRL-OM (known P) + HT-FTRL-UOB (unknown P) in one vectorized batch."""
    ns = nseeds
    is_uob = np.r_[np.zeros(ns, bool), np.ones(ns, bool)]
    is_skip = np.ones(2 * ns, bool)
    ro, ru = [], []
    for T in horizons:
        G = C0 * (float(T) ** (1.0 / alpha - 1.0))
        P, lb, V, adv, nS = m.build_mdp(H, S, A, G, c_state=_c_state(S, normalize_c))
        o, _ = m.run_episodes(alpha, P, lb, adv, nS, is_uob, is_skip, G,
                              seed=4000 + T, tmax=T, ckpts=[T], nbis=NBIS)
        ro.append(float(o[:ns, 0].mean())); ru.append(float(o[ns:, 0].mean()))
    s_om, r2_om = slope(horizons, ro); s_uob, r2_uob = slope(horizons, ru)
    tgt = 1.0 / alpha
    return dict(regime='adv', alpha=alpha, H=H, S=S, A=A, C0=C0,
                n_states_total=1 + S * (H - 1), nseeds=ns,
                c_state=_c_state(S, normalize_c), horizons=list(horizons),
                reg_om=ro, reg_uob=ru,
                slope_om=s_om, r2_slope_om=r2_om,
                slope_uob=s_uob, r2_slope_uob=r2_uob, target=tgt,
                match_om=bool(abs(s_om - tgt) <= 0.12),
                match_uob=bool(abs(s_uob - tgt) <= 0.15),
                uob_minus_om=[ru[i] - ro[i] for i in range(len(horizons))])


def run_stoch(alpha, H, S, A, tmax, ckpts, asy_t0, nseeds=8, gap=0.9,
              normalize_c=False, seed=20260717):
    """Stochastic regime: fixed gap, one trajectory to tmax;
    HT-FTRL-OM (known P) + HT-FTRL-UOB (unknown P), nseeds each."""
    ns = nseeds
    is_uob = np.r_[np.zeros(ns, bool), np.ones(ns, bool)]
    is_skip = np.ones(2 * ns, bool)
    P, lb, V, adv, nS = m.build_mdp(H, S, A, gap, c_state=_c_state(S, normalize_c))
    o, _ = m.run_episodes(alpha, P, lb, adv, nS, is_uob, is_skip, gap,
                          seed=seed, tmax=tmax, ckpts=ckpts, nbis=NBIS)
    sd_om = o[:ns]; sd_uob = o[ns:]
    reg_om = sd_om.mean(0); reg_uob = sd_uob.mean(0)
    widx = [i for i, T in enumerate(ckpts) if T >= asy_t0]
    wck = [ckpts[i] for i in widx]
    s_om_w, r2w_om = slope(wck, reg_om[widx]); s_uo_w, r2w_uo = slope(wck, reg_uob[widx])
    s_om_f, _ = slope(ckpts, reg_om); s_uo_f, _ = slope(ckpts, reg_uob)
    pf_om, _, rs_om = peak_fall(ckpts, reg_om); pf_uo, _, rs_uo = peak_fall(ckpts, reg_uob)
    return dict(regime='stoch', alpha=alpha, H=H, S=S, A=A, gap=gap,
                n_states_total=1 + S * (H - 1), nseeds=ns,
                c_state=_c_state(S, normalize_c), tmax=tmax,
                ckpts=list(ckpts), asy_t0=asy_t0,
                reg_om=[float(v) for v in reg_om],
                reg_uob=[float(v) for v in reg_uob],
                std_om=float(sd_om[:, -1].std()), std_uob=float(sd_uob[:, -1].std()),
                slope_om_win=s_om_w, r2_slope_om_win=r2w_om,
                slope_uob_win=s_uo_w, r2_slope_uob_win=r2w_uo,
                slope_om_full=s_om_f, slope_uob_full=s_uo_f,
                r2_om=fits(ckpts, reg_om), r2_uob=fits(ckpts, reg_uob),
                om_falls=bool(pf_om), uob_falls=bool(pf_uo),
                regsqrt_om=rs_om, regsqrt_uob=rs_uo)


def factor_fit(factors, reg_ends):
    """Empirical polynomial exponent: fit log(Reg@Tend) = a + b*log(factor)."""
    f = np.asarray(factors, float); rg = np.asarray(reg_ends, float)
    co = np.polyfit(np.log(f), np.log(rg), 1)
    yh = np.polyval(co, np.log(f))
    return float(co[0]), r2(np.log(rg), yh)


def finish(ev, t0, tag):
    ev['runtime_s'] = round(time.time() - t0, 1)
    ev['numpy'] = np.__version__
    ev['python'] = sys.version.split()[0]
    os.makedirs(os.path.join(_HERE, '_cache'), exist_ok=True)
    fn = os.path.join(_HERE, '_cache', tag + '.json')
    json.dump(ev, open(fn, 'w'), indent=1)
    print('WROTE', os.path.relpath(fn, _HERE), 'runtime', ev['runtime_s'], 's')
    return fn
