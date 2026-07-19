#!/usr/bin/env python3
"""
Independent NumPy reproduction of the STOCHASTIC half of Theorem 4.1 (HT-FTRL-OM,
known transitions) of
  "Best-of-Both-Worlds for Heavy-Tailed Markov Decision Processes"
  OpenReview j6gXeiPJ3z / arXiv 2602.01295 (ICML 2026).

Theorem 4.1 gives BoBW guarantees for HT-FTRL-OM:
  * adversarial regime:  Reg_T = O~(sigma * T^{1/alpha})   (reproduced in
        artifacts/repro.py: slopes 0.752/0.649/0.501 vs 1/alpha 0.769/0.667/0.500);
  * stochastic / self-bounding regime:
        Reg_T = O(sigma^{alpha/(alpha-1)} * w_Delta(alpha) * log T) = O(log T).
THIS script reproduces the stochastic half: the SAME algorithm on a FIXED-gap
i.i.d. stochastic instance must have regret growing like log(T), NOT like the
adversarial T^{1/alpha} and NOT like sqrt(T).  Together the two halves are BoBW.

Reduction (identical to the adversarial script): H=1,S=1,A=K bandit.  HT-FTRL-OM =
FTRL with (1/alpha)-Tsallis regularizer + skipping importance-weighted estimator:
    eta_t=beta/(sigma t^{1/a}); tau_t=C sigma t^{1/a} x^{1/a};
    b=C^{1-a} sigma t^{1/a-1} x^{1/a-1}.
The ONLY change vs adversarial: STOCHASTIC fixed gap Delta (i.i.d.) instead of the
shrinking worst-case gap.  Instance fixed across horizons -> ONE trajectory to T_max,
read cumulative pseudo-regret at checkpoints.  Real executed sim; no hand numbers.
"""
import os, sys, json, time
import numpy as np

K      = 4
BASE   = 0.5
ALPHAS = [1.3, 1.5, 2.0]
BETA   = 1.0
CC     = 1.0
SIGMA  = 1.0
MCAP   = 1.0e6
DELTA  = float(os.environ.get('DELTA', '1.75'))
N_SEED = int(os.environ.get('N_SEED', '48'))
N_BIS  = int(os.environ.get('N_BIS', '8'))
TMAX   = int(os.environ.get('TMAX', '96000'))
CKPTS  = [int(x) for x in os.environ.get(
    'CKPTS', '1000,2000,4000,8000,16000,32000,64000,96000').split(',')]

def solve_ftrl(w, pexp, c, n_iter):
    wmin = w.min(axis=1, keepdims=True)
    d    = w - wmin
    logc = np.log(c)
    lo   = logc.copy()
    hi   = logc + (1.0/pexp)*np.log(K)
    for _ in range(n_iter):
        mid = 0.5*(lo+hi)
        x   = np.exp(pexp*(logc - np.log(d+np.exp(mid))))
        too_small = x.sum(axis=1, keepdims=True) > 1.0
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
    x = np.exp(pexp*(logc - np.log(d+np.exp(0.5*(lo+hi)))))
    return x / x.sum(axis=1, keepdims=True)

def make_noise(rng, alpha_row, s_row, M):
    U   = rng.random(M)
    mag = U**(-1.0/alpha_row)
    np.minimum(mag, MCAP, out=mag)
    sgn = np.where(rng.random(M) < 0.5, -1.0, 1.0)
    return sgn * mag / s_row

def run_stochastic(alpha_row, delta, seed, tmax, ckpts, n_bis):
    rng   = np.random.default_rng(seed)
    M     = alpha_row.size
    q     = 1.0/alpha_row
    pexp  = (1.0/(1.0-q))[:, None]
    c     = (q/(1.0-q))[:, None]
    inv_a = 1.0/alpha_row
    s_row = (alpha_row*np.log(MCAP) + 1.0)**(1.0/alpha_row)
    Cpow  = CC**(1.0-alpha_row)
    ridx  = np.arange(M)
    Lhat  = np.zeros((M, K))
    cumreg= np.zeros(M)
    out   = np.zeros((M, len(ckpts)))
    ck    = 0
    ckset = set(ckpts)
    for t in range(1, tmax+1):
        eta = (BETA/(SIGMA * t**inv_a))[:, None]
        x   = solve_ftrl(eta*Lhat, pexp, c, n_bis)
        cumreg += delta*(1.0 - x[:, 0])
        cdf  = np.cumsum(x, axis=1)
        a_t  = np.minimum((cdf < rng.random((M, 1))).sum(axis=1), K-1)
        x_at = x[ridx, a_t]
        l    = BASE + (a_t != 0).astype(float)*delta + make_noise(rng, alpha_row, s_row, M)
        tau  = CC*SIGMA*(t**inv_a)*(x_at**inv_a)
        l_sk = np.where(np.abs(l) <= tau, l, 0.0)
        iw   = l_sk / x_at
        bonus= (Cpow*SIGMA*(t**(inv_a-1.0)))[:, None]*(x**(inv_a-1.0)[:, None])
        lhat = -bonus
        lhat[ridx, a_t] += iw
        Lhat += lhat
        if t in ckset:
            out[:, ck] = cumreg; ck += 1
    return out

def r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss_res = np.sum((y-yhat)**2); ss_tot = np.sum((y-np.mean(y))**2)
    return float(1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0

def fit_models(Ts, reg):
    Ts = np.asarray(Ts, float); reg = np.asarray(reg, float)
    lnT = np.log(Ts); sqT = np.sqrt(Ts)
    Alog = np.vstack([np.ones_like(lnT), lnT]).T
    cl, *_ = np.linalg.lstsq(Alog, reg, rcond=None); r2_log = r2(reg, Alog@cl)
    Asq = np.vstack([np.ones_like(sqT), sqT]).T
    cs, *_ = np.linalg.lstsq(Asq, reg, rcond=None); r2_sq = r2(reg, Asq@cs)
    Alin = np.vstack([np.ones_like(Ts), Ts]).T
    cli, *_ = np.linalg.lstsq(Alin, reg, rcond=None); r2_lin = r2(reg, Alin@cli)
    m = reg > 0
    p = float(np.polyfit(lnT[m], np.log(reg[m]), 1)[0]) if m.sum() >= 2 else float('nan')
    return dict(r2_log=r2_log, r2_sqrt=r2_sq, r2_linT=r2_lin, loglog_slope=p,
                b_log=float(cl[1]), a_log=float(cl[0]))

def main():
    t0 = time.time()
    print("== heavy-tailed noise check (E[|noise|^a]<=sigma^a=1) ==  DELTA={} (fixed), K={}".format(DELTA, K))
    rng = np.random.default_rng(7); NS = 400000
    noise_moment = {}
    for a in ALPHAS:
        s  = (a*np.log(MCAP)+1.0)**(1.0/a)
        nz = make_noise(rng, np.full(NS, a), s, NS)
        mm = float(np.mean(np.abs(nz)**a)); noise_moment[str(a)] = mm
        print("  alpha={:>4}:  E[|noise|^a]={:6.3f}   var={:11.1f}".format(a, mm, float(np.var(nz))))

    A = np.array(ALPHAS)
    alpha_row = np.repeat(A, N_SEED)
    print("\n== STOCHASTIC regime (fixed gap Delta={}, i.i.d.): single run to T={}, {} seeds/alpha, bisect={} ==".format(DELTA, TMAX, N_SEED, N_BIS))
    print("   checkpoints T = {}".format(CKPTS))
    out = run_stochastic(alpha_row, DELTA, seed=20260717, tmax=TMAX, ckpts=CKPTS, n_bis=N_BIS)
    reg = {}
    for ai, a in enumerate(ALPHAS):
        reg[a] = out[ai*N_SEED:(ai+1)*N_SEED].mean(axis=0)

    print("\n== Cumulative pseudo-regret Reg_T (mean over seeds) ==")
    print("   {:>6}".format("alpha") + "".join("{:>9}".format(T) for T in CKPTS))
    for a in ALPHAS:
        print("   {:>6}".format(a) + "".join("{:9.2f}".format(v) for v in reg[a]))

    ratio_log  = float(np.log(CKPTS[-1])/np.log(CKPTS[0]))
    ratio_sqrt = float(np.sqrt(CKPTS[-1]/CKPTS[0]))
    ratio_pow  = float((CKPTS[-1]/CKPTS[0])**(1.0/1.5))
    print("\n== growth over T range {}->{} ({}x): log-ratio={:.2f}, sqrt-ratio={:.2f}, T^(1/1.5)-ratio={:.2f} ==".format(
        CKPTS[0], CKPTS[-1], CKPTS[-1]//CKPTS[0], ratio_log, ratio_sqrt, ratio_pow))

    print("\n== FIT: measured Reg_T vs models (target O(log T)) ==")
    print("   {:>6} {:>9} {:>9} {:>10} {:>7} {:>9} {:>10}".format(
        "alpha", "measured", "R2(logT)", "R2(sqrtT)", "R2(T)", "loglog_p", "logT PASS?"))
    n_pass = 0; detail = {}; stoch_slope = {}
    for a in ALPHAS:
        f = fit_models(CKPTS, reg[a])
        rs = reg[a]/np.sqrt(CKPTS); rl = reg[a]/np.log(CKPTS)
        cv_over_logT = float(np.std(rl)/np.mean(rl))
        sqrt_decay = float(rs[-1]/rs[0])
        sqrt_tail_dec = bool(rs[-1] < rs[len(rs)//2])
        growth = float(reg[a][-1]/reg[a][0])
        stoch_slope[str(a)] = f['loglog_slope']
        ok = (f['r2_log'] >= 0.97) and (f['r2_log'] >= f['r2_sqrt']) and \
             (f['loglog_slope'] < 0.5) and sqrt_tail_dec
        n_pass += int(ok)
        detail[str(a)] = dict(measured=[float(v) for v in reg[a]],
                              cv_reg_over_logT=cv_over_logT, sqrt_decay_ratio=sqrt_decay,
                              sqrt_tail_decreasing=sqrt_tail_dec, growth_ratio=growth,
                              pass_=bool(ok), **f)
        print("   {:>6} {:>9.2f} {:>9.4f} {:>10.4f} {:>7.3f} {:>9.3f} {:>10}".format(
            a, float(reg[a][-1]), f['r2_log'], f['r2_sqrt'], f['r2_linT'],
            f['loglog_slope'], "yes" if ok else "no"))

    print("\n== Normalized regret: Reg_T/ln(T) (should be ~flat) vs Reg_T/sqrt(T) (should fall) ==")
    for a in ALPHAS:
        rl = reg[a]/np.log(CKPTS); rs = reg[a]/np.sqrt(CKPTS)
        print("   alpha={}:  Reg/lnT  = [".format(a) + ", ".join("{:.2f}".format(v) for v in rl) + "]")
        print("             Reg/sqrtT = [" + ", ".join("{:.3f}".format(v) for v in rs) + "]")

    verdict = n_pass >= 2
    print("\n== RULE: O(log T) signature (R2(a+b logT)>=0.97 AND log-fit>=sqrt-fit AND loglog slope<0.5 AND Reg/sqrt(T) decreasing) for >=2 of 3 alphas ==")
    print("   passes={}/{}".format(n_pass, len(ALPHAS)))

    adv = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        advp = os.path.normpath(os.path.join(here, "..", "..", "..", "..", "artifacts", "evidence.json"))
        aj = json.load(open(advp))
        adv = {}
        for a in ALPHAS:
            adv[str(a)] = float(aj["fitted_slope_p"][str(a)])
        print("\n== BoBW contrast (SAME algorithm), log-log slope of Reg_T vs T ==")
        print("   {:>6} {:>16} {:>17}".format("alpha", "adv slope (~1/a)", "stoch slope (~0)"))
        for a in ALPHAS:
            print("   {:>6} {:>16.3f} {:>17.3f}".format(a, adv[str(a)], stoch_slope[str(a)]))
    except Exception as e:
        print("[adversarial reference not loaded: {}]".format(e))

    print("\nRESULT: {}".format("PASS" if verdict else "FAIL"))
    print("[runtime {:.1f}s]  numpy={}".format(time.time()-t0, np.__version__))

    ev = dict(
        orid="j6gXeiPJ3z",
        claim="Theorem 4.1 stochastic half: HT-FTRL-OM Reg_T=O(log T) on fixed-gap i.i.d. instance",
        regime="stochastic/self-bounding (fixed gap, i.i.d.)",
        reduction="H=1,S=1,A=K bandit; 1/alpha-Tsallis FTRL + skipping estimator",
        K=K, base=BASE, delta=DELTA, beta=BETA, C=CC, sigma=SIGMA, mcap=MCAP,
        n_seed=N_SEED, n_bisect=N_BIS, tmax=TMAX, checkpoints=CKPTS, alphas=ALPHAS,
        noise_moment_empirical=noise_moment,
        ratio_log=ratio_log, ratio_sqrt=ratio_sqrt,
        regret_curves={str(a): [float(v) for v in reg[a]] for a in ALPHAS},
        fits=detail, n_pass=n_pass, verdict_pass=bool(verdict),
        adversarial_slope_recorded=adv, stochastic_loglog_slope=stoch_slope,
        numpy=np.__version__)
    if os.environ.get('WRITE_JSON', '1') == '1':
        json.dump(ev, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=2)
        print("[wrote results.json]")
    print("EVIDENCE_BEGIN"); print(json.dumps(ev)); print("EVIDENCE_END")

if __name__ == "__main__":
    main()
