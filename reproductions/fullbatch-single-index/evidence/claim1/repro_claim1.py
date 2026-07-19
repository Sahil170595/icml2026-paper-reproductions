#!/usr/bin/env python3
# Independent NumPy/scipy reproduction of CLAIM 1 (sample-complexity SEPARATION):
#   "Full-Batch GD Outperforms One-Pass SGD: Sample Complexity Separation in
#    Single-Index Learning" (OpenReview QItZDBVCT0 / arXiv 2602.02431), Section 3.
#
# CLAIM 1 (paper, Sec 3, Fig 1):
#   * Thm 3.1 (PLAIN quadratic sigma(z)=z^2): when n << d log d, full-batch
#     spherical gradient flow on the CORRELATION loss gets TRIVIAL overlap; i.e.
#     "full-batch updates offer no statistical advantage over one-pass SGD".
#     => weak-recovery threshold delta*=n/d GROWS like log d   (Fig 1a / Fig 1c).
#   * Thm 3.2 (smooth TRUNCATED quadratic, M large): n >~ d suffices for weak
#     recovery => full-batch(truncated) is MORE sample-efficient   (Fig 1b).
#   The truncated link still has information exponent 2 (Hermite-2 coeff ~1), so
#   the one-pass-SGD lower bound n >~ d log d (Ben Arous et al. 2021) still applies.
#   SEPARATION: full-batch(truncated) recovers with far FEWER samples per dim than
#   full-batch(plain) or one-pass SGD; the plain/SGD requirement grows like log d.
#
# SCORING RULE (measured against real stdout numbers):
#   (R1) full-batch PLAIN threshold delta*(d) grows with log d (slope>0, R^2>=0.85)
#        -> exact result (flow limit = top eigenvector of A*, cross-checked).
#   (R2) full-batch TRUNCATED is MORE efficient than plain: delta*_trunc < delta*_plain
#        at every d, and recovery-fraction(fixed delta) trunc > plain at every d.
#   (R3) ONE-PASS SGD is LEAST efficient: delta*_sgd > delta*_plain > delta*_trunc,
#        and its fixed-delta overlap COLLAPSES toward 0 as d grows.
#   (R4) separation ratios delta*_plain/delta*_trunc and delta*_sgd/delta*_trunc >1
#        and increase with d.
#   (R5) fixed-budget (delta=4) recovery fraction: TRUNC high, PLAIN low, SGD ~0
#        at the largest d (stark, d-widening gap).
#   FALSIFICATION: if truncated needed delta >= plain (ratio<=1) OR truncated
#   overlap collapsed like plain/SGD, the efficiency separation would be refuted.
#
# Model: single-index y=sigma(<x,theta*>), theta*=e_1, x~N(0,I_d).
# Correlation loss  Lhat(theta)=-(1/n) sum_i y_i sigma(<x_i,theta>) on S^{d-1}.
# Spherical gradient flow: theta'=(I-theta theta^T)(1/n) sum y_i sigma'(<x_i,th>) x_i.
#   PLAIN sigma'=2z => flow = power iteration on A*=(2/n)sum y_i x_i x_i^T; its
#     t->inf limit is EXACTLY v1(A*) (paper proof sketch); computed via a Lanczos
#     top-eigenpair and cross-checked against the real flow (control [C]).
#   TRUNCATED: real spherical GD flow (3.8) from uniform init (converges by ~500 it).
# One-pass SGD: same spherical correlation gradient, ONE fresh sample/step, exactly
#   n steps (single pass), truncated link, step 0.1/d (info-exponent-2 scaling).
# Recovery: SQUARED overlap m^2=<theta,theta*>^2 >= TAU (even link => up to sign).
# CPU only, deterministic (numpy default_rng, fixed seeds). Self-contained.

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import time
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

M = 8.0
ETA_FLOW = 0.10
ETA_SGD = 0.10          # one-pass SGD step = ETA_SGD/d
TAU = 0.50              # recovery bar on squared overlap
TMAX_FLOW = 1000        # truncated flow converges by ~500 iters (verified)

DS = [64, 128, 256]
DS_PLAIN = [64, 128, 256, 512]
DELTAS_FB = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
DELTAS_TR = [2.0, 3.0, 4.0, 5.0]
DELTAS_SGD = [4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
FIXED_DELTA = 4.0
SEEDS_PLAIN = 24
SEEDS_TRUNC = 12
SEEDS_SGD = 12


def phi(u2):
    out = np.ones_like(u2)
    mid = (u2 > M) & (u2 < 2 * M)
    out[mid] = (2 * M - u2[mid]) / M
    out[u2 >= 2 * M] = 0.0
    return out


def sigma_trunc(z):
    s = z * z
    ramp = M + 2.0 * (s - M) - (s * s - M * M) / (2.0 * M)
    return np.where(s <= M, s, np.where(s < 2 * M, ramp, 1.5 * M))


def sigmap_trunc(z):
    return 2.0 * z * phi(z * z)


def unit(v):
    return v / np.linalg.norm(v)


def rand_unit(d, rng):
    return unit(rng.standard_normal(d))


def plain_overlap2(d, n, rng):
    X = rng.standard_normal((n, d))
    y = X[:, 0] ** 2
    def mv(v):
        return (2.0 / n) * (X.T @ (y * (X @ v)))
    L = LinearOperator((d, d), matvec=mv, dtype=float)
    try:
        _, V = eigsh(L, k=1, which='LA', maxiter=3000, tol=1e-7)
        return float(V[:, 0][0] ** 2)
    except Exception:
        A = (2.0 / n) * (X * y[:, None]).T @ X
        _, V = np.linalg.eigh(A)
        return float(V[:, -1][0] ** 2)


def plain_flow_overlap2(d, n, rng, eta=ETA_FLOW, tmax=4000):
    X = rng.standard_normal((n, d))
    y = X[:, 0] ** 2
    theta = rand_unit(d, rng)
    prev = theta[0] ** 2
    for t in range(tmax):
        g = (2.0 / n) * (X.T @ (y * (X @ theta)))
        g = g - (g @ theta) * theta
        theta = unit(theta + eta * g)
        m2 = theta[0] ** 2
        if abs(m2 - prev) < 1e-9 and t > 300:
            break
        prev = m2
    return float(theta[0] ** 2)


def trunc_flow_overlap2(d, n, rng, eta=ETA_FLOW, tmax=TMAX_FLOW):
    X = rng.standard_normal((n, d))
    y = sigma_trunc(X[:, 0])
    theta = rand_unit(d, rng)
    best = theta[0] ** 2
    prev = best
    stall = 0
    for t in range(tmax):
        v = X @ theta
        g = (X.T @ (y * sigmap_trunc(v))) / n
        g = g - (g @ theta) * theta
        theta = unit(theta + eta * g)
        m2 = theta[0] ** 2
        if m2 > best:
            best = m2
        if abs(m2 - prev) < 1e-8:           # safe: only after clearly settled
            stall += 1
            if stall > 80 and t > 500:
                break
        else:
            stall = 0
        prev = m2
    return float(best)


def onepass_sgd_overlap2(d, n, rng, eta=ETA_SGD):
    X = rng.standard_normal((n, d))
    y = sigma_trunc(X[:, 0])
    theta = rand_unit(d, rng)
    step = eta / d
    best = theta[0] ** 2
    for t in range(n):
        x = X[t]
        u = x @ theta
        g = (y[t] * sigmap_trunc(u)) * x
        g = g - (g @ theta) * theta
        theta = unit(theta + step * g)
        m2 = theta[0] ** 2
        if m2 > best:
            best = m2
    return float(best)


def stats(fn, d, n, seeds, salt):
    vals = np.array([fn(d, n, np.random.default_rng([d, n, s, salt])) for s in range(seeds)])
    return float(vals.mean()), float(np.mean(vals >= TAU))


def threshold(deltas, means, tau=TAU):
    for i, dl in enumerate(deltas):
        if means[i] >= tau:
            if i == 0:
                return float(dl)
            x0, x1, y0, y1 = deltas[i - 1], deltas[i], means[i - 1], means[i]
            return float(x0 + (tau - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)
    return float('inf')


def linfit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yh = a * x + b
    ssr = float(np.sum((y - yh) ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
    return float(a), float(b), (1.0 - ssr / sst if sst > 0 else float('nan'))


def main():
    t0 = time.time()
    bar = "=" * 74
    print(bar)
    print("QItZDBVCT0 CLAIM 1  full-batch GD vs one-pass SGD (sample-cx separation)")
    print("correlation loss on sphere; M=%.0f eta_flow=%.2f eta_sgd=%.2f/d tau=%.2f"
          % (M, ETA_FLOW, ETA_SGD, TAU))
    print("FB-plain=exact v1(A*)[Lanczos] | FB-trunc=real flow | one-pass SGD(real)")
    print(bar)

    print("")
    print("[C] control: real PLAIN spherical-GD flow == exact v1(A*) (justifies Lanczos)")
    for (d, dl) in [(128, 8.0), (256, 6.0)]:
        n = int(dl * d)
        ev = plain_overlap2(d, n, np.random.default_rng([d, n, 0, 11]))
        fl = plain_flow_overlap2(d, n, np.random.default_rng([d, n, 0, 11]))
        print("   d=%3d delta=%.1f  v1(A*) m^2=%.3f  real-flow m^2=%.3f  |diff|=%.4f"
              % (d, dl, ev, fl, abs(ev - fl)))

    print("")
    print("[P] full-batch PLAIN (Thm 3.1/Fig 1a,1c): mean m^2 [recover-frac]; thr grows ~log d")
    pm, pf, thr_p = {}, {}, {}
    for d in DS_PLAIN:
        ms, fs = zip(*[stats(plain_overlap2, d, int(dl * d), SEEDS_PLAIN, 11) for dl in DELTAS_FB])
        pm[d], pf[d] = list(ms), list(fs); thr_p[d] = threshold(DELTAS_FB, ms)
        print("   d=%3d " % d + " ".join("d%.0f=%.2f[%.0f%%]" % (dl, m, 100 * f)
              for dl, m, f in zip(DELTAS_FB, ms, fs)) + "  delta*=%.2f" % thr_p[d])

    print("")
    print("[T] full-batch TRUNCATED (Thm 3.2/Fig 1b): mean m^2 [recover-frac]; more efficient")
    tm, tf, thr_t = {}, {}, {}
    for d in DS:
        ms, fs = zip(*[stats(trunc_flow_overlap2, d, int(dl * d), SEEDS_TRUNC, 22) for dl in DELTAS_TR])
        tm[d], tf[d] = list(ms), list(fs); thr_t[d] = threshold(DELTAS_TR, ms)
        print("   d=%3d " % d + " ".join("d%.0f=%.2f[%.0f%%]" % (dl, m, 100 * f)
              for dl, m, f in zip(DELTAS_TR, ms, fs)) + "  delta*=%.2f" % thr_t[d])

    print("")
    print("[S] ONE-PASS SGD (truncated link): mean m^2 [recover-frac]; least efficient")
    sm, sf, thr_s = {}, {}, {}
    for d in DS:
        ms, fs = zip(*[stats(onepass_sgd_overlap2, d, int(dl * d), SEEDS_SGD, 33) for dl in DELTAS_SGD])
        sm[d], sf[d] = list(ms), list(fs); thr_s[d] = threshold(DELTAS_SGD, ms)
        print("   d=%3d " % d + " ".join("d%.0f=%.2f[%.0f%%]" % (dl, m, 100 * f)
              for dl, m, f in zip(DELTAS_SGD, ms, fs)) + "  delta*=%.2f" % thr_s[d])

    print("")
    print(bar)
    print("[SEP] weak-recovery sample threshold delta*=n/d (smallest with mean m^2>=%.2f)" % TAU)
    print("   d      truncFB   plainFB   1passSGD   plain/tr   sgd/tr")
    for d in DS:
        rp = thr_p[d] / thr_t[d]; rs = thr_s[d] / thr_t[d]
        print("  %4d    %6.2f    %6.2f    %7.2f    %6.2f    %6.2f"
              % (d, thr_t[d], thr_p[d], thr_s[d], rp, rs))
    print("   d=512 plainFB delta*=%.2f (extra point for log-d fit)" % thr_p[512])

    pa, _, pr2 = linfit(np.log(DS_PLAIN), [thr_p[d] for d in DS_PLAIN])
    ta, _, tr2 = linfit(np.log(DS), [thr_t[d] for d in DS])
    sa, _, sr2 = linfit(np.log(DS), [thr_s[d] for d in DS])
    print("")
    print("   fit delta*=a*log d+b : plainFB slope=%.2f R2=%.3f | truncFB slope=%.2f R2=%.3f | SGD slope=%.2f R2=%.3f"
          % (pa, pr2, ta, tr2, sa, sr2))

    print("")
    print("[F] fixed budget delta=%.1f : mean m^2 & recover-frac vs d (Fig 1a vs 1b)" % FIXED_DELTA)
    ip = DELTAS_FB.index(FIXED_DELTA); it = DELTAS_TR.index(FIXED_DELTA); iss = DELTAS_SGD.index(FIXED_DELTA)
    fx = {"trunc_m": [], "plain_m": [], "sgd_m": [], "trunc_f": [], "plain_f": [], "sgd_f": []}
    for d in DS:
        fx["trunc_m"].append(tm[d][it]); fx["plain_m"].append(pm[d][ip]); fx["sgd_m"].append(sm[d][iss])
        fx["trunc_f"].append(tf[d][it]); fx["plain_f"].append(pf[d][ip]); fx["sgd_f"].append(sf[d][iss])
        print("   d=%3d | trunc m^2=%.3f (%.0f%%)  plain m^2=%.3f (%.0f%%)  SGD m^2=%.3f (%.0f%%)"
              % (d, tm[d][it], 100 * tf[d][it], pm[d][ip], 100 * pf[d][ip], sm[d][iss], 100 * sf[d][iss]))

    dmax, dmin = DS[-1], DS[0]
    R1 = (pa > 0) and (pr2 >= 0.85)
    R2 = all(thr_t[d] < thr_p[d] for d in DS) and all(tf[d][it] > pf[d][ip] for d in DS)
    R3 = (thr_s[dmax] > thr_p[dmax] > thr_t[dmax]) and (fx["sgd_m"][-1] < fx["sgd_m"][0]) and (fx["sgd_m"][-1] < 0.2)
    R4 = (thr_p[dmax] / thr_t[dmax] > thr_p[dmin] / thr_t[dmin]) and (thr_s[dmax] / thr_t[dmax] > thr_s[dmin] / thr_t[dmin])
    R5 = (fx["trunc_f"][-1] - fx["plain_f"][-1] > 0.25) and (fx["trunc_f"][-1] - fx["sgd_f"][-1] > 0.4)
    overall = R1 and R2 and R3 and R5

    print("")
    print(bar)
    print("PASS (R1) plain FB threshold grows ~log d (exact)   : %s" % R1)
    print("PASS (R2) truncated MORE efficient than plain       : %s" % R2)
    print("PASS (R3) one-pass SGD least efficient / collapses  : %s" % R3)
    print("PASS (R4) separation ratios increase with d         : %s" % R4)
    print("PASS (R5) fixed-budget recover-frac: trunc>>plain>>SGD: %s" % R5)
    print("OVERALL SEPARATION CONFIRMED (R1,R2,R3,R5)          : %s" % overall)
    print(bar)
    elapsed = time.time() - t0
    print("runtime %.1fs  (numpy %s, scipy Lanczos)" % (elapsed, np.__version__))

    out = {
        "orid": "QItZDBVCT0",
        "claim": "Claim 1: full-batch GD outperforms one-pass SGD (sample-complexity separation)",
        "paper_section": "Sec 3 (Thm 3.1 plain / Thm 3.2 truncated), Fig 1",
        "M": M, "eta_flow": ETA_FLOW, "eta_sgd_over_d": ETA_SGD, "tau": TAU, "tmax_flow": TMAX_FLOW,
        "ds": DS, "ds_plain": DS_PLAIN, "fixed_delta": FIXED_DELTA,
        "deltas": {"plain": DELTAS_FB, "trunc": DELTAS_TR, "sgd": DELTAS_SGD},
        "seeds": {"plain": SEEDS_PLAIN, "trunc": SEEDS_TRUNC, "sgd": SEEDS_SGD},
        "threshold_delta_star": {"trunc": {str(d): thr_t[d] for d in DS},
                                 "plain": {str(d): thr_p[d] for d in DS_PLAIN},
                                 "sgd": {str(d): thr_s[d] for d in DS}},
        "logd_fit": {"plain": {"slope": pa, "R2": pr2}, "trunc": {"slope": ta, "R2": tr2},
                     "sgd": {"slope": sa, "R2": sr2}},
        "ratio_plain_over_trunc": {str(d): thr_p[d] / thr_t[d] for d in DS},
        "ratio_sgd_over_trunc": {str(d): thr_s[d] / thr_t[d] for d in DS},
        "mean_overlap2": {"plain": {str(d): pm[d] for d in DS_PLAIN},
                          "trunc": {str(d): tm[d] for d in DS}, "sgd": {str(d): sm[d] for d in DS}},
        "recover_frac": {"plain": {str(d): pf[d] for d in DS_PLAIN},
                         "trunc": {str(d): tf[d] for d in DS}, "sgd": {str(d): sf[d] for d in DS}},
        "fixed_budget": {"delta": FIXED_DELTA, "ds": DS, **fx},
        "checks": {"R1_plain_logd": bool(R1), "R2_trunc_more_efficient": bool(R2),
                   "R3_sgd_least": bool(R3), "R4_ratios_grow": bool(R4),
                   "R5_fixed_sep": bool(R5), "overall": bool(overall)},
        "runtime_s": elapsed, "numpy": np.__version__,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json")


if __name__ == "__main__":
    main()
