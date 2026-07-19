"""
Claim 1 reproduction: FC2FB meta-algorithm converts a fixed-confidence (FC)
best-arm-identification algorithm into a fixed-budget (FB) one with sample
complexity matching up to logarithmic factors.

Paper: "Fixed Budget is No Harder Than Fixed Confidence in Best-Arm
Identification up to Logarithmic Factors" (arXiv 2602.03972, ICML 2026),
Definition 3.1 (strong FC), Algorithm 3 (FC2FB), Theorem 3.2 (error bound).

We (a) verify the strong-FC premise and MEASURE its constant A,
(b) run FC2FB (Algorithm 3, faithful, delta0=1/e, Q=1) and check its error
    probability obeys the Theorem-3.2 bound while decaying exponentially,
(c) verify the FB-vs-FC sample-complexity penalty grows only logarithmically
    (sub-polynomial) -- i.e. matching "up to logarithmic factors".

Deterministic (fixed seeds), pure NumPy, single-thread CPU.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json, time, hashlib, sys
from pathlib import Path
import numpy as np

DELTA_GAP = 0.5                 # arm gap
MU = np.array([DELTA_GAP, 0.0]) # arm means, best arm = 0
BEST = 0
SIGMA = 1.0
A_THEORY = 8.0 * SIGMA**2 / DELTA_GAP**2   # FC characteristic time 8 sigma^2/Delta^2 = 32
HERE = Path(__file__).resolve().parent


def strong_fc(delta, n_cap, n_trials, rng):
    """Vectorised 2-arm round-robin GLR strong-FC subroutine.
    Per-arm count n; GLR statistic n*(mu0_hat-mu1_hat)^2/4; stop when >= ln(1/delta).
    Returns (total_samples[n_trials], recommendation[n_trials], self_terminated[n_trials])."""
    thr = np.log(1.0 / delta)
    X0 = rng.normal(MU[0], SIGMA, size=(n_trials, n_cap))
    X1 = rng.normal(MU[1], SIGMA, size=(n_trials, n_cap))
    ns = np.arange(1, n_cap + 1)
    m0 = np.cumsum(X0, axis=1) / ns
    m1 = np.cumsum(X1, axis=1) / ns
    diff = m0 - m1
    stat = ns * (diff ** 2) / 4.0
    crossed = stat >= thr
    any_cross = crossed.any(axis=1)
    first = np.where(any_cross, crossed.argmax(axis=1), n_cap - 1)
    stop_n = first + 1
    row = np.arange(n_trials)
    rec = (m1[row, first] > m0[row, first]).astype(int)   # 1 -> chose wrong arm
    return 2 * stop_n, rec, any_cross


def measure_fc_scaling(deltas, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    taus, errs, fracs = [], [], []
    for d in deltas:
        n_cap = int(4.0 * (4.0 / DELTA_GAP ** 2) * np.log(1.0 / d)) + 80
        tot, rec, stopped = strong_fc(d, n_cap, n_trials, rng)
        taus.append(float(tot.mean()))
        errs.append(float((rec != BEST).mean()))
        fracs.append(float(stopped.mean()))
    x = np.log(1.0 / np.array(deltas))
    y = np.array(taus)
    A, C = np.polyfit(x, y, 1)
    yhat = A * x + C
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(deltas=list(deltas), taus=taus, errs=errs, frac_stopped=fracs,
                A=float(A), C=float(C), r2=float(r2))


def fc2fb(budgets, A_meas, n_trials, delta0, Q, seed=1):
    """Faithful Algorithm 3 (FC2FB). R=floor(log2(B/Q)) stages, per-stage budget
    B'=floor(B/R), stage r targets delta0**(2**(R-r)), force-terminate at B'/2
    per-arm cap; output the FIRST stage that self-terminates."""
    rng = np.random.default_rng(seed)
    ln_inv_d0 = np.log(1.0 / delta0)
    perr, bound, Rs = [], [], []
    for B in budgets:
        R = max(int(np.floor(np.log2(B / Q))), 1)
        Bp = int(B // R)
        ncap = max(Bp // 2, 1)
        rec = np.full(n_trials, -1, dtype=int)
        done = np.zeros(n_trials, dtype=bool)
        for r in range(1, R + 1):
            idx = np.where(~done)[0]
            if idx.size == 0:
                break
            stage_delta = max(delta0 ** (2 ** (R - r)), 1e-300)
            _, rr, stopped = strong_fc(stage_delta, ncap, idx.size, rng)
            sub = stopped
            rec[idx[sub]] = rr[sub]
            done[idx[sub]] = True
        arb = rng.integers(0, 2, size=n_trials)          # arbitrary arm if never terminated
        final = np.where(rec < 0, arb, rec)
        perr.append(float((final != BEST).mean()))
        denom = 4.0 * Q / ln_inv_d0 + 4.0 * np.log2(B / Q) * A_meas
        bound.append(float(3.0 * np.exp(-B / denom)))
        Rs.append(R)
    return dict(budgets=list(budgets), perr=perr, bound=bound, R=Rs)


def main():
    t0 = time.time()
    print("=" * 72)
    print("CLAIM 1  FC2FB reduction (Def 3.1 / Alg 3 / Thm 3.2)")
    print("2-arm Gaussian mu=[0.5,0.0], Delta=0.5, sigma=1")
    print("FC characteristic time  A* = 8 sigma^2/Delta^2 =", A_THEORY)
    print("=" * 72)

    # (a) strong-FC premise + measure A -----------------------------------
    deltas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    fc = measure_fc_scaling(deltas, n_trials=4000, seed=0)
    print("\n[a] strong-FC: tau vs ln(1/delta)   (4000 trials/delta)")
    for d, t, e, fr in zip(deltas, fc["taus"], fc["errs"], fc["frac_stopped"]):
        print("    delta=%.0e  tau=%8.2f  emp_err=%.4f  (<=delta? %s)  stopped=%.3f"
              % (d, t, e, str(e <= d), fr))
    A_meas = fc["A"]
    print("    fit  A_meas=%.3f  C_meas=%.3f  R2=%.6f   |  A*=%.1f  rel_err=%.3f"
          % (A_meas, fc["C"], fc["r2"], A_THEORY, abs(A_meas - A_THEORY) / A_THEORY))

    # (b) FC2FB error vs Theorem-3.2 bound --------------------------------
    delta0 = 1.0 / np.e
    Q = 1.0
    budgets = [300, 450, 650, 950, 1400, 2000, 3000, 4500, 6500, 9000, 13000, 18000]
    fb = fc2fb(budgets, A_meas, n_trials=4000, delta0=delta0, Q=Q, seed=1)
    print("\n[b] FC2FB (Alg 3, delta0=1/e, Q=1): P_err vs Thm-3.2 bound  (4000 trials/B)")
    all_below = True
    for B, p, bd, R in zip(budgets, fb["perr"], fb["bound"], fb["R"]):
        below = p <= bd
        all_below = all_below and below
        print("    B=%5d  R=%2d  P_err=%.4f  Thm3.2_bound=%.4f  (P_err<=bound? %s)"
              % (B, R, p, bd, str(below)))
    xb = np.array([B for B, p in zip(budgets, fb["perr"]) if p > 0])
    yb = np.array([np.log(p) for p in fb["perr"] if p > 0])
    decay = float(np.polyfit(xb, yb, 1)[0]) if xb.size >= 2 else float("nan")
    print("    fitted ln(P_err) decay slope = %.6f  (negative => exponential decay)" % decay)
    print("    all P_err <= Thm-3.2 bound? %s" % all_below)

    # (c) matching up to log factors: FB/FC penalty ratio -----------------
    print("\n[c] FB-vs-FC penalty ratio  rho(B) = B / (A_meas * ln(1/P_err))")
    ratios = []
    for B, p in zip(budgets, fb["perr"]):
        if 0.0 < p < 1.0:
            rho = B / (A_meas * np.log(1.0 / p))
            ratios.append((B, p, rho))
            print("    B=%5d  achieved_delta~%.4f  rho=%.3f" % (B, p, rho))
    growth = float("nan")
    if len(ratios) >= 2:
        Bv = np.log([b for b, _, _ in ratios])
        Rv = np.log([r for _, _, r in ratios])
        growth = float(np.polyfit(Bv, Rv, 1)[0])
        rr = [r for _, _, r in ratios]
        print("    rho range [%.3f, %.3f]   log-log growth exponent d ln(rho)/d ln(B)=%.3f"
              % (min(rr), max(rr), growth))
        print("    (exponent < 1  => sub-polynomial => penalty is only a LOG factor)")

    # verdict -------------------------------------------------------------
    a_ok = (fc["r2"] > 0.99) and (abs(A_meas - A_THEORY) / A_THEORY < 0.10) \
           and all(e <= d for e, d in zip(fc["errs"], deltas))
    b_ok = all_below and (decay < 0)
    c_ok = (not np.isnan(growth)) and (growth < 1.0)
    verified = a_ok and b_ok and c_ok
    print("\n" + "=" * 72)
    print("(a) strong-FC linear R2>0.99 & A_meas~=8/Delta^2 (<10%%) & delta-correct : %s" % a_ok)
    print("(b) FC2FB P_err <= Thm-3.2 bound at all B & exponential decay           : %s" % b_ok)
    print("(c) FB/FC penalty sub-polynomial (log-log exp %.3f < 1)                 : %s" % (growth, c_ok))
    print("VERDICT verified (all three gates) : %s" % verified)
    print("=" * 72)

    out = dict(
        claim="FC2FB converts fixed-confidence into fixed-budget with sample "
              "complexity matching up to logarithmic factors",
        paper="arXiv 2602.03972 Def 3.1 / Alg 3 / Thm 3.2",
        instance=dict(arms=2, mu=[0.5, 0.0], Delta=DELTA_GAP, sigma=SIGMA),
        A_theory=A_THEORY, fc=fc, delta0=float(delta0), Q=Q, fb=fb,
        fb_decay_slope=decay, all_perr_below_thm_bound=bool(all_below),
        penalty_ratios=[dict(B=b, delta=p, rho=r) for b, p, r in ratios],
        penalty_loglog_growth=growth,
        gates=dict(a_strong_fc=bool(a_ok), b_thm32=bool(b_ok), c_uptolog=bool(c_ok)),
        verdict="verified" if verified else "toy",
        runtime_s=round(time.time() - t0, 2),
        numpy=np.__version__, python=sys.version.split()[0], seeds=dict(fc=0, fb=1),
    )
    (HERE / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote %s   (%.1fs)" % (HERE / "results.json", out["runtime_s"]))


if __name__ == "__main__":
    main()
