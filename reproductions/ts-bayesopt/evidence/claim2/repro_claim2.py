"""
Independent NumPy/SciPy reproduction of Claim 2 of
"On Regret Bounds of Thompson Sampling for Bayesian Optimization" (arXiv 2603.09276).

Claim 2 bundles two contributions of the paper:
  (ii)  Theorem 3.2: E[R_T^2] = O(T gamma_T log T), which via Markov's inequality
        yields the improved high-probability bound
            Pr( R_T <= sqrt( E[R_T^2] / delta ) ) >= 1 - delta,
        i.e. the dependence on the failure probability delta is tightened from
        the previously established 1/delta (first-moment Markov) to 1/sqrt(delta).
  (iii) Theorem 3.3: the EXPECTED LENIENT regret is polylogarithmic in T; the
        number of Delta-suboptimal rounds  |T_Delta| = #{t: f(x*)-f(x_t) >= Delta}
        and the lenient regret LR_T = sum over those rounds saturate (bounded /
        polylog in T) rather than growing like the cumulative regret.

Faithful GP-TS in the Bayesian setting: the objective f is a sample path from a
GP prior over a finite domain X (grid on [0,1], squared-exponential kernel), the
exact setting analysed for the |X| < infinity case in the paper. Each round GP-TS
draws a full posterior sample over X (exact Gaussian conjugate posterior),
queries its argmax, observes a noisy reward, and updates the posterior. All runs
are deterministic (numpy.random.default_rng, fixed seeds). CPU only.

Usage:
    python3 repro_claim2.py moment    # experiment 2A (second moment / improved delta)
    python3 repro_claim2.py lenient   # experiment 2B (lenient regret polylog)
    python3 repro_claim2.py all       # both (default)
Results are merged into results.json in this directory.
"""
import numpy as np
import json
import os
import sys
from pathlib import Path

OUT = Path(__file__).with_name("results.json")


def se_gram(xs, ell):
    d = xs[:, None] - xs[None, :]
    return np.exp(-(d ** 2) / (2.0 * ell ** 2))


def gpts_finite(M, K, T, ell, noise, seed, checkpoints=None, Delta=None):
    """Vectorised exact GP-TS on a finite GP bandit (K grid points on [0,1]).
    Bayesian setting: each of the M trials draws its own f ~ GP prior.
    Returns dict with cumulative regret R_T per trial (and, if checkpoints given,
    checkpointed cumulative regret, Delta-bad-pull counts and lenient regret)."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1.0, K)
    C = se_gram(xs, ell) + 1e-9 * np.eye(K)
    L0 = np.linalg.cholesky(C)
    # true objective per trial ~ GP prior
    f = (L0 @ rng.standard_normal((K, M))).T           # (M,K)
    xstar = f.max(1)                                   # (M,)
    m = np.zeros((M, K))
    P = np.broadcast_to(C, (M, K, K)).copy()
    R = np.zeros(M)
    idx = np.arange(M)
    cps = set(checkpoints) if checkpoints else set()
    ck = {}
    nbad = np.zeros(M)          # count of Delta-suboptimal pulls
    lr = np.zeros(M)            # lenient regret accumulator
    for t in range(1, T + 1):
        Lc = np.linalg.cholesky(P + 1e-10 * np.eye(K))
        g = m + np.einsum('mij,mj->mi', Lc, rng.standard_normal((M, K)))
        a = g.argmax(1)
        gap = xstar - f[idx, a]
        R += gap
        if Delta is not None:
            bad = gap >= Delta
            nbad += bad
            lr += gap * bad
        y = f[idx, a] + np.sqrt(noise) * rng.standard_normal(M)
        v = P[idx, :, a]
        den = P[idx, a, a] + noise
        m += v * ((y - m[idx, a]) / den)[:, None]
        P -= np.einsum('mi,mj->mij', v, v) / den[:, None, None]
        if t in cps:
            ck[t] = (R.copy(), nbad.copy(), lr.copy())
    return {"R": R, "xstar": xstar, "checkpoints": ck}


def experiment_moment(results):
    # ---- 2A: second moment => improved delta dependence -------------------
    K, ell, noise, seed = 8, 0.25, 0.05, 20260717
    T_fixed, M_fixed = 300, 40000
    r = gpts_finite(M_fixed, K, T_fixed, ell, noise, seed)
    R = r["R"]
    m1 = float(R.mean())
    m2 = float((R ** 2).mean())
    sd = float(R.std())
    deltas = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
    rows = []
    for d in deltas:
        q = float(np.quantile(R, 1.0 - d))          # empirical (1-delta) quantile
        b_old = m1 / d                              # first-moment Markov bound (old, ~1/delta)
        b_new = np.sqrt(m2 / d)                     # second-moment Markov bound (new, ~1/sqrt(delta))
        exc_new = float(np.mean(R >= b_new))        # empirical exceedance of new bound (must be <= d)
        rows.append({
            "delta": d, "emp_quantile": q,
            "bound_old_1_over_delta": b_old,
            "bound_new_1_over_sqrt_delta": b_new,
            "exceedance_new": exc_new,
            "improve_factor_old_over_new": b_old / b_new,
        })
    # fit exponent of improvement factor vs 1/delta  (theory: 0.5)
    inv = np.array([1.0 / d for d in deltas])
    fac = np.array([row["improve_factor_old_over_new"] for row in rows])
    imp_exp = float(np.polyfit(np.log(inv), np.log(fac), 1)[0])
    # fit empirical quantile exponent q: Q(delta) = c * delta^{-q}
    q_arr = np.array([row["emp_quantile"] for row in rows])
    quant_exp = float(-np.polyfit(np.log(np.array(deltas)), np.log(q_arr), 1)[0])
    bound_valid = all(row["exceedance_new"] <= row["delta"] + 1e-9 for row in rows)

    # ---- second-moment scaling in T (E[R_T^2] = O(T polylog), sub-quadratic)
    Ts = [75, 150, 300, 600]
    m2s = []
    for i, Tv in enumerate(Ts):
        rr = gpts_finite(8000, K, Tv, ell, noise, seed + 101 + i)["R"]
        m2s.append(float((rr ** 2).mean()))
    m2_exp = float(np.polyfit(np.log(Ts), np.log(m2s), 1)[0])   # theory upper bd exponent ~1

    results["moment"] = {
        "setup": {"K": K, "ell": ell, "noise": noise, "seed": seed,
                  "T_fixed": T_fixed, "M_fixed": M_fixed, "domain": "[0,1] grid, SE kernel, Bayesian f~prior"},
        "m1_mean_RT": m1, "m2_mean_RT2": m2, "std_RT": sd,
        "delta_table": rows,
        "improvement_factor_exponent_vs_inv_delta": imp_exp,   # target ~0.5
        "empirical_quantile_exponent_q": quant_exp,            # <= 0.5 (>=0 light tails)
        "new_bound_valid_all_delta": bool(bound_valid),
        "second_moment_T_sweep": {"T": Ts, "m2": m2s, "m2_exponent_in_T": m2_exp},
    }
    print("=== 2A  Second moment => improved delta dependence (Thm 3.2) ===")
    print(f"setup: GP-TS, K={K} arms on [0,1], SE ell={ell}, noise={noise}, "
          f"T={T_fixed}, M={M_fixed}, seed={seed}")
    print(f"E[R_T]={m1:.4f}  E[R_T^2]={m2:.4f}  sd={sd:.4f}")
    print(f"{'delta':>9} {'emp_q(1-d)':>11} {'old 1/d':>10} {'new 1/sqrt(d)':>13} "
          f"{'exceed_new':>10} {'improve x':>9}")
    for row in rows:
        print(f"{row['delta']:>9.5f} {row['emp_quantile']:>11.3f} "
              f"{row['bound_old_1_over_delta']:>10.2f} {row['bound_new_1_over_sqrt_delta']:>13.3f} "
              f"{row['exceedance_new']:>10.5f} {row['improve_factor_old_over_new']:>9.3f}")
    print(f"improvement-factor exponent vs 1/delta = {imp_exp:.4f}  (theory 0.5)")
    print(f"empirical (1-delta) quantile exponent q = {quant_exp:.4f}  (<=0.5 => tail no heavier than 2nd-moment worst case)")
    print(f"new delta^-1/2 bound valid at all delta (exceedance<=delta): {bound_valid}")
    print(f"E[R_T^2] T-sweep T={Ts} -> m2={[round(v,2) for v in m2s]}  exponent_in_T={m2_exp:.4f} (<=1 => O(T polylog) upper bd holds)")
    return results


def experiment_lenient(results):
    # ---- 2B: lenient regret is polylog / bounded (Thm 3.3) ----------------
    K, ell, noise, seed = 12, 0.20, 0.05, 20260718
    M, T = 1500, 1024
    cps = [64, 128, 256, 512, 1024]
    out = {}
    for Delta in [0.3, 0.6]:
        r = gpts_finite(M, K, T, ell, noise, seed, checkpoints=cps, Delta=Delta)
        ck = r["checkpoints"]
        R_series = [float(ck[t][0].mean()) for t in cps]
        nbad_series = [float(ck[t][1].mean()) for t in cps]
        lr_series = [float(ck[t][2].mean()) for t in cps]
        exp_R = float(np.polyfit(np.log(cps), np.log(R_series), 1)[0])
        exp_nbad = float(np.polyfit(np.log(cps), np.log(np.maximum(nbad_series, 1e-9)), 1)[0])
        exp_lr = float(np.polyfit(np.log(cps), np.log(np.maximum(lr_series, 1e-9)), 1)[0])
        out[f"Delta_{Delta}"] = {
            "T": cps, "E_RT": R_series, "E_badpulls": nbad_series, "E_lenient_LR": lr_series,
            "exp_cumulative_RT": exp_R, "exp_badpulls": exp_nbad, "exp_lenient": exp_lr,
        }
        print(f"\n=== 2B  Lenient regret polylog (Thm 3.3), Delta={Delta} ===")
        print(f"setup: GP-TS, K={K} arms on [0,1], SE ell={ell}, noise={noise}, M={M}, seed={seed}")
        print(f"{'T':>6} {'E[R_T] cumul':>13} {'E|T_Delta| bad':>15} {'E[LR_T] lenient':>16}")
        for i, t in enumerate(cps):
            print(f"{t:>6} {R_series[i]:>13.3f} {nbad_series[i]:>15.3f} {lr_series[i]:>16.4f}")
        print(f"growth exponents: cumulative R_T={exp_R:.3f}  bad-pulls={exp_nbad:.3f}  lenient LR_T={exp_lr:.3f}")
        print(f"  (Thm 3.3: lenient/bad-pull exponents ~0 [polylog/bounded] vs positive cumulative exponent)")
    results["lenient"] = {
        "setup": {"K": K, "ell": ell, "noise": noise, "seed": seed, "M": M, "T": T},
        "by_Delta": out,
    }
    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}
    if OUT.exists():
        try:
            results = json.loads(OUT.read_text())
        except Exception:
            results = {}
    if mode in ("moment", "all"):
        results = experiment_moment(results)
    if mode in ("lenient", "all"):
        results = experiment_lenient(results)
    results["_meta"] = {
        "paper": "arXiv 2603.09276", "claim": 2,
        "numpy": np.__version__,
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
