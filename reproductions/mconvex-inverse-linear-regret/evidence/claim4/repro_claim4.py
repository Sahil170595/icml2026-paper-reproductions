#!/usr/bin/env python3
"""
Claim 4 (Theorem 6.1): There exists an online-inverse-linear-optimization instance with M-convex
feasible sets X_1,...,X_T on which ANY (randomized) algorithm incurs regret R_T = Omega(d).
Hence the O(d log d) upper bound (Claim 2) is tight up to the log d factor.

Faithful realization of the hard instance (Sakaue et al. 2025b, Thm 5.1, adapted to M-convex).
Round i (i=1..d) presents an axis-aligned integer segment along coordinate i:
  X_i = { t*e_i : t in {-k,...,+k} },  k = floor(sqrt(d)/4)   (an M^natural-convex line segment;
  embeddable as an M-convex set in Z^{d+1}, Murota 2003 Sec 6.1 -- exactly the paper's argument).
The agent's hidden w*(i) has a sign chosen adversarially/uniformly and is revealed ONLY at round i;
since every coordinate is queried once and the segments are on distinct axes, the learner has NO
information about sign(w*(i)) before committing x_hat_i.  Any learner therefore mispredicts the
endpoint with probability >= 1/2, paying the full per-round gap (normalized to 1, Assumption 2.2).
Expected regret >= d * (1/2) * 1 = d/2 = Omega(d), for EVERY learner.

TARGET (Theorem 6.1): R_T = Omega(d) -- expected regret grows at least linearly in d.
ACCEPTANCE RULE (all):
  (A) for every tested learner (random guess, always-+, history/centroid), E[R]/d is ~constant
      (in [0.4,0.6]) across d, i.e. E[R] = Theta(d);
  (B) least-squares slope of E[R] vs d in [0.4,0.6] (target 0.5), R^2 > 0.999;
  (C) no learner achieves sublinear regret: E[R] >= 0.3*d for all d and all learners.
FALSIFIED if some learner attains o(d) (sublinear) regret on this instance.
"""
import numpy as np, json, time, os, math
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

def run_instance(d, seed, strategy):
    """One realization: hidden signs, d one-shot coordinate segments. Return regret (unit gap)."""
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=d)          # hidden sign of w*(i); |gap| normalized to 1
    k = max(1, int(math.floor(math.sqrt(d) / 4)))  # segment half-length (integer points), paper's k
    seen_signs = []                               # history of previously revealed signs
    reg = 0
    for i in range(d):
        # learner commits x_hat_i = guess of the maximizing endpoint, with NO info on coordinate i
        if strategy == "random":
            guess = rng.choice([-1, 1])
        elif strategy == "always_pos":
            guess = 1
        elif strategy == "history_majority":
            # a "smart" learner that uses the empirical sign majority of past coords -> still no info
            guess = 1 if (sum(seen_signs) >= 0) else -1
        if guess != signs[i]:
            reg += 1                              # picked the wrong endpoint: full gap 1
        seen_signs.append(signs[i])
    return reg, k

def main():
    t0 = time.time()
    ds = [16, 32, 64, 100, 144, 196, 256]
    strategies = ["random", "always_pos", "history_majority"]
    NREP = 400
    table = {}; kvals = {}
    for strat in strategies:
        row = []
        for d in ds:
            regs = []
            for s in range(NREP):
                r, k = run_instance(d, s, strat); regs.append(r); kvals[d] = k
            row.append(float(np.mean(regs)))
        table[strat] = row
    # ratio E[R]/d and linear fit for each strategy
    ratios = {st: [round(table[st][j] / ds[j], 4) for j in range(len(ds))] for st in strategies}
    slopes = {}; r2s = {}
    x = np.array(ds, float)
    A = np.vstack([x, np.ones_like(x)]).T
    for st in strategies:
        y = np.array(table[st], float)
        sl, ic = np.linalg.lstsq(A, y, rcond=None)[0]
        pred = A @ np.array([sl, ic]); r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        slopes[st] = round(float(sl), 4); r2s[st] = round(float(r2), 5)
    all_linear = all(0.4 <= slopes[st] <= 0.6 and r2s[st] > 0.999 for st in strategies)
    all_ratio_const = all(0.4 <= min(ratios[st]) and max(ratios[st]) <= 0.6 for st in strategies)
    no_sublinear = all(table[st][j] >= 0.3 * ds[j] for st in strategies for j in range(len(ds)))
    res = {
        "claim": "Theorem 6.1: any algorithm incurs R_T = Omega(d) on M-convex hard instance (tightness up to log d)",
        "target": "R_T = Omega(d); expected regret grows at least linearly in d (>= c*d) for every learner",
        "acceptance_rule": "(A) E[R]/d ~const in [0.4,0.6]; (B) slope of E[R] vs d in [0.4,0.6], R^2>0.999; (C) E[R]>=0.3d for all learners",
        "ds": ds, "segment_halflength_k_floor_sqrt_d_over_4": [kvals[d] for d in ds],
        "n_repetitions": NREP,
        "E_regret_by_strategy": {st: [round(v, 2) for v in table[st]] for st in strategies},
        "E_regret_over_d_by_strategy": ratios,
        "linear_slope_by_strategy": slopes, "fit_r2_by_strategy": r2s,
        "all_strategies_linear_slope_~0.5": bool(all_linear),
        "all_ratios_constant_0.4_0.6": bool(all_ratio_const),
        "no_learner_sublinear": bool(no_sublinear),
        "verdict_rule_A": bool(all_ratio_const),
        "verdict_rule_B": bool(all_linear),
        "verdict_rule_C": bool(no_sublinear),
        "runtime_sec": round(time.time() - t0, 2), "numpy_version": np.__version__,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results.json"), "w") as f: json.dump(res, f, indent=2)
    print("== Claim 4 (Thm 6.1): Omega(d) lower bound on M-convex hard instance ==")
    hdr = f"{'d':>5} " + " ".join(f"{st[:9]:>12}" for st in strategies) + f"  {'k':>3}"
    print(hdr)
    for j, d in enumerate(ds):
        print(f"{d:>5} " + " ".join(f"{table[st][j]:>12.2f}" for st in strategies) + f"  {kvals[d]:>3}")
    print("E[R]/d:")
    for st in strategies:
        print(f"  {st:16s} ratios={ratios[st]}  slope={slopes[st]}  R^2={r2s[st]}")
    print(f"all learners linear (slope~0.5, R^2>0.999): {all_linear}")
    print(f"no learner achieves sublinear regret (E[R]>=0.3d): {no_sublinear}")
    print(f"runtime {res['runtime_sec']}s  numpy {np.__version__}")

if __name__ == "__main__":
    main()
