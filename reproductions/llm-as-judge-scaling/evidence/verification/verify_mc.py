"""Independent-method cross-check: compare the exact order-statistics
quadrature (model.delta_exact, used for all three claim scripts) against a
completely independent direct Monte Carlo simulation (model.delta_mc: draw k
samples, hard-select by reward, measure squared error -- no quadrature, no
order-statistics formula) for representative (eta, k) pairs in all three
claim regimes. Uses the SAME fitted posterior / test set as the claim
scripts (same seeds), so this isolates disagreement between the two
computation METHODS, not sampling noise from a different draw.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import (build_fixed_posterior, make_wR, test_stats, delta_exact,  # noqa: E402
                    delta_mc, D, N_TRAIN, S_STD, SIGMA, GAMMA,
                    TAU_GRID, SEED_TEST, SEED_MISALIGN)

HERE = Path(__file__).resolve().parent

# Smaller test-point subset + a per-k trial budget for this cross-check only
# (the primary evidence uses the full N_TEST=1200 exact quadrature in
# ../claim{1,2,3}/; this script exists solely to validate that quadrature
# against a completely independent brute-force simulation).
N_TEST_MC = 150
MC_SEED = 2024
TRIAL_BUDGET = 4_000_000  # total (n_test_mc * n_trials * k) kept roughly fixed across k

checks = [
    ("claim1_small_mismatch", 0.05, [1, 5, 20, 100, 500]),
    ("claim2_large_mismatch", 0.15, [1, 4, 20, 100, 1000]),
    ("claim3_teacher_reward", 0.00, [1, 5, 20, 100, 1000]),
]

t0 = time.time()
w_T, w_post, Lam_inv = build_fixed_posterior()

results = {"n_test_mc": N_TEST_MC, "trial_budget": TRIAL_BUDGET, "seed": MC_SEED, "checks": []}
print("== Verification: exact quadrature vs. direct Monte Carlo (independent method) ==")
print("%-24s %6s %10s %12s %12s %10s" % ("regime", "k", "n_trials", "exact", "mc", "rel.diff"))
for name, eta, ks in checks:
    rng_w = np.random.default_rng(SEED_MISALIGN)
    w_R, v = make_wR(w_T, eta, rng_w)
    rng_te = np.random.default_rng(SEED_TEST)
    m, s, yT, yR = test_stats(N_TEST_MC, D, S_STD, w_T, w_R, w_post, Lam_inv, SIGMA, rng_te)
    bp = (yT - m) / s
    tp = (yR - m) / s
    s2 = s**2
    k_arr = np.array(ks)
    exact_vals = delta_exact(bp, tp, s2, k_arr, TAU_GRID)
    rng_mc = np.random.default_rng(MC_SEED)
    for k, ev in zip(ks, exact_vals):
        n_trials = max(500, min(20000, TRIAL_BUDGET // (N_TEST_MC * max(k, 1))))
        mc_vals = delta_mc(m, s, yT, yR, k, n_trials, rng_mc, batch=100)
        mc_mean = float(np.mean(mc_vals))
        rel_diff = float((mc_mean - ev) / ev)
        print("%-24s %6d %10d %12.6f %12.6f %9.2f%%" % (name, k, n_trials, ev, mc_mean, 100 * rel_diff))
        results["checks"].append({"regime": name, "eta": eta, "k": int(k), "n_trials": int(n_trials),
                                   "exact": float(ev), "mc": mc_mean, "rel_diff": rel_diff})

runtime_s = round(time.time() - t0, 2)
results["runtime_s"] = runtime_s
max_rel = max(abs(c["rel_diff"]) for c in results["checks"])
results["max_abs_rel_diff"] = max_rel
(HERE / "verification.json").write_text(json.dumps(results, indent=1))
print()
print("max |relative difference| across all %d checks = %.2f%%  (Monte Carlo sampling noise, per-k trial count above)"
      % (len(results["checks"]), 100 * max_rel))
print("runtime=%.2fs" % runtime_s)
print("[written] verification.json")
