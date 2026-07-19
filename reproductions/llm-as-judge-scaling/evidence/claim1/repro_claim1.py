"""Claim 1 -- "When the reward is not too different from the teacher,
generalization error decreases monotonically with increasing inference-time
samples k." (Halder & Pehlevan, arXiv:2512.19905, ICML 2026, OpenReview
ANVg7NnupP.)

Independent NumPy/SciPy reproduction (see ../model.py for the full model and
method). Small reward misalignment eta=0.05 (reward direction differs from
the teacher direction by a small amount; see model.make_wR). Computes
delta(k) = E_x[(y_selected(x,k) - y_teacher(x))^2] EXACTLY (order-statistics
quadrature, no Monte Carlo noise) for every integer k = 1..K_MAX and checks
strict monotonic decrease.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import build_fixed_posterior, eval_regime, N_TEST, D, N_TRAIN, S_STD, SIGMA, GAMMA  # noqa: E402

HERE = Path(__file__).resolve().parent

ETA = 0.05
K_MAX = 1000

t0 = time.time()
w_T, w_post, Lam_inv = build_fixed_posterior()
k_values = np.arange(1, K_MAX + 1)
d_of_k, (m, s, yT, yR), diag = eval_regime(ETA, k_values, w_T, w_post, Lam_inv)

diffs = np.diff(d_of_k)
strictly_decreasing = bool(np.all(diffs < 0))
n_violations = int(np.sum(diffs >= 0))
floor = diag["mean_reward_teacher_gap2"]

runtime_s = round(time.time() - t0, 2)

report_k = [1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000]
rows = [(int(k), float(d_of_k[k - 1])) for k in report_k]

results = {
    "claim": "1: reward close to teacher -> generalization error decreases monotonically with k",
    "paper": {"arxiv_id": "2512.19905", "openreview_id": "ANVg7NnupP",
              "title": "Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling"},
    "config": {"d": D, "n_train": N_TRAIN, "S": S_STD, "sigma": SIGMA, "gamma": GAMMA,
               "n_test": N_TEST, "eta_misalignment": ETA, "k_max": K_MAX},
    "diagnostics": diag,
    "delta_of_k_report_points": rows,
    "delta_1": float(d_of_k[0]),
    "delta_kmax": float(d_of_k[-1]),
    "min_delta": float(np.min(d_of_k)),
    "argmin_k": int(k_values[np.argmin(d_of_k)]),
    "strictly_decreasing_all_k_1_to_kmax": strictly_decreasing,
    "num_non_decreasing_steps": n_violations,
    "max_nonneg_diff": float(np.max(diffs)),
    "floor_mean_reward_teacher_gap2": floor,
    "runtime_s": runtime_s,
}
(HERE / "results.json").write_text(json.dumps(results, indent=1))

print("== Claim 1: small reward-teacher misalignment (eta=%.3f) -> monotone decrease in k ==" % ETA)
print("d=%d n_train=%d sigma=%.3f gamma=%.3f n_test=%d  runtime=%.2fs" % (D, N_TRAIN, SIGMA, GAMMA, N_TEST, runtime_s))
print("diagnostics: mean teacher-bias^2=%.6f  mean (reward-teacher gap)^2=%.6f  mean s^2=%.6f  max|t'|=%.3f"
      % (diag["mean_teacher_bias2"], diag["mean_reward_teacher_gap2"], diag["mean_s2"], diag["max_abs_tprime"]))
print()
print("  k     delta(k)")
for k, v in rows:
    print("%5d   %.6f" % (k, v))
print()
print("Strictly decreasing for every k=1..%d: %s (%d non-decreasing steps out of %d)"
      % (K_MAX, strictly_decreasing, n_violations, K_MAX - 1))
print("delta(1)=%.6f -> delta(%d)=%.6f  (floor = mean(y_R-y_T)^2 = %.6f)" % (d_of_k[0], K_MAX, d_of_k[-1], floor))
print("[written] results.json")
