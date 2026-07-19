"""Claim 2 -- "Substantial reward misspecification induces a finite optimal
k beyond which more sampling increases generalization error." (Halder &
Pehlevan, arXiv:2512.19905, ICML 2026, OpenReview ANVg7NnupP.)

Independent NumPy/SciPy reproduction (see ../model.py). Large reward
misalignment eta=0.15 (3x the Claim-1 misalignment, same fixed direction).
Computes delta(k) exactly (order-statistics quadrature) on a grid dense at
small k (to pinpoint k*) and log-spaced out to k=5000 (to show the
subsequent rise and its plateau toward the reward-teacher mismatch floor
E[(y_R-y_T)^2]).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import build_fixed_posterior, eval_regime, N_TEST, D, N_TRAIN, S_STD, SIGMA, GAMMA  # noqa: E402

HERE = Path(__file__).resolve().parent

ETA = 0.15

t0 = time.time()
w_T, w_post, Lam_inv = build_fixed_posterior()

k_dense = np.arange(1, 61)
k_sparse = np.unique(np.round(np.geomspace(61, 5000, 40)).astype(int))
k_values = np.unique(np.concatenate([k_dense, k_sparse]))

d_of_k, (m, s, yT, yR), diag = eval_regime(ETA, k_values, w_T, w_post, Lam_inv)

argmin_idx = int(np.argmin(d_of_k))
kstar = int(k_values[argmin_idx])
delta_star = float(d_of_k[argmin_idx])
floor = diag["mean_reward_teacher_gap2"]

# check: (a) decreasing from k=1 to k*, (b) increasing from k* to k_max
pre = d_of_k[: argmin_idx + 1]
post = d_of_k[argmin_idx:]
decreasing_to_star = bool(np.all(np.diff(pre) <= 1e-12))
increasing_after_star = bool(np.all(np.diff(post) >= -1e-12))
rise_relative = float((d_of_k[-1] - delta_star) / delta_star)

runtime_s = round(time.time() - t0, 2)

report_k = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
report_k = [k for k in report_k if k in set(k_values.tolist())]
rows = [(int(k), float(d_of_k[np.searchsorted(k_values, k)])) for k in report_k]

results = {
    "claim": "2: substantial reward misspecification -> finite optimal k* then error increases",
    "paper": {"arxiv_id": "2512.19905", "openreview_id": "ANVg7NnupP",
              "title": "Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling"},
    "config": {"d": D, "n_train": N_TRAIN, "S": S_STD, "sigma": SIGMA, "gamma": GAMMA,
               "n_test": N_TEST, "eta_misalignment": ETA, "k_values_used": k_values.tolist()},
    "diagnostics": diag,
    "delta_of_k_report_points": rows,
    "delta_1": float(d_of_k[0]),
    "delta_kmax": float(d_of_k[-1]),
    "kstar": kstar,
    "delta_at_kstar": delta_star,
    "decreasing_up_to_kstar": decreasing_to_star,
    "increasing_after_kstar": increasing_after_star,
    "relative_rise_from_kstar_to_kmax": rise_relative,
    "floor_mean_reward_teacher_gap2": floor,
    "runtime_s": runtime_s,
}
(HERE / "results.json").write_text(json.dumps(results, indent=1))

print("== Claim 2: substantial reward-teacher misalignment (eta=%.3f) -> finite optimal k* ==" % ETA)
print("d=%d n_train=%d sigma=%.3f gamma=%.3f n_test=%d  runtime=%.2fs" % (D, N_TRAIN, SIGMA, GAMMA, N_TEST, runtime_s))
print("diagnostics: mean teacher-bias^2=%.6f  mean (reward-teacher gap)^2=%.6f  mean s^2=%.6f  max|t'|=%.3f"
      % (diag["mean_teacher_bias2"], diag["mean_reward_teacher_gap2"], diag["mean_s2"], diag["max_abs_tprime"]))
print()
print("  k     delta(k)")
for k, v in rows:
    marker = "  <-- k*" if k == kstar else ""
    print("%5d   %.6f%s" % (k, v, marker))
print()
print("k* = argmin_k delta(k) = %d  (delta(k*)=%.6f)" % (kstar, delta_star))
print("Monotone decreasing for k=1..k*: %s" % decreasing_to_star)
print("Monotone increasing for k=k*..%d: %s" % (int(k_values[-1]), increasing_after_star))
print("Relative rise delta(kmax)/delta(k*) - 1 = %.1f%%" % (100 * rise_relative))
print("Plateau (large k) = %.6f, approaching mismatch floor E[(y_R-y_T)^2] = %.6f" % (d_of_k[-1], floor))
print("[written] results.json")
