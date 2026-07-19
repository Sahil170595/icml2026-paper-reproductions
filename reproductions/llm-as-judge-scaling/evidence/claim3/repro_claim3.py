"""Claim 3 -- "In the best-of-k limit with the teacher as reward,
generalization error decays as Theta(1/k^2)." (Halder & Pehlevan,
arXiv:2512.19905, ICML 2026, OpenReview ANVg7NnupP.)

Independent NumPy/SciPy reproduction (see ../model.py). eta=0, i.e. w_R=w_T
identically (reward IS the teacher). Computes delta(k) exactly
(order-statistics quadrature) on a log-spaced k grid from 2 to 20000 and
fits the log-log slope; also derives the closed-form k->infinity
coefficient independently (Laplace / extreme-value argument, see below) and
compares it to the fitted prefactor.

Closed-form check (derived independently, not taken from the paper's code):
as k->infinity the density of the selected standardized draw concentrates
at the target t'=b' (since reward==teacher here) with local rate
lambda = k*phi(b'); the squared deviation of the nearest of k iid N(0,1)
draws from a fixed point has E[gap^2] -> 1/(2*lambda^2) (nearest-neighbor
spacing of a rate-lambda Poisson process on the line), so in original units
   delta(x,k) ~ s(x)^2 / (2*k^2*phi(b'(x))^2) = (pi*s(x)^2/k^2) * exp(b'(x)^2)
   delta(k)   ~ (pi/k^2) * E_x[ s(x)^2 * exp(b'(x)^2) ]  =: C_theory / k^2
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import build_fixed_posterior, eval_regime, test_stats, make_wR, N_TEST, D, N_TRAIN, S_STD, SIGMA, GAMMA, TAU_GRID, SEED_TEST, SEED_MISALIGN  # noqa: E402

HERE = Path(__file__).resolve().parent

ETA = 0.0

t0 = time.time()
w_T, w_post, Lam_inv = build_fixed_posterior()

k_values = np.unique(np.round(np.geomspace(2, 20000, 70)).astype(int))
d_of_k, (m, s, yT, yR), diag = eval_regime(ETA, k_values, w_T, w_post, Lam_inv)

bp = (yT - m) / s
C_theory = float(np.pi * np.mean(s**2 * np.exp(bp**2)))

mask = (k_values >= 20) & (k_values <= 5000) & (d_of_k > 0)
slope, intercept = np.polyfit(np.log(k_values[mask]), np.log(d_of_k[mask]), 1)
C_fit = float(np.exp(intercept))

# also fit restricted to a later window to show the slope is stable, not a
# transient artifact of the fitting window
mask2 = (k_values >= 200) & (k_values <= 20000) & (d_of_k > 0)
slope2, intercept2 = np.polyfit(np.log(k_values[mask2]), np.log(d_of_k[mask2]), 1)

runtime_s = round(time.time() - t0, 2)

report_k = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
rows = []
seen_idx = set()
for k in report_k:
    idx = int(np.argmin(np.abs(k_values - k)))
    if idx not in seen_idx:
        seen_idx.add(idx)
        rows.append((int(k_values[idx]), float(d_of_k[idx])))

results = {
    "claim": "3: teacher-as-reward -> generalization error decays as Theta(1/k^2)",
    "paper": {"arxiv_id": "2512.19905", "openreview_id": "ANVg7NnupP",
              "title": "Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling"},
    "config": {"d": D, "n_train": N_TRAIN, "S": S_STD, "sigma": SIGMA, "gamma": GAMMA,
               "n_test": N_TEST, "eta_misalignment": ETA, "k_values_used": k_values.tolist()},
    "diagnostics": diag,
    "delta_of_k_report_points": rows,
    "loglog_slope_k_20_to_5000": float(slope),
    "loglog_amplitude_C_fit_20_to_5000": C_fit,
    "loglog_slope_k_200_to_20000": float(slope2),
    "C_theory_closed_form": C_theory,
    "C_fit_over_C_theory": float(C_fit / C_theory),
    "runtime_s": runtime_s,
}
(HERE / "results.json").write_text(json.dumps(results, indent=1))

print("== Claim 3: teacher-as-reward (eta=0) -> Theta(1/k^2) best-of-k decay ==")
print("d=%d n_train=%d sigma=%.3f gamma=%.3f n_test=%d  runtime=%.2fs" % (D, N_TRAIN, SIGMA, GAMMA, N_TEST, runtime_s))
print()
print("     k     delta(k)")
for k, v in rows:
    print("%7d   %.8e" % (k, v))
print()
print("log-log slope of delta(k) vs k, fit over k in [20,5000]   = %.4f  (theory: -2)" % slope)
print("log-log slope of delta(k) vs k, fit over k in [200,20000] = %.4f  (theory: -2, stability check)" % slope2)
print("fitted amplitude C (delta ~ C/k^2), k in [20,5000]         = %.6f" % C_fit)
print("closed-form C_theory = pi * E_x[s(x)^2 * exp(b'(x)^2)]     = %.6f" % C_theory)
print("C_fit / C_theory = %.4f" % (C_fit / C_theory))
print("[written] results.json")
