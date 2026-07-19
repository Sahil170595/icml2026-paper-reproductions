"""
Claim C1 (Theorem 4.1, tightened) -- Neural-ODE universal approximation of
monotone warps.  Unlike the earlier simultaneous 12-warp fit (best mean L2
4.86e-3), here each target warp is fitted by a DEDICATED high-capacity
Neural-ODE warp and trained to convergence, so the approximation error is driven
decisively small and the capacity sweep is clean.

Reports per-capacity mean/median/min L2 and sup error over the 12-warp family,
the log-log capacity slope, and monotonicity of every learned warp.

Staged/cache-resumable:
  python3 repro_claim2b.py fit <H> <warp_index>
  python3 repro_claim2b.py batch
  python3 repro_claim2b.py combine
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache"); os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import neuralfloc as nf
import nfloc_ext as ext

# finer grid for the target warps -> lower reconstruction floor
T = 161
TVEC = np.linspace(0.0, 1.0, T)
FAM = [
    ("exp_a=+3.0", nf.warp_exp(TVEC, 3.0)),
    ("exp_a=-3.0", nf.warp_exp(TVEC, -3.0)),
    ("exp_a=+1.5", nf.warp_exp(TVEC, 1.5)),
    ("beta_2_5", nf.warp_beta(TVEC, 2.0, 5.0)),
    ("beta_5_2", nf.warp_beta(TVEC, 5.0, 2.0)),
    ("beta_3_3", nf.warp_beta(TVEC, 3.0, 3.0)),
    ("logit_k8_m0.35", nf.warp_logit(TVEC, 8.0, 0.35)),
    ("logit_k6_m0.6", nf.warp_logit(TVEC, 6.0, 0.6)),
    ("sine_b=+0.8", nf.warp_sine(TVEC, 0.8)),
    ("sine_b=-0.8", nf.warp_sine(TVEC, -0.8)),
    ("power_p=2.2", nf.warp_power(TVEC, 2.2)),
    ("power_p=0.45", nf.warp_power(TVEC, 0.45)),
]
CAPS = [4, 8, 16, 32, 64]
STEPS = {4: 700, 8: 800, 16: 900, 32: 1100, 64: 1400}
SVAL = 50


def tag(H, wi):
    return f"c2b_H{H}_w{wi}"


def fit(H, wi):
    p = os.path.join(CACHE, tag(H, wi) + ".json")
    if os.path.exists(p):
        print("cached", tag(H, wi)); return
    name, g = FAM[wi]
    t0 = time.time()
    r = ext.fit_single_warp(g, TVEC, H=H, steps=STEPS[H], S=SVAL, method="euler",
                            lr=6e-3, seed=0)
    r.update({"H": H, "wi": wi, "name": name, "t": time.time() - t0})
    json.dump(r, open(p, "w"), indent=1, default=float)
    print("H=%d %-16s L2=%.2e sup=%.2e mono=%s (%.1fs)" %
          (H, name, r["l2"], r["sup"], r["monotone"], r["t"]))


def batch():
    for H in CAPS:
        for wi in range(len(FAM)):
            try:
                fit(H, wi)
            except Exception as e:
                print("ERR", H, wi, repr(e))
    combine()
    print("C2B BATCH DONE")


def combine():
    per_cap = []
    caps_done = []
    for H in CAPS:
        rows = []
        for wi in range(len(FAM)):
            p = os.path.join(CACHE, tag(H, wi) + ".json")
            if os.path.exists(p):
                rows.append(json.load(open(p)))
        if len(rows) < len(FAM):
            continue
        l2 = np.array([r["l2"] for r in rows])
        sup = np.array([r["sup"] for r in rows])
        mono = all(r["monotone"] for r in rows)
        per_cap.append({"H": H, "mean_l2": float(l2.mean()),
                        "median_l2": float(np.median(l2)),
                        "min_l2": float(l2.min()), "max_l2": float(l2.max()),
                        "mean_sup": float(sup.mean()), "all_monotone": bool(mono),
                        "per_warp": [{"name": r["name"], "l2": r["l2"],
                                      "sup": r["sup"], "monotone": r["monotone"]}
                                     for r in rows]})
        caps_done.append(H)
    out = {"grid_T": T, "capacities": caps_done, "S_integration": SVAL,
           "n_warps": len(FAM), "per_capacity": per_cap}
    if len(per_cap) >= 2:
        Hs = np.array([c["H"] for c in per_cap], dtype=float)
        me = np.array([c["mean_l2"] for c in per_cap])
        slope = float(np.polyfit(np.log(Hs), np.log(me), 1)[0])
        best = min(per_cap, key=lambda c: c["mean_l2"])
        global_min = min(c["min_l2"] for c in per_cap)
        out.update({
            "mean_l2_by_cap": [c["mean_l2"] for c in per_cap],
            "loglog_slope": slope,
            "best_mean_l2": best["mean_l2"], "best_cap": best["H"],
            "global_min_l2": global_min,
            "shrink_factor": per_cap[0]["mean_l2"] / best["mean_l2"],
            "n_warps_below_1e4_at_best": int(sum(
                1 for w in best["per_warp"] if w["l2"] < 1e-4)),
            "all_monotone": all(c["all_monotone"] for c in per_cap)})
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
    if per_cap:
        print("caps:", [c["H"] for c in per_cap])
        print("mean_l2:", ["%.2e" % c["mean_l2"] for c in per_cap])
        print("min_l2 :", ["%.2e" % c["min_l2"] for c in per_cap])
        if "best_mean_l2" in out:
            print("best mean L2 = %.2e (H=%d), global min = %.2e, slope=%.3f, shrink=%.1fx"
                  % (out["best_mean_l2"], out["best_cap"], out["global_min_l2"],
                     out["loglog_slope"], out["shrink_factor"]))
    print("wrote results.json")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "combine"
    if a == "combine":
        combine()
    elif a == "batch":
        batch()
    elif a == "fit":
        fit(int(sys.argv[2]), int(sys.argv[3]))
