"""
Claim 2 reproduction -- the algorithm models utility as an unknown LINEAR function
of item-agent features and LEARNS the parameter from limited observed utilities.

Paper: Verma, Saha, Yokoo, Low, "Keep Everyone Happy: Online Fair Division of
Numerous Items with Few Copies" (ICML 2026 / arXiv 2408.12845).
Utility model: u_{t,n} = m_{t,n}^T theta*, with m_{t,n} the concatenated
item-agent feature vector (item and agent features ~ U(0,10)) and theta* unknown.
OFD-UCB keeps a ridge estimate theta_hat_t = M_t^{-1} b_t updated ONLY from the
selected agent's noisy reward y_t = m_{t,n_t}^T theta* + eta (R-sub-Gaussian). Only
ONE of the N agents' utilities is observed per arriving item -> "few copies /
limited observed utilities". The linear-feature model is what lets the learner
estimate utilities for ALL item-agent pairs, including never-selected ones.

TARGET / COMPARISON RULE (falsifiable):
  (a) RECOVERY: ||theta_hat_t - theta*|| -> 0 as T grows; the log-log decay
      exponent p (err ~ t^{-p}) must be > 0 (statistically consistent; the
      OLS/ridge rate is p ~ 0.5).
  (b) GENERALISATION: with theta_hat the held-out utility-prediction RMSE over a
      FIXED pool of unseen item-agent pairs must fall to the noise floor sigma,
      and be far below a no-learning baseline (predict 0) and a shuffled-feedback
      control (ridge fit on permuted rewards, which carries no signal).
  (c) The learning-driven allocation (OFD-UCB) must OUTPERFORM the no-learning
      allocation baseline (OFD-Uniform) in cumulative regret.
  FALSIFIED if the recovery error does not decrease (p<=0), or the learned
  held-out RMSE is not below the no-learning / shuffled controls, or OFD-UCB does
  not beat OFD-Uniform.

Independent NumPy implementation; deterministic; CPU-only; single BLAS thread.
"""
import json, os, time
from pathlib import Path
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
OUT = Path(__file__).with_name("results.json")


def make_pairs(rng, P, dm, dn, N, agent_feats):
    """P items x N agents -> normalized concatenated item-agent feature matrix."""
    items = rng.uniform(0, 10, size=(P, dm))
    it = np.repeat(items, N, axis=0)
    ag = np.tile(agent_feats, (P, 1))
    M = np.concatenate([it, ag], axis=1)
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def run(seed, T, dm, dn, N, sigma=0.1, lam=1.0, P_test=2000, algo="ucb", shuffle=False):
    rng = np.random.default_rng(seed)
    d = dm + dn
    theta = rng.standard_normal(d); theta /= np.linalg.norm(theta)   # ||theta*||=S=1
    agent_feats = rng.uniform(0, 10, size=(N, dn))                   # fixed agent features
    # fixed held-out test pool of unseen item-agent pairs
    tg = np.random.default_rng(90000 + seed)
    Mtest = make_pairs(tg, P_test, dm, dn, N, agent_feats)
    u_test = Mtest @ theta
    rms_u = float(np.sqrt(np.mean(u_test ** 2)))                     # no-learning (predict 0) RMSE

    Ainv = (1.0 / lam) * np.eye(d); bvec = np.zeros(d); th = np.zeros(d)
    reg = np.empty(T); cum = 0.0; delta = 0.05
    ckpts = [c for c in [250, 500, 1000, 2000, 4000, T] if c <= T]
    rec = {}
    for t in range(1, T + 1):
        item = rng.uniform(0, 10, size=dm)
        M = np.concatenate([np.broadcast_to(item, (N, dm)), agent_feats], axis=1)
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        u = M @ theta; astar = int(np.argmax(u))
        if algo == "uniform":
            a = int(rng.integers(N))
        else:
            alpha = sigma * np.sqrt(d * np.log((1.0 + t / lam) / delta)) + np.sqrt(lam)
            quad = np.einsum("ij,jk,ik->i", M, Ainv, M)
            a = int(np.argmax(M @ th + alpha * np.sqrt(np.maximum(quad, 0.0))))
        x = M[a]
        y = u[a] + sigma * rng.standard_normal()
        if shuffle:                       # control: feedback carries no signal
            y = sigma * rng.standard_normal() + rms_u * rng.standard_normal()
        cum += u[astar] - u[a]; reg[t - 1] = cum
        Ax = Ainv @ x; Ainv -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        bvec += y * x; th = Ainv @ bvec
        if t in ckpts:
            err = float(np.linalg.norm(th - theta))
            rmse = float(np.sqrt(np.mean((Mtest @ th - u_test) ** 2)))
            rec[str(t)] = {"theta_err": err, "heldout_rmse": rmse}
    return {"reg": reg, "rec": rec, "rms_u": rms_u, "final_theta_err": float(np.linalg.norm(th - theta))}


def avg_reg(seeds, T, dm, dn, N, algo):
    acc = np.zeros(T)
    for s in seeds:
        acc += run(s, T, dm, dn, N, algo=algo)["reg"]
    return acc / len(seeds)


def avg_curves(seeds, T, dm, dn, N, shuffle=False):
    errs, rmses, keys, rmsu = {}, {}, None, []
    for s in seeds:
        r = run(s, T, dm, dn, N, algo="ucb", shuffle=shuffle)
        keys = list(r["rec"].keys())
        for k in keys:
            errs.setdefault(k, []).append(r["rec"][k]["theta_err"])
            rmses.setdefault(k, []).append(r["rec"][k]["heldout_rmse"])
        rmsu.append(r["rms_u"])
    ea = {k: float(np.mean(errs[k])) for k in keys}
    ra = {k: float(np.mean(rmses[k])) for k in keys}
    return ea, ra, float(np.mean(rmsu))


def decay_exponent(err_by_t):
    ts = np.array([int(k) for k in err_by_t], float)
    ev = np.array([err_by_t[k] for k in err_by_t], float)
    half = ts >= ts[len(ts) // 2]
    slope, _ = np.polyfit(np.log(ts[half]), np.log(ev[half]), 1)
    return float(-slope)   # p, so err ~ t^{-p}


def main():
    t0 = time.time(); seeds = [0, 1, 2, 3, 4]
    T, dm, dn, N = 8000, 4, 4, 10; sigma = 0.1
    print(f"[Claim 2] linear-utility recovery  T={T} d={dm+dn} (dm={dm},dn={dn}) N={N} sigma={sigma} seeds={seeds}")
    print(f"  per round only 1/{N} = {1.0/N:.2f} of the arriving item's agent utilities is observed (few copies)")

    err_learn, rmse_learn, rms_u = avg_curves(seeds, T, dm, dn, N, shuffle=False)
    err_shuf,  rmse_shuf,  _     = avg_curves(seeds, T, dm, dn, N, shuffle=True)

    print("\n  t       ||theta_hat-theta*||   held-out RMSE (learned) | RMSE shuffled-ctrl | RMSE no-learn(0)")
    for k in err_learn:
        print(f"  {int(k):<6d}  {err_learn[k]:10.4f}          {rmse_learn[k]:10.4f}        | {rmse_shuf[k]:10.4f}       | {rms_u:8.4f}")

    p = decay_exponent(err_learn)
    final_err = err_learn[list(err_learn)[-1]]
    final_rmse = rmse_learn[list(rmse_learn)[-1]]
    final_rmse_shuf = rmse_shuf[list(rmse_shuf)[-1]]

    # allocation quality: learning (UCB) vs no-learning (Uniform)
    r_ucb = avg_reg(seeds, T, dm, dn, N, "ucb")
    r_uni = avg_reg(seeds, T, dm, dn, N, "uniform")
    print(f"\n  recovery decay exponent p (err ~ t^-p, 2nd half) = {p:.3f}  (target > 0, OLS rate ~0.5)")
    print(f"  final ||theta_hat-theta*|| = {final_err:.4f}   final held-out RMSE = {final_rmse:.4f} (noise floor sigma={sigma})")
    print(f"  shuffled-feedback control held-out RMSE = {final_rmse_shuf:.4f} (stays ~ RMS(u)={rms_u:.3f}: no signal)")
    print(f"  allocation: OFD-UCB regret={r_ucb[-1]:.2f}  vs  OFD-Uniform regret={r_uni[-1]:.2f}  ({r_uni[-1]/max(r_ucb[-1],1e-9):.0f}x)")

    recovers = p > 0.05 and final_err < 0.5 * err_learn[list(err_learn)[0]]
    generalises = final_rmse < 0.25 * rms_u and final_rmse < 0.5 * final_rmse_shuf
    beats = r_ucb[-1] < r_uni[-1] / 10.0
    verified = recovers and generalises and beats
    print(f"\n  RECOVERY (p>0, err halves)? {recovers}   GENERALISES (RMSE<<no-learn & shuffled)? {generalises}   BEATS no-learning alloc? {beats}")
    print(f"  CLAIM 2 VERIFIED? {verified}")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    res = {
        "claim": "The algorithm models utility as an unknown linear function of item-agent features and learns theta* from limited observed (selected-arm) utilities.",
        "target_rule": "(a) ||theta_hat-theta*||->0 with decay exponent p>0 (OLS rate ~0.5); (b) held-out utility RMSE -> noise floor sigma and far below no-learning (predict-0) and shuffled-feedback controls; (c) OFD-UCB beats OFD-Uniform in regret. Falsified if p<=0, or learned RMSE not below controls, or UCB does not beat Uniform.",
        "config": {"T": T, "dm": dm, "dn": dn, "d": dm+dn, "N": N, "sigma_R": sigma, "lambda": 1.0, "seeds": seeds,
                    "observed_fraction_per_item": 1.0/N, "held_out_pool_items": 2000},
        "recovery": {"theta_err_by_t": err_learn, "decay_exponent_p": p, "final_theta_err": final_err},
        "generalisation": {"heldout_rmse_learned_by_t": rmse_learn, "final_rmse_learned": final_rmse,
                            "final_rmse_shuffled_control": final_rmse_shuf, "no_learning_predict0_rmse": rms_u,
                            "noise_floor_sigma": sigma},
        "allocation": {"ofd_ucb_regret": float(r_ucb[-1]), "ofd_uniform_regret": float(r_uni[-1]),
                        "ratio_uni_over_ucb": float(r_uni[-1]/max(r_ucb[-1],1e-9))},
        "verdict": {"recovers": bool(recovers), "generalises": bool(generalises),
                     "beats_no_learning": bool(beats), "verified": bool(verified)},
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
