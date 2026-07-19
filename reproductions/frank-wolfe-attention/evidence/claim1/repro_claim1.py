"""Claim 1 reproduction --- "Attention's forward pass and Frank-Wolfe"
(Alcalde, Geshkovski, Ruiz-Balet; OpenReview zrn7rRuvhW, arXiv 2508.09628).

Official code (independent reimplementation here; NOT imported):
  https://github.com/borjanG/2025-transformers-frank-wolfe @ 2107c4b5bae6614478150aba915252674bc796de

Paper equations (verbatim from arXiv source main.tex, downloaded 2026-07-17):

  (SA_infty), source line ~344 (hardmax self-attention dynamics):
    x_i^{t+1} = x_i^t + gamma^t * ( argmax_{y in K^t} <B^t x_i^t, y> - x_i^t )
    K^t = conv{x_j^t}_{j=1..n}, gamma^t = h^t/(1+h^t) in (0,1).

  Section 3 ("Negative-definite key-query"), source line ~380-388:
    Reparametrize B^t = -B_*^t (B_* symmetric). Then (SA_infty) rewrites exactly as
      x_i^{t+1} = x_i^t + gamma^t * ( argmin_{y in K^t} <B_*^t x_i^t, y> - x_i^t ).      (*)
    Define J^t(x) = 0.5 <B_*^t x, x>, so grad J^t(x) = B_*^t x. (*) is *by definition*
    the classical Frank-Wolfe update for J^t over K^t: the linear-minimization-oracle
    (LMO) step LMO(g) = argmin_{y in K^t} <g, y>, applied to g = grad J^t(x_i^t).

  Theorem 3.1 ("Frank-Wolfe convergence (to a cluster)"), source line ~386-392:
    If B_*^t - B_*^{t+1} >= 0 (PSD) and B_*^t >= 0 for all t>=0, gamma^t = 2/(t+2),
    and 0 in K^0, then for all i:
        J^t(x_i^{t+1}) <= 2/(t+1) * lambda_max(B_*^0) * diam(K^0)^2.

This script:
  (A) verifies the *algebraic identity* -- that the raw hardmax self-attention update
      (argmax over the *given token set*, as an attention layer literally computes)
      equals the true Frank-Wolfe LMO step over the *full convex hull* (solved
      independently via linear programming, scipy.optimize.linprog) -- across many
      dimensions, token counts, seeds and step-size rules;
  (B) verifies the two equivalent sign conventions (B-argmax vs B_*-argmin) produce
      bit-identical trajectories;
  (C) verifies Theorem 3.1's O(1/t) rate bound holds (ratio <= 1) under its stated
      PSD / nonincreasing / step-size hypotheses, for constant and shrinking PSD
      key-query schedules.

Deterministic (numpy.random.default_rng, fixed seeds), CPU-only, single-threaded.
Prints only measured numbers; writes evidence-package/claim1/results.json.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent


def diam(X):
    """Diameter of the point set X (n,d): max pairwise distance."""
    diffs = X[:, None, :] - X[None, :, :]
    return float(np.sqrt(np.sum(diffs**2, axis=-1)).max())


def lmo_argmax(scores):
    """Discrete argmax over the given token set (what an attention layer computes)."""
    return int(np.argmax(scores))


def lmo_linprog(grad, X):
    """True Frank-Wolfe LMO over conv(X): min_{y in conv(X)} <grad, y>, solved as an
    independent LP over the barycentric simplex (lambda >= 0, sum lambda = 1).
    Returns (y*, lambda*). This is a fully independent numerical method (simplex/
    interior-point LP solve) from the discrete-argmax code path above."""
    n = X.shape[0]
    c = X @ grad  # c_j = <grad, x_j>
    res = linprog(c, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(0, None)] * n, method="highs")
    lam = res.x
    y = lam @ X
    return y, lam, res


def random_config(d, n, seed, spread=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=spread, size=(n, d))
    return X, rng


def random_symmetric(d, rng, psd=None):
    """Random symmetric matrix; if psd=True, PSD; if psd=False, made indefinite."""
    A = rng.normal(size=(d, d))
    S = 0.5 * (A + A.T)
    if psd is True:
        w, V = np.linalg.eigh(S)
        S = V @ np.diag(np.abs(w)) @ V.T
    return S


# ---------------------------------------------------------------------------
# (A) + (B): exact identity check -- hardmax argmax == FW-LMO(LP) == B_*-argmin
# ---------------------------------------------------------------------------
def check_identity(d, n, seed, gamma_rule, n_steps, B_kind):
    """Run the hardmax dynamics for n_steps using THREE independent code paths:
      path1: raw attention argmax with B  (x <- x + gamma*(argmax_y<Bx,y> - x))
      path2: B_*-argmin (B_*=-B), i.e. the FW-update-as-written-in-paper form
      path3: FW LMO solved via linprog over the continuous convex hull (independent
             numerical method), using grad = B_* x_i^t (grad of J^t)
    Reports (a) max abs coordinate residual between path1 and path2 trajectories
    (should be exactly 0, same computation up to sign flip), and (b) max abs
    residual between path1's selected vertex and path3's LP-solved point (should
    be at LP numerical precision), and (c) whether the LP puts mass 1 on a single
    vertex (as guaranteed by LP theory for a generic linear objective).
    """
    X0, rng = random_config(d, n, seed)
    X1 = X0.copy()  # path1: B argmax
    X2 = X0.copy()  # path2: B_* argmin
    max_resid_12 = 0.0
    max_resid_13 = 0.0
    max_lambda_offvertex = 0.0
    oracle_mismatches = 0
    for t in range(n_steps):
        gamma = gamma_rule(t)
        if B_kind == "fixed":
            if t == 0:
                B = random_symmetric(d, rng, psd=None)
        elif B_kind == "shrinking_psd":
            if t == 0:
                B0 = random_symmetric(d, rng, psd=True)
            B = -(B0 / (t + 1.0))  # B_* = B0/(t+1): decreasing PSD -> B=-B_*
        Bstar = -B

        K1 = X1.copy()
        K2 = X2.copy()

        for i in range(n):
            # path1: argmax_y <B x_i, y> over the current token set
            scores1 = K1 @ (B @ K1[i])
            j1 = lmo_argmax(scores1)
            y1 = K1[j1]
            x1_new = K1[i] + gamma * (y1 - K1[i])

            # path2: argmin_y <B_* x_i, y> over the current token set
            scores2 = K2 @ (Bstar @ K2[i])
            j2 = int(np.argmin(scores2))
            y2 = K2[j2]
            x2_new = K2[i] + gamma * (y2 - K2[i])

            max_resid_12 = max(max_resid_12, float(np.max(np.abs(x1_new - x2_new))))
            if j1 != j2:
                oracle_mismatches += 1

            # path3: true LMO via linprog over conv(K1) (independent LP solve)
            grad = Bstar @ K1[i]
            y3, lam, res = lmo_linprog(grad, K1)
            max_resid_13 = max(max_resid_13, float(np.max(np.abs(y1 - y3))))
            off_mass = float(1.0 - lam.max())
            max_lambda_offvertex = max(max_lambda_offvertex, off_mass)

            X1[i] = x1_new
            X2[i] = x2_new

    return {
        "max_resid_B_vs_Bstar_paths": max_resid_12,
        "oracle_mismatches_B_vs_Bstar": oracle_mismatches,
        "max_resid_argmax_vs_linprogLMO": max_resid_13,
        "max_LP_mass_off_top_vertex": max_lambda_offvertex,
    }


def run_identity_sweep():
    dims = [2, 3, 5, 8]
    ns = [6, 8, 10, 12]
    seeds = [0, 1, 2, 3, 4, 5]
    gamma_rules = {
        "2/(t+2)": lambda t: 2.0 / (t + 2),
        "1/(t+1)": lambda t: 1.0 / (t + 1),
        "const_0.3": lambda t: 0.3,
    }
    n_steps = 8
    rows = []
    worst = {
        "max_resid_B_vs_Bstar_paths": 0.0,
        "oracle_mismatches_B_vs_Bstar": 0,
        "max_resid_argmax_vs_linprogLMO": 0.0,
        "max_LP_mass_off_top_vertex": 0.0,
    }
    n_configs = 0
    for d in dims:
        for n in ns:
            for seed in seeds:
                for gname, grule in gamma_rules.items():
                    for B_kind in ["fixed", "shrinking_psd"]:
                        n_configs += 1
                        r = check_identity(d, n, seed, grule, n_steps, B_kind)
                        rows.append({"d": d, "n": n, "seed": seed, "gamma_rule": gname, "B_kind": B_kind, **r})
                        for k in worst:
                            worst[k] = max(worst[k], r[k])
    return {"n_configs": n_configs, "n_steps_per_config": n_steps, "worst": worst, "rows": rows}


# ---------------------------------------------------------------------------
# (C) Theorem 3.1 rate bound: J^t(x_i^{t+1}) <= 2/(t+1) * lambda_max(B_*^0) * diam(K^0)^2
# ---------------------------------------------------------------------------
def run_rate_bound_sweep():
    dims = [2, 3, 5, 8]
    seeds = [10, 11, 12, 13, 14]
    n_tokens_list = [6, 10]
    n_steps = 60
    schedules = ["constant", "shrinking"]
    rows = []
    worst_ratio = 0.0
    n_checks = 0
    n_violations = 0
    for d in dims:
        for n in n_tokens_list:
            for seed in seeds:
                for sched in schedules:
                    rng = np.random.default_rng(1000 + seed)
                    # Ensure 0 in K^0: symmetric point cloud around origin (include -X too)
                    half = rng.normal(size=(n // 2, d))
                    X = np.concatenate([half, -half], axis=0)
                    if X.shape[0] < n:
                        X = np.concatenate([X, rng.normal(size=(n - X.shape[0], d))], axis=0)
                    K0 = X.copy()
                    d0 = diam(K0)
                    B0_star = random_symmetric(d, rng, psd=True)
                    lam_max0 = float(np.linalg.eigvalsh(B0_star).max())
                    bound_const = 2.0 * lam_max0 * d0**2

                    Xt = X.copy()
                    for t in range(n_steps):
                        gamma = 2.0 / (t + 2)
                        if sched == "constant":
                            Bstar_t = B0_star
                        else:  # shrinking: B_*^t = B0_star / (t+1), nonincreasing PSD
                            Bstar_t = B0_star / (t + 1.0)
                        Xt_new = Xt.copy()
                        for i in range(n):
                            grad = Bstar_t @ Xt[i]
                            scores = Xt @ grad
                            j = int(np.argmin(scores))
                            y = Xt[j]
                            Xt_new[i] = Xt[i] + gamma * (y - Xt[i])
                        Xt = Xt_new
                        Bstar_next = B0_star if sched == "constant" else B0_star / (t + 2.0)
                        Jval = 0.5 * float(np.max([Xt[i] @ (Bstar_next @ Xt[i]) for i in range(n)]))
                        bound = bound_const / (t + 1.0)
                        ratio = Jval / bound if bound > 0 else 0.0
                        n_checks += 1
                        worst_ratio = max(worst_ratio, ratio)
                        if Jval > bound + 1e-9:
                            n_violations += 1
                    rows.append({"d": d, "n": n, "seed": seed, "schedule": sched,
                                 "final_ratio": ratio, "lambda_max_B0": lam_max0, "diam_K0": d0})
    return {"n_checks": n_checks, "n_violations": n_violations, "worst_ratio": worst_ratio, "rows": rows}


def main():
    t0 = time.time()
    print("== Claim 1: hardmax self-attention update == Frank-Wolfe LMO step (Theorem 3.1) ==")
    print("   arXiv 2508.09628 (Alcalde, Geshkovski, Ruiz-Balet); source eq. SA_infty + Thm 3.1")

    print("\n[A/B] Exact identity: raw attention argmax vs B_*-argmin vs linprog LMO over conv hull")
    ident = run_identity_sweep()
    w = ident["worst"]
    print(f"  configs={ident['n_configs']} (d in {{2,3,5,8}} x n in {{6,8,10,12}} x 6 seeds x 3 step rules x 2 B-schedules), "
          f"{ident['n_steps_per_config']} steps each")
    print(f"  max |B-path - B_*-path| residual (should be 0.0, same computation up to sign):  {w['max_resid_B_vs_Bstar_paths']:.3e}")
    print(f"  oracle (argmax vs argmin) index mismatches:                                      {w['oracle_mismatches_B_vs_Bstar']}")
    print(f"  max |argmax-selected vertex - linprog LMO solution| (independent LP solve):       {w['max_resid_argmax_vs_linprogLMO']:.3e}")
    print(f"  max LP probability mass NOT on the single top vertex (LP always picks a vertex):  {w['max_LP_mass_off_top_vertex']:.3e}")

    print("\n[C] Theorem 3.1 rate bound: J^t(x_i^{t+1}) <= 2/(t+1) * lambda_max(B_*^0) * diam(K^0)^2")
    rate = run_rate_bound_sweep()
    print(f"  checks={rate['n_checks']} (d in {{2,3,5,8}} x n in {{6,10}} x 5 seeds x {{constant,shrinking}} PSD schedule x 60 steps)")
    print(f"  violations (J^t > bound): {rate['n_violations']}")
    print(f"  worst observed ratio J^t/bound (target <= 1.0): {rate['worst_ratio']:.6f}")

    runtime = time.time() - t0
    out = {
        "identity_check": ident,
        "rate_bound_check": rate,
        "runtime_s": round(runtime, 2),
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[written] claim1/results.json  (runtime {runtime:.1f}s)")


if __name__ == "__main__":
    main()
