"""Claim 1 reproduction --- "On Structured State-Space Duality"
(Hu, Zhang, ElSheikh, Wu, Liu; OpenReview `DKathyl3XN`, arXiv 2510.04944).

Scored claim: "Extends Structured State-Space Duality from the scalar-identity
case to general diagonal SSMs."

Paper targets reproduced (independent NumPy implementation, no paper code used):
  - Proposition 3.1 (Dao & Gu 2024, restated as the paper's background result):
    scalar-identity SSM  x_t = a x_{t-1} + b u_t,  y_t = c x_t
    is EXACTLY the 1-semiseparable masked-attention operator
    y = c*b * (M @ u),  M[t,s] = a^(t-s) for t>=s, else 0.
  - Section 4.1 "Structured State-Space Duality for General Diagonal SSMs":
    a diagonal SSM (state dim N, fixed diagonal A=diag(lambda_1..N)) is exactly
    the SUM of N rank-1 (1-SS) masked-attention heads
    M = sum_n C_n B_n * 1SS(lambda_n),  y = M @ u.
  - The paper's own numerical study (Appendix B.2, "Time-Varying Extension")
    extends this further to TIME-VARYING diagonal SSMs A_t = diag(lambda_{t,1..N}):
    this is the genuinely NEW result of Section 4.1 (Prop 3.1 only covers scalar
    a_t, not a per-mode-and-per-time diagonal). We stress-test exactly this case.

Design (deterministic, CPU-only, pure NumPy/SciPy, single-threaded BLAS):
  Part A - scalar-identity baseline (Prop 3.1): 1000 seeds x 3 horizons x 3 decays
           = 9000 runs.
  Part B - fixed-diagonal SSM (Section 4.1, N=2..8): 1000 seeds x 4 state widths
           = 4000 runs; also cross-checks generator rank == matrix rank == N.
  Part C - TIME-VARYING diagonal SSM (Section 4.1's full generality, N=4, T=32,
           per-step random A_t, B_t, C_t): 1500 seeds -- this is the headline
           "diagonal SSM" stress test named in the challenge (N=4, T up to 32).

Every number printed below is the actual stdout of this script; results are
also written to results.json next to this file.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent


def causal_mask(T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_t = np.arange(T)[:, None]
    idx_s = np.arange(T)[None, :]
    return (idx_t >= idx_s), idx_t, idx_s


def semiseparable_order(M: np.ndarray) -> int:
    """Definition 3.1's semiseparable rank: max rank of the corner blocks
    M[t:, :t] (entries strictly below the row-t/col-t cut, on or below the
    main diagonal), t = 1..T-1. NOTE: this is *not* np.linalg.matrix_rank(M)
    on the whole matrix -- a lower-triangular matrix with a nonzero diagonal
    is trivially full rank T as an ordinary matrix (its diagonal alone is a
    basis), which would make every SSM kernel "rank T" and is not the
    quantity the paper's theorems are about.
    """
    T = M.shape[0]
    return max((int(np.linalg.matrix_rank(M[t:, :t])) for t in range(1, T)), default=0)


# ---------------------------------------------------------------------------
# Part A: scalar-identity SSM ==  1-SS attention  (Proposition 3.1 / Dao & Gu Prop 3.1)
# ---------------------------------------------------------------------------
def part_a_scalar_identity() -> dict:
    seeds = range(1000)
    horizons = (20, 50, 100)
    decays = (0.5, 0.8, 0.95)
    b, c = 1.0, 1.0
    errors = []
    n_runs = 0
    for T in horizons:
        mask, t_idx, s_idx = causal_mask(T)
        diff = np.where(mask, t_idx - s_idx, 0)
        for a in decays:
            M = (a**diff) * mask
            for seed in seeds:
                rng = np.random.default_rng(1_000_000 * int(T) + int(round(a * 100)) * 10_000 + seed)
                u = rng.standard_normal(T)
                x = 0.0
                y = np.empty(T)
                for t in range(T):
                    x = a * x + b * u[t]
                    y[t] = c * x
                y_att = c * b * (M @ u)
                errors.append(float(np.max(np.abs(y - y_att))))
                n_runs += 1
    errors = np.asarray(errors)
    return {
        "description": "Prop 3.1: scalar-identity SSM recurrence vs c*b*(M@u), M[t,s]=a^(t-s)",
        "n_runs": n_runs,
        "seeds_per_config": len(list(seeds)),
        "horizons": list(horizons),
        "decays": list(decays),
        "max_abs_error": float(errors.max()),
        "mean_abs_error": float(errors.mean()),
        "failures_at_1e-12": int(np.sum(errors >= 1e-12)),
    }


# ---------------------------------------------------------------------------
# Part B: fixed-diagonal SSM == sum of N rank-1 (1-SS) attention heads (Sec. 4.1)
# ---------------------------------------------------------------------------
def part_b_fixed_diagonal() -> dict:
    T = 20
    seeds = range(1000)
    width_configs = {
        2: (0.5, 0.8),
        3: (0.4, 0.6, 0.9),
        5: (0.2, 0.4, 0.6, 0.8, 0.95),
        8: tuple(float(v) for v in np.round(np.linspace(0.15, 0.9, 8), 6)),
    }
    mask, t_idx, s_idx = causal_mask(T)
    diff = np.where(mask, t_idx - s_idx, 0)
    per_width = {}
    all_errors = []
    n_runs = 0
    for N, decays in width_configs.items():
        A_vals = np.asarray(decays, dtype=np.float64)
        errors = []
        rank_mismatches = 0
        for seed in seeds:
            rng = np.random.default_rng(2_000_000 + N * 100_000 + seed)
            u = rng.standard_normal(T)
            B = np.ones(N)
            C = np.ones(N)
            x = np.zeros(N)
            y = np.empty(T)
            for t in range(T):
                x = A_vals * x + B * u[t]
                y[t] = float(C @ x)
            M = np.zeros((T, T))
            for n in range(N):
                M += C[n] * B[n] * (A_vals[n] ** diff) * mask
            y_att = M @ u
            err = float(np.max(np.abs(y - y_att)))
            errors.append(err)
            all_errors.append(err)
            gen_rank = int(np.linalg.matrix_rank(np.column_stack([A_vals**k for k in range(T)])))
            ss_rank = semiseparable_order(M) if seed < 20 else None  # expensive; sample only
            if gen_rank != N or (ss_rank is not None and ss_rank != N):
                rank_mismatches += 1
            n_runs += 1
        per_width[str(N)] = {
            "decays": list(decays),
            "max_abs_error": float(np.max(errors)),
            "rank_mismatches_vs_N": rank_mismatches,
            "note": (
                "generator rank (Vandermonde of the decays) checked on all 1000 seeds; "
                "semiseparable order (corner-block rank of M, Definition 3.1) additionally "
                "checked on first 20 seeds since it costs O(T) rank calls per instance. "
                "Ordinary np.linalg.matrix_rank(M) is NOT used here -- it is trivially T "
                "because M is lower triangular with a nonzero diagonal."
            ),
        }
    all_errors = np.asarray(all_errors)
    return {
        "description": "Sec 4.1: fixed-diagonal SSM (N modes, distinct decays) vs sum of N 1-SS heads",
        "T": T,
        "n_runs": n_runs,
        "seeds_per_width": len(list(seeds)),
        "per_width": per_width,
        "max_abs_error": float(all_errors.max()),
        "failures_at_1e-12": int(np.sum(all_errors >= 1e-12)),
    }


# ---------------------------------------------------------------------------
# Part C: TIME-VARYING diagonal SSM == sum of N time-varying 1-SS heads
#         (Sec 4.1 in full generality; paper's own B.2 "time-varying extension")
# ---------------------------------------------------------------------------
def _time_varying_mask(A_vals: np.ndarray) -> np.ndarray:
    """mask[t, s, m] = prod_{k=s+1..t} A_vals[k, m] for t>=s, else 0. A_vals: (T,N)."""
    T, N = A_vals.shape
    mask = np.zeros((T, T, N), dtype=np.float64)
    for m in range(N):
        for t in range(T):
            mask[t, t, m] = 1.0
            prod = 1.0
            for s in range(t - 1, -1, -1):
                prod *= A_vals[s + 1, m]
                mask[t, s, m] = prod
    return mask


def part_c_time_varying(n_seeds: int = 1500, T: int = 32, N: int = 4) -> dict:
    errors = []
    matrix_ranks = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(3_000_000 + seed)
        A_vals = rng.uniform(0.3, 0.95, size=(T, N))
        B_mat = rng.uniform(-1.0, 1.0, size=(T, N))
        C_mat = rng.uniform(-1.0, 1.0, size=(T, N))
        u = rng.standard_normal(T)

        x = np.zeros(N)
        y = np.empty(T)
        for t in range(T):
            x = A_vals[t] * x + B_mat[t] * u[t]
            y[t] = float(C_mat[t] @ x)

        tv_mask = _time_varying_mask(A_vals)  # (T,T,N)
        contrib = tv_mask * C_mat[:, None, :] * B_mat[None, :, :]
        M = np.sum(contrib, axis=2)
        y_att = M @ u

        errors.append(float(np.max(np.abs(y - y_att))))
        if seed < 50:
            matrix_ranks.append(int(np.linalg.matrix_rank(M)))

    errors = np.asarray(errors)
    return {
        "description": (
            "Sec 4.1 (full generality) / paper App. B.2 time-varying extension: "
            "diagonal SSM with per-timestep A_t=diag(lambda_t,1..N), B_t, C_t vs "
            "sum of N time-varying 1-SS masked-attention heads"
        ),
        "T": T,
        "N": N,
        "n_runs": n_seeds,
        "max_abs_error": float(errors.max()),
        "mean_abs_error": float(errors.mean()),
        "median_abs_error": float(np.median(errors)),
        "failures_at_1e-12": int(np.sum(errors >= 1e-12)),
        "sample_matrix_ranks_first_50_seeds": matrix_ranks,
    }


def main() -> None:
    t0 = time.time()
    print("== Claim 1: scalar-identity SSD extends exactly to (fixed and time-varying) diagonal SSMs ==")

    print("\n[A] Scalar-identity SSM (Prop 3.1) vs 1-SS attention -- 1000 seeds x T in {20,50,100} x a in {0.5,0.8,0.95}")
    a_res = part_a_scalar_identity()
    print(json.dumps({k: v for k, v in a_res.items() if k != "per_width"}, sort_keys=True))

    print("\n[B] Fixed-diagonal SSM (Sec 4.1) vs sum of N 1-SS heads -- 1000 seeds x N in {2,3,5,8}, T=20")
    b_res = part_b_fixed_diagonal()
    print(json.dumps({k: v for k, v in b_res.items() if k != "per_width"}, sort_keys=True))
    for width, info in b_res["per_width"].items():
        print(f"    N={width}: decays={info['decays']} max_err={info['max_abs_error']:.3e} rank_mismatches={info['rank_mismatches_vs_N']}")

    print("\n[C] TIME-VARYING diagonal SSM (N=4, T=32) vs sum of N time-varying 1-SS heads -- 1500 seeds")
    c_res = part_c_time_varying(n_seeds=1500, T=32, N=4)
    print(json.dumps({k: v for k, v in c_res.items() if k != "sample_matrix_ranks_first_50_seeds"}, sort_keys=True))

    total_runs = a_res["n_runs"] + b_res["n_runs"] + c_res["n_runs"]
    overall_max_error = max(a_res["max_abs_error"], b_res["max_abs_error"], c_res["max_abs_error"])
    elapsed = time.time() - t0

    summary = {
        "paper": "arxiv:2510.04944 (OpenReview DKathyl3XN) -- On Structured State-Space Duality",
        "claim": "Extends Structured State-Space Duality from the scalar-identity case to general diagonal SSMs",
        "total_runs": total_runs,
        "overall_max_abs_error": overall_max_error,
        "part_a_scalar_identity": a_res,
        "part_b_fixed_diagonal": b_res,
        "part_c_time_varying_diagonal": c_res,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "elapsed_seconds": elapsed,
    }
    out = HERE / "results.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"\n[summary] total_runs={total_runs} overall_max_abs_error={overall_max_error:.3e} elapsed={elapsed:.1f}s")
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
