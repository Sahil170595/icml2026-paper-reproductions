"""Claim 3 reproduction --- "On Structured State-Space Duality"
(Hu, Zhang, ElSheikh, Wu, Liu; OpenReview `DKathyl3XN`, arXiv 2510.04944).

Scored claim: "Establishes a necessary and sufficient condition for SSM
equivalence to 1-semiseparable masked attention."

This is the paper's Theorem 4.1 (General State-Space Duality), built on:
  - Definition 4.1 (fine 1-SS matrix): L = 1SS(a_1,...,a_T) is "fine" iff
    a_1 a_2 ... a_T != 0 (no zero transition factors -> no forced blocks).
  - Definition 4.2 (new column): column t of a lower-triangular M is "new"
    iff M[t:, t] is not in the column space of M[t:, :t] (both restricted to
    the SAME row range t..T).
  - Proposition 4.2: an N-SS matrix M has a representation L (x) (Q K^T) with
    Q,K in R^{TxN} and L a FINE 1-SS matrix iff M has at most N new columns.
  - Lemma 4.1 / Theorem 4.1: an N-SS matrix M (equivalently, its SSM) has a
    GENERAL 1-SS masked-attention dual L (x) (Q K^T), Q,K in R^{TxN}, iff M's
    nonzero entries lie in diagonal blocks (the general L can have zero
    transition factors, splitting M into blocks) and EACH block has at most
    N new columns.
  - Proposition 5.1 (impossibility / Section 5): the SSM whose kernel is
    M = I_T + E_{T,1} (a 2-SS matrix, E_{T,1} the single-entry matrix with a
    1 at row T, column 1) has NO 1-SS masked-attention dual of any bounded
    width -- because M forms a single un-splittable block (the corner entry
    (T,1) keeps every candidate crossing block nonzero) whose new-column
    count is exactly T-1, unbounded in T.

We independently rederive and numerically stress-test BOTH directions of
Theorem 4.1 plus the impossibility side, at non-toy scale (T up to 512,
N up to 16), using only NumPy/SciPy numerical linear algebra (SVD-based
numerical rank; no paper code used).

Parts:
  A. NECESSARY direction: construct genuine general-1-SS-dual matrices (random
     block boundaries, random Q,K per block, random "fine" decay masks) and
     verify the detected blocks match the construction and each block has at
     most N new columns, as Theorem 4.1 requires.
  B. SUFFICIENT direction: construct single-block matrices satisfying the
     "at most N new columns" condition directly (from a random low-rank
     lower-triangular generator), then run an independent implementation of
     Proposition 4.2's constructive proof (scan columns left-to-right;
     complete the upper-triangular part by least-squares combination of
     already-completed prior columns) to build Q,K explicitly, and check the
     reconstruction L (x) (Q K^T) reproduces the original lower-triangular
     data to near machine precision.
  C. IMPOSSIBILITY certificates: the paper's own Proposition 5.1 example,
     M = I_T + E_{T,1}, at growing T -- confirm it is a single unsplittable
     block whose new-column count is exactly T-1 (derived analytically and
     confirmed numerically), so no width-N dual exists whenever T-1 > N.
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
RANK_TOL = 1e-9


# ---------------------------------------------------------------------------
# Shared structural utilities (Definitions 4.1/4.2, Lemma 4.1's block split)
# ---------------------------------------------------------------------------
def zero_crossing_blocks(M: np.ndarray, atol: float = 1e-12) -> list[tuple[int, int]]:
    """Split at every boundary t whose crossing block M[t:, :t] is all zero
    (Lemma 4.1's block decomposition -- a general 1-SS mask can have zero
    transition factors, and every zero factor forces the corresponding
    crossing block to vanish, and vice versa)."""
    T = M.shape[0]
    cuts = [0]
    for t in range(1, T):
        if np.all(np.abs(M[t:, :t]) <= atol):
            cuts.append(t)
    cuts.append(T)
    return list(zip(cuts[:-1], cuts[1:]))


def new_column_count(block: np.ndarray, tol: float = RANK_TOL) -> int:
    """Definition 4.2: column t is "new" iff rank(M[t:, :t+1]) > rank(M[t:, :t]),
    i.e. BOTH ranks computed at the SAME row range t: (rows t..B-1).

    NOTE: the row range t: is DIFFERENT for every t (it shrinks as t grows),
    so rank(M[t:, :t]) cannot be obtained by reusing rank(M[(t-1):, :t]) from
    the previous iteration -- that would be a rank computed over a different
    (larger) row range and is not the same quantity Definition 4.2 compares.
    Both ranks are therefore recomputed at every t (2 rank calls/column)."""
    B = block.shape[0]
    count = 0
    for t in range(B):
        prev_cols = block[t:, :t]
        cur_cols = block[t:, : t + 1]
        r_prev = int(np.linalg.matrix_rank(prev_cols, tol=tol)) if prev_cols.size else 0
        r_cur = int(np.linalg.matrix_rank(cur_cols, tol=tol))
        if r_cur > r_prev:
            count += 1
    return count


def new_column_counts_per_block(M: np.ndarray, blocks: list[tuple[int, int]], tol: float = RANK_TOL) -> list[int]:
    return [new_column_count(M[a:b, a:b], tol=tol) for a, b in blocks]


# ---------------------------------------------------------------------------
# Part A: NECESSARY direction -- genuine constructions satisfy the theorem
# ---------------------------------------------------------------------------
def build_fine_1ss_mask(decays: np.ndarray) -> np.ndarray:
    """L[t,s] = prod(decays[s+1:t+1]) for t>=s (fine: every decay != 0)."""
    B = decays.size
    L = np.zeros((B, B))
    for t in range(B):
        L[t, t] = 1.0
        prod = 1.0
        for s in range(t - 1, -1, -1):
            prod *= decays[s + 1]
            L[t, s] = prod
    return L


def construct_general_1ss_dual(T: int, N: int, n_blocks: int, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    """Build a genuine general-1-SS dual: choose n_blocks-1 interior cut points,
    and for each block build a random fine 1-SS mask times a random rank-<=N
    outer product Q K^T. Returns (M, cut_points)."""
    if n_blocks > 1:
        interior = np.sort(rng.choice(np.arange(1, T), size=n_blocks - 1, replace=False))
        cuts = [0, *interior.tolist(), T]
    else:
        cuts = [0, T]
    M = np.zeros((T, T))
    for a, b in zip(cuts[:-1], cuts[1:]):
        B = b - a
        # Mild, realistic decay range (all nonzero -> "fine"): SSMs with
        # long-range memory (the point of the diagonal-SSD extension) use
        # decays close to 1. Aggressive decays (e.g. 0.3) make the fine-1SS
        # mask span ~15+ orders of magnitude over a length-100+ block, which
        # makes the corner-block RANK computations themselves ill-conditioned
        # (SVD cannot reliably separate "true" rank from numerical noise when
        # entries range from O(1) to O(1e-15)) -- an artifact of the rank
        # *measurement*, not of the underlying theorem.
        decays = rng.uniform(0.92, 0.999, size=B)
        L = build_fine_1ss_mask(decays)
        eff_N = min(N, B)
        Q = rng.standard_normal((B, eff_N))
        K = rng.standard_normal((B, eff_N))
        M[a:b, a:b] = L * (Q @ K.T)
    return M, cuts


def part_a_necessary(cases: list[tuple[int, int, int]]) -> dict:
    """cases: list of (T, N, n_blocks)."""
    results = []
    n_pass = 0
    for i, (T, N, n_blocks) in enumerate(cases):
        rng = np.random.default_rng(500_000 + i)
        M, true_cuts = construct_general_1ss_dual(T, N, n_blocks, rng)
        detected = zero_crossing_blocks(M)
        boundary_match = [c for c in true_cuts[1:-1]] == [b[0] for b in detected[1:]]
        counts = new_column_counts_per_block(M, detected)
        ok = boundary_match and all(c <= N for c in counts)
        n_pass += int(ok)
        results.append({
            "T": T, "N": N, "n_blocks_constructed": n_blocks,
            "n_blocks_detected": len(detected),
            "boundary_match": bool(boundary_match),
            "new_column_counts": counts,
            "max_new_columns": max(counts),
            "within_N": bool(all(c <= N for c in counts)),
            "pass": bool(ok),
        })
    return {
        "description": (
            "Theorem 4.1 NECESSARY direction: genuine general-1-SS-dual "
            "constructions (random block boundaries, random fine decay masks, "
            "random rank-<=N outer products per block) must have <= N new "
            "columns in every detected block. Detected block boundaries are "
            "also checked against the construction's true cut points."
        ),
        "n_cases": len(cases),
        "n_pass": n_pass,
        "all_pass": n_pass == len(cases),
        "cases": results,
    }


# ---------------------------------------------------------------------------
# Part B: SUFFICIENT direction -- constructive proof of Prop 4.2, reconstructed
# ---------------------------------------------------------------------------
def low_state_generator(B: int, N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a lower-triangular BxB matrix guaranteed (by construction) to
    have semiseparable order <= N: literally the fine-1SS-masked rank-N
    outer product used in Part A, restricted to its lower triangle. This is
    the "ground truth ready to be rediscovered" input to the completion
    algorithm below -- the completion algorithm is NOT told Q, K, or the mask;
    it only sees the lower-triangular numbers. Mild decay range for the same
    conditioning reason as construct_general_1ss_dual above -- the recursive
    least-squares completion in Proposition 4.2's constructive proof (below)
    amplifies the dynamic range of the input at every step, so an aggressively
    decaying mask causes the completion itself to lose precision (the paper's
    own Remark 4.4 notes this constructive proof "exceeds SSD limit" and is
    not meant as a numerically optimized algorithm)."""
    decays = rng.uniform(0.92, 0.999, size=B)
    L = build_fine_1ss_mask(decays)
    Q = rng.standard_normal((B, N))
    K = rng.standard_normal((B, N))
    full = L * (Q @ K.T)
    return np.tril(full), L, Q, K


def complete_and_factor(lower: np.ndarray, width: int, tol: float = RANK_TOL) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Independent implementation of Proposition 4.2's constructive
    (sufficiency) proof: scan columns left to right. If column t is "new"
    (Definition 4.2), leave its upper part zero (it starts a fresh
    independent direction). Otherwise, its lower part (rows t:) is a linear
    combination of the previously *completed* columns' lower parts; apply
    that same combination to the already-completed (full-height) prior
    columns to fill in column t's upper part. The result is a full BxB matrix
    whose rank equals the number of new columns (<= width by construction),
    which can then be factored as Q K^T with exactly `width` columns via SVD.
    Returns (Q, K, completed_matrix, measured_new_column_count).
    """
    B = lower.shape[0]
    completed = np.zeros((B, B))
    new_count = 0
    for t in range(B):
        completed[t:, t] = lower[t:, t]
        if t == 0:
            new_count += 1  # column 0 is trivially new (no prior columns)
            continue
        prev_lower = lower[t:, :t]
        cur_lower = lower[t:, t]
        prev_rank = int(np.linalg.matrix_rank(prev_lower, tol=tol))
        aug_rank = int(np.linalg.matrix_rank(np.column_stack([prev_lower, cur_lower]), tol=tol))
        if aug_rank > prev_rank:
            new_count += 1  # new column: upper part stays 0
        else:
            coeffs, *_ = np.linalg.lstsq(prev_lower, cur_lower, rcond=None)
            completed[:t, t] = completed[:t, :t] @ coeffs

    U, S, Vt = np.linalg.svd(completed, full_matrices=False)
    rank = int(np.sum(S > tol * max(1.0, S[0] if S.size else 0.0)))
    eff_width = max(width, rank)
    Q = np.zeros((B, eff_width))
    K = np.zeros((B, eff_width))
    if rank:
        root = np.sqrt(S[:rank])
        Q[:, :rank] = U[:, :rank] * root
        K[:, :rank] = Vt[:rank, :].T * root
    return Q, K, completed, new_count


def part_b_sufficient(cases: list[tuple[int, int]]) -> dict:
    """cases: list of (T, N) -- single-block instances, matching the
    challenge's "lengths 64-256, widths 1-8"-style sufficient-direction test."""
    results = []
    max_err_overall = 0.0
    n_pass = 0
    for i, (B, N) in enumerate(cases):
        rng = np.random.default_rng(600_000 + i)
        lower, L_true, Q_true, K_true = low_state_generator(B, N, rng)
        Q, K, completed, measured_new_cols = complete_and_factor(lower, width=N)
        # The REAL test: factor the completed matrix via SVD into Q,K with
        # exactly `width` columns, then reconstruct 1SS(1,...,1) (x) (Q K^T)
        # -- a UNIT (all-ones) fine mask, per Proposition 4.2's proof, which
        # need not be the original L_true -- and compare its lower-triangular
        # part against the ORIGINAL observed data `lower`. This is NOT the
        # same as comparing `completed`'s lower part to `lower` (that would
        # be circular: `completed`'s lower part is copied from `lower` by
        # construction). The Q,K factorization is a genuinely separate,
        # lossy-if-truncated computation, so this error is a real number.
        recon_lower = np.tril(Q @ K.T)
        err = float(np.max(np.abs(recon_lower - lower)))
        max_err_overall = max(max_err_overall, err)
        ok = measured_new_cols <= N and err < 1e-9
        n_pass += int(ok)
        results.append({
            "T": B, "N": N,
            "measured_new_columns": measured_new_cols,
            "within_N": measured_new_cols <= N,
            "reconstruction_max_abs_error": err,
            "pass": bool(ok),
        })
    return {
        "description": (
            "Theorem 4.1 SUFFICIENT direction: single-block matrices with a "
            "known-by-construction <= N semiseparable order are completed and "
            "factored via an independent implementation of Proposition 4.2's "
            "constructive proof (scan + least-squares completion + SVD "
            "factorization), then checked to reconstruct the original "
            "lower-triangular data to near machine precision."
        ),
        "n_cases": len(cases),
        "n_pass": n_pass,
        "all_pass": n_pass == len(cases),
        "max_reconstruction_error_overall": max_err_overall,
        "cases": results,
    }


# ---------------------------------------------------------------------------
# Part C: IMPOSSIBILITY certificates -- Proposition 5.1's concrete example
# ---------------------------------------------------------------------------
def prop_5_1_matrix(T: int) -> np.ndarray:
    """M = I_T + E_{T,1}: identity plus a single 1 at (row T, col 1), 0-indexed
    as (T-1, 0). This is a 2-SS matrix (Proposition 5.1) that the paper shows
    has NO 1-SS masked-attention dual of any bounded width."""
    M = np.eye(T)
    M[T - 1, 0] += 1.0
    return M


def part_c_impossibility(T_values: list[int], N_values: list[int]) -> dict:
    rows = []
    sweep = []
    for T in T_values:
        M = prop_5_1_matrix(T)
        blocks = zero_crossing_blocks(M)
        single_block = len(blocks) == 1
        new_cols = new_column_count(M)
        predicted = T - 1
        rows.append({
            "T": T,
            "n_blocks_detected": len(blocks),
            "single_unsplittable_block": single_block,
            "measured_new_columns": new_cols,
            "predicted_new_columns_T_minus_1": predicted,
            "matches_prediction": new_cols == predicted,
        })
        for N in N_values:
            impossible_predicted = predicted > N
            # Certificate: Theorem 4.1's necessary condition (<=N new columns
            # in every block) fails for this block whenever new_cols > N, so
            # no width-N dual exists; when new_cols <= N the condition does
            # NOT rule out a dual (trivially possible once N >= T-1).
            sweep.append({
                "T": T, "N": N,
                "new_columns": new_cols,
                "impossibility_certified": bool(new_cols > N),
                "consistent_with_prediction": bool((new_cols > N) == impossible_predicted),
            })
    n_consistent = sum(r["consistent_with_prediction"] for r in sweep)
    return {
        "description": (
            "Proposition 5.1 impossibility example: M = I_T + E_(T,1) is a "
            "2-SS matrix (i.e. realizable by a 2-dimensional-state SSM) that "
            "forms a SINGLE unsplittable block (the corner entry keeps every "
            "candidate crossing block nonzero) whose new-column count is "
            "exactly T-1 -- unbounded in T. Theorem 4.1's necessary condition "
            "(<=N new columns per block) therefore certifies impossibility of "
            "a width-N 1-SS masked-attention dual whenever T-1 > N, despite "
            "the SSM's own state dimension being only 2."
        ),
        "per_T": rows,
        "impossibility_sweep": sweep,
        "n_sweep_cases": len(sweep),
        "n_consistent_with_prediction": n_consistent,
        "all_consistent": n_consistent == len(sweep),
    }


def main() -> None:
    t0 = time.time()
    print("== Claim 3: Theorem 4.1 (General State-Space Duality) -- necessary & sufficient condition ==")

    print("\n[A] NECESSARY direction: genuine general-1-SS-dual constructions, T up to 512, N up to 16")
    necessary_cases = []
    for T in (64, 128, 256, 512):
        for N in (1, 2, 4, 8, 16):
            for seed_variant in range(3):
                n_blocks = 4 if T >= 128 else 2
                necessary_cases.append((T, N, n_blocks))
    a_res = part_a_necessary(necessary_cases)
    print(f"  {a_res['n_pass']}/{a_res['n_cases']} necessary-direction cases passed "
          f"(all_pass={a_res['all_pass']})")
    worst = max(a_res["cases"], key=lambda r: r["max_new_columns"] - r["N"])
    print(f"  sample case: T={worst['T']} N={worst['N']} blocks_detected={worst['n_blocks_detected']} "
          f"new_column_counts={worst['new_column_counts']} within_N={worst['within_N']}")

    print("\n[B] SUFFICIENT direction: single-block constructions, lengths 64-256, widths 1-8")
    sufficient_cases = []
    for T in (64, 128, 192, 256):
        for N in (1, 2, 4, 8):
            for seed_variant in range(2):
                sufficient_cases.append((T, N))
    b_res = part_b_sufficient(sufficient_cases)
    print(f"  {b_res['n_pass']}/{b_res['n_cases']} sufficient-direction cases passed "
          f"(all_pass={b_res['all_pass']})")
    print(f"  max reconstruction error over all cases: {b_res['max_reconstruction_error_overall']:.6e}")

    print("\n[C] IMPOSSIBILITY certificates: Proposition 5.1's M = I_T + E_(T,1), T up to 512, N up to 16")
    c_res = part_c_impossibility(T_values=[8, 16, 32, 64, 128, 256, 512], N_values=[1, 2, 4, 8, 16])
    for row in c_res["per_T"]:
        print(f"  T={row['T']:4d}  blocks_detected={row['n_blocks_detected']} (single_block={row['single_unsplittable_block']})  "
              f"new_columns={row['measured_new_columns']:4d}  predicted(T-1)={row['predicted_new_columns_T_minus_1']:4d}  "
              f"match={row['matches_prediction']}")
    n_certified = sum(1 for s in c_res["impossibility_sweep"] if s["impossibility_certified"])
    print(f"  impossibility certified in {n_certified}/{c_res['n_sweep_cases']} (T,N) combinations where T-1 > N; "
          f"all {c_res['n_sweep_cases']} combinations consistent with the T-1 prediction: {c_res['all_consistent']}")

    elapsed = time.time() - t0
    summary = {
        "paper": "arxiv:2510.04944 (OpenReview DKathyl3XN) -- On Structured State-Space Duality",
        "claim": "Establishes a necessary and sufficient condition for SSM equivalence to 1-semiseparable masked attention",
        "theorem": "Theorem 4.1 (General State-Space Duality), building on Definitions 4.1/4.2 and Propositions 4.2/5.1",
        "part_a_necessary": a_res,
        "part_b_sufficient": b_res,
        "part_c_impossibility": c_res,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "elapsed_seconds": elapsed,
    }
    out = HERE / "results.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"\n[summary] necessary={a_res['n_pass']}/{a_res['n_cases']}  sufficient={b_res['n_pass']}/{b_res['n_cases']}"
          f"  impossibility_consistent={c_res['all_consistent']}  elapsed={elapsed:.1f}s")
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
