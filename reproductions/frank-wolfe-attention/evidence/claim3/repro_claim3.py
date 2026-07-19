"""Claim 3 reproduction --- "Attention's forward pass and Frank-Wolfe"
(Alcalde, Geshkovski, Ruiz-Balet; OpenReview zrn7rRuvhW, arXiv 2508.09628).

Official code (independent reimplementation here; NOT imported):
  https://github.com/borjanG/2025-transformers-frank-wolfe @ 2107c4b5bae6614478150aba915252674bc796de

Paper equations (verbatim from arXiv source main.tex, downloaded 2026-07-17):

  (SA_beta), source line ~731 (finite-temperature self-attention, B=I_d, V^t=gamma*I_d):
    x_i^{t+1} = (1-gamma) x_i^t + gamma * sum_j softmax_j(beta<x_i^t,x_j^t>) x_j^t.

  (SA_P), source line ~767 (Gumbel-trick categorical Markov chain -- the actual finite-beta
  process the metastability theorems are about):
    P( x_i^{t+1} = (1-gamma) x_i^t + gamma x_j^t ) = softmax_j(beta<x_i^t,x_j^t>).

  Theorem 5.2 ("Clustering"), source line ~779-836: with high probability
  (>= 1 - beta^{-1/8}), interior points reach a ball of radius O(tau) around their
  assigned vertex within
        T_1 = floor( (1/log(1-gamma)) * log( tau / min_j||x_j^0 - v_sigma(j)|| ) )
  steps -- i.e. O(1), independent of beta (beyond a threshold beta_*).

  Theorem 5.4 ("Metastability"), source line ~876-898: once near-vertex, the residence
  time T_2 before an epsilon-escape satisfies, for eps/gamma >= 2*diam(K),
        P(T_2 >= t) >= 1 - exp( (1+eps/gamma)*log(gamma*t/eps) + (1+eps/gamma)*log(n)
                                 - beta*c_0*eps/(2*gamma) )
  i.e. the metastable window scales as t ~ exp(c*beta) for fixed eps -- "exponentially
  long in beta".

This script:
  (A) Phase 1 / clustering: runs the literal deterministic (SA_beta) dynamics from an
      explicit near-vertex-but-displaced configuration across a wide beta grid and
      measures the first-passage step count to an epsilon0-ball around each token's
      assigned vertex -- shows this count is bounded / does NOT grow with beta (O(1)),
      in contrast to Phase 2.
  (B) Phase 2 / metastability: derives the *exact* per-step escape probability
      p_escape(beta) for a token sitting exactly at a regular-kappa-gon vertex (n=kappa,
      one token per vertex, no duplicates) under the categorical (SA_P) chain -- exact
      because a self-pick is a no-op (x_i <- (1-gamma)x_i + gamma*x_i = x_i), so the
      chain is stationary until the first non-self ("cross-cluster") pick, making the
      first-exit time *exactly* Geometric(p_escape) -- matching the paper's own
      "exactly geometric" argument. Choosing epsilon < gamma*d_min(K) makes any single
      cross-pick an epsilon-escape (per Thm 5.4's remark).
      - For SMALL beta (where median exit time is a few thousand steps), this is
        cross-checked against a literal step-by-step Monte-Carlo simulation of the
        actual categorical draws (real softmax probabilities, real rng.choice draws,
        vectorized over many independent replicate chains).
      - For the FULL beta sweep (used for the exponential-rate fit), the exact
        closed-form Geometric-distribution median is used (no simulation needed --
        argued exact above, and empirically validated against simulation for small
        beta in the same run).
      - Reports log(median trapping time) vs beta: linear fit, slope compared to the
        theoretical nearest-neighbor score gap 1-cos(2*pi/kappa).

Deterministic (numpy.random.default_rng, fixed seeds), CPU-only, single-threaded.
Prints only measured numbers; writes evidence-package/claim3/results.json.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent


def regular_polygon_vertices(kappa, radius=1.0):
    angles = 2 * np.pi * np.arange(kappa) / kappa
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def softmax_rows(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


def linfit(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, ys, rcond=None)
    yhat = A @ [slope, intercept]
    ss_res = np.sum((ys - yhat) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


# ---------------------------------------------------------------------------
# (A) Phase 1: clustering in O(1) steps (deterministic SA_beta), across beta
# ---------------------------------------------------------------------------
def run_phase1(kappa, betas, gamma=0.1, offset=0.35, eps0=0.05, max_steps=2000, seed=0):
    """n = 2*kappa tokens: kappa vertices (start exactly at v_i) + kappa interior
    points (start at distance `offset` from their assigned vertex, toward centroid),
    evolving under the literal deterministic SA_beta expectation dynamics."""
    rng = np.random.default_rng(seed)
    V = regular_polygon_vertices(kappa, radius=1.0)
    sigma = np.arange(kappa)  # interior point i is assigned to vertex i
    centroid = V.mean(axis=0)
    interior0 = np.array([
        V[i] + offset * (centroid - V[i]) / np.linalg.norm(centroid - V[i])
        for i in range(kappa)
    ])
    X0 = np.concatenate([V, interior0], axis=0)  # first kappa rows = vertices, next kappa = interior
    rows = []
    for beta in betas:
        X = X0.copy()
        T1 = None
        vertex_drift_at_T1 = None
        for t in range(1, max_steps + 1):
            S = beta * (X @ X.T)
            P = softmax_rows(S)
            X = (1 - gamma) * X + gamma * (P @ X)
            interior = X[kappa:]
            dists = np.linalg.norm(interior - V[sigma], axis=1)
            if np.all(dists < eps0):
                T1 = t
                vertex_drift_at_T1 = float(np.max(np.linalg.norm(X[:kappa] - V, axis=1)))
                break
        rows.append({
            "kappa": kappa, "beta": beta, "gamma": gamma, "offset": offset, "eps0": eps0,
            "T1_steps": T1, "max_vertex_drift_at_T1": vertex_drift_at_T1,
        })
    return rows


# ---------------------------------------------------------------------------
# (B) Phase 2: exact escape probability + exponential trapping-time fit
# ---------------------------------------------------------------------------
def escape_prob_exact(kappa, beta):
    """Exact P(non-self pick) for a token sitting exactly at a regular-kappa-gon vertex
    (unit circle), n=kappa (no duplicates), categorical SA_P chain, B=I. Numerically
    stable: self-score is always the max (score_self - score_j = beta*(1-cos(theta_j)) >= 0),
    so shift by the self-score before exponentiating."""
    angles = 2 * np.pi * np.arange(kappa) / kappa
    cos_to_self = np.cos(angles)  # cos(theta_{0,j}) for vertex 0 vs all others (incl self=1.0)
    gaps = 1 - cos_to_self  # >=0, gaps[0]=0 (self)
    other = np.exp(-beta * gaps[1:])  # kappa-1 terms, all in (0,1]
    denom = 1.0 + other.sum()
    p_escape = other.sum() / denom
    return float(p_escape), float(np.min(gaps[1:]))  # also return nearest-neighbor gap


def geometric_median(p):
    """Median of Geometric(p) (number of trials to first success): ceil(log(0.5)/log(1-p)).
    Uses log1p(-p) instead of log(1-p): for the tiny p seen at large beta (down to ~1e-30
    or smaller), naive 1-p rounds to exactly 1.0 in float64 (catastrophic cancellation,
    since p is far below machine epsilon relative to 1), which would incorrectly give
    log(1)=0 and a divide-by-zero. log1p is accurate in this regime down to p as small
    as the smallest positive float64."""
    if p <= 0:
        return np.inf
    if p >= 1:
        return 1.0
    log1mp = np.log1p(-p)
    return float(np.ceil(np.log(0.5) / log1mp))


def simulate_categorical_chain(kappa, beta, gamma, n_replicates, max_steps, eps, seed):
    """Literal step-by-step simulation of the (SA_P) categorical Markov chain for a
    single representative token starting exactly at vertex 0 of a regular kappa-gon
    (n=kappa tokens total, no duplicates). At each step draws a real categorical
    outcome from the real softmax(beta * <x, v_j>) distribution via inverse-CDF
    sampling (rng.random + cumulative probabilities), vectorized across independent
    replicate chains. Records the first step at which the token's position leaves
    the eps-ball around vertex 0. Self-picks are no-ops (position doesn't change),
    so this is literally simulating the Markov chain, not shortcutting it."""
    V = regular_polygon_vertices(kappa, radius=1.0)
    v0 = V[0]
    scores = beta * (V @ v0)  # fixed while token sits exactly at v0 (pre-escape)
    probs = softmax_rows(scores[None, :])[0]  # length kappa, probs[0] = self
    rng = np.random.default_rng(seed)
    cum = np.cumsum(probs)
    active = np.ones(n_replicates, dtype=bool)
    exit_step = np.full(n_replicates, -1, dtype=np.int64)
    exit_dist = np.zeros(n_replicates, dtype=float)
    n_active = n_replicates
    t = 0
    while n_active > 0 and t < max_steps:
        t += 1
        u = rng.random(n_active)
        picks = np.searchsorted(cum, u, side="right")
        non_self = picks != 0
        idx_active = np.where(active)[0]
        newly_escaped = idx_active[non_self]
        if newly_escaped.size > 0:
            j_picks = picks[non_self]
            new_pos = (1 - gamma) * v0[None, :] + gamma * V[j_picks]
            d = np.linalg.norm(new_pos - v0[None, :], axis=1)
            really_escaped = d > eps
            escaped_idx = newly_escaped[really_escaped]
            exit_step[escaped_idx] = t
            exit_dist[escaped_idx] = d[really_escaped]
            active[escaped_idx] = False
        n_active = int(active.sum())
    return exit_step, probs


def run_phase2(kappas, gamma=0.1):
    diam_K = 2.0  # regular polygon on unit circle: diameter <= 2 (achieved for even kappa)
    eps = 0.3 * gamma * (2 * np.sin(np.pi / max(kappas)))  # conservative small eps << gamma*d_min
    rows_sim = []
    rows_analytic = []
    for kappa in kappas:
        d_min = 2 * np.sin(np.pi / kappa)  # nearest-neighbor chord length, regular polygon, radius 1
        eps_k = 0.3 * gamma * d_min  # ensures gamma*d_min > eps_k comfortably (paper's Thm 5.4 condition)
        theoretical_gap = 1 - np.cos(2 * np.pi / kappa)  # nearest-neighbor score gap

        # --- (i) small-beta direct simulation cross-check ---
        sim_betas = [1.0, 2.0, 3.0, 4.0] if kappa <= 4 else [1.0, 1.5, 2.0, 2.5]
        for beta in sim_betas:
            exit_step, probs = simulate_categorical_chain(
                kappa, beta, gamma, n_replicates=6000, max_steps=400000, eps=eps_k, seed=int(1000 * kappa + beta * 10)
            )
            censored = int(np.sum(exit_step < 0))
            valid = exit_step[exit_step > 0]
            emp_median = float(np.median(valid)) if valid.size > 0 else np.nan
            p_exact, gap_nn = escape_prob_exact(kappa, beta)
            analytic_median = geometric_median(p_exact)
            rel_err = abs(emp_median - analytic_median) / analytic_median if analytic_median > 0 else np.nan
            rows_sim.append({
                "kappa": kappa, "beta": beta, "gamma": gamma, "eps": eps_k,
                "n_replicates": 6000, "n_censored": censored,
                "empirical_median_exit_step": emp_median,
                "analytic_p_escape": p_exact, "analytic_median_exit_step": analytic_median,
                "relative_error": rel_err,
            })

        # --- (ii) full beta sweep: exact analytic median (validated above) ---
        full_betas = list(np.arange(1.0, 20.5, 1.0)) + [24.0, 28.0, 32.0, 36.0, 40.0]
        log_medians = []
        for beta in full_betas:
            p_exact, gap_nn = escape_prob_exact(kappa, beta)
            med = geometric_median(p_exact)
            rows_analytic.append({"kappa": kappa, "beta": beta, "p_escape": p_exact, "median_exit_step": med})
            log_medians.append(np.log(med))
        slope, intercept, r2 = linfit(full_betas, log_medians)
        rows_analytic.append({
            "kappa": kappa, "summary": True,
            "fitted_slope": slope, "theoretical_gap": theoretical_gap,
            "slope_rel_err": abs(slope - theoretical_gap) / theoretical_gap,
            "R2": r2,
            "median_at_beta1": geometric_median(escape_prob_exact(kappa, 1.0)[0]),
            "median_at_beta40": geometric_median(escape_prob_exact(kappa, 40.0)[0]),
        })
    return rows_sim, rows_analytic


def main():
    t0 = time.time()
    print("== Claim 3: finite-beta dynamic metastability (Theorems 5.2 & 5.4) ==")
    print("   arXiv 2508.09628; SA_beta / SA_P (source lines ~731, ~767)")

    print("\n[A] Phase 1 (clustering): steps to reach eps0-ball of assigned vertex, across beta")
    phase1_rows = []
    for kappa in [3, 4, 5]:
        rows = run_phase1(kappa, betas=[1, 2, 4, 8, 16, 32, 64, 128], gamma=0.1)
        phase1_rows.extend(rows)
        Ts = [r["T1_steps"] for r in rows]
        print(f"  kappa={kappa}: T1(beta) for beta in {[r['beta'] for r in rows]} -> {Ts}")
    T1_all = [r["T1_steps"] for r in phase1_rows if r["T1_steps"] is not None]
    print(f"  T1 range across ALL configs: min={min(T1_all)}, max={max(T1_all)} "
          f"(bounded / O(1) -- contrast with Phase 2 below)")

    print("\n[B] Phase 2 (metastability): exact escape probability, small-beta simulation cross-check,")
    print("    then exponential fit of log(median trapping time) vs beta over the full sweep")
    rows_sim, rows_analytic = run_phase2(kappas=[3, 4, 5, 6], gamma=0.1)
    print("\n  (i) Small-beta direct Monte-Carlo simulation vs exact analytic Geometric median:")
    worst_relerr = 0.0
    for r in rows_sim:
        print(f"    kappa={r['kappa']} beta={r['beta']:.1f}: empirical_median={r['empirical_median_exit_step']:.1f}  "
              f"analytic_median={r['analytic_median_exit_step']:.1f}  rel_err={r['relative_error']:.4f}  "
              f"(censored {r['n_censored']}/{r['n_replicates']})")
        if not np.isnan(r["relative_error"]):
            worst_relerr = max(worst_relerr, r["relative_error"])
    print(f"  worst empirical-vs-analytic relative error across all small-beta sim configs: {worst_relerr:.4f}")

    print("\n  (ii) Full beta sweep, exact analytic median trapping time (validated above):")
    summaries = [r for r in rows_analytic if r.get("summary")]
    for s in summaries:
        print(f"    kappa={s['kappa']}: fitted slope={s['fitted_slope']:.6f}  "
              f"theoretical nearest-neighbor gap (1-cos(2pi/kappa))={s['theoretical_gap']:.6f}  "
              f"rel_err={s['slope_rel_err']:.4f}  R2={s['R2']:.6f}  "
              f"median@beta=1: {s['median_at_beta1']:.1f}  median@beta=40: {s['median_at_beta40']:.3e}")

    runtime = time.time() - t0
    out = {
        "phase1_clustering": phase1_rows,
        "phase2_simulation_crosscheck": rows_sim,
        "phase2_analytic_sweep": rows_analytic,
        "runtime_s": round(runtime, 2),
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[written] claim3/results.json  (runtime {runtime:.1f}s)")


if __name__ == "__main__":
    main()
