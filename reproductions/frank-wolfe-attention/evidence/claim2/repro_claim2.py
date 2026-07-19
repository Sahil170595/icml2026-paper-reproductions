"""Claim 2 reproduction --- "Attention's forward pass and Frank-Wolfe"
(Alcalde, Geshkovski, Ruiz-Balet; OpenReview zrn7rRuvhW, arXiv 2508.09628).

Official code (independent reimplementation here; NOT imported):
  https://github.com/borjanG/2025-transformers-frank-wolfe @ 2107c4b5bae6614478150aba915252674bc796de

Paper equations (verbatim from arXiv source main.tex, downloaded 2026-07-17):

  Proposition 4.5 ("Voronoi cells"), source line ~536:
    Suppose B>0 (SPD), v=(v_1,...,v_kappa) vertices of polytope K, and J(v_i)=c>0 for
    all i where J(x)=0.5<Bx,x>. Define the B-norm Voronoi cell
        Vor_B(v_i) = {x : ||x-v_i||_B <= ||x-v_j||_B  for all j}.
    Then the *dominance cell* C_i(v) = {x in K : <Bx,v_i> = max_y<Bx,y>} satisfies
        C_i(v) = Vor_B(v_i) intersect K.

  Theorem 4.2 ("Super-exponential convergence to vertices"), source line ~433-455:
    Let B^t = B > 0 constant, gamma^t in (0,1) for all t. If the initial vertices
    v_1..v_kappa each lie in their own cell (v_j in C_j(v) \\ union_{i!=j} C_i(v)) and
    no interior point lies on a cell boundary, then with sigma(i) the cell assignment,
        x_i^t = ( prod_{tau=0}^{t-1}(1-gamma^tau) ) x_i^0
                 + sum_{tau=0}^{t-1} ( gamma^tau * prod_{s=tau+1}^{t-1}(1-gamma^s) ) v_sigma(i)
    In particular x_i^t -> v_sigma(i) *at least exponentially fast*.
    [Remark, same section]: for constant gamma this closed form reduces to
        x_i^t - v_sigma(i) = (1-gamma)^t (x_i^0 - v_sigma(i))         -- exponential, rate log(1-gamma).
    For the increasing schedule gamma^t = 1 - exp(-a(t+1)) (an allowed sequence in (0,1)),
        prod_{tau=0}^{t-1}(1-gamma^tau) = exp(-a * t(t+1)/2)          -- super-exponential.

This script:
  (A) verifies Proposition 4.5: dominance-cell label == B-Voronoi label, on random
      convex-hull samples, for regular-polygon / simplex vertex sets with equal
      B-quadratic-form value J(v_i)=c;
  (B) runs the ACTUAL hardmax dynamics (via literal per-step argmax over the current
      token set -- not the closed form) for interior points under constant gamma,
      verifies (i) the cell label never changes, (ii) the trajectory matches the
      closed-form product exactly, (iii) log||x_i^t - v|| vs t is linear with fitted
      slope == log(1-gamma) (log-linear / exponential decay, per Theorem 4.2);
  (C) repeats with the increasing schedule gamma^t=1-exp(-a(t+1)) and verifies
      log||x_i^t-v|| vs t(t+1)/2 is linear with slope == -a (super-exponential decay),
      while log||.|| vs plain t is visibly *not* linear (curvature), contrasting the
      two regimes named in the paper's "(super-)exponential" language.

Deterministic (numpy.random.default_rng, fixed seeds), CPU-only, single-threaded.
Prints only measured numbers; writes evidence-package/claim2/results.json.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent


def spd_from_eigs(d, eigs, rng):
    """Random SPD matrix with prescribed eigenvalues (condition number = max/min eig)."""
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    return Q @ np.diag(eigs) @ Q.T


def regular_polygon_vertices(kappa, radius=1.0):
    """kappa vertices of a regular polygon in R^2 on a circle of given radius."""
    angles = 2 * np.pi * np.arange(kappa) / kappa
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def bnorm2(B, x):
    return float(x @ (B @ x))


def dominance_label(Bx, V):
    """argmax_i <Bx, v_i>."""
    return int(np.argmax(V @ Bx))


def voronoi_label(B, x, V):
    """argmin_i ||x - v_i||_B^2."""
    diffs = V - x[None, :]
    d2 = np.einsum("ij,jk,ik->i", diffs, B, diffs)
    return int(np.argmin(d2))


# ---------------------------------------------------------------------------
# (A) Proposition 4.5: dominance cell == B-Voronoi cell (equal-J vertices)
# ---------------------------------------------------------------------------
def check_voronoi_partition(seed):
    rng = np.random.default_rng(seed)
    rows = []
    worst_mismatch_frac = 0.0
    total_pts = 0
    total_mismatch = 0
    for kappa in [3, 4, 5, 6, 7, 9, 10]:
        for cond in [1.0, 4.0, 6.0, 8.0]:
            d = 2
            eigs = np.array([1.0, cond]) if cond != 1.0 else np.array([1.0, 1.0])
            B = spd_from_eigs(d, eigs, rng)
            V_dir = regular_polygon_vertices(kappa, radius=1.0)
            # Rescale each vertex along its ray so that J(v_i) = c = 1 for all i (equal B-norm)
            V = np.zeros_like(V_dir)
            for i in range(kappa):
                v = V_dir[i]
                j0 = 0.5 * bnorm2(B, v)
                V[i] = v / np.sqrt(2 * j0)
            # sanity: all J(v_i) equal
            Js = np.array([0.5 * bnorm2(B, V[i]) for i in range(kappa)])
            assert np.allclose(Js, Js[0], atol=1e-10)

            # sample points in the convex hull via random convex combinations (strict interior
            # samples: dirichlet with alpha>1 avoids the boundary/vertices)
            n_samples = 200
            alpha = np.full(kappa, 2.0)
            lambdas = rng.dirichlet(alpha, size=n_samples)
            pts = lambdas @ V

            mismatches = 0
            for x in pts:
                Bx = B @ x
                lbl_dom = dominance_label(Bx, V)
                lbl_vor = voronoi_label(B, x, V)
                if lbl_dom != lbl_vor:
                    mismatches += 1
            total_pts += n_samples
            total_mismatch += mismatches
            frac = mismatches / n_samples
            worst_mismatch_frac = max(worst_mismatch_frac, frac)
            rows.append({"kappa": kappa, "cond_number": cond, "n_samples": n_samples,
                         "mismatches": mismatches, "mismatch_frac": frac})
    return {"rows": rows, "total_points": total_pts, "total_mismatches": total_mismatch,
            "worst_mismatch_frac": worst_mismatch_frac}


# ---------------------------------------------------------------------------
# (B) + (C) Real hardmax dynamics: exponential and super-exponential convergence
# ---------------------------------------------------------------------------
def make_config(kappa, n_interior, seed, radius=1.0):
    """Regular-kappa-gon vertices (own their cell by construction/symmetry) plus
    n_interior points placed strictly inside one target cell, away from any boundary."""
    rng = np.random.default_rng(seed)
    V = regular_polygon_vertices(kappa, radius=radius)
    # place interior points near the centroid-ward region of cell 0, biased toward vertex 0
    # but strictly inside (avoid boundaries): convex combo of v_0 (weight in [0.55,0.85]) and
    # centroid of the OTHER vertices scaled down (weight complement), then jitter slightly.
    centroid_others = V[1:].mean(axis=0)
    pts = []
    for k in range(n_interior):
        w = rng.uniform(0.55, 0.85)
        base = w * V[0] + (1 - w) * (0.15 * centroid_others)
        jitter = rng.normal(scale=0.01, size=2)
        pts.append(base + jitter)
    X0 = np.array(pts)
    return V, X0


def run_hardmax_step(X, V, B, gamma):
    """One literal hardmax step for every interior point: argmax over the *vertex set*
    V (the token set defining K^0, since K^t shrinks toward vertices under this dynamics
    and the vertices themselves are stationary -- see Lemma "convex hull shrinks")."""
    Xnew = np.empty_like(X)
    labels = np.empty(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        scores = V @ (B @ X[i])
        j = int(np.argmax(scores))
        labels[i] = j
        Xnew[i] = X[i] + gamma * (V[j] - X[i])
    return Xnew, labels


def closed_form_const_gamma(x0, v, gamma, t):
    return (1 - gamma) ** t * (x0 - v) + v


def closed_form_prod(x0, v, gammas_used):
    """gammas_used: list of gamma^0..gamma^{t-1} actually applied, in order."""
    prod = 1.0
    for g in gammas_used:
        prod *= (1 - g)
    return prod * (x0 - v) + v


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


def run_exponential_case():
    rows = []
    for kappa in [3, 4, 5, 6]:
        for gamma in [0.1, 0.35, 0.7]:
            rng = np.random.default_rng(500 + kappa)
            B = spd_from_eigs(2, np.array([1.0, 3.0]), rng)
            V, X0 = make_config(kappa, n_interior=4, seed=42 + kappa)
            n_steps = 40
            X = X0.copy()
            label_changes = 0
            dists = {i: [float(np.linalg.norm(X0[i] - V[0]))] for i in range(X0.shape[0])}
            closed_form_max_err = 0.0
            for t in range(n_steps):
                X, labels = run_hardmax_step(X, V, B, gamma)
                if np.any(labels != 0):
                    label_changes += int(np.sum(labels != 0))
                for i in range(X.shape[0]):
                    dists[i].append(float(np.linalg.norm(X[i] - V[0])))
                    cf = closed_form_const_gamma(X0[i], V[0], gamma, t + 1)
                    closed_form_max_err = max(closed_form_max_err, float(np.max(np.abs(X[i] - cf))))
            # log-linear fit for particle 0
            ts = np.arange(0, n_steps + 1)
            d0 = np.array(dists[0])
            mask = d0 > 1e-14
            slope, intercept, r2 = linfit(ts[mask], np.log(d0[mask]))
            target = float(np.log(1 - gamma))
            rows.append({
                "kappa": kappa, "gamma": gamma, "n_steps": n_steps,
                "label_changes_total": label_changes,
                "closed_form_max_abs_err": closed_form_max_err,
                "fitted_slope": slope, "target_slope_log1mgamma": target,
                "slope_rel_err": abs(slope - target) / abs(target),
                "R2": r2,
            })
    return rows


def run_super_exponential_case():
    rows = []
    for kappa in [3, 4, 5, 6]:
        for a in [0.05, 0.1, 0.2]:
            rng = np.random.default_rng(700 + kappa)
            B = spd_from_eigs(2, np.array([1.0, 3.0]), rng)
            V, X0 = make_config(kappa, n_interior=3, seed=142 + kappa)
            # Cap steps so a*t(t+1)/2 stays well above float64's ~1e-16 relative-precision
            # floor (otherwise log(||x-v||) plateaus due to cancellation, not model error).
            n_steps = max(8, int(np.sqrt(2 * 20.0 / a)))
            X = X0.copy()
            gammas_used = []
            closed_form_max_err = 0.0
            dist_hist = {i: [float(np.linalg.norm(X0[i] - V[0]))] for i in range(X0.shape[0])}
            for t in range(n_steps):
                gamma_t = 1 - np.exp(-a * (t + 1))
                gammas_used.append(gamma_t)
                X, labels = run_hardmax_step(X, V, B, gamma_t)
                for i in range(X.shape[0]):
                    dist_hist[i].append(float(np.linalg.norm(X[i] - V[0])))
                    cf = closed_form_prod(X0[i], V[0], gammas_used)
                    closed_form_max_err = max(closed_form_max_err, float(np.max(np.abs(X[i] - cf))))
            ts = np.arange(0, n_steps + 1)
            quad = ts * (ts + 1) / 2.0  # t(t+1)/2, matches product exp(-a*t(t+1)/2)
            d0 = np.array(dist_hist[0])
            mask = d0 > 1e-300
            slope_quad, intercept_quad, r2_quad = linfit(quad[mask], np.log(d0[mask]))
            # contrast: plain-t fit should be visibly worse (curvature) for larger a/t
            slope_lin, intercept_lin, r2_lin = linfit(ts[mask], np.log(d0[mask]))
            rows.append({
                "kappa": kappa, "a": a, "n_steps": n_steps,
                "closed_form_max_abs_err": closed_form_max_err,
                "fitted_slope_vs_quad": slope_quad, "target_slope_vs_quad": -a,
                "slope_rel_err_vs_quad": abs(slope_quad - (-a)) / a,
                "R2_vs_quad": r2_quad,
                "R2_vs_plain_t": r2_lin,
            })
    return rows


def main():
    t0 = time.time()
    print("== Claim 2: PSD key-query -> Voronoi partition + (super-)exponential convergence to vertices ==")
    print("   arXiv 2508.09628; source Prop. 4.5 (Voronoi cells) + Thm 4.2 (super-exponential convergence)")

    print("\n[A] Proposition 4.5: dominance-cell label == B-norm Voronoi label (equal-J vertices)")
    vor = check_voronoi_partition(seed=1)
    print(f"  regular polygons kappa in {{3,4,5,6,7,9,10}}, SPD condition numbers {{1,4,6,8}}, "
          f"200 interior samples/config")
    print(f"  total points checked: {vor['total_points']}, total label mismatches: {vor['total_mismatches']}, "
          f"worst per-config mismatch fraction: {vor['worst_mismatch_frac']:.4f}")

    print("\n[B] Constant gamma: literal hardmax dynamics vs closed form; log||x_i^t-v|| vs t slope == log(1-gamma)")
    exp_rows = run_exponential_case()
    worst_cf_err = max(r["closed_form_max_abs_err"] for r in exp_rows)
    worst_slope_relerr = max(r["slope_rel_err"] for r in exp_rows)
    worst_label_changes = max(r["label_changes_total"] for r in exp_rows)
    min_r2 = min(r["R2"] for r in exp_rows)
    print(f"  configs: kappa in {{3,4,5,6}} x gamma in {{0.1,0.35,0.7}} = {len(exp_rows)} rows, 40 steps each")
    print(f"  cell-label changes across all runs (target: 0, particles never leave their cell): {worst_label_changes}")
    print(f"  max |simulated - closed-form (1-gamma)^t| trajectory error: {worst_cf_err:.3e}")
    print(f"  worst |fitted slope - log(1-gamma)| relative error: {worst_slope_relerr:.3e}   (min R^2 = {min_r2:.6f})")
    for r in exp_rows:
        print(f"    kappa={r['kappa']} gamma={r['gamma']:.2f}: fitted_slope={r['fitted_slope']:.6f} "
              f"target={r['target_slope_log1mgamma']:.6f} R2={r['R2']:.6f}")

    print("\n[C] Increasing schedule gamma_t=1-exp(-a(t+1)): super-exponential, slope vs t(t+1)/2 == -a")
    sup_rows = run_super_exponential_case()
    worst_cf_err2 = max(r["closed_form_max_abs_err"] for r in sup_rows)
    worst_slope_relerr2 = max(r["slope_rel_err_vs_quad"] for r in sup_rows)
    min_r2_quad = min(r["R2_vs_quad"] for r in sup_rows)
    max_r2_lin = max(r["R2_vs_plain_t"] for r in sup_rows)
    print(f"  configs: kappa in {{3,4,5,6}} x a in {{0.05,0.1,0.2}} = {len(sup_rows)} rows, "
          f"steps per row capped so a*t(t+1)/2 stays in a safe float64 range (8-28 steps)")
    print(f"  max |simulated - closed-form prod(1-gamma^tau)| trajectory error: {worst_cf_err2:.3e}")
    print(f"  worst |fitted slope(vs t(t+1)/2) - (-a)| relative error: {worst_slope_relerr2:.3e}  (min R^2={min_r2_quad:.6f})")
    print(f"  contrast: R^2 of log-dist vs PLAIN t (should be visibly worse, curvature): max = {max_r2_lin:.6f}")
    for r in sup_rows:
        print(f"    kappa={r['kappa']} a={r['a']:.2f}: fitted_slope_vs_quad={r['fitted_slope_vs_quad']:.6f} "
              f"target=-{r['a']:.2f} R2_quad={r['R2_vs_quad']:.6f} R2_plain_t={r['R2_vs_plain_t']:.6f}")

    runtime = time.time() - t0
    out = {
        "voronoi_partition_check": vor,
        "exponential_case": exp_rows,
        "super_exponential_case": sup_rows,
        "runtime_s": round(runtime, 2),
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[written] claim2/results.json  (runtime {runtime:.1f}s)")


if __name__ == "__main__":
    main()
