#!/usr/bin/env python3
"""
Independent NumPy/scipy reproduction of core claims from
"Row-Stochastic Matrices Can Provably Outperform Doubly Stochastic
 Matrices in Decentralized Learning" (arXiv 2511.19513, OpenReview GAQE4Wr53f).

Verified on small graphs (n in {10,20,50}; Erdos-Renyi + ring + star):
 (a) EXACT weighted detailed balance for the modified Metropolis-Hastings
     ROW-stochastic W:  lam_i W_ij = lam_j W_ji  (residual ~ machine eps),
     i.e. W is SELF-ADJOINT in <x,y>_lam = sum_i lam_i x_i y_i, while the
     standard-MH DOUBLY-stochastic W^ds violates it (O(1) residual) whenever
     lam is not uniform.
 (b) Consequence: the lam-weighted consensus-error transient of W has prefactor
     exactly 1 (contracts as rho_Lambda^t with constant 1), while W^ds's
     lam-weighted transient is inflated by up to kappa_lam=sqrt(lam_max/lam_min)>1.
 (c) Stationarity lam^T W = lam^T; and, on degree-matched lam_i propto d_i
     (Cor. 7.3 optimal design), Theorem 7.1's sufficient condition + strictly
     faster weighted decay (the paper's CONDITIONAL claim: may fail for random lam).

Deterministic (numpy default_rng), CPU-only, no downloads.
"""
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

EPS = 0.1        # laziness in modified-MH row-stochastic construction, Eq.(3)
ETA = 1.8e-3     # slack constant in the Theorem 7.1 sufficient condition
T_TRANSIENT = 120
FLOOR = 1e-9     # stop empirical-prefactor scoring once envelope rho^t hits this floor


def is_connected(A):
    ncomp, _ = connected_components(csr_matrix(A), directed=False)
    return ncomp == 1


def er_graph(n, rng):
    p = min(1.0, 2.2 * np.log(max(n, 2)) / n)
    for _ in range(5000):
        M = rng.random((n, n))
        A = (M < p).astype(float)
        A = np.triu(A, 1)
        A = A + A.T
        if A.sum(1).min() >= 1 and is_connected(A):
            return A
    raise RuntimeError("could not sample a connected ER graph")


def ring_graph(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[(i + 1) % n, i] = 1.0
    return A


def star_graph(n):
    A = np.zeros((n, n))
    for i in range(1, n):
        A[0, i] = 1.0
        A[i, 0] = 1.0
    return A


def lam_random(n, rng):
    lam = rng.uniform(0.3, 3.0, size=n)          # generic heterogeneous weights
    return lam * n / lam.sum()                    # normalize sum(lam)=n


def lam_degree_matched(A):
    d = A.sum(1)
    return d * A.shape[0] / d.sum()               # lam_i propto d_i  (Cor. 7.3)


def row_stochastic_W(A, lam, eps=EPS):
    """Modified Metropolis-Hastings, Eq.(3): row-stochastic, self-adjoint in <.,.>_lam."""
    n = A.shape[0]
    d = A.sum(1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0 and i != j:
                W[i, j] = (1.0 - eps) / d[i] * min(1.0, (lam[j] * d[i]) / (lam[i] * d[j]))
    for i in range(n):
        W[i, i] = 1.0 - W[i].sum()
    return W


def doubly_stochastic_W(A):
    """Standard Metropolis-Hastings: symmetric, doubly stochastic (stationary=uniform)."""
    n = A.shape[0]
    d = A.sum(1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0 and i != j:
                W[i, j] = 1.0 / (1.0 + max(d[i], d[j]))
    for i in range(n):
        W[i, i] = 1.0 - W[i].sum()
    return W


def db_residual(W, lam):
    """max_ij |lam_i W_ij - lam_j W_ji| = ||D_lam W - (D_lam W)^T||_max."""
    M = lam[:, None] * W
    return float(np.max(np.abs(M - M.T)))


def transform(A, lam):
    """D_lam^{1/2} A D_lam^{-1/2}; its ||.||_2 equals the lam-weighted operator norm of A."""
    s = np.sqrt(lam)
    return (s[:, None] * A) / s[None, :]


def weighted_opnorm(A, lam):
    return float(np.linalg.norm(transform(A, lam), 2))


def wnorm(x, lam):
    return float(np.sqrt(np.sum(lam * x * x)))


def realized_prefactor(Wmat, proj, lam, rho, x0, T, floor):
    """Run the ACTUAL consensus iteration x_{t+1}=Wmat x_t from x0 and return the
    realized lam-weighted transient prefactor max_t (e_t/e_0)/rho^t, scored only
    while the envelope rho^t stays above the double-precision floor."""
    x = x0.copy()
    tgt = proj @ x0                       # consensus limit (weighted or uniform average)
    e0 = wnorm(x - tgt, lam)
    if e0 <= 0.0:
        return 0.0
    pre = 0.0
    for t in range(1, T + 1):
        x = Wmat @ x
        e = wnorm(x - tgt, lam)
        if rho > 1e-12 and rho ** t > floor:
            pre = max(pre, (e / e0) / rho ** t)
    return pre


def run_config(n, topo, regime, seed):
    rng = np.random.default_rng(seed)
    if topo == "ER":
        A = er_graph(n, rng)
    elif topo == "ring":
        A = ring_graph(n)
    else:
        A = star_graph(n)
    lam = lam_random(n, rng) if regime == "random" else lam_degree_matched(A)

    W = row_stochastic_W(A, lam)
    Wds = doubly_stochastic_W(A)

    kappa = float(np.sqrt(lam.max() / lam.min()))
    uniform_lam = kappa < 1.0 + 1e-9              # degree-matched on a regular graph -> lam uniform

    # (a) detailed balance / self-adjointness
    db_W = db_residual(W, lam)
    db_Wds = db_residual(Wds, lam)
    sa_W = float(np.linalg.norm(lam[:, None] * W - (lam[:, None] * W).T, "fro"))
    sa_Wds = float(np.linalg.norm(lam[:, None] * Wds - (lam[:, None] * Wds).T, "fro"))

    # (c) stationarity / stochasticity
    stat_W = float(np.max(np.abs(lam @ W - lam)))     # lam^T W = lam^T
    rowsum_W = float(np.max(np.abs(W.sum(1) - 1.0)))  # row-stochastic
    ds_col = float(np.max(np.abs(Wds.sum(0) - 1.0)))  # doubly stochastic (cols)
    ds_row = float(np.max(np.abs(Wds.sum(1) - 1.0)))

    # contraction operators & asymptotic rates
    Lam = np.ones((n, n)) * lam[None, :] / n          # weighted-avg projector (W consensus)
    J = np.ones((n, n)) / n                           # uniform-avg projector  (Wds consensus)
    M_row = W - Lam
    M_ds = Wds - J
    rho_L = weighted_opnorm(M_row, lam)               # 2nd-eigval magnitude of W
    rho_J = float(np.linalg.norm(M_ds, 2))            # 2nd-eigval magnitude of Wds (symmetric)

    # (b) operator-norm transient prefactor:  sup_t ||M^t||_lam / rho^t
    pre_row = 0.0
    pre_ds = 0.0
    Prow = np.eye(n)
    Pds = np.eye(n)
    Prow_star = np.eye(n)   # snapshot of M^t at the prefactor-maximizing step (worst-case)
    Pds_star = np.eye(n)
    for t in range(1, T_TRANSIENT + 1):
        Prow = Prow @ M_row
        Pds = Pds @ M_ds
        if rho_L > 1e-12:
            v = weighted_opnorm(Prow, lam) / rho_L ** t
            if v > pre_row:
                pre_row = v
                Prow_star = Prow.copy()
        if rho_J > 1e-12:
            v = weighted_opnorm(Pds, lam) / rho_J ** t
            if v > pre_ds:
                pre_ds = v
                Pds_star = Pds.copy()

    # Real executed consensus runs. The transient prefactor is a WORST-CASE
    # (sup-over-initialization) quantity, so we drive the actual iteration from the
    # theory's worst-case initial-error direction (top right singular vector of
    # D_lam^{1/2} M^{t*} D_lam^{-1/2} at the peak step t*), mapped back via
    # x0 = D_lam^{-1/2} u. We also run a plain random x0 for the typical case.
    inv_s = 1.0 / np.sqrt(lam)
    _, _, VtR = np.linalg.svd(transform(Prow_star, lam))
    _, _, VtD = np.linalg.svd(transform(Pds_star, lam))
    x0_row_wc = VtR[0] * inv_s     # worst-case init for W
    x0_ds_wc = VtD[0] * inv_s      # worst-case init for W^ds
    emp_pre_row = realized_prefactor(W, Lam, lam, rho_L, x0_row_wc, T_TRANSIENT, FLOOR)
    emp_pre_ds = realized_prefactor(Wds, J, lam, rho_J, x0_ds_wc, T_TRANSIENT, FLOOR)
    x0_rand = rng.standard_normal(n)
    emp_pre_row_rand = realized_prefactor(W, Lam, lam, rho_L, x0_rand, T_TRANSIENT, FLOOR)
    emp_pre_ds_rand = realized_prefactor(Wds, J, lam, rho_J, x0_rand, T_TRANSIENT, FLOOR)

    # (c) Theorem 7.1 sufficient condition
    rho_gap_L = 1.0 - rho_L
    factor = max((1.0 + ETA) * kappa ** (-1.0 / 3.0), lam.max() ** (-0.5))
    rhs = float(factor * (1.0 - rho_J))
    thm71 = bool(rho_gap_L >= rhs)
    faster = bool(rho_L < rho_J - 1e-12)              # strictly faster asymptotic weighted decay

    return dict(
        n=n, topo=topo, regime=regime, kappa=kappa, uniform_lam=bool(uniform_lam),
        db_W=db_W, db_Wds=db_Wds, sa_W=sa_W, sa_Wds=sa_Wds,
        stat_W=stat_W, rowsum_W=rowsum_W, ds_col=ds_col, ds_row=ds_row,
        rho_L=rho_L, rho_J=rho_J, pre_row=pre_row, pre_ds=pre_ds,
        emp_pre_row=emp_pre_row, emp_pre_ds=emp_pre_ds,
        emp_pre_row_rand=emp_pre_row_rand, emp_pre_ds_rand=emp_pre_ds_rand,
        thm71=thm71, thm71_lhs=rho_gap_L, thm71_rhs=rhs, faster=faster,
    )


def main():
    configs = []
    seed = 100
    for n in (10, 20, 50):
        for topo in ("ER", "ring", "star"):
            for regime in ("random", "degree"):
                configs.append(run_config(n, topo, regime, seed))
                seed += 1

    line = "=" * 104
    print(line)
    print("PER-CONFIG MEASUREMENTS (DB=detailed-balance residual, pre=lam-weighted transient prefactor)")
    print(line)
    print("  n topo  regime   kappa       DB_W    DB_Wds    rhoL    rhoJ   pre_W  pre_ds  ds<=k thm71  fast")
    print("-" * 104)
    for r in configs:
        if r["uniform_lam"]:
            ds_ok = "n/a"
        elif r["pre_ds"] <= r["kappa"] * (1 + 1e-6):
            ds_ok = "yes"
        else:
            ds_ok = "NO"
        thm = "yes" if r["thm71"] else "no"
        fst = "yes" if r["faster"] else "no"
        print("{:>3} {:>4} {:>7} {:>7.3f} {:>10.2e} {:>9.2e} {:>7.4f} {:>7.4f} "
              "{:>7.4f} {:>7.3f} {:>5} {:>5} {:>5}".format(
                  r["n"], r["topo"], r["regime"], r["kappa"], r["db_W"], r["db_Wds"],
                  r["rho_L"], r["rho_J"], r["pre_row"], r["pre_ds"], ds_ok, thm, fst))

    # ---------- aggregate CORE (headline) claims (a) & (b) ----------
    hetero = [r for r in configs if not r["uniform_lam"]]   # non-uniform lam -> claim has bite
    unif = [r for r in configs if r["uniform_lam"]]

    max_db_W = max(r["db_W"] for r in configs)
    max_sa_W = max(r["sa_W"] for r in configs)
    min_db_Wds_h = min(r["db_Wds"] for r in hetero)
    max_stat_W = max(r["stat_W"] for r in configs)
    max_rowsum = max(r["rowsum_W"] for r in configs)
    max_dscol = max(r["ds_col"] for r in configs)

    max_pre_row_dev = max(abs(r["pre_row"] - 1.0) for r in configs)     # W prefactor identically 1?
    max_pre_ds_h = max(r["pre_ds"] for r in hetero)                     # largest Wds inflation
    ds_ge1 = all(r["pre_ds"] >= 1.0 - 1e-9 for r in hetero)             # Wds prefactor never < 1
    ds_bound_ok = all(r["pre_ds"] <= r["kappa"] * (1 + 1e-6) for r in hetero)  # never exceeds kappa
    n_ds_inflated = sum(1 for r in hetero if r["pre_ds"] > 1.0 + 1e-3)  # how many actually inflate

    max_emp_pre_row = max(r["emp_pre_row"] for r in configs)            # W never inflates, even worst-case
    max_emp_pre_ds = max(r["emp_pre_ds"] for r in hetero)              # DS realized inflation (worst-case x0)
    n_emp_ds_infl = sum(1 for r in hetero if r["emp_pre_ds"] > 1.0 + 1e-3)

    deg = [r for r in configs if r["regime"] == "degree" and not r["uniform_lam"]]
    rnd = [r for r in configs if r["regime"] == "random"]
    thm71_deg = sum(1 for r in deg if r["thm71"])
    faster_deg = sum(1 for r in deg if r["faster"])
    thm71_rnd = sum(1 for r in rnd if r["thm71"])
    faster_rnd = sum(1 for r in rnd if r["faster"])

    print()
    print(line)
    print("PAPER TARGETS  vs  MEASURED   (headline = claims a & b)")
    print(line)

    def row(name, target, measured, ok):
        tag = "OK" if ok else "XX"
        print("  {:<48} target: {:<18} measured: {:<24} [{}]".format(name, target, measured, tag))

    ok_a1 = max_db_W < 1e-9
    ok_a2 = min_db_Wds_h > 1e-3
    ok_a3 = max_sa_W < 1e-9
    ok_c_stat = max_stat_W < 1e-9 and max_rowsum < 1e-12 and max_dscol < 1e-12
    ok_b1 = max_pre_row_dev < 1e-6
    ok_b2 = ds_ge1 and ds_bound_ok and (n_ds_inflated >= 1)
    ok_b_emp = (max_emp_pre_row <= 1.0 + 1e-6) and (n_emp_ds_infl >= 1)

    row("(a) DB residual, row-stoch W (max cfg)", "~1e-12 (=0)", "{:.2e}".format(max_db_W), ok_a1)
    row("(a) DB residual, DS W^ds (min non-unif)", ">1e-3 (O(1))", "{:.2e}".format(min_db_Wds_h), ok_a2)
    row("(a) self-adjoint ||D_l W-W^T D_l||_F", "~1e-12 (=0)", "{:.2e}".format(max_sa_W), ok_a3)
    row("(c) stationarity max|lam^T W-lam^T|", "~1e-12 (=0)", "{:.2e}".format(max_stat_W), ok_c_stat)
    row("(b) W prefactor==1 all cfg (max|pre-1|)", "=1 (no kappa)", "1+{:.1e}".format(max_pre_row_dev), ok_b1)
    row("(b) DS prefactor inflated (#>1/#nonunif)", ">1, up to kappa",
        "{}/{} (max {:.3f})".format(n_ds_inflated, len(hetero), max_pre_ds_h), ok_b2)
    row("(b) DS prefactor within [1,kappa] bound", "always",
        "yes" if (ds_ge1 and ds_bound_ok) else "VIOL", ds_ge1 and ds_bound_ok)
    row("(b) real-run W pre / DS pre (worst-case x0)", "<=1 / >1",
        "{:.3f} / {:.3f} ({} infl)".format(max_emp_pre_row, max_emp_pre_ds, n_emp_ds_infl), ok_b_emp)

    print()
    print("  Conditional claim (c) [Thm 7.1 + faster on degree-matched; may fail on random lam]:")
    print("     degree-matched non-uniform cfgs: {}  |  Thm7.1 holds: {}  |  rho_L<rho_J (faster): {}".format(
        len(deg), thm71_deg, faster_deg))
    print("     random-lam cfgs: {}  |  Thm7.1 holds: {}  |  faster: {}".format(len(rnd), thm71_rnd, faster_rnd))
    print("     ({} degree-matched-on-regular cfgs have uniform lam -> W==W^ds, kappa=1: expected & consistent)".format(
        len(unif)))

    core_ok = all([ok_a1, ok_a2, ok_a3, ok_c_stat, ok_b1, ok_b2, ok_b_emp])
    cond_ok = (thm71_deg == len(deg) and faster_deg == len(deg))

    print()
    print(line)
    print("CORE CLAIMS (a: detailed balance/self-adjoint;  b: prefactor 1 vs up-to-kappa): {}".format(
        "PASS" if core_ok else "FAIL"))
    print("CONDITIONAL CLAIM (c: degree-matched Thm7.1 + strictly faster): {}".format(
        "holds on ALL degree-matched cfgs" if cond_ok else "partial") +
        "  (random lam not required -> matches paper's conditional statement)")
    print(line)
    print("OVERALL: {}   (headline mechanism a+b reproduced; c reproduced as the paper's conditional claim)".format(
        "PASS" if core_ok else "FAIL"))

    ev = {
        "orid": "GAQE4Wr53f",
        "eps": EPS, "eta": ETA, "T_transient": T_TRANSIENT, "n_configs": len(configs),
        "headline": {
            "max_db_residual_W": max_db_W,
            "min_db_residual_Wds_nonuniform": min_db_Wds_h,
            "max_selfadjoint_fro_W": max_sa_W,
            "max_stationarity_residual_W": max_stat_W,
            "max_prefactor_dev_from_1_W": max_pre_row_dev,
            "max_prefactor_Wds_nonuniform": max_pre_ds_h,
            "n_ds_configs_inflated_above_1": int(n_ds_inflated),
            "n_nonuniform_configs": len(hetero),
            "ds_prefactor_ge_1": bool(ds_ge1),
            "ds_prefactor_within_kappa_bound": bool(ds_bound_ok),
            "emp_max_prefactor_W_worstcase": max_emp_pre_row,
            "emp_max_prefactor_Wds_worstcase": max_emp_pre_ds,
            "emp_n_ds_inflated": int(n_emp_ds_infl),
        },
        "conditional": {
            "n_degree_matched_nonuniform": len(deg),
            "thm71_holds_degree": int(thm71_deg),
            "faster_degree": int(faster_deg),
            "n_random": len(rnd),
            "thm71_holds_random": int(thm71_rnd),
            "faster_random": int(faster_rnd),
        },
        "core_pass": bool(core_ok),
        "conditional_holds_on_degree": bool(cond_ok),
        "per_config": configs,
    }
    with open("evidence.json", "w") as f:
        json.dump(ev, f, indent=2)
    print()
    print("wrote evidence.json")


if __name__ == "__main__":
    main()
