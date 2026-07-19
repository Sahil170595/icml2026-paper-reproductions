"""
Reproduction of CLAIM 1 (Section 4) of
  "Anytime-Valid Inference Under Outcome Delay: A Design-Based Approach"
  Michael Lindon, Nathan Kallus. arXiv 2603.25971v1 (ICML 2026), OpenReview FXWnvznHMW.

CLAIM 1 (anchored, Section 4):
  "The IPW treatment-effect estimation error is NOT a martingale with respect to
   any single filtration, but EACH ARM's IPW estimation error IS a martingale with
   respect to an arm-specific event-time filtration."

Paper anchors used:
  Def 4.3  Single-arm filtration F_t(w): w_i revealed at potential event time t_i(w).
  Thm 4.4  M_t(w)=r^hat_t(w)-r_t(w) is a martingale wrt F_t(w)  (IPW = m_it(w)=0 case).
  Lem 4.7  For a martingale with M_0=0:  Cov(M_s,M_t) = Var(M_s)  for s<=t.
  Prop 4.8 For the difference D_t = Dhat_t - Delta_t:  Cov(D_s,D_t) != Var(D_s)  (s<t)
           => by contrapositive of Lem 4.7, D_t is NOT a martingale under ANY filtration.
  Thm 4.6  Confidence sequence  r^hat_t(w) +/- b(V^hat_t(w);alpha),
           b(V;a) = sqrt( (V*eta^2+1)/eta^2 * log( (V*eta^2+1)/a^2 ) )   (mixture boundary).

DESIGN-BASED setup (the ONLY randomness is treatment assignment):
  Potential outcomes (t_i(0),t_i(1),y_i(0),y_i(1)) and entry E_i are FIXED constants
  (generated once, fixed seed). Treatment is accelerated (pull-forward) and lower
  magnitude, matching Dataset 5.1's structure, so arms have ASYMMETRIC arrival clocks.
  We then Monte-Carlo over R independent assignment vectors w ~ Bernoulli(pi).

WHAT IS MEASURED (all numbers below are printed from this run; no fabrication):
  (A) Per-arm IPW error is a martingale wrt the single-arm event-time filtration:
      (A1) increment conditional-mean-zero  E[dM_k | past] ~ 0  (standardized z),
      (A2) increments unpredictable from the past (regression slope ~ 0, standardized z),
      (A3) Lemma 4.7 identity  Cov(M_s,M_t)/Var(M_s) ~ 1  (max dev + standardized z).
  (B) NEGATIVE result: the treatment-effect (difference) error violates Lemma 4.7,
      Cov(D_s,D_t) != Var(D_s)  (large standardized z) => not a martingale under any filtration.
  (C) The martingale structure yields ANYTIME-VALID coverage: the single-arm CS (Thm 4.6)
      has uniform-over-time coverage >= 1-alpha, while a naive fixed-n pointwise CI
      OVER-REJECTS under continuous monitoring (uniform coverage < 1-alpha), even though
      its single-look coverage ~ 1-alpha.

PASS rule (Claim 1 VERIFIED) if ALL hold:
  A1 |z| <= 4 ,  A2 |z| <= 4 ,  A3 standardized |z| <= 4 (Lemma 4.7 identity within MC error),
  B  max standardized |z| >= 6  (decisive difference-process violation),
  C  CS uniform coverage >= 1-alpha  AND  naive-pointwise uniform coverage < 1-alpha
     AND naive-pointwise single-look coverage within [1-alpha-0.03, 1].
Otherwise the relevant sub-claim is FALSIFIED and reported as such.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
import hashlib
import numpy as np

ALPHA = 0.05
PI = 0.5
ETA = 0.1          # free boundary constant; Thm 4.6 holds for ANY eta>0. Chosen to
                   # match the information scale so the CS is non-vacuous (not tuned to data).
N = 400            # units
R = 20000          # Monte-Carlo assignment draws
SEED_DESIGN = 0
SEED_ASSIGN = 12345


def boundary(V, alpha=ALPHA, eta=ETA):
    """Thm 4.6 normal-mixture boundary b(V;alpha), eq (20)."""
    zz = (V * eta**2 + 1.0) / eta**2
    return np.sqrt(zz * np.log((V * eta**2 + 1.0) / alpha**2))


def cov_se(A, B):
    """Sample covariance of paired samples and a standard error for it."""
    Am = A - A.mean()
    Bm = B - B.mean()
    g = Am * Bm
    c = g.mean()
    se = g.std(ddof=1) / np.sqrt(A.shape[0])
    return float(c), float(se)


def main():
    t0wall = time.time()
    rng = np.random.default_rng(SEED_DESIGN)

    # ---- FIXED potential outcomes (design-based constants) ----
    E = rng.uniform(0.0, 10.0, N)                       # staggered entry times
    s0 = np.exp(rng.normal(1.40, 0.50, N))              # control internal delay (slower)
    s1 = np.exp(rng.normal(0.20, 0.30, N))              # treatment internal delay (pull-forward)
    t_pot0 = E + s0                                     # potential event time, control
    t_pot1 = E + s1                                     # potential event time, treatment
    eps = rng.uniform(0.9, 1.1, N)
    y0 = 1.0 * (1.0 + 0.15 * np.log1p(s0)) * eps        # control magnitude (higher, beta0=1.0)
    y1 = 0.6 * (1.0 + 0.15 * np.log1p(s1)) * eps        # treatment magnitude (lower, beta1=0.6)

    print("=" * 74)
    print("CLAIM 1 - per-arm IPW error martingale wrt single-arm filtration")
    print("Anytime-Valid Inference Under Outcome Delay (arXiv 2603.25971v1, FXWnvznHMW)")
    print("=" * 74)
    print(f"design: N={N} fixed potential outcomes; only randomness = assignment")
    print(f"        pi={PI}, R={R} MC draws, alpha={ALPHA}, eta(boundary)={ETA}")
    print(f"        arm-1 median event time {np.median(t_pot1):.2f} (pull-forward) "
          f"< arm-0 median {np.median(t_pot0):.2f}  -> asymmetric clocks")
    print(f"        mean |y0|={np.mean(np.abs(y0)):.3f}  mean |y1|={np.mean(np.abs(y1)):.3f} "
          f"(treatment lower magnitude), bounded")

    # ---- Randomization: the ONLY source of randomness ----
    rng2 = np.random.default_rng(SEED_ASSIGN)
    W = (rng2.random((R, N)) < PI).astype(np.float64)   # 1 = treatment (arm 1)
    obs1 = W
    obs0 = 1.0 - W
    pi1 = PI
    pi0 = 1.0 - PI
    Z1 = obs1 / pi1 - 1.0                                # (R,N) arm-1 IPW centering weights
    Z0 = obs0 / pi0 - 1.0                                # (R,N) arm-0

    results = {"config": {"N": N, "R": R, "alpha": ALPHA, "pi": PI, "eta": ETA,
                          "seed_design": SEED_DESIGN, "seed_assign": SEED_ASSIGN}}

    # =====================================================================
    # (A) Per-arm martingale, single-arm event-time order (use arm w=1)
    # =====================================================================
    o1 = np.argsort(t_pot1, kind="stable")              # order units by arm-1 event time
    y1o = y1[o1]
    Z1o = Z1[:, o1]                                      # (R,N) reordered
    dM1 = Z1o * y1o                                      # (R,N) per-event increments dM_k
    M1 = np.cumsum(dM1, axis=1)                          # (R,N) error at each arm-1 event time
    truth_r1 = np.cumsum(y1o)                            # (N,) fixed true cumulative reward

    # (A1) increment conditional-mean-zero: E[dM_k] over reps ~ 0 for every k
    dbar = dM1.mean(axis=0)                              # (N,)
    dse = dM1.std(axis=0, ddof=1) / np.sqrt(R)
    zk = dbar / dse
    a1_max_abs_z = float(np.max(np.abs(zk)))
    a1_frac_within2 = float(np.mean(np.abs(zk) <= 2.0))
    # aggregate terminal-error mean (martingale with M_0=0 => E[M_T]=0)
    mT = M1[:, -1]
    a1_termmean = float(mT.mean())
    a1_term_z = float(mT.mean() / (mT.std(ddof=1) / np.sqrt(R)))

    # (A2) increments unpredictable from the past: Cov(M_{k-1}, dM_k) ~ 0.
    #      Pool a representative set of steps and report the worst standardized z.
    step_idx = np.unique(np.linspace(N // 5, N - 1, 8).astype(int))
    a2_worst_z = 0.0
    a2_worst_slope = 0.0
    for k in step_idx:
        past = M1[:, k - 1]
        incr = dM1[:, k]
        c, se = cov_se(past, incr)
        z = c / se if se > 0 else 0.0
        slope = c / past.var()
        if abs(z) > abs(a2_worst_z):
            a2_worst_z = z
            a2_worst_slope = slope

    # (A3) Lemma 4.7 identity on a calendar grid, single arm.
    #      grid of calendar times spanning the observed horizon.
    tmax = min(t_pot0.max(), t_pot1.max())
    grid = np.quantile(np.concatenate([t_pot0, t_pot1]),
                       np.linspace(0.15, 0.98, 12))
    # M_t(w) at each grid time = sum over units with t_pot_w<=g of Z*y
    def arm_process_on_grid(Z, tpot, y):
        out = np.empty((R, grid.size))
        for j, g in enumerate(grid):
            mask = (tpot <= g).astype(np.float64)
            out[:, j] = (Z * (mask * y)).sum(axis=1)
        return out
    M1g = arm_process_on_grid(Z1, t_pot1, y1)           # (R,Q) single-arm error, arm 1
    M0g = arm_process_on_grid(Z0, t_pot0, y0)           # (R,Q) single-arm error, arm 0
    Dg = M1g - M0g                                      # (R,Q) treatment-effect error

    Q = grid.size
    a3_max_ratdev = 0.0
    a3_max_abs_z = 0.0
    a3_example = None
    for si in range(Q):
        for ti in range(si + 1, Q):
            Ms = M1g[:, si]
            Mt = M1g[:, ti]
            var_s = Ms.var()
            if var_s <= 0:
                continue
            cov_st, _ = cov_se(Ms, Mt)
            ratio = cov_st / var_s
            # orthogonality of increment to past (equivalent identity test)
            c_inc, se_inc = cov_se(Ms, Mt - Ms)
            z = c_inc / se_inc if se_inc > 0 else 0.0
            if abs(ratio - 1.0) > a3_max_ratdev:
                a3_max_ratdev = abs(ratio - 1.0)
            if abs(z) > abs(a3_max_abs_z):
                a3_max_abs_z = z
                a3_example = (float(grid[si]), float(grid[ti]), float(ratio), float(z))

    # =====================================================================
    # (B) NEGATIVE result: difference error violates Lemma 4.7 (Prop 4.8)
    # =====================================================================
    b_max_ratdev = 0.0
    b_max_abs_z = 0.0
    b_example = None
    for si in range(Q):
        for ti in range(si + 1, Q):
            Ds = Dg[:, si]
            Dt = Dg[:, ti]
            var_s = Ds.var()
            if var_s <= 0:
                continue
            cov_st, _ = cov_se(Ds, Dt)
            ratio = cov_st / var_s
            c_inc, se_inc = cov_se(Ds, Dt - Ds)
            z = c_inc / se_inc if se_inc > 0 else 0.0
            if abs(ratio - 1.0) > b_max_ratdev:
                b_max_ratdev = abs(ratio - 1.0)
            if abs(z) > abs(b_max_abs_z):
                b_max_abs_z = z
                b_example = (float(grid[si]), float(grid[ti]), float(ratio), float(z))

    # =====================================================================
    # (C) Anytime-valid coverage of the single-arm CS vs naive pointwise CI
    # =====================================================================
    # V^hat_t(1) at each arm-1 event time (Thm 4.6 / eq 19), IPW form.
    vinc1 = (1.0 - pi1) * (obs1[:, o1] / pi1**2) * (y1o**2)   # (R,N)
    Vhat1 = np.cumsum(vinc1, axis=1)                          # (R,N)
    b1 = boundary(Vhat1)                                     # (R,N) mixture boundary
    # covered_cs at step k  <=>  |r^hat - r| = |M1| <= b1
    covered_cs = np.abs(M1) <= b1                            # (R,N)
    cs_uniform_cover = float(covered_cs.all(axis=1).mean())  # cover at ALL times
    cs_uniform_miscover = 1.0 - cs_uniform_cover

    # naive fixed-n pointwise CI: r^hat_t +/- 1.96*sqrt(V^hat_t)
    zcrit = 1.959963984540054
    pw = zcrit * np.sqrt(Vhat1)
    covered_pw = np.abs(M1) <= pw
    pw_uniform_cover = float(covered_pw.all(axis=1).mean())  # continuous monitoring
    pw_uniform_miscover = 1.0 - pw_uniform_cover
    # single-look pointwise coverage (control): at a fixed mid/late time index
    kmid = int(0.5 * N)
    klate = N - 1
    pw_single_mid = float(covered_pw[:, kmid].mean())
    pw_single_late = float(covered_pw[:, klate].mean())
    # for reference: number of looks (event times where V>0)
    n_looks = int((Vhat1[0] > 0).sum())

    # ---------------------------------------------------------------- print
    print("\n" + "-" * 74)
    print("(A) PER-ARM IPW ERROR IS A MARTINGALE wrt single-arm filtration (arm 1)")
    print("-" * 74)
    print(f"  A1 increment E[dM_k|past]~0 : max|z_k| over {N} event steps = {a1_max_abs_z:.2f} "
          f"(frac |z|<=2 = {a1_frac_within2:.3f})")
    print(f"     terminal error mean E[M_T]  = {a1_termmean:+.4f}  (z = {a1_term_z:+.2f}; expect 0)")
    print(f"  A2 increment unpredictable  : worst Cov(M_(k-1),dM_k) z = {a2_worst_z:+.2f} "
          f"(slope = {a2_worst_slope:+.2e}; expect 0)")
    print(f"  A3 Lemma4.7 Cov(M_s,M_t)=Var(M_s): max|ratio-1| = {a3_max_ratdev:.4f}, "
          f"max|z| = {a3_max_abs_z:+.2f}")
    if a3_example:
        print(f"     example (s={a3_example[0]:.2f}, t={a3_example[1]:.2f}): "
              f"ratio={a3_example[2]:.4f}  z={a3_example[3]:+.2f}")
    a1_ok = (a1_max_abs_z <= 4.0) and (abs(a1_term_z) <= 4.0)
    a2_ok = abs(a2_worst_z) <= 4.0
    a3_ok = abs(a3_max_abs_z) <= 4.0   # Lemma 4.7 identity holds within MC error
    print(f"  => A1 {'PASS' if a1_ok else 'FAIL'} | A2 {'PASS' if a2_ok else 'FAIL'} | "
          f"A3 {'PASS' if a3_ok else 'FAIL'}  (martingale property holds)")

    print("\n" + "-" * 74)
    print("(B) NEGATIVE RESULT: treatment-effect (difference) error is NOT a martingale")
    print("    under ANY filtration  (Prop 4.8: Cov(D_s,D_t) != Var(D_s))")
    print("-" * 74)
    print(f"  max|ratio-1| = {b_max_ratdev:.4f}, max standardized |z| = {b_max_abs_z:+.2f}")
    if b_example:
        print(f"     example (s={b_example[0]:.2f}, t={b_example[1]:.2f}): "
              f"Cov/Var ratio={b_example[2]:.4f}  z={b_example[3]:+.2f}  (!= 1 => violation)")
    b_ok = abs(b_max_abs_z) >= 6.0
    print(f"  => {'PASS' if b_ok else 'FAIL'} (violation is decisive; martingale fails for the difference)")

    print("\n" + "-" * 74)
    print("(C) ANYTIME-VALID coverage from the martingale CS  vs  naive pointwise CI")
    print(f"    single-arm CS (Thm 4.6) checked at all {n_looks} event times (continuous monitoring)")
    print("-" * 74)
    print(f"  confidence SEQUENCE  uniform coverage = {cs_uniform_cover*100:.2f}%  "
          f"(miscover {cs_uniform_miscover*100:.2f}% <= {ALPHA*100:.0f}%?  "
          f"{'yes' if cs_uniform_miscover <= ALPHA else 'NO'})")
    print(f"  naive pointwise CI   uniform coverage = {pw_uniform_cover*100:.2f}%  "
          f"(miscover {pw_uniform_miscover*100:.2f}% -> over-rejects under optional stopping)")
    print(f"  naive pointwise CI   single-look coverage: mid={pw_single_mid*100:.2f}%  "
          f"late={pw_single_late*100:.2f}%  (~{(1-ALPHA)*100:.0f}% => pointwise valid, only sequential use breaks)")
    c_ok = (cs_uniform_miscover <= ALPHA) and (pw_uniform_cover < 1 - ALPHA) \
        and (pw_single_late >= 1 - ALPHA - 0.03)
    print(f"  => {'PASS' if c_ok else 'FAIL'} (martingale CS is anytime-valid; naive CI is not)")

    overall = a1_ok and a2_ok and a3_ok and b_ok and c_ok
    print("\n" + "=" * 74)
    print(f"OVERALL CLAIM 1: {'VERIFIED' if overall else 'NOT FULLY VERIFIED'}")
    print("  per-arm IPW error IS a martingale wrt the single-arm filtration (A);")
    print("  the treatment-effect error is NOT a martingale under any filtration (B);")
    print("  the martingale CS is anytime-valid while the naive CI over-rejects (C).")
    print("=" * 74)

    results.update({
        "A1_increment_max_abs_z": a1_max_abs_z,
        "A1_frac_z_within2": a1_frac_within2,
        "A1_terminal_error_mean": a1_termmean,
        "A1_terminal_error_z": a1_term_z,
        "A2_worst_predict_z": float(a2_worst_z),
        "A2_worst_slope": float(a2_worst_slope),
        "A3_single_arm_max_ratio_dev": float(a3_max_ratdev),
        "A3_single_arm_max_abs_z": float(a3_max_abs_z),
        "A3_example_s_t_ratio_z": a3_example,
        "B_diff_max_ratio_dev": float(b_max_ratdev),
        "B_diff_max_abs_z": float(b_max_abs_z),
        "B_example_s_t_ratio_z": b_example,
        "C_cs_uniform_coverage": cs_uniform_cover,
        "C_cs_uniform_miscoverage": cs_uniform_miscover,
        "C_pw_uniform_coverage": pw_uniform_cover,
        "C_pw_uniform_miscoverage": pw_uniform_miscover,
        "C_pw_single_look_mid": pw_single_mid,
        "C_pw_single_look_late": pw_single_late,
        "C_n_looks": n_looks,
        "pass_A1": bool(a1_ok), "pass_A2": bool(a2_ok), "pass_A3": bool(a3_ok),
        "pass_B": bool(b_ok), "pass_C": bool(c_ok),
        "overall_verified": bool(overall),
        "runtime_s": round(time.time() - t0wall, 3),
    })
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote results.json  (runtime {results['runtime_s']}s)")


if __name__ == "__main__":
    main()
