"""
Claim 3 (MaxRL, OpenReview EeuLO2BjFN / arXiv 2602.02710):

  "[the MaxRL objectives] converge to maximum likelihood optimization in the
   infinite-compute limit."  (Abstract)

MaxRL family:  J_N(theta) = E_{a_1..a_N ~ pi}[ log((1/N) sum_i r_{a_i}) ].
Exact maximum-likelihood objective:  J_ML(theta) = log E[r] = log p.

We verify (i) J_N -> J_ML = log p as N -> infinity, and (ii) the convergence
RATE is O(1/N): the gap Delta_N = log p - J_N obeys Delta_N ~ C/N with
C = Var_{a~pi}(r_a)/(2 p^2) (second-order delta method), so the log-log slope of
Delta_N vs N is ~ -1.  Everything is measured (Monte-Carlo J_N) and cross-checked
against the analytic constant; the analytic Box-Cox value family and the
gradient gap are shown to converge at the same O(1/N) rate.
"""
import json, numpy as np

def softmax(theta):
    z = theta - theta.max(); e = np.exp(z); return e / e.sum()

def J_N_mc(theta, r, N, groups, rng, chunk=8000):
    pi = softmax(theta); cdf = np.cumsum(pi); cdf[-1] = 1.0
    tot = 0.0; tot2 = 0.0; done = 0
    while done < groups:
        b = min(chunk, groups - done)
        u = rng.random((b, N))
        idx = np.searchsorted(cdf, u); np.clip(idx, 0, len(r) - 1, out=idx)
        lm = np.log(r[idx].mean(axis=1))
        tot += lm.sum(); tot2 += (lm * lm).sum(); done += b
    mean = tot / groups
    var = tot2 / groups - mean * mean
    return float(mean), float(np.sqrt(max(var, 0.0) / groups))   # mean, standard error

def main():
    out = {"claim": "MaxRL converges to maximum-likelihood optimization in the infinite-compute limit (rate O(1/N))",
           "tests": {}}
    print("=" * 74)
    print("MaxRL Claim 3 — infinite-compute limit = maximum likelihood  (EeuLO2BjFN)")
    print("=" * 74)

    rng = np.random.default_rng(3)
    K = 10
    r = np.round(rng.uniform(0.05, 1.0, size=K), 4)
    theta = rng.normal(0, 1.0, size=K)
    pi = softmax(theta)
    p = float(pi @ r)
    logp = float(np.log(p))
    var_r = float(pi @ (r - p) ** 2)                 # Var_{a~pi}(r_a)
    C_delta = var_r / (2 * p ** 2)                    # predicted gap constant
    print(f"K={K}, p=E[r]={p:.6f}, log p (ML objective) = {logp:.6f}")
    print(f"Var_pi(r) = {var_r:.6f}  ->  delta-method gap constant C = Var/(2 p^2) = {C_delta:.6f}")
    print(f"Prediction: Delta_N = log p - J_N  ~  C/N  (log-log slope -1)\n")

    # ---- measured sample-based gap Delta_N -----------------------------------
    Ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    G = 400_000
    rng2 = np.random.default_rng(999)
    rows = []
    print(f"  {'N':>6} {'J_N (MC)':>12} {'Delta_N=logp-J_N':>18} {'N*Delta_N':>12} {'SE':>10}")
    for N in Ns:
        JN, se = J_N_mc(theta, r, N, G, rng2)
        d = logp - JN
        rows.append((N, JN, d, N * d, se))
        print(f"  {N:6d} {JN:12.6f} {d:18.6e} {N*d:12.6f} {se:10.2e}")
    Ns_a = np.array([x[0] for x in rows], float)
    D_a = np.array([x[2] for x in rows], float)
    # fit slope over a clean window (small enough that MC gap >> SE)
    win = (Ns_a >= 4) & (Ns_a <= 256)
    slope, intercept = np.polyfit(np.log10(Ns_a[win]), np.log10(D_a[win]), 1)
    NdeltaN = np.array([x[3] for x in rows])
    mean_NdeltaN_win = float(np.mean(NdeltaN[(Ns_a >= 16) & (Ns_a <= 256)]))
    print(f"\n  measured log-log slope of Delta_N vs N over [4,256] = {slope:.4f}  (target ~ -1)")
    print(f"  measured N*Delta_N over [16,256] ~ {mean_NdeltaN_win:.5f}   vs predicted C = {C_delta:.5f}")
    rate_ok = -1.2 <= slope <= -0.8
    const_ok = abs(mean_NdeltaN_win - C_delta) / C_delta < 0.20
    conv_ok = D_a[-1] < D_a[0] and D_a[-1] < 5e-3     # converged toward 0
    out["tests"]["sample_based_gap"] = dict(
        p=p, log_p=logp, var_r=var_r, C_delta=C_delta, G=G,
        Ns=[int(x) for x in Ns_a], J_N=[x[1] for x in rows], Delta_N=D_a.tolist(),
        N_times_Delta=NdeltaN.tolist(), SE=[x[4] for x in rows],
        loglog_slope_4_256=float(slope), mean_NDelta_16_256=mean_NdeltaN_win,
        rate_slope_ok=bool(rate_ok), const_ok=bool(const_ok), converged=bool(conv_ok))

    # ---- analytic Box-Cox value gap (independent O(1/N) confirmation) ----------
    lam = 1.0 / Ns_a
    phi = (p ** lam - 1.0) / lam
    gap_phi = phi - logp                              # (p^lam-1)/lam - log p ~ ln^2 p /(2N)
    sphi, _ = np.polyfit(np.log10(Ns_a), np.log10(gap_phi), 1)
    print(f"\n  analytic Box-Cox value gap slope (exact family) = {sphi:.4f}  "
          f"(N*gap -> ln^2 p /2 = {logp**2/2:.5f}; measured {float(Ns_a[-1]*gap_phi[-1]):.5f})")
    out["tests"]["boxcox_value_gap"] = dict(loglog_slope=float(sphi),
        pred_const_ln2p_over_2=float(logp ** 2 / 2), N_times_gap_last=float(Ns_a[-1] * gap_phi[-1]),
        slope_ok=bool(-1.1 <= sphi <= -0.9))

    # ---- analytic gradient gap ||g_N - g_ML|| ~ O(1/N) ------------------------
    # MaxRL-N per-problem gradient weight is p^{1/N}; g_N = p^{1/N} g_ML, so
    # ||g_N - g_ML|| = |p^{1/N}-1| * ||g_ML||  ~  |ln p|/N.
    gw_gap = np.abs(p ** (1.0 / Ns_a) - 1.0)
    sg, _ = np.polyfit(np.log10(Ns_a), np.log10(gw_gap), 1)
    print(f"  analytic gradient-gap slope |p^(1/N)-1| = {sg:.4f}  (target -1; N*gap -> |ln p| = {abs(logp):.5f})")
    out["tests"]["gradient_gap"] = dict(loglog_slope=float(sg), pred_const_abs_lnp=float(abs(logp)),
        N_times_gap_last=float(Ns_a[-1] * gw_gap[-1]), slope_ok=bool(-1.1 <= sg <= -0.9))

    verdict = rate_ok and const_ok and conv_ok and (-1.1 <= sphi <= -0.9) and (-1.1 <= sg <= -0.9)
    out["verdict"] = dict(infinite_compute_limit_is_ML=bool(verdict),
        summary=f"J_N -> log p at O(1/N): measured slope {slope:.3f}, N*Delta -> delta-method C={C_delta:.4f}; analytic value & gradient gaps slope ~ -1")
    print("\n" + "=" * 74)
    print(f"CLAIM 3 verified: J_N -> ML (log p) at rate O(1/N): {verdict}")
    print("=" * 74)
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json")

if __name__ == "__main__":
    main()
