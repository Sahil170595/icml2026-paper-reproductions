#!/usr/bin/env python3
"""
Claim 3 (Section 4.1 / Proposition 4.1): for a finite number of distinct packet
types K>=2, ALG^theta breaks the golden-ratio barrier Phi=(1+sqrt5)/2 (Hajek 2001)
and attains a competitive ratio theta_K in [sqrt(2), Phi), with theta_2=sqrt(2),
theta_3=3/2, theta_K strictly increasing and theta_K -> Phi as K -> infinity.

Paper: arXiv 2606.00835 (rZTiFcDihH). theta_K is the unique root in (1,Phi) of the
Hajek system (Eq. 1):
    x_0=1, x_1=1/(theta-1),
    x_j=((theta+1)/(theta-1))(x_{j-1}-x_{j-2})  (2<=j<=K-1),
    boundary  x_{K-1}=(theta+1) x_{K-2}.

Two deterministic (seeded) CPU experiments:
 (A) Solve the recurrence for K=2..30; check theta_2=sqrt2, theta_3=3/2 (machine
     precision), monotone-increasing, sqrt(2) <= theta_K < Phi, theta_K -> Phi.
 (B) Faithful ALG^theta (Algorithm 2, position-dependent thresholds x_j/x_{j+1})
     on random K-type 2-bounded instances: empirical competitive ratio G_OPT/G_ALG
     stays strictly below Phi (barrier broken) and <= theta_K (Prop 4.1).
"""
import json, math, time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from collections import defaultdict

OUT = Path(__file__).with_name("results.json")
PHI = (1.0 + 5.0**0.5) / 2.0
SQRT2 = math.sqrt(2.0)
t0 = time.time()

def solve(K):
    """Return (theta_K, x[0..K-1]) solving the Hajek system (Eq. 1)."""
    def resid(t):
        x = np.zeros(K); x[0] = 1.0; x[1] = 1.0/(t-1.0); r = (t+1.0)/(t-1.0)
        for j in range(2, K): x[j] = r*(x[j-1]-x[j-2])
        return x[K-1] - (t+1.0)*x[K-2]
    th = brentq(resid, 1.35, PHI-1e-13, xtol=1e-15, rtol=1e-15)  # all theta_K in [sqrt2,Phi)
    x = np.zeros(K); x[0] = 1.0; x[1] = 1.0/(th-1.0); r = (th+1.0)/(th-1.0)
    for j in range(2, K): x[j] = r*(x[j-1]-x[j-2])
    return th, x

def opt_2b(pk, T):
    """Exact offline optimum (max-weight unit-job schedule, transversal matroid greedy)."""
    free = [False]*(T+2); tot = 0.0
    for (r, d, w) in sorted(pk, key=lambda p: -p[2]):
        s = min(d, T)
        while s >= r:
            if not free[s]: free[s] = True; tot += w; break
            s -= 1
    return tot

def alg_theta(pk, x):
    """Faithful ALG^theta (Algorithm 2). Epoch counter j; threshold x_j/x_{j+1} (x_K:=x_{K-1}).
    Schedule slack b_t iff w(v_t) < (x_j/x_{j+1}) w(b_t); epoch continues iff v_t scheduled and
    w(v_t) < w(b_t) (paper text: epoch ends when v_t scheduled with v_t >= b_t)."""
    K = len(x); T = max(d for (_, d, _) in pk)
    rel = defaultdict(list)
    for (r, d, w) in pk: rel[r].append([d, w])
    def rt(j): return x[j]/(x[j+1] if j+1 < K else x[K-1])
    buf = []; j = 0; tot = 0.0
    for t in range(1, T+1):
        buf += rel.get(t, []); buf = [p for p in buf if p[0] >= t]
        if not buf: continue
        V = [p for p in buf if p[0] == t]; B = [p for p in buf if p[0] == t+1]
        v = max(V, key=lambda p: p[1]) if V else None
        b = max(B, key=lambda p: p[1]) if B else None
        vw = v[1] if v else 0.0; bw = b[1] if b else 0.0
        if vw < rt(j)*bw and b is not None: ch = b; sb = True
        else: ch = v if v is not None else b; sb = (v is None)
        tot += ch[1]; buf.remove(ch)
        j = 0 if sb else (min(j+1, K-1) if (b is not None and v is not None and vw < bw) else 0)
    return tot, T

def main():
    res = {"paper": "arXiv 2606.00835 (rZTiFcDihH)", "phi": PHI, "sqrt2": SQRT2}

    # ---- (A) recurrence: theta_K in [sqrt2, Phi) ----
    Ks = list(range(2, 31))
    th = {K: solve(K)[0] for K in Ks}
    recA = {
        "theta_2": th[2], "theta_2_err_vs_sqrt2": abs(th[2]-SQRT2),
        "theta_3": th[3], "theta_3_err_vs_1.5": abs(th[3]-1.5),
        "theta_30": th[30], "phi_minus_theta_30": PHI-th[30],
        "all_theta_K": {K: th[K] for K in Ks},
        "monotone_increasing": all(th[Ks[i]] < th[Ks[i+1]] for i in range(len(Ks)-1)),
        "all_ge_sqrt2": all(th[K] >= SQRT2-1e-12 for K in Ks),
        "all_below_phi": all(th[K] < PHI for K in Ks),
    }
    res["experimentA_recurrence"] = recA
    print("=== (A) recurrence: theta_K in [sqrt2, Phi) ===")
    print(f"theta_2 = {th[2]:.12f}   (sqrt2={SQRT2:.12f}, err {abs(th[2]-SQRT2):.1e})")
    print(f"theta_3 = {th[3]:.12f}   (3/2, err {abs(th[3]-1.5):.1e})")
    print(f"theta_30= {th[30]:.12f}  (Phi={PHI:.12f}, Phi-theta_30={PHI-th[30]:.2e})")
    print(f"monotone increasing (K=2..30): {recA['monotone_increasing']}")
    print(f"all sqrt2 <= theta_K < Phi   : {recA['all_ge_sqrt2'] and recA['all_below_phi']}")
    for K in [2,3,4,5,8,12,20,30]:
        print(f"   theta_{K:<2d} = {th[K]:.9f}   in[sqrt2,Phi): {SQRT2-1e-12<=th[K]<PHI}")

    # ---- (B) faithful ALG^theta stays below Phi and <= theta_K on random K-type instances ----
    print("\n=== (B) ALG^theta empirical competitive ratio: < Phi (barrier broken) and <= theta_K ===")
    rng = np.random.default_rng(20260717)
    B = {}
    for K in range(2, 9):
        thK, x = solve(K)
        wvals = np.sort(rng.random(K)*9 + 1.0)
        worst = 0.0; nviol_phi = 0; nviol_th = 0; N = 2500
        for _ in range(N):
            Tn = int(rng.integers(2, 8)); pk = []
            for s in range(1, Tn+1):
                for _ in range(int(rng.integers(0, 3))):
                    pk.append((s, s+int(rng.integers(0, 2)), float(wvals[rng.integers(0, K)])))
            if not pk: continue
            a, T = alg_theta(pk, x)
            if a <= 0: continue
            r = opt_2b(pk, T)/a
            if r > worst: worst = r
            nviol_phi += (r > PHI + 1e-9); nviol_th += (r > thK + 1e-9)
        B[K] = {"theta_K": thK, "max_ratio": worst, "below_phi": bool(worst < PHI),
                "phi_violations": nviol_phi, "thetaK_violations": nviol_th, "n_instances": N}
        print(f"K={K}: theta_K={thK:.6f} Phi={PHI:.6f} | max ratio={worst:.6f} "
              f"| <Phi: {worst<PHI} (viol {nviol_phi}/{N}) | <=theta_K viol {nviol_th}/{N}")
    res["experimentB_alg_below_phi"] = B
    res["runtime_s"] = round(time.time()-t0, 2)
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nWrote {OUT}  ({res['runtime_s']}s)")

if __name__ == "__main__":
    main()
