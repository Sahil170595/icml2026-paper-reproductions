#!/usr/bin/env python3
"""
Claim 5 (headline / abstract; resolution of the open problem): under M-convex action sets the
regret is FINITE -- bounded INDEPENDENT of the horizon T -- whereas all prior bounds grew as
O(d log T).  This is the paper's central contribution ("Whether a finite regret bound polynomial
in d is achievable ... has remained an open question. We partially resolve this ...").  Both the
O(d^2) (Thm 3.1) and O(d log d) (Thm 4.2) bounds are T-free; the checkable consequence is that
cumulative regret R_T PLATEAUS as T -> infinity and never grows with T.

Faithful realization.  Fixed d, fixed hidden order w*.  A stream of two-action M-convex sets
X_t = {e_i, e_j} (uniformly random pairs) arrives for T rounds.  We track cumulative regret R_T
for the topological-sort learner (Algorithm 1) and the center-of-gravity learner (Thm 4.2).
Because regret can only be incurred while some queried pair is still undetermined, once the
learner has recovered the full order (A_t is a total order) EVERY later round is predicted
correctly and R_T is frozen -- so R_T is computed exactly for all T up to 10^6 by simulating
until determination and holding the value constant thereafter.  Reference: prior art d*ln(T).

TARGET: R_T bounded independent of T; R_T <= C(d,2) (toposort), <= log_{e/(e-1)}(d!) (centroid);
d*ln(T) diverges.
ACCEPTANCE RULE (all):
  (A) PLATEAU: R_{10^6} == R_{10^4} == R_{10^3}, ratio R_{10^6}/R_{10^3} in [0.98,1.02];
  (B) finite cap: R_T <= C(d,2) (toposort) and <= log_{e/(e-1)}(d!) (centroid) at all T;
  (C) prior O(d log T) reference grows >= 1.5x over T in [10^3,10^6] while ours stays flat.
FALSIFIED if R_T keeps growing with T (ratio R_{10^6}/R_{10^3} > 1.05).
"""
import numpy as np, json, time, os, math
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

def transitive_add(R, i, j, d):
    if R[i, j]: return
    anc = np.where(R[:, i])[0]; anc = np.append(anc, i)
    desc = np.where(R[j, :])[0]; desc = np.append(desc, j)
    for a in anc: R[a, desc] = True

def gibbs_centroid(R, d, rng, sweeps=90, burn=25):
    downs = [np.where(R[x, :])[0] for x in range(d)]
    ups   = [np.where(R[:, x])[0] for x in range(d)]
    w = np.full(d, 0.5); acc = np.zeros(d); n = 0
    for s in range(sweeps):
        for x in range(d):
            lo = 0.0 if downs[x].size == 0 else max(0.0, w[downs[x]].max())
            hi = 1.0 if ups[x].size == 0 else min(1.0, w[ups[x]].min())
            if hi < lo: hi = lo
            w[x] = rng.uniform(lo, hi)
        if s >= burn: acc += w; n += 1
    return acc / max(n, 1)

def run_stream(d, seed, Tmax, learner):
    """Cumulative regret over a random two-action stream, snapshots at powers of ten.
    Early-stop: once A_t is a total order (R.sum()==C(d,2)) regret is frozen, so all larger
    checkpoints inherit the frozen value (exact, T-independent by construction)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(d); rank = np.empty(d, int); rank[order] = np.arange(d)
    R = np.zeros((d, d), bool); reg = 0
    binom = d * (d - 1) // 2
    checkpoints = [100, 1000, 10000, 100000, 1000000]
    snaps = {}; ci = 0; determined_at = None
    T = 0
    while T < Tmax:
        T += 1
        # if fully determined, all remaining rounds add 0 regret: fill checkpoints and stop
        if R.sum() == binom:
            determined_at = determined_at or T
            while ci < len(checkpoints):
                snaps[checkpoints[ci]] = reg; ci += 1
            break
        a, b = int(rng.integers(0, d)), int(rng.integers(0, d))
        while a == b: b = int(rng.integers(0, d))
        if R[a, b]:    pred = True
        elif R[b, a]:  pred = False
        else:                                   # undetermined pair: learner must predict
            if learner == "toposort":
                pred = (a < b)                      # arbitrary tie-break
            else:
                w = gibbs_centroid(R, d, rng); pred = bool(w[a] > w[b])   # center of gravity
        true = (rank[a] < rank[b])
        if pred != true: reg += 1
        i, j = (a, b) if true else (b, a)
        transitive_add(R, i, j, d)
        if ci < len(checkpoints) and T == checkpoints[ci]:
            snaps[T] = reg; ci += 1
    # if loop ended without hitting all checkpoints (Tmax reached before determination), fill rest
    while ci < len(checkpoints):
        snaps[checkpoints[ci]] = reg; ci += 1
    return snaps, determined_at

def grunbaum(d):
    return math.lgamma(d + 1) / math.log(math.e / (math.e - 1))

def main():
    t0 = time.time()
    Tmax = 1_000_000
    cps = [100, 1000, 10000, 100000, 1000000]
    out = {}
    for d in [12, 20]:
        topo, det_t = run_stream(d, 3, Tmax, "toposort")
        cent, det_c = run_stream(d, 3, Tmax, "centroid")
        binom = d * (d - 1) // 2; gru = grunbaum(d)
        dlnT = {T: round(d * math.log(T), 1) for T in cps}
        plateau_topo = topo[1000000] / max(topo[1000], 1)
        plateau_cent = cent[1000000] / max(cent[1000], 1)
        out[d] = {
            "binom_d_2": binom, "grunbaum_bound": round(gru, 1),
            "R_T_toposort": {str(T): topo[T] for T in cps},
            "R_T_centroid": {str(T): cent[T] for T in cps},
            "prior_reference_d_lnT": {str(T): dlnT[T] for T in cps},
            "order_determined_at_round_topo": det_t, "order_determined_at_round_cent": det_c,
            "plateau_ratio_toposort_1e6_over_1e3": round(plateau_topo, 4),
            "plateau_ratio_centroid_1e6_over_1e3": round(plateau_cent, 4),
            "prior_dlnT_growth_1e3_to_1e6": round(dlnT[1000000] / dlnT[1000], 3),
            "toposort_le_binom": bool(topo[1000000] <= binom),
            "centroid_le_grunbaum": bool(cent[1000000] <= gru),
        }
    p = out[20]
    verdict_A = bool(0.98 <= p["plateau_ratio_toposort_1e6_over_1e3"] <= 1.02 and 0.98 <= p["plateau_ratio_centroid_1e6_over_1e3"] <= 1.02)
    verdict_B = bool(p["toposort_le_binom"] and p["centroid_le_grunbaum"] and out[12]["toposort_le_binom"] and out[12]["centroid_le_grunbaum"])
    verdict_C = bool(p["prior_dlnT_growth_1e3_to_1e6"] >= 1.5)
    res = {
        "claim": "Headline: under M-convex sets regret is FINITE (T-independent), resolving the open problem; prior bounds were O(d log T)",
        "target": "R_T bounded independent of T; R_T<=C(d,2) (toposort), <=log_{e/(e-1)}(d!) (centroid); d*ln(T) diverges",
        "acceptance_rule": "(A) plateau ratio R_1e6/R_1e3 in [0.98,1.02]; (B) R_T<=finite cap at all T; (C) prior d*ln(T) grows >=1.5x",
        "Tmax": Tmax, "by_dimension": out,
        "verdict_rule_A_plateau": verdict_A, "verdict_rule_B_finite_cap": verdict_B, "verdict_rule_C_prior_grows": verdict_C,
        "runtime_sec": round(time.time() - t0, 2), "numpy_version": np.__version__,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results.json"), "w") as f: json.dump(res, f, indent=2)
    print("== Claim 5 (headline): FINITE (T-independent) regret, open-problem resolution ==")
    for d in [12, 20]:
        o = out[d]
        print(f"-- d={d}  binom={o['binom_d_2']}  Grunbaum={o['grunbaum_bound']}  (order learned by round ~{o['order_determined_at_round_topo']}) --")
        print(f"  {'T':>9} {'R_T topo':>9} {'R_T cent':>9} {'d*ln T prior':>14}")
        for T in cps:
            print(f"  {T:>9} {o['R_T_toposort'][str(T)]:>9} {o['R_T_centroid'][str(T)]:>9} {o['prior_reference_d_lnT'][str(T)]:>14}")
        print(f"  plateau R_1e6/R_1e3: topo={o['plateau_ratio_toposort_1e6_over_1e3']} cent={o['plateau_ratio_centroid_1e6_over_1e3']}  | prior d*lnT grows x{o['prior_dlnT_growth_1e3_to_1e6']}")
    print(f"verdicts: A(plateau)={verdict_A} B(finite cap)={verdict_B} C(prior grows)={verdict_C}")
    print(f"runtime {res['runtime_sec']}s  numpy {np.__version__}")

if __name__ == "__main__":
    main()
