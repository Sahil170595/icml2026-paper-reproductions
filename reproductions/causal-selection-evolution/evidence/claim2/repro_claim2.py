"""
Claim 2 (Lemma 1) - "Causal Modeling of Selection in Evolution" (OpenReview mOcTXKawFY).

CLAIM: Lemma 1 shows repeated evolutionary selection induces conditional dependencies among
variables that are ABSENT under static selection, so applying static-selection graphical
models to evolutionary data can yield FALSE causal discoveries.

MECHANISM ("Bulmer effect" / selection-induced association): the phenotype
X^(t) = sum_k eps_k^(t) + env is a COLLIDER of its heritable factors. Truncation selection
S^(t)=1{X^(t)>c} conditions on a descendant of that collider, inducing (negative) dependence
among the heritable factors eps_k^(t) that are marginally independent without selection.
Repeated selection (every generation) spreads this across generations; one-shot static
selection (single event) does not.

CHECKABLE CONSEQUENCES (executed numbers):
  (A) Un-selected: cross-chain factor pairs are independent (|corr|~0, CI test ACCEPTS).
  (B) Evolutionary selection (condition on all S^(0..T-1)): the SAME pairs become strongly
      dependent (|corr| large, Fisher-z p<1e-6) -> induced dependence.
  (C) One-shot static selection (final S* only): early-generation factor pairs stay
      ~independent -> the extra dependence is specific to REPEATED selection.
  (D) FALSE causal discoveries: a static-selection-blind skeleton search on evolutionary data
      draws spurious edges between factors that are NON-adjacent in the true causal DAG.

PASS RULE: (min|corr| induced pairs under evol > 0.05 with p<1e-6) AND (those pairs ~indep
unselected, max|corr|<0.02) AND (#spurious causal edges on evol data > on unselected).

Independent NumPy implementation, CPU-only, deterministic. Partial correlations are computed
from a correlation matrix built ONCE per dataset (fast, exact for linear-Gaussian CI tests).
"""
import numpy as np, json, os
from math import log, sqrt, erfc

def fac(k, t): return f"e{k}_{t}"
def Xn(t):     return f"X_{t}"

def simulate(T, K, rho, se, mopt, sw, n, seed, mode):
    """Gaussian stabilizing selection: reproduce with prob exp(-(X-mopt)^2/(2 sw^2)).
    Keeps the selected joint exactly Gaussian, so Fisher-z CI is valid.
    mode: 'none' (no selection), 'evol' (S^0..S^{T-1}), 'static' (final S only)."""
    rng = np.random.default_rng(seed)
    cols = {}
    prev = {k: rng.standard_normal(n) for k in range(K)}
    sel = np.ones(n, bool)
    for t in range(T + 1):
        if t > 0:
            for k in range(K):
                prev[k] = rho * prev[k] + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        for k in range(K):
            cols[fac(k, t)] = prev[k].copy()
        X = sum(prev[k] for k in range(K)) + se * rng.standard_normal(n)
        cols[Xn(t)] = X
        if t < T:
            accept = rng.random(n) < np.exp(-((X - mopt) ** 2) / (2.0 * sw ** 2))
            if mode == "evol":
                sel = sel & accept
            elif mode == "static" and t == T - 1:
                sel = sel & accept
    keep = {k: v[sel] for k, v in cols.items()}
    return keep, int(sel.sum())

class CIOracle:
    """Correlation-matrix-based partial-correlation CI test (linear-Gaussian)."""
    def __init__(self, data, names):
        self.names = names
        self.idx = {nm: i for i, nm in enumerate(names)}
        M = np.column_stack([data[nm] for nm in names]).astype(float)
        self.n = M.shape[0]
        self.R = np.corrcoef(M, rowvar=False)
    def pcorr(self, a, b, Z):
        ids = [self.idx[a], self.idx[b]] + [self.idx[z] for z in Z]
        sub = self.R[np.ix_(ids, ids)]
        P = np.linalg.pinv(sub)
        return float(-P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]))
    def pvalue(self, a, b, Z):
        r = self.pcorr(a, b, Z)
        r = min(max(r, -0.999999), 0.999999)
        z = 0.5 * log((1 + r) / (1 - r)) * sqrt(max(self.n - len(Z) - 3, 1))
        return float(erfc(abs(z) / sqrt(2)))

def naive_skeleton_spurious(oracle, factors, alpha=1e-6):
    """Static-blind skeleton search over heritable factors (order<=1 CI). Count edges between
    factors that are NON-adjacent in the true causal DAG (true adjacency = same index, |dt|=1)."""
    def true_adj(u, v):
        ku, tu = int(u[1]), int(u[3:]); kv, tv = int(v[1]), int(v[3:])
        return ku == kv and abs(tu - tv) == 1
    spurious = drawn = 0
    for i, u in enumerate(factors):
        for v in factors[i + 1:]:
            others = [w for w in factors if w not in (u, v)]
            indep = oracle.pvalue(u, v, ()) > alpha
            if not indep:
                for w in others:
                    if oracle.pvalue(u, v, (w,)) > alpha:
                        indep = True; break
            if not indep:
                drawn += 1
                if not true_adj(u, v):
                    spurious += 1
    return spurious, drawn

def main():
    T, K, rho, se, mopt, sw = 3, 3, 0.6, 0.7, 1.5, 1.2
    N = 1_500_000
    print("=" * 78)
    print("CLAIM 2  (Lemma 1)  repeated selection induces dependencies absent under static")
    print("paper: Causal Modeling of Selection in Evolution (mOcTXKawFY); independent NumPy")
    print("=" * 78)
    names = [fac(k, t) for t in range(T + 1) for k in range(K)] + [Xn(t) for t in range(T + 1)]
    d_none, n_none = simulate(T, K, rho, se, mopt, sw, N, 0, "none")
    d_evol, n_evol = simulate(T, K, rho, se, mopt, sw, N, 0, "evol")
    d_stat, n_stat = simulate(T, K, rho, se, mopt, sw, N, 0, "static")
    o_none, o_evol, o_stat = (CIOracle(d, names) for d in (d_none, d_evol, d_stat))
    print(f"samples kept: none={n_none}  evolutionary={n_evol}  static={n_stat}  (init pop N={N})")

    pairs = [("e0_0", "e1_0"), ("e0_0", "e2_0"), ("e1_0", "e2_0"),
             ("e0_1", "e1_1"), ("e0_0", "e1_2"), ("e0_0", "e2_2")]
    print("\n[induced conditional dependence] cross-chain factor pairs (causally NON-adjacent):")
    print(f"  {'pair':<16}{'corr:none':>11}{'corr:static':>13}{'corr:evol':>11}{'p_evol':>11}")
    rows = []
    for a, b in pairs:
        rn = o_none.pcorr(a, b, ()); rs = o_stat.pcorr(a, b, ()); re = o_evol.pcorr(a, b, ())
        pe = o_evol.pvalue(a, b, ())
        rows.append((a, b, rn, rs, re, pe))
        print(f"  {a+'-'+b:<16}{rn:>+11.4f}{rs:>+13.4f}{re:>+11.4f}{pe:>11.1e}")

    early = [r for r in rows if r[0].endswith("_0") and r[1].endswith("_0")]
    max_none = max(abs(r[2]) for r in rows)
    min_evol = min(abs(r[4]) for r in rows)
    max_p    = max(r[5] for r in rows)
    max_static_early = max(abs(r[3]) for r in early)
    min_evol_early   = min(abs(r[4]) for r in early)
    print(f"\n  max|corr| unselected (expect ~0)             : {max_none:.4f}")
    print(f"  min|corr| under evolutionary selection       : {min_evol:.4f}  (max p={max_p:.1e})")
    print(f"  gen-0 pairs: max|corr| STATIC = {max_static_early:.4f}  vs  min|corr| EVOL = {min_evol_early:.4f}")
    print(f"  -> repeated selection induces early-generation dependence static selection does not")

    print("\n[false causal discoveries] static-blind skeleton search (order<=1 CI, alpha=1e-6):")
    factors = [fac(k, t) for t in range(T + 1) for k in range(K)]
    sp_none, dr_none = naive_skeleton_spurious(o_none, factors)
    sp_evol, dr_evol = naive_skeleton_spurious(o_evol, factors)
    sp_stat, dr_stat = naive_skeleton_spurious(o_stat, factors)
    print(f"  unselected data   : spurious factor-factor edges = {sp_none}")
    print(f"  static selection  : spurious factor-factor edges = {sp_stat}")
    print(f"  evolutionary data : spurious factor-factor edges = {sp_evol}  <-- false discoveries")

    passed = (min_evol > 0.05 and max_p < 1e-6 and max_none < 0.02 and sp_evol > sp_none)
    print("=" * 78)
    print("PASS RULE: min|corr|evol>0.05 & p<1e-6 & max|corr|none<0.02 & spurious_evol>spurious_none")
    print(f"           {min_evol:.3f}>0.05 & {max_p:.1e}<1e-6 & {max_none:.3f}<0.02 & {sp_evol}>{sp_none}")
    print(f"OVERALL CLAIM 2: {'VERIFIED' if passed else 'FAILED'}")
    print("=" * 78)

    out = dict(T=T, K=K, rho=rho, se=se, mopt=mopt, sw=sw, N=N, n_none=n_none, n_evol=n_evol, n_static=n_stat,
               induced_pairs=[dict(a=a, b=b, corr_none=rn, corr_static=rs, corr_evol=re, p_evol=pe)
                              for (a, b, rn, rs, re, pe) in rows],
               max_corr_none=max_none, min_corr_evol=min_evol, max_p_evol=max_p,
               static_early_maxcorr=max_static_early, evol_early_mincorr=min_evol_early,
               spurious_none=sp_none, spurious_static=sp_stat, spurious_evol=sp_evol,
               drawn_none=dr_none, drawn_static=dr_stat, drawn_evol=dr_evol,
               verified=bool(passed))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("JSON_SUMMARY=" + json.dumps({k: out[k] for k in out if k != "induced_pairs"}))

if __name__ == "__main__":
    main()
