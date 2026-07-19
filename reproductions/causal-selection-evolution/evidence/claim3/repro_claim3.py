"""
Claim 3 (Theorem 1 / Definition 2) - "Causal Modeling of Selection in Evolution" (mOcTXKawFY).

CLAIM: The clique-augmented DAG G^+ (Definition 2) FULLY captures all d-separation /
conditional-independence constraints implied by the evolutionary selection model, WITHOUT
explicitly modelling the selection variables.

GROUND TRUTH: the CI model of the selected distribution P(observed | S^(0..T-1)=1) is, by the
Markov property of a conditional, exactly d-separation in the full DAG G^(T) with the selection
nodes S placed in the conditioning set.  We verify that G^+ (no S nodes) reproduces this EXACTLY.

G^+ construction (Definition 2, clique augmentation of selection):
  * keep all directed edges of G^(T) over observed nodes {eps, X}; drop selection nodes S;
  * for every generation t that underwent selection (0..T-1), the heritable factors that
    feed the selected phenotype X^(t) form a CLIQUE (undirected selection edges) -- because
    conditioning on a descendant of the collider X^(t) couples its parents inseparably.
  * m-separation of this mixed graph is computed via its canonical DAG: replace each
    selection clique by a common child W^(t) that is held in the conditioning set.

CHECKABLE CONSEQUENCE (executed):
  (1) EXHAUSTIVE over ALL ordered pairs and ALL conditioning subsets of the observed nodes,
      dsep_{G^+}(i,j|Z)  ==  dsep_{G^(T)}(i,j|Z + all S).   Target: 100% agreement.
  (2) The selection-blind NAIVE DAG (directed edges only, no clique) does NOT match
      (< 100%), i.e. G^+ is necessary.
  (3) Empirically, on simulated selected data, Fisher-z CI decisions match G^+'s d-sep.

PASS RULE: exact agreement fraction G^+ vs ground truth == 1.0 (all triples), naive < 1.0,
and empirical CI-vs-G^+ agreement >= 0.95.

FALSIFICATION: any triple where G^+ and the ground-truth selected model disagree.

Independent NumPy/networkx implementation, CPU-only, deterministic.
"""
import numpy as np, itertools, json, os, networkx as nx
from math import log, sqrt, erfc

def fac(k, t): return f"e{k}_{t}"
def Xn(t):     return f"X_{t}"
def Sn(t):     return f"S_{t}"

def build_full(T, K):
    G = nx.DiGraph()
    for t in range(T + 1):
        for k in range(K):
            if t > 0: G.add_edge(fac(k, t - 1), fac(k, t))
            G.add_edge(fac(k, t), Xn(t))
        if t < T: G.add_edge(Xn(t), Sn(t))
    return G

def build_plus_canonical(T, K):
    """Clique-augmented G^+ rendered as a canonical DAG: selection cliques -> conditioned W^(t)."""
    G = nx.DiGraph(); W = []
    for t in range(T + 1):
        for k in range(K):
            if t > 0: G.add_edge(fac(k, t - 1), fac(k, t))
            G.add_edge(fac(k, t), Xn(t))
        if t < T:
            w = f"W_{t}"; W.append(w)
            for k in range(K): G.add_edge(fac(k, t), w)   # factors -> selection clique node
    return G, W

def build_naive(T, K):
    G = nx.DiGraph()
    for t in range(T + 1):
        for k in range(K):
            if t > 0: G.add_edge(fac(k, t - 1), fac(k, t))
            G.add_edge(fac(k, t), Xn(t))
    return G

def dsep(G, x, y, Z):
    try:    return nx.is_d_separator(G, {x}, {y}, set(Z))
    except Exception: return nx.d_separated(G, {x}, {y}, set(Z))

def simulate_selected(T, K, rho, se, mopt, sw, n, seed):
    """Gaussian stabilizing selection: reproduce w.p. exp(-(X-mopt)^2/(2 sw^2)); selected joint
    stays exactly Gaussian, so the Fisher-z partial-correlation CI test is valid."""
    rng = np.random.default_rng(seed)
    cols = {}; prev = {k: rng.standard_normal(n) for k in range(K)}; sel = np.ones(n, bool)
    for t in range(T + 1):
        if t > 0:
            for k in range(K):
                prev[k] = rho * prev[k] + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        for k in range(K): cols[fac(k, t)] = prev[k].copy()
        X = sum(prev[k] for k in range(K)) + se * rng.standard_normal(n); cols[Xn(t)] = X
        if t < T:
            sel = sel & (rng.random(n) < np.exp(-((X - mopt) ** 2) / (2.0 * sw ** 2)))
    return {k: v[sel] for k, v in cols.items()}, int(sel.sum())

def main():
    T, K = 2, 3    # trait gens 0..2, selection 0..1, 3 heritable factors/gen -> 12 observed nodes
    rho, se, mopt, sw = 0.6, 0.7, 1.5, 1.2
    print("=" * 78)
    print("CLAIM 3  (Theorem 1 / Def 2)  clique-augmented G^+ captures ALL d-separations")
    print("paper: Causal Modeling of Selection in Evolution (mOcTXKawFY); independent NumPy")
    print("=" * 78)
    obs = [fac(k, t) for t in range(T + 1) for k in range(K)] + [Xn(t) for t in range(T + 1)]
    Gf = build_full(T, K); Sall = [Sn(t) for t in range(T)]
    Gp, Wall = build_plus_canonical(T, K)
    Gn = build_naive(T, K)
    print(f"T={T}, K={K}: observed nodes = {len(obs)} ; selection gens = {list(range(T))}")
    print(f"ground truth = d-sep in G^(T) with S{ {t for t in range(T)} } conditioned\n")

    # (1)+(2) EXHAUSTIVE enumeration
    pairs = [(a, b) for i, a in enumerate(obs) for b in obs[i + 1:]]
    ag_plus = ag_naive = tot = 0
    disag_plus = []; disag_naive = 0
    for a, b in pairs:
        rest = [o for o in obs if o not in (a, b)]
        for r in range(len(rest) + 1):
            for Z in itertools.combinations(rest, r):
                truth = dsep(Gf, a, b, set(Z) | set(Sall))
                dp    = dsep(Gp, a, b, set(Z) | set(Wall))
                dn    = dsep(Gn, a, b, set(Z))
                tot += 1
                if dp == truth: ag_plus += 1
                elif len(disag_plus) < 5: disag_plus.append((a, b, Z))
                if dn == truth: ag_naive += 1
                else: disag_naive += 1
    frac_plus  = ag_plus / tot
    frac_naive = ag_naive / tot
    print(f"[1] EXHAUSTIVE over ALL pairs x ALL conditioning subsets: {tot} triples")
    print(f"    G^+  vs ground-truth selected model : {ag_plus}/{tot} = {frac_plus:.6f}")
    print(f"    disagreements: {len(disag_plus)}  {disag_plus[:5]}")
    print(f"[2] NAIVE DAG (ignores selection) vs truth: {ag_naive}/{tot} = {frac_naive:.6f}")
    print(f"    naive disagreements (selection genuinely matters): {disag_naive}")

    # (3) empirical CI on selected data vs G^+ d-sep
    data, nsel = simulate_selected(T, K, rho, se, mopt, sw, 1_500_000, 0)
    M = np.column_stack([data[o] for o in obs]).astype(float); R = np.corrcoef(M, rowvar=False)
    idx = {o: i for i, o in enumerate(obs)}
    def pval(a, b, Z):
        ids = [idx[a], idx[b]] + [idx[z] for z in Z]
        P = np.linalg.pinv(R[np.ix_(ids, ids)])
        r = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]); r = min(max(r, -0.999999), 0.999999)
        zt = 0.5 * log((1 + r) / (1 - r)) * sqrt(max(nsel - len(Z) - 3, 1))
        return erfc(abs(zt) / sqrt(2))
    rng = np.random.default_rng(7)
    trip = []
    for a, b in pairs:
        rest = [o for o in obs if o not in (a, b)]
        for r in (0, 1, 2):
            combs = list(itertools.combinations(rest, r)); rng.shuffle(combs)
            for Z in combs[:3]: trip.append((a, b, Z))
    alpha = 1e-3; emp_ok = 0
    for a, b, Z in trip:
        emp_indep = pval(a, b, Z) > alpha
        pred_indep = dsep(Gp, a, b, set(Z) | set(Wall))
        if emp_indep == pred_indep: emp_ok += 1
    emp_frac = emp_ok / len(trip)
    print(f"[3] empirical Fisher-z CI (selected data n={nsel}, alpha={alpha}) vs G^+ d-sep:")
    print(f"    {emp_ok}/{len(trip)} triples agree = {emp_frac:.4f}")

    passed = (frac_plus == 1.0) and (frac_naive < 1.0) and (emp_frac >= 0.95)
    print("=" * 78)
    print(f"PASS RULE: G^+==1.0 & naive<1.0 & empirical>=0.95 : "
          f"{frac_plus:.4f}==1.0 & {frac_naive:.4f}<1.0 & {emp_frac:.4f}>=0.95")
    print(f"OVERALL CLAIM 3: {'VERIFIED' if passed else 'FAILED'}")
    print("=" * 78)

    out = dict(T=T, K=K, rho=rho, se=se, mopt=mopt, sw=sw, n_obs=len(obs), triples=tot,
               agree_plus=ag_plus, frac_plus=frac_plus, agree_naive=ag_naive,
               frac_naive=frac_naive, naive_disagree=disag_naive, n_selected=nsel,
               empirical_triples=len(trip), empirical_agree=emp_ok, empirical_frac=emp_frac,
               alpha=alpha, verified=bool(passed))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("JSON_SUMMARY=" + json.dumps(out))

if __name__ == "__main__":
    main()
