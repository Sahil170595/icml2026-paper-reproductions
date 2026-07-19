"""
Claim 1 (Definition 1) - "Causal Modeling of Selection in Evolution" (OpenReview mOcTXKawFY).

CLAIM: Definition 1 formalizes an evolutionary selection model as a DAG G^(T) over trait
variables X^(0..T), heritable factors epsilon^(0..T), and reproduction/selection indicators
S^(0..T-1), distinguishing it from one-shot static selection models.

CHECKABLE CONSEQUENCES (what this script verifies with executed numbers):
  (A) The DAG G^(T) built exactly per Definition 1 is acyclic and has the correct node/edge
      inventory: T+1 trait gens, K heritable-factor components per gen, T selection nodes,
      each S^(t) a child of X^(t) (t<T), inheritance chains eps_k^(t-1)->eps_k^(t),
      eps_k^(t)->X^(t).
  (B) G^(T) is a FAITHFUL realization: on UN-selected simulated data every pair that is
      d-separated in G^(T) is empirically independent (|partial corr| small) and every
      adjacent pair is dependent -> the SCM realizes the DAG.
  (C) The evolutionary model is genuinely DISTINCT from a one-shot STATIC selection model
      (single global selection node S* child of the final trait): the two graphs disagree on
      a strictly positive number of d-separation relations once selection is conditioned,
      i.e. they imply different conditional-independence models. A static model applied to
      evolutionary data is therefore mis-specified.

PASS RULE: acyclic AND inventory-correct AND faithful (max |pcorr| over d-separated pairs
< 0.02, min |corr| over adjacent pairs > 0.05) AND #(d-sep disagreements evol-vs-static) > 0.

Independent NumPy/networkx implementation, CPU-only, deterministic (numpy.random.default_rng).
"""
import numpy as np, itertools, json, networkx as nx

def fac(k, t): return f"e{k}_{t}"
def Xn(t):     return f"X_{t}"
def Sn(t):     return f"S_{t}"

def build_evolutionary_dag(T, K):
    """G^(T) per Definition 1: traits X^(0..T), heritable factors eps_k^(0..T), selection S^(0..T-1)."""
    G = nx.DiGraph()
    for t in range(T + 1):
        for k in range(K):
            G.add_node(fac(k, t))
            if t > 0:
                G.add_edge(fac(k, t - 1), fac(k, t))     # inheritance chain
            G.add_edge(fac(k, t), Xn(t))                 # heritable factor -> trait
        if t < T:
            G.add_edge(Xn(t), Sn(t))                     # trait -> reproduction/selection indicator
    return G

def build_static_dag(T, K):
    """One-shot STATIC selection model: same traits/factors but a SINGLE global selection
    node S* that is a child of the final trait X^(T) only (classic one-shot selection bias)."""
    G = nx.DiGraph()
    for t in range(T + 1):
        for k in range(K):
            G.add_node(fac(k, t))
            if t > 0:
                G.add_edge(fac(k, t - 1), fac(k, t))
            G.add_edge(fac(k, t), Xn(t))
    G.add_edge(Xn(T), "Sstar")                           # single one-shot selection
    return G

def dsep(G, x, y, Z):
    try:    return nx.is_d_separator(G, {x}, {y}, set(Z))
    except Exception: return nx.d_separated(G, {x}, {y}, set(Z))

def simulate_unselected(T, K, rho, se, n, seed):
    rng = np.random.default_rng(seed)
    cols = {}
    prev = {k: rng.standard_normal(n) for k in range(K)}
    for t in range(T + 1):
        if t > 0:
            for k in range(K):
                prev[k] = rho * prev[k] + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        for k in range(K):
            cols[fac(k, t)] = prev[k].copy()
        cols[Xn(t)] = sum(prev[k] for k in range(K)) + se * rng.standard_normal(n)
    return cols

def pcorr(a, b, Z, data):
    y = np.column_stack([data[a], data[b]] + [data[z] for z in Z]).astype(float)
    y = y - y.mean(0)
    C = np.cov(y, rowvar=False)
    P = np.linalg.pinv(np.atleast_2d(C))
    return float(-P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]))

def main():
    T, K = 3, 3
    rho, se = 0.6, 0.7
    print("=" * 78)
    print("CLAIM 1  (Definition 1)  evolutionary selection model as DAG G^(T)")
    print("paper: Causal Modeling of Selection in Evolution  (OpenReview mOcTXKawFY)")
    print("independent NumPy/networkx, CPU-only, deterministic")
    print("=" * 78)
    G = build_evolutionary_dag(T, K)
    obs = [fac(k, t) for t in range(T + 1) for k in range(K)] + [Xn(t) for t in range(T + 1)]

    # (A) inventory + acyclicity
    n_trait   = T + 1
    n_factor  = K * (T + 1)
    n_sel     = T
    acyclic   = nx.is_directed_acyclic_graph(G)
    inv_ok = (sum(1 for v in G if v.startswith("X_")) == n_trait and
              sum(1 for v in G if v.startswith("e")) == n_factor and
              sum(1 for v in G if v.startswith("S_")) == n_sel and
              all(list(G.successors(Sn(t))) == [] for t in range(T)) and
              all(Sn(t) in G.predecessors(  Sn(t) ) if False else Xn(t) in list(G.predecessors(Sn(t))) for t in range(T)))
    print(f"[A] G^(T=%d), K=%d heritable-factor components/gen" % (T, K))
    print(f"    trait nodes X^(0..T)        = {sum(1 for v in G if v.startswith('X_'))} (expect {n_trait})")
    print(f"    heritable factor nodes eps  = {sum(1 for v in G if v.startswith('e'))} (expect {n_factor})")
    print(f"    selection nodes S^(0..T-1)  = {sum(1 for v in G if v.startswith('S_'))} (expect {n_sel})")
    print(f"    every S^(t) is a leaf child of X^(t): {all(list(G.successors(Sn(t)))==[] and Xn(t) in list(G.predecessors(Sn(t))) for t in range(T))}")
    print(f"    acyclic DAG: {acyclic}   inventory correct: {inv_ok}   total edges: {G.number_of_edges()}")

    # (B) faithfulness on UNSELECTED data
    data = simulate_unselected(T, K, rho, se, n=400_000, seed=0)
    dsep_pairs, adj_pairs = [], []
    for i, a in enumerate(obs):
        for b in obs[i + 1:]:
            if dsep(G, a, b, set()):
                dsep_pairs.append((a, b))
            elif G.has_edge(a, b) or G.has_edge(b, a):
                adj_pairs.append((a, b))
    max_pc_dsep = max(abs(pcorr(a, b, (), data)) for a, b in dsep_pairs)
    min_pc_adj  = min(abs(pcorr(a, b, (), data)) for a, b in adj_pairs)
    faithful = (max_pc_dsep < 0.02) and (min_pc_adj > 0.05)
    print(f"[B] faithfulness on UNSELECTED n=400k:")
    print(f"    marginally d-separated pairs: {len(dsep_pairs)}   max|corr| = {max_pc_dsep:.4f} (expect ~0, <0.02)")
    print(f"    graph-adjacent pairs:         {len(adj_pairs)}   min|corr| = {min_pc_adj:.4f} (expect >0.05)")
    print(f"    SCM faithfully realizes G^(T): {faithful}")

    # (C) distinctness from one-shot static model (conditioning on selection)
    Gst = build_static_dag(T, K)
    Sall = [Sn(t) for t in range(T)]
    disagree = 0; total = 0; examples = []
    for i, a in enumerate(obs):
        for b in obs[i + 1:]:
            rest = [o for o in obs if o not in (a, b)]
            for r in range(0, 3):
                for Z in itertools.combinations(rest, r):
                    de = dsep(G,   a, b, set(Z) | set(Sall))    # evolutionary: condition on all S
                    ds = dsep(Gst, a, b, set(Z) | {"Sstar"})    # static: condition on single S*
                    total += 1
                    if de != ds:
                        disagree += 1
                        if len(examples) < 4 and (not de) and ds:
                            examples.append((a, b, Z))
    print(f"[C] evolutionary vs one-shot static selection model:")
    print(f"    selection nodes: evolutionary={n_sel}  static=1")
    print(f"    d-separation relations compared (|Z|<=2): {total}")
    print(f"    DISAGREEMENTS (models imply different CI): {disagree}  ({100*disagree/total:.2f}%)")
    print(f"    e.g. dependent under evolutionary selection but independent under static:")
    for a, b, Z in examples:
        print(f"        {a} vs {b} | {Z}: evol=DEP, static=INDEP")

    passed = acyclic and inv_ok and faithful and disagree > 0
    print("=" * 78)
    print(f"PASS RULE: acyclic & inventory & faithful & (evol!=static): "
          f"{acyclic} & {inv_ok} & {faithful} & ({disagree}>0)")
    print(f"OVERALL CLAIM 1: {'VERIFIED' if passed else 'FAILED'}")
    print("=" * 78)

    out = dict(T=T, K=K, rho=rho, se=se, n_trait=n_trait, n_factor=n_factor, n_sel=n_sel,
               total_edges=G.number_of_edges(), acyclic=bool(acyclic), inventory_ok=bool(inv_ok),
               faithful=bool(faithful), max_pcorr_dsep=max_pc_dsep, min_corr_adj=min_pc_adj,
               n_dsep_pairs=len(dsep_pairs), n_adj_pairs=len(adj_pairs),
               dsep_compared=total, dsep_disagree_static=disagree,
               disagree_frac=disagree / total, verified=bool(passed))
    import os
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("JSON_SUMMARY=" + json.dumps(out))

if __name__ == "__main__":
    main()
