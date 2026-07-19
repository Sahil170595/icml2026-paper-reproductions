"""
Claim 5 (Theorem 4 / Algorithm 2) - "Causal Modeling of Selection in Evolution" (mOcTXKawFY).

CLAIM: Combining heterogeneous data from multiple environments/domains via the CDNOD-based
procedure (Algorithm 2) IMPROVES identifiability of the evolutionary selection model compared
to single-environment data.

CDNOD idea: pool E environments and add a domain index C (a known exogenous ROOT). C becomes
adjacent to any variable whose causal MECHANISM changes across environments. Here the natural
heterogeneity is that different populations/domains start from different genetic means, so the
ROOT heritable factors eps_k^(0) have environment-specific means (C -> eps_k^(0)); the
inheritance and trait mechanisms are invariant. Because C is a known root, orienting C->eps^(0)
and propagating with Meek rules orients the inheritance chains eps_k^(t)->eps_k^(t+1) that are
Markov-equivalent (hence UNORIENTED) with single-environment data. Selection-clique edges are a
known edge type in G^+ and are never oriented as causal.

CHECKABLE CONSEQUENCE (executed): count of correctly-oriented true-causal edges, and SHD to the
ground-truth DAG, SINGLE-environment vs CDNOD (multi-environment). CDNOD must orient strictly
more causal edges (lower SHD) with NO loss of soundness (no wrong-direction edge, no selection
edge oriented as causal).

We verify at two layers:
  LAYER 1 (oracle, exact): d-separation oracle of G^+ / CDNOD-augmented G^+.
  LAYER 2 (finite sample): pooled Gaussian data with env-specific root means + Fisher-z CI.

PASS RULE: oracle CDNOD oriented-causal > single-env oriented-causal AND both sound
(wrong-dir==0, selection-oriented==0) AND CDNOD SHD < single-env SHD; finite-sample CDNOD also
orients strictly more causal edges than single-env with 0 wrong-direction edges.
FALSIFICATION: CDNOD does not orient more causal edges than single-env, or introduces
wrong-direction / selection-as-causal edges.

Independent NumPy/networkx implementation, CPU-only, deterministic.
"""
import numpy as np, itertools, json, os, networkx as nx
from math import log, sqrt, erfc

def fac(k, t): return f"e{k}_{t}"
def Xn(t):     return f"X_{t}"

def ground_truth(T, K):
    causal = set()
    for t in range(T + 1):
        for k in range(K):
            if t > 0: causal.add((fac(k, t - 1), fac(k, t)))
            causal.add((fac(k, t), Xn(t)))
    sel = set(frozenset((fac(i, t), fac(j, t)))
              for t in range(T) for i in range(K) for j in range(K) if i < j)
    return causal, sel

# ---------------- skeleton + orientation (selection edges frozen, roots exogenous) -------------
def pc_skeleton(nodes, ci):
    V = list(nodes); adj = {v: set(V) - {v} for v in V}; sep = {}; d = 0
    while True:
        edges = [(a, b) for i, a in enumerate(V) for b in V[i + 1:] if b in adj[a]]
        if not edges or all(max(len(adj[a] - {b}), len(adj[b] - {a})) < d for a, b in edges): break
        snap = {v: set(adj[v]) for v in V}; rem = []
        for a, b in edges:
            for base in (snap[a] - {b}, snap[b] - {a}):
                if len(base) < d: continue
                ok = False
                for S in itertools.combinations(sorted(base), d):
                    if ci(a, b, S): rem.append((a, b)); sep[frozenset((a, b))] = set(S); ok = True; break
                if ok: break
        for a, b in rem: adj[a].discard(b); adj[b].discard(a)
        d += 1
    return adj, sep

def orient(nodes, adj, sep, roots, frozen, ci=None, conservative=False):
    V = list(nodes); und = set(frozenset((a, b)) for a in V for b in adj[a] if a < b); arr = set()
    for r in roots:                                   # background: domain index is exogenous
        for nb in list(adj[r]): arr.add((r, nb))
    def is_collider(a, c, b):
        if not conservative:
            return c not in sep.get(frozenset((a, b)), set())
        base = (set(adj[a]) | set(adj[b])) - {a, b}; seps = []
        for d in range(0, min(len(base), 4) + 1):
            for S in itertools.combinations(sorted(base), d):
                if ci(a, b, S): seps.append(set(S))
        return bool(seps) and all(c not in S for S in seps)
    for c in V:
        nb = [x for x in V if frozenset((x, c)) in und]
        for a, b in itertools.combinations(nb, 2):
            if frozenset((a, b)) in und: continue
            if is_collider(a, c, b):
                if frozenset((a, c)) not in frozen: arr.add((a, c))
                if frozenset((b, c)) not in frozen: arr.add((b, c))
    for e in list(arr): und.discard(frozenset(e))
    changed = True
    while changed:
        changed = False
        for e in list(und):
            if e in frozen: continue
            a, b = tuple(e)
            if (a, b) in arr or (b, a) in arr: continue
            for (x, y) in [(a, b), (b, a)]:
                fired = False
                for z in V:                            # Meek R1
                    if (z, x) in arr and (x, z) not in arr and frozenset((z, y)) not in und \
                       and (z, y) not in arr and (y, z) not in arr and z not in (x, y):
                        arr.add((x, y)); changed = fired = True; break
                if fired: break
                for z in V:                            # Meek R2
                    if (x, z) in arr and (z, y) in arr:
                        arr.add((x, y)); changed = fired = True; break
                if fired: break
        for e in list(arr): und.discard(frozenset(e))
    return arr, und

def eval_orient(arr, causal, sel):
    oe = [(a, b) for (a, b) in arr if (b, a) not in arr]
    oc = [e for e in oe if not (e[0] == "C" or e[1] == "C")]
    correct = sum(1 for e in oc if e in causal)
    wrong   = sum(1 for e in oc if e not in causal and (e[1], e[0]) in causal)
    sel_or  = sum(1 for e in oc if frozenset(e) in sel)
    shd = (len(causal) - correct) + wrong + sel_or            # unoriented/mis-oriented causal + spurious
    return correct, wrong, sel_or, shd

def simulate_env(T, K, rho, se, mopt, sw, n, seed, root_mean):
    rng = np.random.default_rng(seed)
    cols = {}; prev = {k: rng.standard_normal(n) + root_mean for k in range(K)}; sel = np.ones(n, bool)
    for t in range(T + 1):
        if t > 0:
            for k in range(K):
                prev[k] = rho * prev[k] + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        for k in range(K): cols[fac(k, t)] = prev[k].copy()
        X = sum(prev[k] for k in range(K)) + se * rng.standard_normal(n); cols[Xn(t)] = X
        if t < T: sel = sel & (rng.random(n) < np.exp(-((X - mopt) ** 2) / (2.0 * sw ** 2)))
    return {k: v[sel] for k, v in cols.items()}, int(sel.sum())

def build_canonical(T, K, cdnod):
    G = nx.DiGraph(); W = []
    for t in range(T + 1):
        for k in range(K):
            if t > 0: G.add_edge(fac(k, t - 1), fac(k, t))
            G.add_edge(fac(k, t), Xn(t))
        if t < T:
            w = f"W_{t}"; W.append(w)
            for k in range(K): G.add_edge(fac(k, t), w)
    if cdnod:
        for k in range(K): G.add_edge("C", fac(k, 0))
    return G, W

def main():
    T, K, rho, se, mopt, sw = 2, 3, 0.6, 0.7, 1.5, 1.2
    print("=" * 78)
    print("CLAIM 5  (Theorem 4 / Alg 2)  CDNOD multi-environment improves identifiability")
    print("paper: Causal Modeling of Selection in Evolution (mOcTXKawFY); independent NumPy")
    print("=" * 78)
    obs = [fac(k, t) for t in range(T + 1) for k in range(K)] + [Xn(t) for t in range(T + 1)]
    causal, sel = ground_truth(T, K)
    print(f"T={T}, K={K}: {len(obs)} observed nodes; {len(causal)} true causal edges; {len(sel)} selection edges")

    # -------- LAYER 1: oracle --------
    def oracle_ci(G, W):
        def ci(a, b, S):
            try:    return nx.is_d_separator(G, {a}, {b}, set(S) | set(W))
            except Exception: return nx.d_separated(G, {a}, {b}, set(S) | set(W))
        return ci
    Gs, Ws = build_canonical(T, K, False); cis = oracle_ci(Gs, Ws)
    adj, sepp = pc_skeleton(obs, cis); arr_s, _ = orient(obs, adj, sepp, [], sel)
    c_s, w_s, so_s, shd_s = eval_orient(arr_s, causal, sel)
    Gc, Wc = build_canonical(T, K, True); cic = oracle_ci(Gc, Wc); nodesC = ["C"] + obs
    adjC, sepC = pc_skeleton(nodesC, cic); arr_c, _ = orient(nodesC, adjC, sepC, ["C"], sel)
    c_c, w_c, so_c, shd_c = eval_orient(arr_c, causal, sel)
    print(f"\n[LAYER 1 - oracle d-separation]")
    print(f"    SINGLE-env : oriented causal = {c_s}/{len(causal)}  wrong-dir={w_s}  sel-oriented={so_s}  SHD={shd_s}")
    print(f"    CDNOD (multi-env, +C root) : oriented causal = {c_c}/{len(causal)}  wrong-dir={w_c}  sel-oriented={so_c}  SHD={shd_c}")
    print(f"    identifiability gain: {c_s} -> {c_c} oriented causal edges;  SHD {shd_s} -> {shd_c}")
    oracle_pass = (c_c > c_s and w_s == 0 and w_c == 0 and so_s == 0 and so_c == 0 and shd_c < shd_s)

    # -------- LAYER 2: finite sample --------
    def fisher_ci(nodes, R, n, alpha):
        idx = {v: i for i, v in enumerate(nodes)}
        def ci(a, b, S):
            ids = [idx[a], idx[b]] + [idx[s] for s in S]
            P = np.linalg.pinv(R[np.ix_(ids, ids)])
            r = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]); r = min(max(r, -0.999999), 0.999999)
            z = 0.5 * log((1 + r) / (1 - r)) * sqrt(max(n - len(S) - 3, 1))
            return erfc(abs(z) / sqrt(2)) > alpha
        return ci
    E = 4; beta = 1.2; alpha = 1e-2; per = 15000
    means = [beta * (e - (E - 1) / 2.0) for e in range(E)]
    print(f"\n[LAYER 2 - finite sample]  E={E} environments, env root means={['%.2f'%m for m in means]}, alpha={alpha}")
    gains = []
    for seed in range(4):
        datasets = [simulate_env(T, K, rho, se, mopt, sw, per, seed * 10 + e, means[e]) for e in range(E)]
        # single env (middle-ish environment)
        d0, n0 = datasets[E // 2]
        R0 = np.corrcoef(np.column_stack([d0[v] for v in obs]).astype(float), rowvar=False)
        a0, s0 = pc_skeleton(obs, fisher_ci(obs, R0, n0, alpha))
        arr0, _ = orient(obs, a0, s0, [], sel, ci=fisher_ci(obs, R0, n0, alpha), conservative=True)
        cc0, ww0, ss0, sh0 = eval_orient(arr0, causal, sel)
        # CDNOD pooled
        allc = {v: np.concatenate([d[v] for d, _ in datasets]) for v in obs}
        allc["C"] = np.concatenate([np.full(n, float(e)) for e, (d, n) in enumerate(datasets)])
        nt = sum(n for _, n in datasets)
        Rc = np.corrcoef(np.column_stack([allc[v] for v in nodesC]).astype(float), rowvar=False)
        ciC = fisher_ci(nodesC, Rc, nt, alpha)
        aC, sC = pc_skeleton(nodesC, ciC)
        arrC, _ = orient(nodesC, aC, sC, ["C"], sel, ci=ciC, conservative=True)
        ccC, wwC, ssC, shC = eval_orient(arrC, causal, sel)
        gains.append((cc0, cc0 and ww0, ww0, sh0, ccC, wwC, shC))
    import statistics as st
    m_s = st.mean(g[0] for g in gains); m_c = st.mean(g[4] for g in gains)
    w_single = sum(g[2] for g in gains); w_cd = sum(g[5] for g in gains)
    sh_single = st.mean(g[3] for g in gains); sh_cd = st.mean(g[6] for g in gains)
    print(f"    SINGLE-env : mean oriented causal = {m_s:.2f}/{len(causal)}  total wrong-dir={w_single}  mean SHD={sh_single:.2f}")
    print(f"    CDNOD      : mean oriented causal = {m_c:.2f}/{len(causal)}  total wrong-dir={w_cd}  mean SHD={sh_cd:.2f}")
    print(f"    identifiability gain (finite sample): {m_s:.2f} -> {m_c:.2f} oriented causal; SHD {sh_single:.2f} -> {sh_cd:.2f}")
    fs_pass = (m_c > m_s and w_cd == 0)

    passed = oracle_pass and fs_pass
    print("=" * 78)
    print(f"PASS RULE: oracle(CDNOD>single, sound, SHD lower)={oracle_pass}; finite(CDNOD>single, 0 wrong)={fs_pass}")
    print(f"OVERALL CLAIM 5: {'VERIFIED' if passed else 'FAILED'}")
    print("=" * 78)

    out = dict(T=T, K=K, rho=rho, se=se, mopt=mopt, sw=sw, n_causal=len(causal), n_selection=len(sel),
               oracle=dict(single_oriented=c_s, cdnod_oriented=c_c, single_shd=shd_s, cdnod_shd=shd_c,
                           single_wrong=w_s, cdnod_wrong=w_c, single_sel_oriented=so_s, cdnod_sel_oriented=so_c),
               finite_sample=dict(E=E, alpha=alpha, per_env=per, env_means=means,
                                  single_oriented=m_s, cdnod_oriented=m_c, single_wrong_total=w_single,
                                  cdnod_wrong_total=w_cd, single_shd=sh_single, cdnod_shd=sh_cd),
               verified=bool(passed))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("JSON_SUMMARY=" + json.dumps({k: out[k] for k in out if k not in ("oracle", "finite_sample")})
          + " | oracle " + json.dumps(out["oracle"]) + " | finite " + json.dumps(out["finite_sample"]))

if __name__ == "__main__":
    main()
