"""
Claim 5 / addendum B -- ACCURACY on HETEROGENEOUS / DIRECTIONAL digraphs:
P-RWDKC vs PIC and symmetrized spectral clustering (SC-SYM), in the regimes
the paper actually targets (App A.2: real graphs are degree-heterogeneous and
flow-asymmetric; "spectral clustering mostly fails for these graphs").

Three graph families (all directed, deterministic seeds, dense NumPy):

  HET  - degree-heterogeneous directed DC-SBM: Pareto(1.3) out- AND in-degree
         propensities (power-law hubs + many low-degree vertices), k=4,
         citation/web-like topology. The regime where eigenvector localization
         hurts spectral methods.
  HUB  - citation-style core-periphery with REDUCIBLE flow: per community a
         strongly-linked directed core (30%) + periphery (70%) that only CITES
         (out-edges only: 3 into own core, 2 into other cores; no in-edges).
         Periphery is transient for the raw walk (PIC's failure mode) and its
         symmetrized attachment signal is a noisy 3-vs-2.
  FLOW - flow-defined 2-block digraph: dense A->B drift (0.16), sparse B->A
         (0.002), within 0.04. After symmetrization the between-block density
         (~=0.081) EXCEEDS the within density (0.04): the undirected view is
         non-assortative, so symmetrization provably destroys the signal that
         only edge DIRECTION carries.

Methods (all given k = true number of communities):
  P-RWDKC : paper Alg 1 with the parametrized operator P_(nu); diffusion-map
            embedding Psi_t(i) = (lambda_l^t psi_l(i))_l of the kernel
            K_t = P_(nu)^t D_{nu+xi}^{-1} (Euclidean distance on Psi_t equals
            the paper's diffusion distance -- Prop 4.1, verified in Claim 4);
            (vertex measure nu, diffusion time t) selected UNSUPERVISED over
            nu in {uniform, out-degree, Eq-8 fwd/bwd, Eq-8 forward} and
            t in {1,2,4,8,16} (the paper grid-searches nu the same way).
            Selection criterion: DIRECTED NEWMAN MODULARITY of the candidate
            partition (degree-corrected null model; uses only W and the
            labels, never ground truth). We report this deviation openly:
            the paper's CH index is monotone-inflating in t on heavy-tailed
            graphs (measured), so CH is unreliable there; modularity is the
            standard unsupervised alternative and is applied IDENTICALLY to
            P-RWDKC and PIC.
  PIC     : power-iteration clustering on the RAW directed walk P^t
            (t selected by the same directed-modularity criterion).
  SC-SYM1 : unnormalized spectral clustering of the symmetrized graph.
  SC-SYM2 : normalized (Ng-Jordan-Weiss) spectral clustering, symmetrized.
  raw-P   : k-means on the rows of raw P (control).

Staged execution (cache _cache_hetero.json; each stage < 40 s):
    python3 repro_claim5_hetero.py stage HET|HUB|FLOW [seed]
    python3 repro_claim5_hetero.py report
or  python3 repro_claim5_hetero.py all          (~1-2 min)
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache_hetero.json")
SEEDS = [0, 1, 2]
TGRID = [1, 2, 4, 8, 16]

# ---------------------------------------------------------------- metrics ----
def contingency(a, b):
    ca = np.unique(a, return_inverse=True)[1]
    cb = np.unique(b, return_inverse=True)[1]
    M = np.zeros((ca.max() + 1, cb.max() + 1))
    np.add.at(M, (ca, cb), 1.0)
    return M

def ari(a, b):
    M = contingency(a, b); n = M.sum(); c2 = lambda x: x * (x - 1) / 2.0
    sij = c2(M).sum(); ai = c2(M.sum(1)).sum(); bj = c2(M.sum(0)).sum()
    exp = ai * bj / c2(n); mx = 0.5 * (ai + bj)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0

def nmi(a, b):
    M = contingency(a, b); n = M.sum()
    Pxy = M / n; Px = Pxy.sum(1); Py = Pxy.sum(0); nz = Pxy > 0
    MI = float(np.sum(Pxy[nz] * np.log(Pxy[nz] / np.outer(Px, Py)[nz])))
    Hx = -float(np.sum(Px[Px > 0] * np.log(Px[Px > 0])))
    Hy = -float(np.sum(Py[Py > 0] * np.log(Py[Py > 0])))
    return MI / np.sqrt(Hx * Hy) if Hx > 0 and Hy > 0 else 0.0

def kmeans(X, k, rng, restarts=5, iters=50):
    n = X.shape[0]; best = None; binert = np.inf
    for _ in range(restarts):
        idx = [int(rng.integers(n))]
        d2 = ((X - X[idx[0]]) ** 2).sum(1)
        for _ in range(k - 1):
            s = d2.sum()
            j = int(rng.choice(n, p=d2 / s)) if s > 0 else int(rng.integers(n))
            idx.append(j); d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(1))
        C = X[idx].copy(); lab = np.full(n, -1)
        for _ in range(iters):
            D = (X * X).sum(1)[:, None] - 2.0 * (X @ C.T) + (C * C).sum(1)[None, :]
            nl = D.argmin(1)
            if np.array_equal(nl, lab):
                break
            lab = nl
            for c in range(k):
                m = lab == c
                C[c] = X[m].mean(0) if m.any() else X[int(rng.integers(n))]
        inert = float(((X - C[lab]) ** 2).sum())
        if inert < binert:
            binert = inert; best = lab.copy()
    return best

def ch_index(X, lab):
    n = X.shape[0]; u = np.unique(lab); k = len(u); mu = X.mean(0)
    if k < 2 or k >= n:
        return -1.0
    bss = wss = 0.0
    for c in u:
        Xc = X[lab == c]; muc = Xc.mean(0)
        bss += len(Xc) * np.sum((muc - mu) ** 2); wss += np.sum((Xc - muc) ** 2)
    return float((bss / (k - 1)) / (wss / (n - k))) if wss > 0 else -1.0

# ------------------------------------------------------------- operators ----
def natural_rw(W):
    d = W.sum(1); d = np.where(d > 0, d, 1.0)
    return W / d[:, None]

def eq8_vertex_measure(W, t=3, gamma=0.7, alpha=0.5):
    dout = W.sum(1); din = W.sum(0)
    dout = np.where(dout > 0, dout, 1.0); din = np.where(din > 0, din, 1.0)
    Pg = gamma * (W / dout[:, None]) + (1.0 - gamma) * (W.T / din[:, None])
    N = W.shape[0]
    v = np.ones(N) / N
    for _ in range(t):
        v = v @ Pg
    return np.clip(v, 1e-12, None) ** alpha

def P_nu(P, nu):
    xi = nu @ P
    inner = P + (P.T * nu[None, :]) / nu[:, None]
    Pn = inner / (1.0 + xi / nu)[:, None]
    return Pn, nu + xi

def transient_count(W):
    P = natural_rw(W)
    w, V = np.linalg.eig(P.T)
    v = np.real(V[:, int(np.argmin(np.abs(w - 1.0)))]); v = v / v.sum()
    return int(np.sum(v < 1e-8))

def dir_modularity(W, lab):
    """Directed Newman modularity (degree-corrected null model). Unsupervised:
    uses only W and the candidate labels."""
    m = W.sum(); dout = W.sum(1); din = W.sum(0)
    Q = 0.0
    for c in np.unique(lab):
        idx = lab == c
        Q += W[np.ix_(idx, idx)].sum() / m \
            - (dout[idx].sum() * din[idx].sum()) / m ** 2
    return float(Q)

def prwdkc(W, k, rng):
    """Alg 1 with diffusion-map embedding of K_t = P_(nu)^t D_{nu+xi}^{-1};
    (nu, t) selected unsupervised by directed modularity."""
    N = W.shape[0]
    P = natural_rw(W)
    dout = np.maximum(W.sum(1), 1.0)
    measures = {"uniform": np.ones(N),
                "outdeg": dout,
                "eq8": eq8_vertex_measure(W, t=3, gamma=0.7, alpha=0.5),
                "eq8fwd": eq8_vertex_measure(W, t=5, gamma=1.0, alpha=1.0)}
    best = (None, -np.inf, "", 0)
    for mname, nu in measures.items():
        Pn, m = P_nu(P, nu)
        d12 = np.sqrt(m)
        T = d12[:, None] * Pn / d12[None, :]
        T = 0.5 * (T + T.T)
        w, V = np.linalg.eigh(T)
        B = V / d12[:, None]                      # psi_l = D^{-1/2} v_l
        for t in TGRID:
            Psi = B * (w ** t)[None, :]           # diffusion-map coords at t
            lab = kmeans(Psi, k, rng)
            q = dir_modularity(W, lab)
            if q > best[1]:
                best = (lab, q, mname, t)
    return best[0], best[2], best[3]

def pic(W, k, rng):
    """PIC: cluster rows of raw P^t, t by the same directed-modularity
    criterion (full-matrix variant, favorable to PIC)."""
    P = natural_rw(W)
    run = np.eye(W.shape[0]); last = 0
    best = (None, -np.inf, 0)
    for t in sorted(TGRID):
        while last < t:
            run = run @ P; last += 1
        lab = kmeans(run, k, rng)
        q = dir_modularity(W, lab)
        if q > best[1]:
            best = (lab, q, t)
    return best[0], best[2]

def sc_sym(W, k, rng, normalized):
    Ws = 0.5 * (W + W.T)
    d = Ws.sum(1); d = np.where(d > 0, d, 1e-12)
    if normalized:
        L = np.eye(len(d)) - (1 / np.sqrt(d))[:, None] * Ws * (1 / np.sqrt(d))[None, :]
    else:
        L = np.diag(d) - Ws
    _, V = np.linalg.eigh(L)
    U = V[:, :k]
    if normalized:
        nr = np.linalg.norm(U, axis=1, keepdims=True); nr[nr == 0] = 1
        U = U / nr
    return kmeans(U, k, rng)

# ------------------------------------------------------------------ graphs ---
def gen_het(seed):
    """Degree-heterogeneous directed DC-SBM (power-law in/out propensities)."""
    rng = np.random.default_rng(31000 + seed)
    N, k, R, deg = 600, 4, 6.0, 7.0
    nk = N // k
    z = np.repeat(np.arange(k), nk)
    th_out = np.clip(rng.pareto(1.3, N) + 1.0, None, 50.0)
    th_in = np.clip(rng.pareto(1.3, N) + 1.0, None, 50.0)
    for c in range(k):                             # per-block mean 1
        m = z == c
        th_out[m] /= th_out[m].mean(); th_in[m] /= th_in[m].mean()
    base = deg / (R * nk + (N - nk))
    F = np.where(z[:, None] == z[None, :], R, 1.0)
    Pm = np.clip(base * np.outer(th_out, th_in) * F, 0.0, 0.8)
    np.fill_diagonal(Pm, 0.0)
    W = (rng.random((N, N)) < Pm).astype(float)
    for i in np.where(W.sum(1) == 0)[0]:
        W[i, (i + 1) % N] = 1.0
    return W, z

def gen_hub(seed):
    """Citation-style core-periphery, periphery has OUT-edges only (transient
    for the raw walk); 3 citations into own core vs 2 into other cores."""
    rng = np.random.default_rng(32000 + seed)
    N, k = 600, 3
    nk = N // k; ncore = int(0.3 * nk)
    z = np.repeat(np.arange(k), nk)
    core = np.zeros(N, bool)
    for c in range(k):
        core[c * nk: c * nk + ncore] = True
    W = np.zeros((N, N))
    ci = [np.where(core & (z == c))[0] for c in range(k)]
    for c in range(k):                             # directed core blocks
        for cc in range(k):
            p = 0.15 if c == cc else 0.01
            blk = rng.random((len(ci[c]), len(ci[cc]))) < p
            W[np.ix_(ci[c], ci[cc])] = blk
    np.fill_diagonal(W, 0.0)
    for i in np.where(~core)[0]:                   # periphery: cites only
        own = ci[z[i]]
        oth = np.concatenate([ci[c] for c in range(k) if c != z[i]])
        W[i, rng.choice(own, 3, replace=False)] = 1.0
        W[i, rng.choice(oth, 2, replace=False)] = 1.0
    for i in np.where(W.sum(1) == 0)[0]:
        W[i, (i + 1) % N] = 1.0
    return W, z

def gen_flow(seed):
    """Flow-defined 2-block digraph: symmetrized between-density exceeds
    within-density, so the undirected view carries no assortative signal."""
    rng = np.random.default_rng(33000 + seed)
    N, k = 600, 2
    nk = N // k
    z = np.repeat(np.arange(k), nk)
    Pm = np.zeros((N, N))
    A_ = z == 0; B_ = z == 1
    Pm[np.ix_(A_, A_)] = 0.04;  Pm[np.ix_(B_, B_)] = 0.04
    Pm[np.ix_(A_, B_)] = 0.16;  Pm[np.ix_(B_, A_)] = 0.002
    np.fill_diagonal(Pm, 0.0)
    W = (rng.random((N, N)) < Pm).astype(float)
    for i in np.where(W.sum(1) == 0)[0]:
        W[i, (i + 1) % N] = 1.0
    return W, z

REGIMES = {"HET": (gen_het, 4), "HUB": (gen_hub, 3), "FLOW": (gen_flow, 2)}

# ------------------------------------------------------------------ driver ---
def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {}

def save_cache(c):
    with open(CACHE, "w") as f:
        json.dump(c, f, indent=1, sort_keys=True)

def run_seed(cache, regime, seed):
    key = f"{regime}:{seed}"
    if key in cache:
        return cache[key]
    gen, k = REGIMES[regime]
    W, z = gen(seed)
    rng = np.random.default_rng(500 + seed)
    rec = {"regime": regime, "seed": seed, "N": int(W.shape[0]), "k": k,
           "transient_raw_walk": transient_count(W)}
    if regime == "FLOW":                          # measured symmetrized densities
        Ws = 0.5 * (W + W.T); A_ = z == 0; B_ = z == 1
        nk = int(A_.sum())
        rec["sym_density_within"] = float((Ws[np.ix_(A_, A_)].sum()
                                           + Ws[np.ix_(B_, B_)].sum())
                                          / (2 * nk * (nk - 1)))
        rec["sym_density_between"] = float((Ws[np.ix_(A_, B_)].sum()
                                            + Ws[np.ix_(B_, A_)].sum())
                                           / (2 * nk * nk))
    lab, mname, td = prwdkc(W, k, rng)
    rec["P-RWDKC"] = dict(ari=ari(z, lab), nmi=nmi(z, lab), measure=mname, t=td)
    lab, td = pic(W, k, rng)
    rec["PIC"] = dict(ari=ari(z, lab), nmi=nmi(z, lab), t=td)
    lab = sc_sym(W, k, rng, normalized=False)
    rec["SC-SYM1"] = dict(ari=ari(z, lab), nmi=nmi(z, lab))
    lab = sc_sym(W, k, rng, normalized=True)
    rec["SC-SYM2"] = dict(ari=ari(z, lab), nmi=nmi(z, lab))
    lab = kmeans(natural_rw(W), k, rng)
    rec["raw-P"] = dict(ari=ari(z, lab), nmi=nmi(z, lab))
    cache[key] = rec
    save_cache(cache)
    print(f"  done {key}: " + "  ".join(
        f"{m}={rec[m]['nmi']*100:.1f}" for m in
        ("P-RWDKC", "PIC", "SC-SYM1", "SC-SYM2", "raw-P")))
    return rec

METHODS = ["P-RWDKC", "PIC", "SC-SYM1", "SC-SYM2", "raw-P"]

def report(cache):
    print("=" * 78)
    print("Claim 5 addendum B  P-RWDKC accuracy on heterogeneous/directional digraphs")
    print(f"N=600, seeds={SEEDS}, numpy {np.__version__}, single CPU thread")
    print("=" * 78)
    out = {"config": dict(seeds=SEEDS, t_grid=TGRID, numpy=np.__version__)}
    summary = {}
    for regime in REGIMES:
        recs = [cache[f"{regime}:{s}"] for s in SEEDS]
        summ = {m: dict(ari=float(np.mean([r[m]["ari"] for r in recs])),
                        nmi=float(np.mean([r[m]["nmi"] for r in recs])),
                        nmi_sd=float(np.std([r[m]["nmi"] for r in recs])))
                for m in METHODS}
        tr = float(np.mean([r["transient_raw_walk"] for r in recs]))
        summary[regime] = summ
        out[regime] = dict(summary=summ, mean_transient_raw_walk=tr,
                           per_seed=recs)
        print(f"\n[{regime}]  k={recs[0]['k']}  raw-walk transient vertices "
              f"= {tr:.1f}/{recs[0]['N']}")
        if regime == "FLOW":
            dw = float(np.mean([r["sym_density_within"] for r in recs]))
            db = float(np.mean([r["sym_density_between"] for r in recs]))
            out[regime]["sym_density_within"] = dw
            out[regime]["sym_density_between"] = db
            print(f"  symmetrized density within={dw:.4f} < between={db:.4f}"
                  f"  -> undirected view is NON-assortative (signal is in the"
                  f" direction)")
        print(f"  {'method':10s} {'ARI':>8s} {'NMI':>8s}")
        for m in METHODS:
            print(f"  {m:10s} {summ[m]['ari']*100:7.2f} {summ[m]['nmi']*100:7.2f}")
        best_sc = max(summ["SC-SYM1"]["nmi"], summ["SC-SYM2"]["nmi"])
        print(f"  -> P-RWDKC - best SC = {(summ['P-RWDKC']['nmi']-best_sc)*100:+.2f} pp"
              f";  P-RWDKC - PIC = {(summ['P-RWDKC']['nmi']-summ['PIC']['nmi'])*100:+.2f} pp")
    gaps_sc = {r: summary[r]["P-RWDKC"]["nmi"]
               - max(summary[r]["SC-SYM1"]["nmi"], summary[r]["SC-SYM2"]["nmi"])
               for r in REGIMES}
    gaps_pic = {r: summary[r]["P-RWDKC"]["nmi"] - summary[r]["PIC"]["nmi"]
                for r in REGIMES}
    beats_pic_all = all(g > 0 for g in gaps_pic.values())
    competitive = all(g >= -0.05 for g in gaps_sc.values())
    flow_win = gaps_sc["FLOW"] >= 0.15
    recover = all(summary[r]["P-RWDKC"]["nmi"] >= 0.70 for r in REGIMES)
    ok = beats_pic_all and competitive and flow_win and recover
    print("\n" + "-" * 78)
    print(f"(1) beats PIC in all 3 heterogeneous regimes        : {beats_pic_all}"
          f"  ({', '.join(f'{r}:{g*100:+.1f}pp' for r, g in gaps_pic.items())})")
    print(f"(2) competitive with best SC (gap >= -5 pp) in all  : {competitive}"
          f"  ({', '.join(f'{r}:{g*100:+.1f}pp' for r, g in gaps_sc.items())})")
    print(f"(3) decisive win over SC where signal is directional: {flow_win}"
          f"  (FLOW: {gaps_sc['FLOW']*100:+.1f} pp >= +15 pp)")
    print(f"(4) recovery NMI >= 70 in all regimes               : {recover}")
    print("VERDICT:", "PASS" if ok else "PARTIAL/FAIL")
    print("=" * 78)
    out.update(gap_vs_best_SC_pp={r: g * 100 for r, g in gaps_sc.items()},
               gap_vs_PIC_pp={r: g * 100 for r, g in gaps_pic.items()},
               beats_pic_all=bool(beats_pic_all), competitive_with_SC=bool(competitive),
               flow_decisive_win=bool(flow_win), recovery=bool(recover),
               verdict="PASS" if ok else "PARTIAL",
               claim=("P-RWDKC is competitive with symmetrized spectral clustering "
                      "on heterogeneous digraphs and decisively better where the "
                      "cluster signal is directional (symmetrization-destroyed); "
                      "beats PIC everywhere"))
    with open(os.path.join(HERE, "results_hetero.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results_hetero.json")
    return out

def main():
    args = sys.argv[1:]
    cache = load_cache()
    mode = args[0] if args else "all"
    if mode == "stage":
        regime = args[1]
        seeds = [int(args[2])] if len(args) > 2 else SEEDS
        for s in seeds:
            run_seed(cache, regime, s)
    elif mode == "report":
        report(cache)
    elif mode == "all":
        for regime in REGIMES:
            for s in SEEDS:
                run_seed(cache, regime, s)
        report(cache)
    else:
        raise SystemExit("usage: repro_claim5_hetero.py "
                         "[stage HET|HUB|FLOW [seed]] | report | all")

if __name__ == "__main__":
    main()
