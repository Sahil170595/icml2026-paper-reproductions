"""
Claim 5 / addendum A -- SCALABILITY of Parametrized Power-Iteration Clustering
(ParPIC / P-RWDKC) vs spectral clustering, on sparse directed SBMs.

Paper (arXiv 2210.00310 / OpenReview 5vI6ApLOg8, "Parametrized Power-Iteration
Clustering for Directed Graphs") claims COMPETITIVE accuracy with IMPROVED
SCALABILITY relative to spectral methods: one ParPIC iteration is a sparse
mat-vec, O(|E|), whereas classical spectral clustering needs an
eigendecomposition of a dense N x N (symmetrized) Laplacian, O(N^3) time and
O(N^2) memory.

WHAT IS MEASURED (all wall-clock, CPU, single thread, deterministic seeds):
  * ParPIC      : sparse P_(nu) (nu=1) -> reversible symmetrization
                  T = D^{1/2} P_(nu) D^{-1/2}; block POWER ITERATION
                  (t sparse mat-vecs on k+2 vectors + thin QR) -> k-means.
                  Cost per iteration O(|E| r + N r^2), memory O(|E| + N r).
  * SC-DENSE    : dense symmetrized normalized-adjacency eigendecomposition
                  (numpy.linalg.eigh, full spectrum -- the classical O(N^3)
                  spectral pipeline) -> k-means on top-k eigenvectors.
  * SC-SPARSE   : scipy.sparse.linalg.eigsh (ARPACK/Lanczos, k+1 evecs) --
                  the strong sparse baseline. (Lanczos is itself a Krylov /
                  power-method descendant, which is the paper's point.)
Each timed job also records NMI/ARI vs ground truth -> accuracy AT SCALE.
Runtime exponents are fitted as the log-log slope of time vs N.
Peak traced memory (tracemalloc) is compared for all methods at N=2000.

Graphs: sparse directed SBM, k=4 equal blocks, expected out-degree 12,
in/out edge-probability ratio 8 (assortative, recoverable), N = 500..50,000.

STAGED EXECUTION (each stage < 40 s CPU; results cached in
_cache_scalability.json keyed by method:N:seed; any stage order works):
    python3 repro_claim5_scalability.py stage small     # all methods, N<=1500
    python3 repro_claim5_scalability.py stage mid       # N<=16000 (+dense 2000)
    python3 repro_claim5_scalability.py stage large     # ParPIC+eigsh 32k, 50k
    python3 repro_claim5_scalability.py stage dense3000 # dense eigh N=3000
    python3 repro_claim5_scalability.py stage mem       # tracemalloc @ N=2000
    python3 repro_claim5_scalability.py report          # fit + results json
or  python3 repro_claim5_scalability.py all             # everything (~2-3 min)
"""
import json, os, sys, time, tracemalloc
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache_scalability.json")
K = 4
AVG_OUT_DEG = 12.0
RATIO = 8.0

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

def kmeans(X, k, rng, restarts=4, iters=60):
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

# ------------------------------------------------------------------ graph ----
def sparse_directed_sbm(N, seed):
    """Sparse directed SBM: K equal blocks, expected out-degree AVG_OUT_DEG,
    within/between probability ratio RATIO. O(|E|) sampling."""
    rng = np.random.default_rng(seed)
    nk = N // K
    z = np.repeat(np.arange(K), nk)
    qout = AVG_OUT_DEG / (RATIO * nk + (N - nk))
    qin = RATIO * qout
    rows, cols = [], []
    for a in range(K):
        for b in range(K):
            p = qin if a == b else qout
            m = int(rng.binomial(nk * nk, p))
            if m == 0:
                continue
            rows.append(rng.integers(0, nk, m) + a * nk)
            cols.append(rng.integers(0, nk, m) + b * nk)
    r = np.concatenate(rows); c = np.concatenate(cols)
    keep = r != c
    A = sp.csr_matrix((np.ones(keep.sum()), (r[keep], c[keep])), shape=(N, N))
    A.sum_duplicates(); A.data[:] = 1.0
    dout = np.asarray(A.sum(1)).ravel()
    z0 = np.where(dout == 0)[0]
    if len(z0):
        A = (A + sp.csr_matrix((np.ones(len(z0)), (z0, (z0 + 1) % N)),
                               shape=(N, N))).tocsr()
    return A, z

# ---------------------------------------------------------------- methods ----
def build_parametrized_operator(A):
    """Sparse P_(nu) with nu=1:  P_(nu) = (I + D_xi)^{-1} (P + P^T),
    xi = 1^T P.  Reversible w.r.t. m = 1 + xi (Prop 3.2/Claim 3)."""
    dout = np.asarray(A.sum(1)).ravel(); dout[dout == 0] = 1.0
    P = sp.diags(1.0 / dout) @ A
    S = (P + P.T).tocsr()
    xi = np.asarray(P.sum(0)).ravel()
    m = 1.0 + xi
    return sp.diags(1.0 / m) @ S, m

def run_parpic(A, z, seed, t_iters=25):
    """Block power iteration on the reversible symmetrization of P_(nu):
    only sparse mat-vecs + thin QR (O(|E| r + N r^2) per iteration)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    Pn, m = build_parametrized_operator(A)
    d12 = np.sqrt(m)
    T = sp.diags(d12) @ Pn @ sp.diags(1.0 / d12)
    T = (0.5 * (T + T.T)).tocsr()
    N = A.shape[0]; r = K + 2
    Z = rng.standard_normal((N, r))
    for _ in range(t_iters):
        Z = T @ Z
        Z, _ = np.linalg.qr(Z)
    B = Z.T @ (T @ Z)
    w, Q = np.linalg.eigh(B)
    Z = Z @ Q[:, np.argsort(-w)]
    U = Z[:, 1:K] / d12[:, None]           # drop stationary direction
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    lab = kmeans(U, K, rng)
    dt = time.perf_counter() - t0
    return dt, nmi(z, lab), ari(z, lab)

def run_sc_dense(A, z, seed):
    """Classical spectral clustering: DENSE symmetrized normalized adjacency,
    full eigendecomposition (O(N^3) time, O(N^2) memory)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    W = 0.5 * np.asarray((A + A.T).todense())
    d = W.sum(1); d[d == 0] = 1e-12
    di = 1.0 / np.sqrt(d)
    M = di[:, None] * W * di[None, :]
    _, V = np.linalg.eigh(M)
    U = V[:, -K:]
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    lab = kmeans(U, K, rng)
    dt = time.perf_counter() - t0
    return dt, nmi(z, lab), ari(z, lab)

def run_sc_sparse(A, z, seed):
    """Sparse spectral clustering via ARPACK Lanczos (strong baseline)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    Ws = (0.5 * (A + A.T)).tocsr()
    d = np.asarray(Ws.sum(1)).ravel(); d[d == 0] = 1e-12
    di = 1.0 / np.sqrt(d)
    M = sp.diags(di) @ Ws @ sp.diags(di)
    M = (0.5 * (M + M.T)).tocsr()
    _, V = spla.eigsh(M, k=K + 1, which="LA",
                      v0=np.random.default_rng(seed).standard_normal(A.shape[0]))
    U = V[:, -K:]
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    lab = kmeans(U, K, rng)
    dt = time.perf_counter() - t0
    return dt, nmi(z, lab), ari(z, lab)

RUNNERS = {"parpic": run_parpic, "sc_dense": run_sc_dense,
           "sc_sparse": run_sc_sparse}

# ------------------------------------------------------------------ cache ----
def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {}

def save_cache(c):
    with open(CACHE, "w") as f:
        json.dump(c, f, indent=1, sort_keys=True)

def run_job(cache, method, N, seed=0, mem=False):
    key = ("mem:" if mem else "") + f"{method}:{N}:{seed}"
    if key in cache:
        return cache[key]
    A, z = sparse_directed_sbm(N, 77000 + N)
    if mem:
        tracemalloc.start()
        RUNNERS[method](A, z, seed)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rec = dict(method=method, N=N, seed=seed, peak_mb=peak / 1e6,
                   edges=int(A.nnz))
    else:
        dt, nm, ar = RUNNERS[method](A, z, seed)
        rec = dict(method=method, N=N, seed=seed, time_s=dt, nmi=nm, ari=ar,
                   edges=int(A.nnz))
    cache[key] = rec
    save_cache(cache)
    print(f"  done {key}: " + ", ".join(f"{k2}={v2:.4g}" if isinstance(v2, float)
          else f"{k2}={v2}" for k2, v2 in rec.items() if k2 not in ("method", "seed")))
    return rec

STAGES = {
    "small": [("parpic", n) for n in (500, 1000, 1500)]
             + [("sc_sparse", n) for n in (500, 1000, 1500)]
             + [("sc_dense", n) for n in (500, 1000, 1500)],
    "mid": [("parpic", n) for n in (2000, 4000, 8000, 16000)]
           + [("sc_sparse", n) for n in (2000, 4000, 8000, 16000)]
           + [("sc_dense", 2000)],
    "large": [("parpic", 32000), ("sc_sparse", 32000),
              ("parpic", 50000), ("sc_sparse", 50000)],
    "dense3000": [("sc_dense", 3000)],
    "denseXL": [("sc_dense", 4000), ("sc_dense", 5000),
                ("parpic", 3000), ("parpic", 5000),
                ("sc_sparse", 3000), ("sc_sparse", 5000)],
}

def fit_exponent(recs, nmin=0):
    pts = sorted((r["N"], r["time_s"]) for r in recs if r["N"] >= nmin)
    if len(pts) < 2:
        return float("nan")
    x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0])

def report(cache):
    timed = [r for k2, r in cache.items() if not k2.startswith("mem:")]
    mem = [r for k2, r in cache.items() if k2.startswith("mem:")]
    by = {m: sorted([r for r in timed if r["method"] == m], key=lambda r: r["N"])
          for m in RUNNERS}
    print("=" * 78)
    print("Claim 5 addendum A  SCALABILITY  ParPIC vs spectral clustering")
    print("sparse directed SBM, k=4, E[out-deg]=12, ratio 8; single CPU thread")
    print("=" * 78)
    print(f"{'N':>7} {'edges':>8} | {'ParPIC s':>9} {'NMI':>6} | "
          f"{'eigsh s':>9} {'NMI':>6} | {'dense-SC s':>10} {'NMI':>6}")
    allN = sorted({r["N"] for r in timed})
    grid = {(r["method"], r["N"]): r for r in timed}
    for n in allN:
        e = next((r["edges"] for r in timed if r["N"] == n), 0)
        cells = []
        for m in ("parpic", "sc_sparse", "sc_dense"):
            r = grid.get((m, n))
            cells.append(f"{r['time_s']:9.3f} {r['nmi']*100:6.1f}" if r
                         else f"{'--':>9} {'--':>6}")
        print(f"{n:>7} {e:>8} | " + " | ".join(cells))
    exps = {
        "parpic": fit_exponent(by["parpic"], nmin=2000),
        "sc_sparse": fit_exponent(by["sc_sparse"], nmin=2000),
        "sc_dense": fit_exponent(by["sc_dense"], nmin=1000),
    }
    print("-" * 78)
    print("fitted runtime exponents  time ~ N^alpha  (log-log slope):")
    ndmax = max((r["N"] for r in by["sc_dense"]), default=0)
    print(f"  ParPIC (block power iteration) : alpha = {exps['parpic']:.2f}   (N in [2000, 50000])")
    print(f"  SC sparse Lanczos (eigsh)      : alpha = {exps['sc_sparse']:.2f}   (N in [2000, 50000])")
    print(f"  SC dense eigendecomposition    : alpha = {exps['sc_dense']:.2f}   (N in [1000, {ndmax}])")
    # speedup at largest common N with dense SC + extrapolation to 50k
    ncap = max(r["N"] for r in by["sc_dense"]) if by["sc_dense"] else 0
    sp_at_cap = ext50k = float("nan")
    if ncap:
        td = grid[("sc_dense", ncap)]["time_s"]; tp = grid[("parpic", ncap)]["time_s"]
        sp_at_cap = td / tp
        t50_dense = td * (50000.0 / ncap) ** exps["sc_dense"]
        t50_par = grid[("parpic", 50000)]["time_s"] if ("parpic", 50000) in grid else float("nan")
        ext50k = t50_dense / t50_par
        print(f"  measured speedup ParPIC vs dense SC at N={ncap}: {sp_at_cap:.1f}x")
        print(f"  extrapolated dense-SC time at N=50000: {t50_dense/60:.1f} min "
              f"-> projected speedup {ext50k:.0f}x (ParPIC measured: {t50_par:.2f} s)")
    if mem:
        print("-" * 78)
        print("peak traced memory (tracemalloc) at N=2000:")
        for r in sorted(mem, key=lambda r: r["peak_mb"]):
            print(f"  {r['method']:10s}: {r['peak_mb']:9.1f} MB   (edges={r['edges']})")
    accs = {m: float(np.mean([r["nmi"] for r in by[m]])) for m in RUNNERS if by[m]}
    minnmi = {m: float(np.min([r["nmi"] for r in by[m]])) for m in RUNNERS if by[m]}
    ok = (exps["parpic"] < 1.5 and exps["sc_dense"] > 2.3
          and minnmi.get("parpic", 0) > 0.95
          and (not np.isnan(sp_at_cap) and sp_at_cap > 10))
    print("-" * 78)
    print(f"mean NMI over all N  : " + "  ".join(f"{m}={v*100:.1f}" for m, v in accs.items()))
    print(f"ACCEPTANCE  ParPIC alpha<1.5 & dense-SC alpha>2.3 & ParPIC min NMI>95"
          f" & speedup@N={ncap}>10x : {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    out = dict(
        claim=("ParPIC (power iteration, O(E) per iteration) scales near-linearly "
               "and matches spectral accuracy on sparse directed SBMs; dense "
               "spectral clustering scales ~cubically"),
        config=dict(k=K, avg_out_deg=AVG_OUT_DEG, ratio=RATIO,
                    numpy=np.__version__),
        runtime_exponents=exps,
        speedup_vs_dense_at_N=dict(N=ncap, speedup=sp_at_cap),
        projected_speedup_at_50000=ext50k,
        mean_nmi=accs, min_nmi=minnmi,
        table=timed, memory_at_2000=mem,
        verdict="PASS" if ok else "FAIL")
    with open(os.path.join(HERE, "results_scalability.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results_scalability.json")
    return out

def main():
    args = sys.argv[1:]
    cache = load_cache()
    mode = args[0] if args else "all"
    if mode == "stage":
        name = args[1]
        if name == "mem":
            for m in RUNNERS:
                run_job(cache, m, 2000, mem=True)
        else:
            for m, n in STAGES[name]:
                run_job(cache, m, n)
    elif mode == "report":
        report(cache)
    elif mode == "all":
        for name in ("small", "mid", "large", "dense3000", "denseXL"):
            for m, n in STAGES[name]:
                run_job(cache, m, n)
        for m in RUNNERS:
            run_job(cache, m, 2000, mem=True)
        report(cache)
    else:
        raise SystemExit("usage: repro_claim5_scalability.py "
                         "[stage small|mid|large|dense3000|mem] | report | all")

if __name__ == "__main__":
    main()
