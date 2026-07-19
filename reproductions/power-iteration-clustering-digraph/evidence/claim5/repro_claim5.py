"""
Claim 5 - P-RWDKC recovers community structure on directed SBMs and, crucially,
outperforms PIC (power-iteration clustering on the RAW directed random walk) --
the paper's central mechanistic claim (Sec. 6.3, verbatim): "the parametrized
random walk is irreducible compared to the random walk used in PIC that is not".

Paper P-RWDKC (arXiv 2210.00310 / OpenReview 5vI6ApLOg8), Alg.1 + Sec.6.
All numpy, CPU-only, deterministic (default_rng). ARI, NMI, k-means and the
Calinski-Harabasz index implemented from scratch (no sklearn).

Methods (all fed k = true #communities):
  * P-RWDKC : rows of K_{t_d}=P_(nu)^{t_d} D_{nu+xi}^{-1}, nu=1; t_d by CH (Alg.2).
  * PIC     : rows of P_raw^{t_d} (raw DIRECTED walk); t_d by CH.
  * SC-SYM1 : unnormalized-Laplacian eigenvectors of symmetrized W.
  * SC-SYM2 : normalized-Laplacian   eigenvectors of symmetrized W.
  * raw-P   : k-means on rows of raw directed P (t=1) [control].

Regimes: (I) assortative directed SBM (recoverable, strongly-ish connected);
(II) sparse assortative directed SBM with source->sink drift that is NOT strongly
connected, so the raw directed walk (PIC) is REDUCIBLE (transient vertices lose
identity) -> the exact setting where irreducibility of P_(nu) matters.

Acceptance rule (paper-faithful; the paper claims it wins in MOST cases, its
decisive namesake comparison being vs PIC):
  (1) recovery: P-RWDKC ARI>0.8 AND NMI>0.8 in regime I;
  (2) beats PIC: P-RWDKC NMI > PIC NMI in BOTH regimes, with a large margin
      (>= 8 NMI points) in the reducible regime II where PIC is provably reducible;
  (3) P-RWDKC NMI >= raw-P control in both regimes.
The SC-SYM comparison is reported as context (symmetrized SC is a strong baseline
on assortative SBMs; the paper's edge over SC is on real heterogeneous graphs).
"""
import json, numpy as np

def contingency(a, b):
    ca = np.unique(a, return_inverse=True)[1]; cb = np.unique(b, return_inverse=True)[1]
    M = np.zeros((ca.max()+1, cb.max()+1)); np.add.at(M, (ca, cb), 1.0); return M
def ari(a, b):
    M = contingency(a, b); n = M.sum(); c2 = lambda x: x*(x-1)/2.0
    sij = c2(M).sum(); ai = c2(M.sum(1)).sum(); bj = c2(M.sum(0)).sum()
    exp = ai*bj/c2(n); mx = 0.5*(ai+bj)
    return float((sij-exp)/(mx-exp)) if mx != exp else 1.0
def nmi(a, b):
    M = contingency(a, b); n = M.sum(); Pxy = M/n; Px = Pxy.sum(1); Py = Pxy.sum(0); nz = Pxy > 0
    MI = float(np.sum(Pxy[nz]*np.log(Pxy[nz]/np.outer(Px, Py)[nz])))
    Hx = -float(np.sum(Px[Px>0]*np.log(Px[Px>0]))); Hy = -float(np.sum(Py[Py>0]*np.log(Py[Py>0])))
    return MI/np.sqrt(Hx*Hy) if Hx > 0 and Hy > 0 else 0.0
def kmeans(X, k, rng, restarts=8, iters=100):
    n = X.shape[0]; best_lab = None; best_in = np.inf
    for _ in range(restarts):
        idx = [int(rng.integers(n))]; d2 = np.sum((X - X[idx[0]])**2, 1)
        for _ in range(k-1):
            pr = d2/d2.sum() if d2.sum() > 0 else np.ones(n)/n
            j = int(rng.choice(n, p=pr)); idx.append(j); d2 = np.minimum(d2, np.sum((X - X[j])**2, 1))
        C = X[idx].copy(); lab = np.zeros(n, int)
        for it in range(iters):
            D = ((X[:, None, :]-C[None, :, :])**2).sum(2); nl = D.argmin(1)
            if it > 0 and np.array_equal(nl, lab): break
            lab = nl
            for c in range(k):
                mm = lab == c; C[c] = X[mm].mean(0) if mm.any() else X[int(rng.integers(n))]
        inertia = float(((X - C[lab])**2).sum())
        if inertia < best_in: best_in = inertia; best_lab = lab.copy()
    return best_lab
def ch_index(X, lab):
    n = X.shape[0]; u = np.unique(lab); k = len(u); mu = X.mean(0)
    if k < 2 or k >= n: return -1.0
    bss = wss = 0.0
    for c in u:
        Xc = X[lab == c]; muc = Xc.mean(0); bss += len(Xc)*np.sum((muc-mu)**2); wss += np.sum((Xc-muc)**2)
    return float((bss/(k-1))/(wss/(n-k))) if wss > 0 else -1.0
def natural_rw(W):
    d = W.sum(1); d = np.where(d > 0, d, 1.0); return W / d[:, None]
def P_nu_uniform(W):
    P = natural_rw(W); nu = np.ones(W.shape[0]); xi = nu @ P
    num = (nu[:, None]*P) + (P.T*nu[None, :]); return num/(nu+xi)[:, None], (nu+xi)
def transient_count(W):
    P = natural_rw(W); w, V = np.linalg.eig(P.T); k = int(np.argmin(np.abs(w-1.0)))
    v = np.real(V[:, k]); v = v/v.sum(); return int(np.sum(v < 1e-8))
def diffusion_select(M, Dinv, k, ts, rng):
    best = None; best_ch = -np.inf; best_t = ts[0]; run = np.eye(M.shape[0]); last = 0
    for t in sorted(ts):
        while last < t: run = run @ M; last += 1
        emb = run * Dinv[None, :] if Dinv is not None else run.copy()
        lab = kmeans(emb, k, rng, restarts=6); ch = ch_index(emb, lab)
        if ch > best_ch: best_ch = ch; best = lab; best_t = t
    return best, best_t
def sc_sym(W, k, rng, normalized=True):
    Ws = 0.5*(W + W.T); d = Ws.sum(1); d = np.where(d > 0, d, 1e-12)
    L = (np.eye(len(d)) - (1/np.sqrt(d))[:, None]*Ws*(1/np.sqrt(d))[None, :]) if normalized else (np.diag(d) - Ws)
    w, V = np.linalg.eigh(L); U = V[:, :k]
    if normalized:
        nr = np.linalg.norm(U, axis=1, keepdims=True); nr[nr == 0] = 1; U = U/nr
    return kmeans(U, k, rng, restarts=8)
def sbm_assortative(N, k, pin, pout, rng):
    z = np.repeat(np.arange(k), N//k)[:N]; A = np.zeros((N, N)); R = rng.random((N, N))
    for i in range(N):
        rp = np.where(z == z[i], pin, pout).astype(float); rp[i] = 0; A[i] = (R[i] < rp).astype(float)
    for i in range(N):
        if A[i].sum() == 0: A[i, (i+1) % N] = 1.0
    return A, z
def sbm_sparse_drift(N, k, pin, pout, drift, rng):
    z = np.repeat(np.arange(k), N//k)[:N]; A = np.zeros((N, N)); R = rng.random((N, N))
    for i in range(N):
        p = np.where(z == z[i], pin, pout).astype(float); p[i] = 0
        p = p*np.where(z > z[i], 1+drift, np.where(z < z[i], max(0.0, 1-drift), 1.0))
        A[i] = (R[i] < p).astype(float)
    for i in range(N):
        if A[i].sum() == 0: A[i, (i+1) % N] = 1.0
    return A, z
def run_regime(gen, args, k, seeds, ts):
    rows = {m: {"ari": [], "nmi": []} for m in ["P-RWDKC", "PIC", "SC-SYM1", "SC-SYM2", "raw-P"]}
    tds = {"P-RWDKC": [], "PIC": []}; transient = []
    for s in seeds:
        rng = np.random.default_rng(1000+s); W, z = gen(*args, rng)
        transient.append(transient_count(W))
        Pn, m = P_nu_uniform(W); Praw = natural_rw(W)
        lab, td = diffusion_select(Pn, 1.0/m, k, ts, rng)
        rows["P-RWDKC"]["ari"].append(ari(z, lab)); rows["P-RWDKC"]["nmi"].append(nmi(z, lab)); tds["P-RWDKC"].append(td)
        lab, td = diffusion_select(Praw, None, k, ts, rng)
        rows["PIC"]["ari"].append(ari(z, lab)); rows["PIC"]["nmi"].append(nmi(z, lab)); tds["PIC"].append(td)
        lab = sc_sym(W, k, rng, False); rows["SC-SYM1"]["ari"].append(ari(z, lab)); rows["SC-SYM1"]["nmi"].append(nmi(z, lab))
        lab = sc_sym(W, k, rng, True);  rows["SC-SYM2"]["ari"].append(ari(z, lab)); rows["SC-SYM2"]["nmi"].append(nmi(z, lab))
        lab = kmeans(Praw, k, rng, 8);  rows["raw-P"]["ari"].append(ari(z, lab)); rows["raw-P"]["nmi"].append(nmi(z, lab))
    summ = {mth: dict(ari=float(np.mean(rows[mth]["ari"])), nmi=float(np.mean(rows[mth]["nmi"])),
                      nmi_sd=float(np.std(rows[mth]["nmi"]))) for mth in rows}
    return summ, tds, float(np.mean(transient))
def main():
    ts = [1, 2, 4, 8, 16, 32, 64]; seeds = [0, 1, 2, 3]
    print("="*74); print("Claim 5  P-RWDKC clustering on directed SBMs vs baselines (ARI/NMI)")
    print("arXiv 2210.00310 / OpenReview 5vI6ApLOg8  (numpy %s)  seeds=%s" % (np.__version__, seeds)); print("="*74)
    out = {}
    for name, gen, args, k in [
        ("REGIME_I_assortative",   sbm_assortative, (240, 4, 0.18, 0.02),      4),
        ("REGIME_II_reducible",    sbm_sparse_drift,(240, 4, 0.08, 0.005, 1.8), 4)]:
        summ, tds, transient = run_regime(gen, args, k, seeds, ts)
        out[name] = dict(summary=summ, td_PRWDKC=tds["P-RWDKC"], td_PIC=tds["PIC"], k=k, N=args[0],
                         mean_transient_vertices_raw_walk=transient)
        print(f"\n[{name}]  N={args[0]} k={k}  mean over {len(seeds)} seeds"
              f"  | raw-walk transient vertices (PIC reducibility) = {transient:.1f}/{args[0]}")
        print(f"  {'method':10s} {'ARI':>8s} {'NMI':>8s}")
        for mth in ["P-RWDKC", "PIC", "SC-SYM1", "SC-SYM2", "raw-P"]:
            print(f"  {mth:10s} {summ[mth]['ari']*100:7.2f} {summ[mth]['nmi']*100:7.2f}")
        pr = summ["P-RWDKC"]
        print(f"  -> P-RWDKC NMI {pr['nmi']*100:.2f} vs PIC {summ['PIC']['nmi']*100:.2f} "
              f"(margin over PIC {(pr['nmi']-summ['PIC']['nmi'])*100:+.2f} pp); t_d P-RWDKC={tds['P-RWDKC']}, PIC={tds['PIC']}")
    r1 = out["REGIME_I_assortative"]["summary"]; r2 = out["REGIME_II_reducible"]["summary"]
    recover = r1["P-RWDKC"]["ari"] > 0.8 and r1["P-RWDKC"]["nmi"] > 0.8
    beats_pic_both = (r1["P-RWDKC"]["nmi"] > r1["PIC"]["nmi"]) and (r2["P-RWDKC"]["nmi"] > r2["PIC"]["nmi"])
    big_gap_II = (r2["P-RWDKC"]["nmi"] - r2["PIC"]["nmi"]) >= 0.08
    ge_raw = r1["P-RWDKC"]["nmi"] >= r1["raw-P"]["nmi"] and r2["P-RWDKC"]["nmi"] >= r2["raw-P"]["nmi"] - 1e-9
    pic_reducible = out["REGIME_II_reducible"]["mean_transient_vertices_raw_walk"] > 0
    verdict = recover and beats_pic_both and big_gap_II and ge_raw and pic_reducible
    # context: SC comparison
    sc_ctx_I = r1["P-RWDKC"]["nmi"] - max(r1["SC-SYM1"]["nmi"], r1["SC-SYM2"]["nmi"])
    sc_ctx_II = r2["P-RWDKC"]["nmi"] - max(r2["SC-SYM1"]["nmi"], r2["SC-SYM2"]["nmi"])
    print("\n" + "-"*74)
    print(f"(1) recovery regime I (ARI>0.8 & NMI>0.8)         : {recover}")
    print(f"(2) beats PIC in BOTH regimes                     : {beats_pic_both}")
    print(f"    large margin in reducible regime II (>=8 pp)  : {big_gap_II}  (PIC provably reducible: {pic_reducible})")
    print(f"(3) >= raw-P control in both regimes              : {ge_raw}")
    print(f"context: P-RWDKC - best SC-SYM  =  {sc_ctx_I*100:+.2f} pp (regime I),  {sc_ctx_II*100:+.2f} pp (regime II)")
    print("VERDICT:", "PASS" if verdict else "PARTIAL/FAIL"); print("="*74)
    out.update(verdict="PASS" if verdict else "PARTIAL", numpy=np.__version__, seeds=seeds,
               diffusion_time_grid=ts, recovery=bool(recover), beats_pic_both=bool(beats_pic_both),
               big_gap_reducible_regime=bool(big_gap_II), ge_raw_control=bool(ge_raw),
               pic_reducible=bool(pic_reducible),
               P_RWDKC_minus_best_SC_regimeI_pp=sc_ctx_I*100, P_RWDKC_minus_best_SC_regimeII_pp=sc_ctx_II*100,
               claim="P-RWDKC recovers directed-SBM communities and outperforms PIC (raw directed walk) in ARI/NMI; decisive in the reducible regime; competitive with symmetrized SC")
    json.dump(out, open("results.json", "w"), indent=2); print("wrote results.json")
if __name__ == "__main__":
    main()
