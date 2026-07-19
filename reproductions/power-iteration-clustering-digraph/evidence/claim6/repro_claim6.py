"""
Claim 6 - The diffusion time controls the scale at which clusters are revealed:
short diffusion times expose fine clusters, longer times expose coarse clusters
(metastability of nearly-uncoupled Markov chains), and the Calinski-Harabasz
(CH/DCH) index of Alg. 2 - evaluated on a FIXED representation - selects a
diffusion time that recovers the ground truth at the requested scale.

Paper P-RWDKC (arXiv 2210.00310 / OpenReview 5vI6ApLOg8), Sec. 5.2 + Alg. 2 +
Sec. 6.2 (multi-scale toy: a short t_{d1} reveals 6 clusters, a LONGER t_{d2}
reveals 2). Setup: hierarchical directed SBM, 2 super-blocks x 3 sub-blocks
(6 fine / 2 coarse). nu=1. numpy only; k-means / ARI / NMI / CH from scratch.

CHECKABLE CONSEQUENCES (deterministic, CPU):
  (A) Metastable spectrum: P_(nu) is nearly-uncoupled - lambda_1=1, one coarse
      mode near 1, then 4 fine modes, then a gap; TWO eigengaps (after 2 and
      after 6 eigenvalues) => two natural scales.
  (B) Effective #metastable clusters N_eff(t)=#{|lambda_i|^t>1/2} decreases
      6 -> 2 -> 1 as diffusion time t grows (fine modes dissolve first).
  (C) Modal diffusion times t*=ln(1/2)/ln(lambda): the coarse mode (lambda_2)
      persists several times LONGER than the fine modes (lambda_6), so the coarse
      scale is revealed at a longer diffusion time than the fine scale.
  (D) Scale recovery: clustering rows of P_(nu)^{t} into 6 at the FINE window
      (short t) recovers the 6 sub-blocks; into 2 at the COARSE window (longer t)
      recovers the 2 super-blocks; and Alg.2's CH selector (scored on the fixed
      conditional-probability representation X=rows of P) recovers each scale.
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
def ch_fixed(Xfixed, lab):
    n = Xfixed.shape[0]; u = np.unique(lab); k = len(u); mu = Xfixed.mean(0)
    if k < 2 or k >= n: return -1.0
    bss = wss = 0.0
    for c in u:
        Xc = Xfixed[lab == c]; muc = Xc.mean(0); bss += len(Xc)*np.sum((muc-mu)**2); wss += np.sum((Xc-muc)**2)
    return float((bss/(k-1))/(wss/(n-k))) if wss > 0 else -1.0
def natural_rw(W):
    d = W.sum(1); d = np.where(d > 0, d, 1.0); return W / d[:, None]
def P_nu_uniform(W):
    P = natural_rw(W); nu = np.ones(W.shape[0]); xi = nu @ P
    num = (nu[:, None]*P) + (P.T*nu[None, :]); return num/(nu+xi)[:, None], (nu+xi)
def hierarchical_digraph(N, rng, p_sub=0.30, p_super=0.05, p_out=0.004):
    nsub = N // 6; fine = np.repeat(np.arange(6), nsub)[:N]; coarse = fine // 3
    A = np.zeros((N, N)); R = rng.random((N, N))
    for i in range(N):
        p = np.full(N, p_out); p[coarse == coarse[i]] = p_super; p[fine == fine[i]] = p_sub
        p[i] = 0; A[i] = (R[i] < p).astype(float)
    for i in range(N):
        if A[i].sum() == 0: A[i, (i+1) % N] = 1.0
    return A, fine, coarse

def main():
    rng = np.random.default_rng(21); N = 300
    W, fine, coarse = hierarchical_digraph(N, rng)
    Pn, m = P_nu_uniform(W); Dinv = 1.0/m; Praw = natural_rw(W)
    ev = np.sort(np.abs(np.linalg.eigvals(Pn)))[::-1]     # moduli (spectrum is real, Claim 3)
    gap2 = float(ev[1] - ev[2]); gap6 = float(ev[5] - ev[6])
    tgrid = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    neff = {t: int(np.sum(ev**t > 0.5)) for t in tgrid}
    def first_t(val):
        for t in tgrid:
            if neff[t] == val: return t
        return None
    t_fine = first_t(6); t_coarse = first_t(2)
    def cluster_at(t, k, seed):
        Pt = np.linalg.matrix_power(Pn, t); emb = Pt * Dinv[None, :]
        return kmeans(emb, k, np.random.default_rng(seed))
    lab_fine = cluster_at(t_fine, 6, 1); lab_coarse = cluster_at(t_coarse, 2, 2)
    ari_f, nmi_f = ari(fine, lab_fine), nmi(fine, lab_fine)
    ari_c, nmi_c = ari(coarse, lab_coarse), nmi(coarse, lab_coarse)
    Xfix = Praw.copy(); J = 8
    def alg2(k, seed):
        rng2 = np.random.default_rng(seed); prof = []; M = Pn.copy()
        for j in range(J+1):
            emb = M * Dinv[None, :]; lab = kmeans(emb, k, rng2, restarts=6)
            prof.append((j, 2**j, ch_fixed(Xfix, lab), lab)); M = M @ M
        jstar = int(np.argmax([p[2] for p in prof])); return prof, jstar
    p6, j6 = alg2(6, 11); p2, j2 = alg2(2, 12)
    td6, td2 = 2**j6, 2**j2
    ari_sel6 = ari(fine, p6[j6][3]); ari_sel2 = ari(coarse, p2[j2][3])
    tstar_coarse = float(np.log(0.5)/np.log(ev[1]))
    tstar_fine = float(np.log(0.5)/np.log(ev[5]))
    ratio = tstar_coarse/tstar_fine

    monotone = all(neff[tgrid[i]] >= neff[tgrid[i+1]] for i in range(len(tgrid)-1))
    seq_6_2_1 = (6 in neff.values()) and (2 in neff.values()) and (1 in neff.values())
    spectrum_ok = gap2 > 0.05 and gap6 > 0.05
    scale_ok = (t_fine is not None and t_coarse is not None and t_fine < t_coarse and
                ari_f > 0.8 and nmi_f > 0.8 and ari_c > 0.8 and nmi_c > 0.8)
    ordering_ok = tstar_coarse > tstar_fine and ratio >= 3.0
    alg2_ok = ari_sel6 > 0.8 and ari_sel2 > 0.8
    verdict = spectrum_ok and monotone and seq_6_2_1 and scale_ok and ordering_ok and alg2_ok

    print("="*74); print("Claim 6  Diffusion time reveals multi-scale clusters (metastability + Alg.2)")
    print("arXiv 2210.00310 / OpenReview 5vI6ApLOg8  (numpy %s)" % np.__version__); print("="*74)
    print(f"hierarchical directed SBM: N={N}, 2 super x 3 sub (6 fine / 2 coarse)")
    print(f"(A) P_(nu) spectrum moduli (top 8): {np.round(ev[:8],4)}")
    print(f"    eigengap after 2 = {gap2:.4f} (coarse) ; after 6 = {gap6:.4f} (fine)  -> two scales")
    print(f"(B) effective #metastable clusters N_eff(t)=#{{|lambda|^t>1/2}}:")
    print("    " + "  ".join(f"t={t}:{neff[t]}" for t in tgrid) + f"   monotone 6->..->1: {monotone}")
    print(f"(C) modal diffusion times t*=ln(1/2)/ln(lambda): fine (lambda_6={ev[5]:.4f}) t*={tstar_fine:.2f}"
          f"  <  coarse (lambda_2={ev[1]:.4f}) t*={tstar_coarse:.2f}   (coarse persists {ratio:.1f}x longer)")
    print(f"(D) scale recovery: FINE window t={t_fine} -> k=6  ARI={ari_f*100:.2f} NMI={nmi_f*100:.2f}")
    print(f"                    COARSE window t={t_coarse} -> k=2  ARI={ari_c*100:.2f} NMI={nmi_c*100:.2f}")
    print(f"                    t_fine({t_fine}) < t_coarse({t_coarse}) : {t_fine < t_coarse}")
    print(f"    Alg.2 (CH on fixed X=rows of raw P) recovers each scale: "
          f"k=6 -> t_d={td6} ARI={ari_sel6*100:.2f} ; k=2 -> t_d={td2} ARI={ari_sel2*100:.2f}")
    print("-"*74)
    print(f"spectrum2gaps={spectrum_ok} Neff_monotone={monotone} seq_6_2_1={seq_6_2_1} "
          f"scale_recovery={scale_ok} coarse_longer_diffusion={ordering_ok} alg2_recovers={alg2_ok}")
    print("VERDICT:", "PASS" if verdict else "PARTIAL/FAIL"); print("="*74)
    out = dict(
        claim="diffusion time controls scale: N_eff drops 6->2->1; coarse mode persists several x longer than fine (t*_coarse>t*_fine); fine clusters at short t, coarse at longer t; Alg.2 CH on fixed representation recovers each requested scale",
        numpy=np.__version__, N=N,
        top_eigenvalue_moduli=[float(x) for x in ev[:8]], eigengap_after2=gap2, eigengap_after6=gap6,
        Neff_vs_t={str(t): neff[t] for t in tgrid}, Neff_monotone=bool(monotone), seq_6_2_1=bool(seq_6_2_1),
        tstar_fine=tstar_fine, tstar_coarse=tstar_coarse, tstar_ratio=ratio,
        coarse_needs_longer_diffusion=bool(ordering_ok),
        t_fine=int(t_fine), t_coarse=int(t_coarse), t_fine_lt_t_coarse=bool(t_fine < t_coarse),
        ari_fine=ari_f, nmi_fine=nmi_f, ari_coarse=ari_c, nmi_coarse=nmi_c,
        alg2_td_k6=int(td6), alg2_td_k2=int(td2), alg2_ari_k6=ari_sel6, alg2_ari_k2=ari_sel2,
        verdict="PASS" if verdict else "PARTIAL")
    json.dump(out, open("results.json", "w"), indent=2); print("wrote results.json")

if __name__ == "__main__":
    main()
