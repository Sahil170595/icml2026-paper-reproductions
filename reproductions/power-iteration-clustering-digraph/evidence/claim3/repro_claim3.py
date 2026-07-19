"""
Claim 3 - The parametrized random walk P_(nu) is self-adjoint in l^2(V, nu+xi)
(reversible), hence has a REAL spectrum in [-1,1] and a unique ergodic measure
pi_nu, whereas the natural directed random walk P has COMPLEX eigenvalues and is
in general reducible on a non-strongly-connected digraph.

Paper P-RWDKC (arXiv 2210.00310 / OpenReview 5vI6ApLOg8):
  Sec.3 (after Prop 3.1) + Prop A.2: "P_(nu) is self-adjoint in l^2(V, nu+xi)"
  and reversible; the associated walk X_nu is ergodic and admits pi_nu.
  This is the fix for the core digraph obstruction stated in Sec.1: directed
  random walks give complex eigenvectors and are usually not irreducible.

CHECKABLE CONSEQUENCES (deterministic, CPU, numpy):
  (A) Detailed balance: D_{nu+xi} P_(nu) is SYMMETRIC (reversibility w.r.t. the
      measure nu+xi), max |asym| ~ machine eps.
  (B) Real spectrum: eigenvalues of P_(nu) are real (max |Im lambda| ~ eps) and
      lie in [-1,1]; the symmetric similarity S=D^{1/2}P_(nu)D^{-1/2} (D=D_{nu+xi})
      shares the same (real) eigenvalues.
  (C) CONTRAST: on the SAME digraph the natural directed walk P=D_out^{-1}W has
      genuinely COMPLEX eigenvalues (max |Im lambda| well above 0).
  (D) Ergodicity: on a weakly- but NOT strongly-connected digraph, P is reducible
      -> its stationary vector puts ZERO mass on transient vertices, whereas
      P_(nu) is irreducible: pi_nu > 0 on every vertex with a spectral gap
      (second-largest eigenvalue modulus < 1).
"""
import json, numpy as np

def natural_rw(W):
    d = W.sum(1); d = np.where(d > 0, d, 1.0); return W / d[:, None]

def build_P_nu(W, nu):
    P = natural_rw(W); xi = nu @ P
    num = (nu[:, None] * P) + (P.T * nu[None, :])   # D_nu P + P^T D_nu
    Pn = num / (nu + xi)[:, None]
    return Pn, P, xi

def stationary(M):
    w, V = np.linalg.eig(M.T)               # left eigenvectors of M
    k = int(np.argmin(np.abs(w - 1.0)))
    v = np.real(V[:, k]); v = v / v.sum()
    return v, w

def second_modulus(M):
    w = np.linalg.eigvals(M); m = np.sort(np.abs(w))[::-1]
    return float(m[1]) if len(m) > 1 else 0.0

def strongly_connected(W):
    # reachability by transitive closure on adjacency support
    A = (W > 0).astype(float); N = W.shape[0]
    R = A.copy(); np.fill_diagonal(R, 1.0)
    for _ in range(int(np.ceil(np.log2(max(2, N)))) + 1):
        R = ((R @ R) > 0).astype(float)
    return bool(np.all(R > 0))

def digraph_sbm(N, k, pin, pout, rng, bias=0.0):
    z = np.repeat(np.arange(k), N // k)[:N]
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            pr = pin if z[i] == z[j] else pout
            if bias and z[i] < z[j]:  # directional bias A->B (source/sink asymmetry)
                pr = pr * (1 + bias)
            if rng.random() < pr: A[i, j] = 1.0
    for i in range(N):
        if A[i].sum() == 0: A[i, (i+1) % N] = 1.0
    return A, z

def block_transient(N, rng):
    # two blocks; edges ONLY A->B (B cannot reach A) => raw walk reducible
    h = N // 2; z = np.array([0]*h + [1]*(N-h))
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if z[i] == z[j] and rng.random() < 0.25: A[i, j] = 1.0
            elif z[i] == 0 and z[j] == 1 and rng.random() < 0.08: A[i, j] = 1.0
    for i in range(N):
        if A[i].sum() == 0: A[i, (i+1) % N] = 1.0
    return A, z

def main():
    rng = np.random.default_rng(3)
    wA = dict(asym=0.0, imP=0.0, sprange=0.0, sim=0.0)
    raw_imag = []; ncase = 0
    for (N, k, pin, pout, bias) in [(60,3,0.35,0.03,0.0),(90,3,0.30,0.02,2.0),(120,4,0.25,0.02,0.0)]:
        for s in range(3):
            W, z = digraph_sbm(N, k, pin, pout, rng, bias=bias)
            for nu in (np.ones(N), rng.uniform(0.2,1.0,N), W.sum(1)+1e-6):
                Pn, P, xi = build_P_nu(W, np.asarray(nu, float))
                D = nu + xi
                # (A) detailed balance
                DB = D[:, None] * Pn
                wA["asym"] = max(wA["asym"], float(np.max(np.abs(DB - DB.T))))
                # (B) real spectrum + range
                ev = np.linalg.eigvals(Pn)
                wA["imP"] = max(wA["imP"], float(np.max(np.abs(ev.imag))))
                wA["sprange"] = max(wA["sprange"], float(max(np.real(ev).max()-1.0, -1.0-np.real(ev).min(), 0.0)))
                s_ = np.sqrt(D)
                S = (s_[:, None] * Pn) / s_[None, :]       # symmetric similarity
                wA["sim"] = max(wA["sim"], float(np.max(np.abs(S - S.T))))
                # (C) contrast: raw directed walk complex eigenvalues
                raw_imag.append(float(np.max(np.abs(np.linalg.eigvals(P).imag))))
                ncase += 1

    # (D) ergodicity contrast on a reducible digraph
    W, z = block_transient(80, rng)
    sc = strongly_connected(W)
    P = natural_rw(W)
    pi_raw, _ = stationary(P)
    Pn, _, xi = build_P_nu(W, np.ones(80))
    pi_nu, _ = stationary(Pn)
    raw_zeros = int(np.sum(pi_raw < 1e-8))     # transient vertices -> 0 mass
    nu_min = float(pi_nu.min())                # P_(nu) stationary strictly positive?
    gap_raw = 1.0 - second_modulus(P)
    gap_nu = 1.0 - second_modulus(Pn)
    Pn_sc = strongly_connected((Pn > 1e-12).astype(float))

    real_ok = wA["imP"] < 1e-9 and wA["asym"] < 1e-9 and wA["sprange"] < 1e-9 and wA["sim"] < 1e-9
    contrast_ok = float(np.max(raw_imag)) > 1e-3
    ergo_ok = (not sc) and raw_zeros > 0 and nu_min > 0 and Pn_sc and gap_nu > 1e-6
    all_ok = real_ok and contrast_ok and ergo_ok

    print("="*74)
    print("Claim 3  Reversibility -> real spectrum + ergodicity of P_(nu)")
    print("arXiv 2210.00310 / OpenReview 5vI6ApLOg8  (numpy %s)  cases=%d" % (np.__version__, ncase))
    print("="*74)
    print(f"(A) detailed balance  max|D_(nu+xi)P_(nu) - (.)^T|     = {wA['asym']:.3e}  (=0 => reversible)")
    print(f"(B) P_(nu) spectrum   max|Im lambda|                   = {wA['imP']:.3e}  (=0 => REAL)")
    print(f"    spectrum outside [-1,1] (excess)                   = {wA['sprange']:.3e}  (=0 => in [-1,1])")
    print(f"    symmetric similarity S=D^1/2 P D^-1/2  max|S-S^T|  = {wA['sim']:.3e}  (=0 => self-adjoint)")
    print(f"(C) CONTRAST raw directed P  max|Im lambda| (worst)    = {np.max(raw_imag):.4f}  (>0 => COMPLEX)")
    print(f"                             mean over cases           = {np.mean(raw_imag):.4f}")
    print("-"*74)
    print(f"(D) ergodicity on reducible digraph (block A->B only):")
    print(f"    strongly connected?                     raw graph  = {sc}  (False => raw walk reducible)")
    print(f"    raw stationary: #vertices with ~0 mass (transient) = {raw_zeros} / 80")
    print(f"    P_(nu) stationary pi_nu : min value                = {nu_min:.3e}  (> 0 on ALL vertices)")
    print(f"    P_(nu) support strongly connected?                 = {Pn_sc}  (True => irreducible)")
    print(f"    spectral gap 1-|lambda_2| : raw = {gap_raw:.3e}   P_(nu) = {gap_nu:.3e}")
    print("-"*74)
    print("VERDICT:", "PASS" if all_ok else "FAIL",
          f"(real={real_ok}, complex-contrast={contrast_ok}, ergodic={ergo_ok})")
    print("="*74)

    out = dict(
        claim="P_(nu) reversible/self-adjoint in l2(V,nu+xi) => real spectrum in [-1,1] and ergodic pi_nu>0; raw directed P complex & reducible",
        numpy=np.__version__, cases=ncase,
        detailed_balance_max_asym=wA["asym"], P_nu_max_abs_imag=wA["imP"],
        P_nu_spectrum_excess_over_pm1=wA["sprange"], symmetric_similarity_max_asym=wA["sim"],
        raw_directed_max_abs_imag=float(np.max(raw_imag)), raw_directed_mean_abs_imag=float(np.mean(raw_imag)),
        reducible_graph_strongly_connected=bool(sc),
        raw_stationary_zero_mass_vertices=raw_zeros,
        P_nu_stationary_min=nu_min, P_nu_support_strongly_connected=bool(Pn_sc),
        spectral_gap_raw=gap_raw, spectral_gap_P_nu=gap_nu,
        verdict="PASS" if all_ok else "FAIL")
    json.dump(out, open("results.json", "w"), indent=2)
    print("wrote results.json")

if __name__ == "__main__":
    main()
