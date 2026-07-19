"""
Independent NumPy reproduction library for:
  "Improved Analysis of the Accelerated Noisy Power Method with Applications
   to Decentralized PCA" (Aguie, Even, Massoulie; arXiv 2602.03682,
   OpenReview UTiEfkfNQ2).

This module implements, from the paper's own equations (Eq. 1, Alg. 1, Alg. 2,
Thm 2.2-2.5, Thm 3.3), independently of the authors' released code:
  - the (Accelerated) Noisy Power Method, ANPM  (Eq. 1)
  - Accelerated Gossip                          (Alg. 1)
  - Decentralized (accelerated) PCA: ADePM, DePM, DeEPCA (Alg. 2 + baselines)
  - principal-angle / subspace-error metrics

The official authors' code (github.com/pierreaguie/ANPM @ 3623010, pinned)
was read for correctness checks (variable/shape conventions, the paper's own
App. E synthetic-instance recipe, sign convention for QR) but every function
below is a fresh implementation, not a copy.

CPU-only, float64, deterministic (numpy.random.default_rng with explicit
per-experiment seeds).
"""
import numpy as np
from scipy.linalg import subspace_angles


# --------------------------------------------------------------------------
# Core ANPM / NPM (Eq. 1 of the paper)
# --------------------------------------------------------------------------

def qr_pos_diag(Y):
    """QR with the sign convention diag(R) > 0 (deterministic factorization;
    the paper leaves the QR sign ambiguity unspecified)."""
    Q, R = np.linalg.qr(Y, mode="reduced")
    signs = np.sign(np.diagonal(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    R = R * signs[:, np.newaxis]
    return Q, R


def anpm(A, beta, T, X0, Xi):
    """Accelerated Noisy Power Method (Eq. 1). beta=0 reduces to the
    (non-accelerated) Noisy Power Method (NPM).

    A: (d,d) PSD.  beta: momentum coefficient (scalar).  T: number of steps.
    X0: (d,k) column-orthonormal init.  Xi: (T,d,k) perturbation sequence
    (Xi[0] is added at the special half-step init, Xi[t] for t=1..T-1 in the
    main recursion, matching the paper's Eq. 1 indexing).

    Returns array of shape (T+1, d, k): [X0, X1, ..., X_T].
    """
    X_prev = X0.copy()
    Y1 = 0.5 * (A @ X0) + Xi[0]
    X, R = qr_pos_diag(Y1)
    X_list = [X0, X]
    for t in range(1, T):
        Y = A @ X - beta * (X_prev @ np.linalg.inv(R)) + Xi[t]
        X_new, R = qr_pos_diag(Y)
        X_prev, X = X, X_new
        X_list.append(X)
    return np.stack(X_list, axis=0)


def anpm_adaptive_beta(A, T, X0, Xi):
    """ANPM with the paper's adaptive momentum beta_t (Eq. 5): beta_t is set
    from the smallest observed Rayleigh coefficient of the current iterate."""
    X_prev = X0.copy()
    Y1 = 0.5 * (A @ X0) + Xi[0]
    X, R = qr_pos_diag(Y1)
    X_list = [X0, X]
    for t in range(1, T):
        rq = np.sort(np.diag(X.T @ (A @ X + Xi[t])))[0]
        beta_t = max(rq, 0.0) ** 2 / 4.0
        Y = A @ X - beta_t * (X_prev @ np.linalg.inv(R)) + Xi[t]
        X_new, R = qr_pos_diag(Y)
        X_prev, X = X, X_new
        X_list.append(X)
    return np.stack(X_list, axis=0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def sin_thetak(X, U, k):
    """sin of the k-th principal angle between span(X) and span(U)."""
    p = min(X.shape[1], U.shape[1])
    return float(np.sin(subspace_angles(X, U)[p - k]))


def tan_thetak(X, U, k):
    s = sin_thetak(X, U, k)
    p = min(X.shape[1], U.shape[1])
    c = float(np.cos(subspace_angles(X, U)[p - k]))
    return s / max(c, 1e-300)


# --------------------------------------------------------------------------
# Synthetic instance generation (paper App. E.1 recipe)
# --------------------------------------------------------------------------

def generate_eigenvectors(d, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q


def generate_X0(d, k, rng):
    X0, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return X0


def generate_matrix(lambdas, U):
    return (U * lambdas[np.newaxis, :]) @ U.T


def generate_adversarial_noise(d, k, T, eta, U, rng):
    """Fixed-direction ('worst case') perturbation: at every step, a random
    positive combination of ALL eigendirections of U, of spectral norm eta,
    with a persistent sign (App. E.1 of the paper)."""
    Z = np.abs(rng.standard_normal((T, d, k)))
    out = np.empty_like(Z)
    for t in range(T):
        out[t] = U @ (Z[t] / np.linalg.norm(Z[t], ord=2))
    return -eta * out


def generate_stochastic_noise(d, k, T, eta, rng):
    Z = rng.standard_normal((T, d, k))
    for t in range(T):
        Z[t] /= np.linalg.norm(Z[t], ord=2)
    return eta * Z


# --------------------------------------------------------------------------
# Gossip (Alg. 1) and decentralized PCA variants (Alg. 2 + baselines)
# --------------------------------------------------------------------------

def compute_omega(W):
    ev = np.linalg.eigvalsh(W)
    lambda_2 = ev[-2]
    lambda_n = ev[0]
    gamma = 1.0 - max(lambda_2, -lambda_n)
    gamma = min(max(gamma, 1e-12), 2.0 - 1e-12)
    omega = (1 - np.sqrt(gamma * (2 - gamma))) / (1 + np.sqrt(gamma * (2 - gamma)))
    return float(omega), float(gamma)


def accelerated_gossip(Y, W, L, omega):
    """Alg. 1. Y: (n, d, k) local vectors at each of n agents."""
    X = Y.copy()
    X_prev = Y.copy()
    for _ in range(L):
        X_new = (1 + omega) * np.einsum("ij,jkl->ikl", W, X) - omega * X_prev
        X_prev, X = X, X_new
    return X


def adepm(A, T, beta, X0, W, omega, L):
    """Accelerated Decentralized PCA (Alg. 2). A: (n,d,d) local matrices."""
    n = A.shape[0]
    X = np.stack([X0.copy() for _ in range(n)], axis=0)
    X_prev = X.copy()
    R = np.stack([np.eye(X0.shape[1]) for _ in range(n)], axis=0)
    X_list = [X]
    for t in range(1, T):
        if t == 1:
            Y = 0.5 * np.einsum("ijk,ikl->ijl", A, X)
        else:
            Rinv = np.linalg.inv(R)
            Y = np.einsum("ijk,ikl->ijl", A, X) - beta * np.einsum("ijl,ilk->ijk", X_prev, Rinv)
        X_prev = X.copy()
        Y = accelerated_gossip(Y, W, L, omega)
        X, R = np.linalg.qr(Y)
        # deterministic sign convention, batched
        signs = np.sign(np.diagonal(R, axis1=1, axis2=2))
        signs[signs == 0] = 1.0
        X = X * signs[:, np.newaxis, :]
        R = R * signs[:, :, np.newaxis]
        X_list.append(X)
    return np.stack(X_list, axis=0)


def depm(A, T, X0, W, omega, L):
    """Non-accelerated decentralized power method, given the SAME gossip
    primitive (Accelerated Gossip) as ADePM for a fair identical-L comparison
    (the paper's Table 2 / the Wai et al. 2017 baseline updated to use
    Alg. 1 as its averaging step)."""
    n = A.shape[0]
    X = np.stack([X0.copy() for _ in range(n)], axis=0)
    X_list = [X]
    for _ in range(T):
        Y = np.einsum("ijk,ikl->ijl", A, X)
        Y = accelerated_gossip(Y, W, L, omega)
        X, _ = np.linalg.qr(Y)
        X_list.append(X)
    return np.stack(X_list, axis=0)


def deepca(A, T, X0, W, omega, L):
    """DeEPCA (Ye & Zhang, 2021) gradient-tracking baseline."""
    n = A.shape[0]
    X = np.stack([X0.copy() for _ in range(n)], axis=0)
    S_prev = X.copy()
    Y_prev = X.copy()
    X_list = [X]
    for _ in range(T):
        Y = np.einsum("ijk,ikl->ijl", A, X)
        S = S_prev + Y - Y_prev
        S = accelerated_gossip(S, W, L, omega)
        X_new, _ = np.linalg.qr(S)
        for col in range(X_new.shape[-1]):
            s = np.sign(np.einsum("ij,ij->i", X_new[:, :, col], X[:, :, col]))
            s[s == 0] = 1.0
            X_new[:, :, col] *= s[:, np.newaxis]
        X_list.append(X_new)
        S_prev, Y_prev, X = S.copy(), Y.copy(), X_new
    return np.stack(X_list, axis=0)


def largest_spnorm(A):
    return max(np.linalg.norm(A[i], ord=2) for i in range(A.shape[0]))


# --------------------------------------------------------------------------
# Communication graphs
# --------------------------------------------------------------------------

def ring_plus_random_graph(n, extra_edges, rng):
    """Ring lattice (each node <-> 2 neighbours) plus `extra_edges` random
    chords, mirroring the paper's App. E.3.2 "G plus ~n log n random extra
    edges" recipe at small scale. Returns adjacency (n,n) 0/1, symmetric,
    zero diagonal, connected by construction."""
    Adj = np.zeros((n, n))
    for i in range(n):
        Adj[i, (i + 1) % n] = 1
        Adj[(i + 1) % n, i] = 1
    added = 0
    tries = 0
    while added < extra_edges and tries < 50 * extra_edges:
        i, j = rng.integers(0, n, size=2)
        tries += 1
        if i == j or Adj[i, j] == 1:
            continue
        Adj[i, j] = 1
        Adj[j, i] = 1
        added += 1
    return Adj


def metropolis_hastings_weights(Adj):
    n = Adj.shape[0]
    deg = Adj.sum(axis=1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if Adj[i, j] == 1:
                W[i, j] = 1.0 / (max(deg[i], deg[j]) + 1.0)
        W[i, i] = 1.0 - W[i].sum()
    return W


# --------------------------------------------------------------------------
# Misc: Xu (2023) prior-work admissible noise bound (paper's Thm B.1 restated)
# --------------------------------------------------------------------------

def xu2023_admissible_noise(t, T, beta, lambda1_plus, sin_theta0, const=1.0):
    """||Xi_t|| = O( sqrt(beta) sin(theta0) (sqrt(beta)/lambda1^+)^t
                      / (T (T-t+1)) )   evaluated with constant `const`."""
    if t > T:
        return 0.0
    rate = np.sqrt(beta) / lambda1_plus
    return const * np.sqrt(beta) * sin_theta0 * (rate ** t) / (T * (T - t + 1))


def loglog_slope(xs, ys):
    """np.polyfit slope of log10(ys) on log10(xs), plus R^2."""
    lx = np.log10(np.asarray(xs, dtype=float))
    ly = np.log10(np.asarray(ys, dtype=float))
    p = np.polyfit(lx, ly, 1)
    fit = np.polyval(p, lx)
    ss_res = np.sum((ly - fit) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(p[0]), float(p[1]), float(r2)
