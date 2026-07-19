"""
mds_common.py  --  Shared machinery for the CPU reproduction of
"Minimum Distance Summaries for Robust Neural Posterior Estimation"
(Khoo, Prangle, Liu, Beaumont; ICML 2026; arXiv:2602.09161; OpenReview lq8fNVME8v).

Everything here is deterministic, single-thread CPU, no network, no HF writes.

Implements, faithfully to the paper (arxiv_main.tex):
  * The two data-generating processes: the conjugate bivariate Gaussian location
    model (App. C, `app:gauss`) and the Ornstein-Uhlenbeck process (App. C, `app:oup`).
  * A genuine conditional-density Neural Posterior Estimator q_psi(theta | s)
    (a neural conditional-Gaussian trained by NLL) -- FROZEN at test time.
  * The Minimum-Distance-Summary (MDS) adapter with Random Fourier Features:
      - scikit-learn RBFSampler (K=512), median-heuristic bandwidth
        (arxiv_main.tex:944),
      - an amortized decoder mean-embedding regressor mu_omega : R^{d_s} -> R^K,
        a 2 x 256 fully-connected net trained by MSE (Algorithm 1, L283/L944),
      - test-time L-BFGS with strong-Wolfe line search, initialised at the observed
        summary (arxiv_main.tex:285 / L877).
  * Exact RBF-kernel MMD, for the RFF-vs-exact approximation-gap check.

The NPE is never mutated at test time: only the *query summary* s changes
(observed summary vs MDS-adapted summary).  This is the plug-in separation.
"""
import os, io, json, time, hashlib, platform
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import torch
from sklearn.kernel_approximation import RBFSampler

torch.set_num_threads(1)
DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
#  Determinism / provenance helpers
# --------------------------------------------------------------------------- #
def set_determinism(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_state_dict(module: torch.nn.Module) -> str:
    """Deterministic SHA-256 over a module's tensor state (sorted keys)."""
    h = hashlib.sha256()
    sd = module.state_dict()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def n_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def environment_report() -> dict:
    import sklearn, scipy
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "device": str(DEVICE),
        "gpu_used": False,
        "torch_threads": torch.get_num_threads(),
    }


# --------------------------------------------------------------------------- #
#  Data-generating processes
# --------------------------------------------------------------------------- #
# ---- Gaussian conjugate bivariate location model (app:gauss) -------------- #
class GaussianModel:
    d = 2               # bivariate
    N = 100             # observations per dataset
    sigma_x = 1.0       # x_i | theta ~ N(theta, I_d)
    prior_sd = 1.0      # theta ~ N(0, I_d)

    @classmethod
    def sample_theta(cls, rng, n):
        return rng.normal(0.0, cls.prior_sd, size=(n, cls.d))

    @classmethod
    def simulate(cls, rng, theta):
        # theta: (n, d) -> x: (n, N, d)
        n = theta.shape[0]
        eps = rng.normal(0.0, cls.sigma_x, size=(n, cls.N, cls.d))
        return theta[:, None, :] + eps

    @staticmethod
    def summary(x):
        # sample mean over the N observations -> (n, d);  sufficient statistic
        return x.mean(axis=1)

    @classmethod
    def analytic_posterior_mean(cls, s):
        # conjugate: prior N(0, I), likelihood mean s with precision N/sigma^2.
        # posterior mean = (N/sigma^2) / (N/sigma^2 + 1/prior_sd^2) * s
        a = cls.N / cls.sigma_x ** 2
        b = 1.0 / cls.prior_sd ** 2
        return (a / (a + b)) * s

    @classmethod
    def contaminate(cls, rng, x_clean, eps, delta):
        """Huber contamination: replace an eps fraction of the N points by
        outliers shifted by L2-magnitude `delta` along a per-dataset random
        direction with a random +/- sign (equal probability), matching
        arxiv_main.tex:403/983 and the Figure-1 shift of 8."""
        x = x_clean.copy()
        n, N, d = x.shape
        k = int(round(eps * N))
        if k == 0:
            return x
        for i in range(n):
            idx = rng.choice(N, size=k, replace=False)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            u = np.array([1.0, 1.0]) / np.sqrt(2.0)   # fixed unit direction
            x[i, idx, :] = x[i, idx, :] + sign * delta * u
        return x


# ---- Ornstein-Uhlenbeck process (app:oup) --------------------------------- #
class OUPModel:
    d_theta = 2
    N = 100             # trajectories per dataset
    T = 25              # trajectory length (timesteps)
    dt = 0.2            # Euler-Maruyama step
    sigma2 = 0.1        # diffusion variance
    X0 = 10.0
    # contamination process (arxiv_main.tex:1030)
    theta_c = np.array([-0.5, 1.0])
    sigma2_c = 0.5

    @classmethod
    def sample_theta(cls, rng, n):
        t1 = rng.uniform(0.0, 2.0, size=(n, 1))
        t2 = rng.uniform(-2.0, 2.0, size=(n, 1))
        return np.concatenate([t1, t2], axis=1)

    @classmethod
    def _simulate_traj(cls, rng, theta, n_traj, sigma2):
        """theta: (2,) -> trajectories (n_traj, T)."""
        t1, t2 = float(theta[0]), float(theta[1])
        sigma = np.sqrt(sigma2)
        X = np.full((n_traj,), cls.X0, dtype=np.float64)
        out = np.empty((n_traj, cls.T), dtype=np.float64)
        level = np.exp(t2)
        sqdt = np.sqrt(cls.dt)
        for t in range(cls.T):
            dW = rng.normal(0.0, 1.0, size=(n_traj,))
            X = X + t1 * (level - X) * cls.dt + sigma * sqdt * dW
            out[:, t] = X
        return out

    @classmethod
    def simulate_dataset(cls, rng, theta):
        return cls._simulate_traj(rng, theta, cls.N, cls.sigma2)  # (N, T)

    @staticmethod
    def summary(X):
        """X: (N, T) trajectories -> (mean, var, lag-1 autocorr). d_s = 3."""
        s1 = X.mean()
        s2 = ((X - s1) ** 2).mean()
        a = X[:, :-1].ravel()
        b = X[:, 1:].ravel()
        am, bm = a.mean(), b.mean()
        denom = np.sqrt(((a - am) ** 2).sum() * ((b - bm) ** 2).sum())
        s3 = ((a - am) * (b - bm)).sum() / denom if denom > 0 else 0.0
        return np.array([s1, s2, s3])

    @classmethod
    def contaminate_dataset(cls, rng, X_clean, eps):
        """Replace an eps fraction of the N trajectories by contaminated ones."""
        X = X_clean.copy()
        k = int(round(eps * cls.N))
        if k == 0:
            return X
        idx = rng.choice(cls.N, size=k, replace=False)
        X[idx, :] = cls._simulate_traj(rng, cls.theta_c, k, cls.sigma2_c)
        return X


# --------------------------------------------------------------------------- #
#  Neural Posterior Estimator  q_psi(theta | s)   (genuine conditional density)
# --------------------------------------------------------------------------- #
class CondGaussNPE(torch.nn.Module):
    """Neural conditional-Gaussian posterior estimator: MLP(s) -> (mu, L),
    with L a lower-triangular Cholesky factor.  Trained by exact NLL.
    A legitimate conditional-density NPE (the density family is Gaussian, which
    is exact for the conjugate Gaussian task and a flexible approximation for OUP)."""

    def __init__(self, d_s, d_theta, hidden=(64, 64)):
        super().__init__()
        self.d_theta = d_theta
        self.n_tri = d_theta * (d_theta + 1) // 2
        layers, prev = [], d_s
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.Tanh()]
            prev = h
        self.body = torch.nn.Sequential(*layers)
        self.head_mu = torch.nn.Linear(prev, d_theta)
        self.head_L = torch.nn.Linear(prev, self.n_tri)
        tril = torch.tril_indices(d_theta, d_theta)
        self.register_buffer("tri_r", tril[0])
        self.register_buffer("tri_c", tril[1])
        self.register_buffer("diag_mask",
                             (tril[0] == tril[1]).to(torch.float32))

    def forward(self, s):
        h = self.body(s)
        mu = self.head_mu(h)
        raw = self.head_L(h)
        B = s.shape[0]
        L = torch.zeros(B, self.d_theta, self.d_theta, dtype=s.dtype)
        vals = raw * (1 - self.diag_mask) + torch.nn.functional.softplus(raw) * self.diag_mask
        L[:, self.tri_r, self.tri_c] = vals
        return mu, L

    def nll(self, theta, s):
        mu, L = self.forward(s)
        diff = (theta - mu).unsqueeze(-1)                       # (B, d, 1)
        sol = torch.linalg.solve_triangular(L, diff, upper=False)
        maha = (sol.squeeze(-1) ** 2).sum(-1)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)
        return 0.5 * maha + 0.5 * logdet + 0.5 * self.d_theta * np.log(2 * np.pi)

    @torch.no_grad()
    def posterior_mean(self, s_np):
        s = torch.as_tensor(s_np, dtype=torch.float32).reshape(1, -1)
        mu, _ = self.forward(s)
        return mu.squeeze(0).numpy()


def train_npe(npe, S, THETA, epochs=200, batch=256, lr=1e-3, seed=0, verbose=False):
    set_determinism(seed)
    S = torch.as_tensor(S, dtype=torch.float32)
    THETA = torch.as_tensor(THETA, dtype=torch.float32)
    opt = torch.optim.Adam(npe.parameters(), lr=lr)
    n = S.shape[0]
    g = torch.Generator().manual_seed(seed)
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            loss = npe.nll(THETA[j], S[j]).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        losses.append(tot / n)
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"  npe epoch {ep:4d}  nll {losses[-1]:.4f}")
    return losses


# --------------------------------------------------------------------------- #
#  Decoder mean-embedding regressor  mu_omega : R^{d_s} -> R^K   (Algorithm 1)
# --------------------------------------------------------------------------- #
class MeanEmbedNet(torch.nn.Module):
    """Fully-connected 2 x 256 net predicting the K-dim conditional mean
    embedding from the summary (arxiv_main.tex:944)."""

    def __init__(self, d_s, K, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_s, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, K),
        )

    def forward(self, s):
        return self.net(s)


def train_decoder(dec, S, ZBAR, epochs=300, batch=256, lr=1e-3, seed=0, verbose=False):
    set_determinism(seed)
    S = torch.as_tensor(S, dtype=torch.float32)
    ZBAR = torch.as_tensor(ZBAR, dtype=torch.float32)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    n = S.shape[0]
    g = torch.Generator().manual_seed(seed)
    last = None
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            pred = dec(S[j])
            loss = ((pred - ZBAR[j]) ** 2).sum(-1).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        last = tot / n
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  dec epoch {ep:4d}  mse {last:.6e}")
    return last


# --------------------------------------------------------------------------- #
#  Random Fourier Features + MMD
# --------------------------------------------------------------------------- #
def median_heuristic_gamma(x_pool, rng, max_pts=1000):
    """gamma = 1/(2 sigma^2), sigma = median pairwise Euclidean distance."""
    x = np.asarray(x_pool, dtype=np.float64)
    if x.shape[0] > max_pts:
        idx = rng.choice(x.shape[0], size=max_pts, replace=False)
        x = x[idx]
    d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(x.shape[0], k=1)
    med = np.median(np.sqrt(d2[iu]))
    sigma = med if med > 1e-8 else 1.0
    return 1.0 / (2.0 * sigma ** 2), sigma


def make_rff(gamma, K, seed, d_in):
    rbf = RBFSampler(gamma=gamma, n_components=K, random_state=seed)
    rbf.fit(np.zeros((1, d_in)))   # only needs dimensionality
    return rbf


def embed_points(rbf, X):
    """X: (m, d_in) points -> mean RFF embedding (K,)."""
    return rbf.transform(np.asarray(X, dtype=np.float64)).mean(axis=0)


def mmd2_exact(A, B, gamma):
    """Unbiased-ish (biased V-statistic) exact RBF MMD^2 between point sets."""
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)

    def k(X, Y):
        d2 = np.sum(X ** 2, 1)[:, None] + np.sum(Y ** 2, 1)[None, :] - 2 * X @ Y.T
        return np.exp(-gamma * np.maximum(d2, 0.0))
    return k(A, A).mean() + k(B, B).mean() - 2.0 * k(A, B).mean()


def mmd2_rff(zbar_a, zbar_b):
    d = zbar_a - zbar_b
    return float(d @ d)


# --------------------------------------------------------------------------- #
#  Test-time MDS adaptation:  s* = argmin_s || mu_omega(s) - zbar_obs ||^2
# --------------------------------------------------------------------------- #
def mds_adapt(dec, zbar_obs, s0, max_iter=100):
    """L-BFGS with strong-Wolfe line search, initialised at observed summary.
    Returns (s_star_np, elapsed_ms).  The decoder is frozen (no grad on params)."""
    for p in dec.parameters():
        p.requires_grad_(False)
    z = torch.as_tensor(zbar_obs, dtype=torch.float32)
    s = torch.as_tensor(np.asarray(s0, dtype=np.float64), dtype=torch.float32).clone()
    s.requires_grad_(True)
    opt = torch.optim.LBFGS([s], lr=1.0, max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        pred = dec(s.reshape(1, -1)).reshape(-1)
        loss = ((pred - z) ** 2).sum()
        loss.backward()
        return loss

    t0 = time.perf_counter()
    opt.step(closure)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    for p in dec.parameters():
        p.requires_grad_(True)
    return s.detach().numpy(), elapsed_ms
