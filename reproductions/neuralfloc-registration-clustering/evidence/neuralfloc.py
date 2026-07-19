"""
neuralfloc.py  --  shared library for the independent NeuralFLoC reproduction.

Independent CPU/torch re-implementation of the core of
"NeuralFLoC: Neural Flow-Based Joint Registration and Clustering of Functional
Data" (arXiv 2602.03169 / OpenReview JIkyyfkeoE).

  * parametric monotone boundary-preserving warp family (Theorem 4.1 targets)
  * Neural-ODE warp: velocity field MLP f(tau, t, w_i) conditioned on a
    per-curve latent w_i (Eq 5), Softplus output -> strictly increasing flow
    -> boundary-normalised gamma in Gamma (Eq 6); RK4/Euler integration on S
    internal steps then interpolated to the sample grid
  * realistic functional-data simulation (C clusters, distinct SHAPES + strong
    phase warps + amplitude variation + noise): phase confounds raw clustering,
    registration reveals shape (oracle-aligned clustering separates perfectly)
  * SRVF, Fourier projection, Student-t soft assignment, cluster-conditional
    SRVF registration loss + DEC KL clustering loss (paper Eqs 3,6-12)
  * full joint NeuralFLoC model + training loop with ablation switches
  * metrics: ARI / NMI / clustering ACC (Hungarian) / ATV (Eq A.1) /
    warp-approximation error / diffeomorphism diagnostics
  * baselines: k-means on raw curves; register-then-cluster (global template)

CPU-only, single thread, deterministic seeds.
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

# ----------------------------------------------------------------------------
# 1. Parametric monotone boundary-preserving warp family (Theorem 4.1 targets)
# ----------------------------------------------------------------------------
def warp_exp(t, a):
    a = float(a)
    if abs(a) < 1e-8:
        return t.copy()
    return (np.exp(a * t) - 1.0) / (np.exp(a) - 1.0)

def warp_beta(t, alpha, beta):
    from scipy.stats import beta as _beta
    return _beta.cdf(t, alpha, beta)

def warp_logit(t, k, m):
    def s(x):
        return 1.0 / (1.0 + np.exp(-k * (x - m)))
    lo, hi = s(0.0), s(1.0)
    return (s(t) - lo) / (hi - lo)

def warp_sine(t, b):
    return t + b * np.sin(np.pi * t) / np.pi          # monotone for |b|<1

def warp_power(t, p):
    return np.power(t, float(p))

def target_warp_family():
    """Return (t_dense, [(name, gamma_values), ...]) -- diverse monotone warps."""
    T = 400
    t = np.linspace(0.0, 1.0, T)
    fam = [
        ("exp_a=+3.0", warp_exp(t, 3.0)),
        ("exp_a=-3.0", warp_exp(t, -3.0)),
        ("exp_a=+1.5", warp_exp(t, 1.5)),
        ("beta_2_5", warp_beta(t, 2.0, 5.0)),
        ("beta_5_2", warp_beta(t, 5.0, 2.0)),
        ("beta_3_3", warp_beta(t, 3.0, 3.0)),
        ("logit_k8_m0.35", warp_logit(t, 8.0, 0.35)),
        ("logit_k6_m0.6", warp_logit(t, 6.0, 0.6)),
        ("sine_b=+0.8", warp_sine(t, 0.8)),
        ("sine_b=-0.8", warp_sine(t, -0.8)),
        ("power_p=2.2", warp_power(t, 2.2)),
        ("power_p=0.45", warp_power(t, 0.45)),
    ]
    return t, fam


# ----------------------------------------------------------------------------
# 2. Neural-ODE warp module (conditioned on per-curve latent w_i)
# ----------------------------------------------------------------------------
class VelocityField(nn.Module):
    """MLP velocity field f(tau, t, w) -> R^C; Softplus output => gamma_dot>0."""
    def __init__(self, cdim, latent, hidden, depth=2, act="elu", gain=1.5):
        super().__init__()
        A = {"elu": nn.ELU, "tanh": nn.Tanh, "relu": nn.ReLU, "softplus": nn.Softplus}[act]
        layers, d = [], cdim + 1 + latent
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), A()]
            d = hidden
        layers += [nn.Linear(d, cdim)]
        self.net = nn.Sequential(*layers)
        self.sp = nn.Softplus()
        with torch.no_grad():                          # larger init -> warps start
            for m in self.net:                         # with energy, escaping the
                if isinstance(m, nn.Linear):           # near-identity local minimum
                    m.weight.mul_(gain)

    def forward(self, tau, t_scalar, w):
        tcol = torch.full((tau.shape[0], 1), float(t_scalar),
                          dtype=tau.dtype, device=tau.device)
        return self.sp(self.net(torch.cat([tau, tcol, w], dim=1)))


def _interp_along_knots(gamma_knots, tgrid):
    """gamma_knots:(N,S+1,C) uniform on [0,1] -> sample at tgrid (T,) -> (N,T,C)."""
    S1 = gamma_knots.shape[1]
    pos = tgrid.clamp(0.0, 1.0) * (S1 - 1)
    lo = pos.floor().long().clamp(0, S1 - 2)
    frac = (pos - lo.to(pos.dtype)).view(1, -1, 1)
    lo_e = lo.view(1, -1, 1).expand(gamma_knots.shape[0], -1, gamma_knots.shape[2])
    glo = torch.gather(gamma_knots, 1, lo_e)
    ghi = torch.gather(gamma_knots, 1, lo_e + 1)
    return glo + frac * (ghi - glo)


def integrate_warp(vf, w, tgrid, cdim, S=40, method="euler"):
    """Integrate dtau/dt = vf(tau,t,w), tau(0)=0 (C-dim), boundary-normalise,
       interpolate gamma onto tgrid.  Returns (N,T,C) in [0,1]."""
    N = w.shape[0]
    h = 1.0 / S
    tau = torch.zeros(N, cdim, dtype=w.dtype, device=w.device)
    traj = [tau]
    for m in range(S):
        t0 = m * h
        if method == "euler":
            tau = tau + h * vf(tau, t0, w)
        else:  # rk4
            k1 = vf(tau, t0, w)
            k2 = vf(tau + 0.5 * h * k1, t0 + 0.5 * h, w)
            k3 = vf(tau + 0.5 * h * k2, t0 + 0.5 * h, w)
            k4 = vf(tau + h * k3, t0 + h, w)
            tau = tau + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(tau)
    tau_all = torch.stack(traj, dim=1)                 # (N,S+1,C)
    lo = tau_all[:, :1, :]; hi = tau_all[:, -1:, :]
    gamma_knots = (tau_all - lo) / (hi - lo + 1e-12)
    return _interp_along_knots(gamma_knots, tgrid)


# ----------------------------------------------------------------------------
# 3. Functional-data simulation
# ----------------------------------------------------------------------------
def cluster_templates(s, C):
    """Distinct base SHAPES: single Gaussian peaks of different widths, slightly
       offset centres.  Discriminative feature = width/shape; strong phase warps
       slide/distort peaks so RAW clustering is confounded but registration
       (which re-centres and un-distorts) reveals the width clusters."""
    widths = [0.06, 0.11, 0.17, 0.09, 0.14]
    centers = [0.42, 0.50, 0.58, 0.46, 0.54]
    B = np.zeros((C, len(s)))
    for c in range(C):
        f = np.exp(-((s - centers[c % 5]) / widths[c % 5]) ** 2)
        B[c] = (f - f.mean()) / (f.std() + 1e-8)
    return B


def random_monotone_warp(rng, T, strength=1.2, n_basis=5):
    """Random smooth strictly-increasing g:[0,1]->[0,1], g(0)=0, g(1)=1."""
    t = np.linspace(0.0, 1.0, T)
    v = np.ones(T)
    for k in range(1, n_basis + 1):
        b = rng.uniform(-strength, strength) / k
        v = v + b * np.sin(k * np.pi * t + rng.uniform(0, np.pi))
    v = np.clip(v, 0.04, None)
    g = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    return (g / g[-1])


def simulate_dataset(N, C, T, seed, noise=0.03, phase=1.2, amp=0.10):
    """x_i(t) = a_i * base_c(g_i(t)) + noise; g_i random monotone phase warp."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, T)
    B = cluster_templates(t, C)
    x = np.zeros((N, T)); lab = np.zeros(N, dtype=int); G = np.zeros((N, T))
    for i in range(N):
        c = i % C
        g = random_monotone_warp(rng, T, strength=phase)
        base_on_g = np.interp(g, t, B[c])
        a = 1.0 + rng.normal(0.0, amp)
        x[i] = a * base_on_g + rng.normal(0.0, noise, size=T)
        lab[i] = c; G[i] = g
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    return x.astype(np.float32), lab, t.astype(np.float32), G.astype(np.float32)


# ----------------------------------------------------------------------------
# 4. Differentiable ops
# ----------------------------------------------------------------------------
def interp1d_uniform(x, q):
    T = x.shape[1]
    pos = q.clamp(0.0, 1.0) * (T - 1)
    lo = pos.floor().long().clamp(0, T - 2)
    frac = pos - lo.to(pos.dtype)
    xlo = torch.gather(x, 1, lo); xhi = torch.gather(x, 1, lo + 1)
    return xlo + frac * (xhi - xlo)

def srvf(x):
    d = x[:, 1:] - x[:, :-1]
    q = torch.sign(d) * torch.sqrt(torch.abs(d) + 1e-8)
    return torch.cat([q, q[:, -1:]], dim=1)

def srvf_warp(q, gamma, T):
    """Proper (norm-preserving) SRVF action of a warp gamma on SRVF q:
       (q o gamma)(t) * sqrt(gamma_dot(t))  (paper Section 2).  This is an
       L2 isometry, so extreme warps cannot shrink ||.|| -> prevents the
       over-warping/flattening degeneracy in the registration loss."""
    qg = interp1d_uniform(q, gamma)
    gd = (gamma[:, 1:] - gamma[:, :-1]).clamp_min(1e-6) * (T - 1)
    gd = torch.cat([gd, gd[:, -1:]], dim=1)
    return qg * torch.sqrt(gd)

def fourier_basis(T, K):
    t = torch.linspace(0.0, 1.0, T)
    cols = []
    for k in range(1, K + 1):
        if k % 2 == 1:
            cols.append(np.sqrt(2.0) * torch.cos(np.pi * ((k + 1) // 2) * t))
        else:
            cols.append(np.sqrt(2.0) * torch.sin(np.pi * (k // 2) * t))
    return torch.stack(cols, dim=1)

def fourier_coeffs(x, Phi):
    return x @ Phi / x.shape[1]

def soft_assign(a, centroids, eps0=1.0):
    d2 = torch.cdist(a, centroids) ** 2
    num = (1.0 + d2 / eps0) ** (-(eps0 + 1.0) / 2.0)
    return num / num.sum(dim=1, keepdim=True).clamp_min(1e-12)

def target_distribution(p):
    g = p.sum(dim=0)
    w = p ** 2 / g.clamp_min(1e-12)
    return w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)


# ----------------------------------------------------------------------------
# 5. Full joint NeuralFLoC model
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    """1D-CNN encoder -> per-curve latent w_i in R^L (Eq 4)."""
    def __init__(self, T, latent=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, 3, padding=1), nn.ELU(),
            nn.Conv1d(8, 16, 3, padding=1), nn.ELU(),
            nn.AdaptiveAvgPool1d(8))
        self.fc = nn.Linear(16 * 8, latent)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.conv(x.unsqueeze(1))
        return self.relu(self.fc(h.reshape(h.shape[0], -1)))


class NeuralFLoC(nn.Module):
    """Encoder -> per-curve latent -> single Neural-ODE diffeomorphic warp;
       learnable cluster centroids for spectral soft-assignment."""
    def __init__(self, C, T, hidden=64, K=10, latent=16, Sode=40):
        super().__init__()
        self.C, self.T, self.K, self.Sode = C, T, K, Sode
        self.enc = Encoder(T, latent)
        self.vf = VelocityField(1, latent, hidden, depth=2, act="elu")
        self.centroids = nn.Parameter(torch.zeros(C, K))
        self.register_buffer("Phi", fourier_basis(T, K))
    def warp(self, x, tgrid):
        w = self.enc(x)
        gh = integrate_warp(self.vf, w, tgrid, 1, S=self.Sode, method="euler")[:, :, 0]
        return w, gh                                    # (N,L), (N,T)


def train_neuralfloc(x_np, tgrid_np, C, seed=0, epochs=200, hidden=64, K=10,
                     latent=16, alpha=0.01, lr=3e-3, use_reg=True, use_warp=True, cond_reg=True,
                     Sode=40, warmup=25, init_centroids=None,
                     verbose=False, log_every=0):
    """Train joint model.  Ablations: use_warp=False -> w/o registration;
       use_reg=False -> clustering only.  A short `warmup` aligns to per-cluster
       reference medoids to seed the SRVF Karcher-mean objective (Eq 10)."""
    torch.manual_seed(seed); np.random.seed(seed)
    x = torch.tensor(x_np, dtype=torch.float32)
    tgrid = torch.tensor(tgrid_np, dtype=torch.float32)
    N, T = x.shape
    model = NeuralFLoC(C, T, hidden=hidden, K=K, latent=latent, Sode=Sode)
    a_raw = fourier_coeffs(x, model.Phi).detach().numpy()
    from scipy.cluster.vq import kmeans2
    if init_centroids is None:
        init_centroids, lab0 = kmeans2(a_raw, C, seed=seed, minit="++", missing="raise")
    else:
        lab0 = kmeans2(a_raw, np.asarray(init_centroids), minit="matrix")[1]
    with torch.no_grad():
        model.centroids.copy_(torch.tensor(init_centroids, dtype=torch.float32))
    # per-cluster reference SRVF (sharpest curve = min total variation) for warmup
    Qraw = srvf(x)
    refs = torch.zeros(C, T)
    for j in range(C):
        idx = np.where(lab0 == j)[0]
        if len(idx) == 0:
            idx = np.arange(N)
        sharp = x_np[idx].max(axis=1)                  # sharpest = tallest peak
        refs[j] = Qraw[idx[int(np.argmax(sharp))]]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, int(epochs * 0.6)), gamma=0.3)
    q_fixed = srvf(x)                                  # fixed SRVF of raw curves
    global_ref = q_fixed[int(np.argmax(x_np.max(axis=1)))].clone()   # sharpest curve
    # warm-start soft assignments one-hot from the k-means init so each
    # class-specific warp specialises instead of diluting into an average
    p_prev = torch.zeros(N, C)
    p_prev[np.arange(N), lab0] = 1.0
    hist = {"total": [], "reg": [], "clu": []}
    for ep in range(epochs):
        if ep == warmup and use_warp and use_reg:
            # re-initialise clustering on the ALIGNED representation once curves
            # have been registered (raw-init centroids are now stale)
            with torch.no_grad():
                w0, gm0 = model.warp(x, tgrid)
                aa = fourier_coeffs(interp1d_uniform(x, gm0), model.Phi).numpy()
                cc2, ll2 = kmeans2(aa, C, seed=seed, minit="++", missing="raise")
                model.centroids.copy_(torch.tensor(cc2, dtype=torch.float32))
                p_prev = torch.zeros(N, C); p_prev[np.arange(N), ll2] = 1.0
        opt.zero_grad()
        if use_warp:
            w, gamma = model.warp(x, tgrid)                          # (N,T) single warp
            xt = interp1d_uniform(x, gamma)
        else:
            gamma = tgrid.unsqueeze(0).expand(N, T); xt = x
        a = fourier_coeffs(xt, model.Phi)
        p = soft_assign(a, model.centroids)
        if use_reg and use_warp:
            Qr = srvf_warp(q_fixed, gamma, T)                       # norm-preserving
            if ep < warmup:
                # robust global alignment: sharp reference then global Karcher mean
                if ep < warmup // 3:
                    tgt = global_ref.unsqueeze(0)
                else:
                    tgt = Qr.mean(dim=0, keepdim=True).detach()
                Lreg = ((Qr - tgt) ** 2).sum() / N
            elif cond_reg:
                w2 = p / p.sum(dim=0, keepdim=True).clamp_min(1e-8)
                mu = (w2.t() @ Qr).detach()                         # (C,T) Karcher mean
                diff = Qr.unsqueeze(1) - mu.unsqueeze(0)
                cond = (p.detach() * (diff ** 2).sum(dim=2)).sum() / N   # Eq 10
                anchor = ((Qr - global_ref.unsqueeze(0)) ** 2).sum() / N # keep centred
                Lreg = cond + 0.8 * anchor
            else:
                tgt = Qr.mean(dim=0, keepdim=True).detach()         # global-only (w/o Clu)
                Lreg = ((Qr - tgt) ** 2).sum() / N
        else:
            Lreg = torch.zeros(())
        q = target_distribution(p)
        Lclu = (q * (torch.log(q.clamp_min(1e-12)) -
                     torch.log(p.clamp_min(1e-12)))).sum() / N       # Eq 11
        loss = Lreg + alpha * Lclu
        loss.backward(); opt.step(); sched.step()
        p_prev = p.detach()
        hist["total"].append(float(loss.detach()))
        hist["reg"].append(float(Lreg.detach()) if torch.is_tensor(Lreg) else float(Lreg))
        hist["clu"].append(float(Lclu.detach()))
        if log_every and verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    ep {ep:4d}  L_total={hist['total'][-1]:.5f}  "
                  f"L_reg={hist['reg'][-1]:.5f}  L_clu={hist['clu'][-1]:.5f}")
    with torch.no_grad():
        if use_warp:
            w, gamma = model.warp(x, tgrid)
            xt = interp1d_uniform(x, gamma)
        else:
            gamma = tgrid.unsqueeze(0).expand(N, T); xt = x
        a = fourier_coeffs(xt, model.Phi)
        p = soft_assign(a, model.centroids)
        pred = p.argmax(dim=1).numpy()
        # k-means on the aligned representation (robust read-out; Alg.2 clusters
        # via soft-assignment, this is a diagnostic cross-check)
        from scipy.cluster.vq import kmeans2 as _km
        try:
            _, pred_km = _km(a.numpy(), C, seed=seed, minit="++", missing="raise")
        except Exception:
            pred_km = pred
    return {"pred": pred, "pred_km": pred_km, "gamma": gamma.detach().numpy(),
            "xt": xt.detach().numpy(), "p": p.detach().numpy(),
            "hist": hist, "model": model}


# ----------------------------------------------------------------------------
# 6. Metrics
# ----------------------------------------------------------------------------
def ari(y, yp):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(y, yp))

def nmi(y, yp):
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(y, yp))

def cluster_acc(y, yp):
    from scipy.optimize import linear_sum_assignment
    y = np.asarray(y); yp = np.asarray(yp)
    D = int(max(y.max(), yp.max())) + 1
    W = np.zeros((D, D), dtype=int)
    for i in range(y.size):
        W[yp[i], y[i]] += 1
    r, c = linear_sum_assignment(-W)
    return float(W[r, c].sum()) / y.size

def atv(xt, labels):
    labels = np.asarray(labels); classes = np.unique(labels)
    means = {c: xt[labels == c].mean(axis=0) for c in classes}
    tv = {c: float(np.mean(np.sum(np.abs(np.diff(xt[labels == c], axis=1)), axis=1)))
          for c in classes}
    pairs, vals = 0, 0.0
    for u in range(len(classes)):
        for v in range(u + 1, len(classes)):
            sep = np.linalg.norm(means[classes[u]] - means[classes[v]]) + 1e-8
            vals += 0.5 * (tv[classes[u]] + tv[classes[v]]) / sep
            pairs += 1
    return vals / max(pairs, 1)

def peak_dispersion(x, labels):
    """Registration alignment error: mean over true clusters of the std of the
       main-peak position (argmax) across curves.  Lower => better within-cluster
       phase alignment (phase variability removed).  Uniform grid, position in [0,1]."""
    labels = np.asarray(labels); T = x.shape[1]
    pos = np.argmax(x, axis=1) / (T - 1.0)
    return float(np.mean([pos[labels == c].std() for c in np.unique(labels)]))


def warp_diffeo_diag(gamma):
    d = np.diff(gamma, axis=1)
    return {"min_increment": float(d.min()),
            "monotone_frac": float((d > 0).all(axis=1).mean()),
            "max_boundary_err": float(max(abs(gamma[:, 0]).max(),
                                          abs(gamma[:, -1] - 1.0).max()))}


# ----------------------------------------------------------------------------
# 7. Baselines
# ----------------------------------------------------------------------------
def kmeans_raw(x, C, K, seed, use_fourier=True):
    from scipy.cluster.vq import kmeans2
    if use_fourier:
        Phi = fourier_basis(x.shape[1], K)
        feat = (torch.tensor(x) @ Phi / x.shape[1]).numpy()
    else:
        feat = x
    cc, lab = kmeans2(feat, C, seed=seed, minit="++", missing="raise")
    return lab, feat

def register_then_cluster(x, tgrid, C, K, seed, epochs=180, hidden=64,
                          latent=16, Sode=40, warmup=40):
    """Register ALL curves to a single global template (Karcher-mean iteration,
       seeded by a sharp reference), then k-means on aligned Fourier coeffs."""
    torch.manual_seed(seed); np.random.seed(seed)
    xt = torch.tensor(x, dtype=torch.float32)
    tg = torch.tensor(tgrid, dtype=torch.float32)
    N, T = xt.shape
    enc = Encoder(T, latent); vf = VelocityField(1, latent, hidden, depth=2, act="elu")
    opt = torch.optim.Adam(list(enc.parameters()) + list(vf.parameters()), lr=3e-3)
    q_fixed = srvf(xt)
    ref = q_fixed[int(np.argmax(x.max(axis=1)))].clone()   # sharpest reference SRVF
    for ep in range(epochs):
        opt.zero_grad()
        w = enc(xt)
        gh = integrate_warp(vf, w, tg, 1, S=Sode, method="euler")[:, :, 0]
        Qr = srvf_warp(q_fixed, gh, T)                     # norm-preserving action
        tgt = ref.unsqueeze(0) if ep < warmup else Qr.mean(dim=0, keepdim=True).detach()
        loss = ((Qr - tgt) ** 2).sum() / N
        loss.backward(); opt.step()
    with torch.no_grad():
        w = enc(xt); gh = integrate_warp(vf, w, tg, 1, S=Sode, method="euler")[:, :, 0]
        xa = interp1d_uniform(xt, gh)
    xa_np = xa.numpy()
    from scipy.cluster.vq import kmeans2
    Phi = fourier_basis(T, K)
    a = (torch.tensor(xa_np) @ Phi / T).numpy()
    cc, lab = kmeans2(a, C, seed=seed, minit="++", missing="raise")
    return lab, xa_np, gh.numpy()
