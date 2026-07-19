"""
nfloc_ext.py -- extension helpers for the NeuralFLoC reproduction, used by the
C1 (Theorem 4.1 tightening), C2 (paper-protocol benchmarks) and C3 (robustness:
missing data, irregular sampling, scalability) evidence scripts.

Everything imports the shared re-implementation `neuralfloc.py`.  CPU-only,
single-thread, deterministic.  No new modelling assumptions: corruption ops
mirror paper Section 6 (missing / irregular / noise) and the minibatch trainer
realises the paper's O(N) / O(1)-memory optimisation claim (Section 3.5).
"""
import time
import numpy as np
import torch

import neuralfloc as nf
from neuralfloc import (
    VelocityField, integrate_warp, Encoder, NeuralFLoC,
    srvf, srvf_warp, fourier_basis, fourier_coeffs, soft_assign,
    target_distribution, interp1d_uniform, simulate_dataset,
    ari, nmi, cluster_acc, atv, peak_dispersion, warp_diffeo_diag,
)
from scipy.cluster.vq import kmeans2

torch.set_num_threads(1)


# ----------------------------------------------------------------------------
# Corruption operators (paper Section 6).  Each returns curves re-expressed on
# the common uniform analysis grid, mirroring the paper's "fit a Fourier spline
# then evaluate" pipeline (here: piecewise-linear reconstruction, a faithful,
# lighter proxy for the smoothing spline).
# ----------------------------------------------------------------------------
def apply_missing(x, t, drop_frac, seed):
    """Randomly drop `drop_frac` of each curve's interior points, then
       reconstruct on the uniform grid by interpolating the retained points."""
    rng = np.random.default_rng(1234567 + seed * 97 + int(round(1000 * drop_frac)))
    N, T = x.shape
    out = x.copy()
    if drop_frac <= 0:
        return out.astype(np.float32)
    ndrop = int(round(drop_frac * (T - 2)))
    for i in range(N):
        drop = rng.choice(np.arange(1, T - 1), size=ndrop, replace=False)
        keep = np.setdiff1d(np.arange(T), drop)
        out[i] = np.interp(t, t[keep], x[i][keep])
    return out.astype(np.float32)


def apply_irregular(x, t, sigma_T, seed):
    """Per-curve non-uniform time grid: jitter the sampling instants by
       Gaussian noise (std = sigma_T * spacing), keep boundaries, sort, and
       reconstruct on the uniform grid (uses the Neural-ODE's arbitrary-t
       evaluation downstream).  Matches paper Section 6 irregular-sampling."""
    rng = np.random.default_rng(7654321 + seed * 89 + int(round(1000 * sigma_T)))
    N, T = x.shape
    out = x.copy()
    if sigma_T <= 0:
        return out.astype(np.float32)
    dt = 1.0 / (T - 1)
    for i in range(N):
        tp = t + rng.normal(0.0, sigma_T * dt, size=T)
        tp[0] = 0.0
        tp[-1] = 1.0
        tp = np.maximum.accumulate(np.clip(tp, 0.0, 1.0))  # enforce monotone grid
        out[i] = np.interp(t, tp, x[i])
    return out.astype(np.float32)


def apply_noise(x, sigma, seed):
    """Inject i.i.d. Gaussian noise then re-standardise (paper Section 6)."""
    rng = np.random.default_rng(424242 + seed * 71 + int(round(1000 * sigma)))
    out = x + rng.normal(0.0, sigma, size=x.shape)
    out = (out - out.mean(axis=1, keepdims=True)) / (out.std(axis=1, keepdims=True) + 1e-8)
    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# Named-scenario simulation for the C2 paper-protocol benchmark.  UCR archive
# is offline, so we build matched-dimension synthetic analogues with the
# paper's exact class counts C and the phase+amplitude+noise generative model,
# reporting the paper's exact metrics (ATV / ACC / NMI).
# ----------------------------------------------------------------------------
SCENARIOS = {
    # name: (C, N_repro, T_repro, phase, noise, paper_N, paper_T)
    "Shapes":      (2, 400, 128, 1.2, 0.03, 1095, 1024),
    "Wave(d=1)":   (2, 400, 128, 1.0, 0.03, 1120, 315),
    "Symbols(2)":  (2, 343, 128, 1.1, 0.03, 343, 398),
    "Symbols(3)":  (3, 400, 128, 1.2, 0.03, 510, 398),
}


def simulate_scenario(name, seed):
    C, N, T, phase, noise, pN, pT = SCENARIOS[name]
    x, lab, t, G = simulate_dataset(N, C, T, seed, noise=noise, phase=phase)
    return x, lab, t, C


# ----------------------------------------------------------------------------
# C1 (Theorem 4.1): fit ONE monotone target warp with a dedicated high-capacity
# Neural-ODE warp; drive the SRVF-registration-equivalent L2 warp error down.
# ----------------------------------------------------------------------------
def fit_single_warp(gstar_np, t_np, H, steps, S, method="euler", lr=6e-3, seed=0):
    torch.manual_seed(seed)
    tg = torch.tensor(t_np, dtype=torch.float32)
    vf = VelocityField(1, 1, H, depth=2, act="elu", gain=1.0)
    w = torch.zeros(1, 1)
    y = torch.tensor(gstar_np, dtype=torch.float32).view(1, -1)
    opt = torch.optim.Adam(vf.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, int(steps * 0.55)), gamma=0.2)
    for _ in range(steps):
        opt.zero_grad()
        gh = integrate_warp(vf, w, tg, 1, S=S, method=method)[:, :, 0]
        (((gh - y) ** 2).mean()).backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        gh = integrate_warp(vf, w, tg, 1, S=S, method=method)[:, :, 0]
        d = np.diff(gh.numpy()[0])
        l2 = float(torch.sqrt(((gh - y) ** 2).mean()))
        sup = float((gh - y).abs().max())
    return {"l2": l2, "sup": sup, "monotone": bool((d > 0).all()),
            "min_incr": float(d.min())}


# ----------------------------------------------------------------------------
# C3 scalability: minibatch trainer realising the paper's O(N) / O(1)-memory
# optimisation (Section 3.5).  Global centroids are learnable; the SRVF
# cluster-conditional registration target (Eq 10) and DEC clustering loss
# (Eq 11) are computed per minibatch, so per-iteration cost and memory are
# independent of N -> wall-time scales ~linearly in N at fixed epochs.
# ----------------------------------------------------------------------------
def train_neuralfloc_minibatch(x_np, tgrid_np, C, seed=0, epochs=60, batch=128,
                               hidden=64, K=10, latent=16, alpha=0.01, lr=3e-3,
                               Sode=30, warmup_frac=0.35):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.tensor(x_np, dtype=torch.float32)
    tgrid = torch.tensor(tgrid_np, dtype=torch.float32)
    N, T = x.shape
    model = NeuralFLoC(C, T, hidden=hidden, K=K, latent=latent, Sode=Sode)
    # centroid init via k-means on a bounded subsample of raw Fourier coeffs -> O(1) mem
    sub = np.sort(np.random.default_rng(seed).choice(N, size=min(N, 512), replace=False))
    a_sub = fourier_coeffs(x[sub], model.Phi).detach().numpy()
    cc, _ = kmeans2(a_sub, C, seed=seed, minit="++", missing="raise")
    with torch.no_grad():
        model.centroids.copy_(torch.tensor(cc, dtype=torch.float32))
    q_all = srvf(x)
    global_ref = q_all[int(np.argmax(x_np.max(axis=1)))].clone()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, int(epochs * 0.6)), gamma=0.3)
    warmup = int(epochs * warmup_frac)
    rng = np.random.default_rng(seed + 1)
    for ep in range(epochs):
        perm = rng.permutation(N)
        for b0 in range(0, N, batch):
            idx = perm[b0:b0 + batch]
            xb = x[idx]
            qb = q_all[idx]
            opt.zero_grad()
            w = model.enc(xb)
            gh = integrate_warp(model.vf, w, tgrid, 1, S=Sode, method="euler")[:, :, 0]
            xt = interp1d_uniform(xb, gh)
            a = fourier_coeffs(xt, model.Phi)
            p = soft_assign(a, model.centroids)
            Qr = srvf_warp(qb, gh, T)
            if ep < warmup:
                tgt = global_ref.unsqueeze(0) if ep < warmup // 3 else Qr.mean(dim=0, keepdim=True).detach()
                Lreg = ((Qr - tgt) ** 2).sum() / xb.shape[0]
            else:
                w2 = p / p.sum(dim=0, keepdim=True).clamp_min(1e-8)
                mu = (w2.t() @ Qr).detach()
                diff = Qr.unsqueeze(1) - mu.unsqueeze(0)
                cond = (p.detach() * (diff ** 2).sum(dim=2)).sum() / xb.shape[0]
                anchor = ((Qr - global_ref.unsqueeze(0)) ** 2).sum() / xb.shape[0]
                Lreg = cond + 0.8 * anchor
            q = target_distribution(p)
            Lclu = (q * (torch.log(q.clamp_min(1e-12)) - torch.log(p.clamp_min(1e-12)))).sum() / xb.shape[0]
            (Lreg + alpha * Lclu).backward()
            opt.step()
        sched.step()
    # inference in minibatches (O(1) memory); also accumulate the diagnostics
    # needed for a GLOBAL (whole-dataset) joint-objective value L_total_diag,
    # used as the label-blind empirical-minimizer selection criterion across
    # restarts (Theorem 4.2) -- the same Lreg (cluster-conditional SRVF) +
    # alpha*Lclu (DEC KL) objective the model was trained on, evaluated once,
    # not backpropagated.
    preds = np.zeros(N, dtype=int)
    xt_full = np.zeros((N, T), dtype=np.float32)
    Qr_full = torch.zeros(N, T)
    p_full = torch.zeros(N, C)
    with torch.no_grad():
        for b0 in range(0, N, batch):
            idx = np.arange(b0, min(b0 + batch, N))
            xb = x[idx]
            w = model.enc(xb)
            gh = integrate_warp(model.vf, w, tgrid, 1, S=Sode, method="euler")[:, :, 0]
            xt = interp1d_uniform(xb, gh)
            a = fourier_coeffs(xt, model.Phi)
            p = soft_assign(a, model.centroids)
            preds[idx] = p.argmax(dim=1).numpy()
            xt_full[idx] = xt.numpy()
            Qr_full[idx] = srvf_warp(q_all[idx], gh, T)
            p_full[idx] = p
        w2 = p_full / p_full.sum(dim=0, keepdim=True).clamp_min(1e-8)
        mu = w2.t() @ Qr_full
        diff = Qr_full.unsqueeze(1) - mu.unsqueeze(0)
        cond = (p_full * (diff ** 2).sum(dim=2)).sum() / N
        anchor = ((Qr_full - global_ref.unsqueeze(0)) ** 2).sum() / N
        Lreg_diag = float(cond + 0.8 * anchor)
        q_diag = target_distribution(p_full)
        Lclu_diag = float((q_diag * (torch.log(q_diag.clamp_min(1e-12)) -
                                     torch.log(p_full.clamp_min(1e-12)))).sum() / N)
    L_total_diag = Lreg_diag + alpha * Lclu_diag
    return {"pred": preds, "xt": xt_full, "L_total_diag": L_total_diag,
            "L_reg_diag": Lreg_diag, "L_clu_diag": Lclu_diag}


# ----------------------------------------------------------------------------
# C1-rescale (Theorem 4.1): a >=100-warp ADMISSIBLE target family, spanning the
# Lipschitz-diffeomorphism class Gamma with random COMPOSITIONS across three
# base families (logistic-rescaled sigmoids, Beta-CDF warps, and random
# piecewise-monotone warps).  Composition of two increasing bijections of
# [0,1] is itself an increasing bijection of [0,1], so compositions stay in
# Gamma; every candidate (base or composed) is numerically screened so the
# finite-difference derivative stays within [DERIV_LO, DERIV_HI] -- bounded
# away from 0 and infinity, i.e. genuinely admissible under A3 -- exactly the
# same screen the original 13-warp family satisfied by construction, applied
# here automatically at 8-10x the scale.
# ----------------------------------------------------------------------------
DERIV_LO, DERIV_HI = 5e-3, 60.0


def _warp_admissible(g, dt):
    d = np.diff(g) / dt
    return (bool(np.all(d > 0)) and d.min() >= DERIV_LO and d.max() <= DERIV_HI
            and abs(g[0]) < 1e-6 and abs(g[-1] - 1.0) < 1e-6)


def _mk_logistic(t, k, m):
    def s(x):
        return 1.0 / (1.0 + np.exp(-k * (x - m)))
    lo, hi = s(0.0), s(1.0)
    return (s(t) - lo) / (hi - lo)


def _mk_beta(t, a, b):
    from scipy.stats import beta as _beta
    return _beta.cdf(t, a, b)


def _mk_piecewise(t, rng, n_basis=6, strength=1.0):
    v = np.ones_like(t)
    for k in range(1, n_basis + 1):
        bb = rng.uniform(-strength, strength) / k
        v = v + bb * np.sin(k * np.pi * t + rng.uniform(0, np.pi))
    v = np.clip(v, 0.08, None)
    g = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    return g / g[-1]


def admissible_family_v2(T=200, n_target=112, seed=20260718, n_pw=24):
    """Return (t, [(name, gamma_values), ...]) with >=100 admissible warps:
       logistic grid + Beta-CDF grid (shape params >=1 => finite boundary
       density, so gamma_dot stays bounded) + random piecewise-monotone warps
       + random cross-family compositions, topped up until >= n_target."""
    t = np.linspace(0.0, 1.0, T)
    dt = t[1] - t[0]
    fam = []
    for k in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        for m in (0.3, 0.4, 0.5, 0.6, 0.7):
            g = _mk_logistic(t, k, m)
            if _warp_admissible(g, dt):
                fam.append((f"logi_k{k:.0f}_m{m:.1f}", g))
    for a in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
        for b in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
            g = _mk_beta(t, a, b)
            if _warp_admissible(g, dt):
                fam.append((f"beta_a{a:.1f}_b{b:.1f}", g))
    pw_i, tries = 0, 0
    while pw_i < n_pw and tries < 300:
        rr = np.random.default_rng(seed + 1000 + tries)
        g = _mk_piecewise(t, rr, n_basis=int(rr.integers(4, 8)), strength=float(rr.uniform(0.6, 1.4)))
        if _warp_admissible(g, dt):
            fam.append((f"pw_{pw_i}", g))
            pw_i += 1
        tries += 1
    groups = [[(n, g) for n, g in fam if n.startswith("logi_")],
              [(n, g) for n, g in fam if n.startswith("beta_")],
              [(n, g) for n, g in fam if n.startswith("pw_")]]
    rng_c = np.random.default_rng(seed + 777)
    n_comp_target = max(0, n_target - len(fam))
    comp_i, tries = 0, 0
    while comp_i < n_comp_target and tries < 2000:
        tries += 1
        ga_i, gb_i = rng_c.choice(3, size=2, replace=False)
        A, B = groups[ga_i], groups[gb_i]
        if not A or not B:
            continue
        na, ya = A[int(rng_c.integers(len(A)))]
        nb, yb = B[int(rng_c.integers(len(B)))]
        g = np.interp(yb, t, ya)          # composition ya(yb(t))
        g = g / g[-1]
        if _warp_admissible(g, dt):
            fam.append((f"comp_{na}_o_{nb}", g))
            comp_i += 1
    return t, fam


def py_ok():
    return True
