#!/usr/bin/env python3
r"""
uci_repro.py <dataset> <model>   -- resumable; writes cells/<dataset>__<model>.json

REAL-BENCHMARK reproduction of "Nonparametric Distribution Regression
Re-calibration" (ICML 2026, OpenReview fTl7NXYtAB, arXiv 2602.13362) following
the OFFICIAL experiment protocol, pinned at:

  official repo : https://github.com/adamgnuj/recalibration_experiment.git
                  revision 12b4a203d5a259cf13e621fccdd0b9a4ab073fa0
  data repo     : https://github.com/yaringal/DropoutUncertaintyExps.git
                  revision 6eb4497628d12b0f300f4b4f6bdc386bebad565c
  model lib     : torch-naut (Kelen et al. 2025), architecture/transforms per
                  proto-n/torch-naut b1cb5ee948d468eb7254edd1558874ae14109973

PROTOCOL MIRRORED FROM THE OFFICIAL REPO (file-by-file):
  data      : experiment/data/uci_datasets/prepare_uci.ipynb -- UCI datasets with
              the PREDEFINED 20 train/test split index files (10% test);
              validation = last 20% of the train rows, no shuffle.
  models    : experiment/models/{gdn,mdn,bnn}/script.py -- MLP d->50->50->3K,
              ReLU; K=1 (gdn), K=100 (mdn), Bayesian layers with K=1 and 100
              stochastic eval passes concatenated into a 100-component mixture
              (bnn); scale = softplus(clamp(.,-15)), mixture logits clamped to
              [-15,15] (torchnaut/mdn.py); batch = int(sqrt(n_train)-5);
              weight decay 1e-4 for bostonHousing/concrete/energy/yacht else
              1e-6; early stopping on validation NLL with patience 50;
              np.random.seed(1) / torch.manual_seed(1).
              drf : experiment/models/drf/script.R (R `drf` package) --
              approximated with a scikit-learn RandomForestRegressor whose
              per-tree leaf co-membership weights give the weighted-empirical
              predictive over y_train atoms (the DRF/GRF weighting scheme).
  CKME      : Calibration/src/ReCalibration/ReCalibration.jl -- calibration set
              = validation; kernel between predictive distributions
              k(q_i,q_j) = Laplace-pdf(0, median)(sqrt(EnergyDistance(q_i,q_j)))
              with median heuristic on the calibration block; observation
              kernel Laplace-pdf median-heuristic; lambda by 5-fold CV
              (cv_mask_seed=42) minimising the embedding loss of
              lambda_cross_validation.jl with per-column euclidean simplex
              projection; beta(Q) = (K_cc + lambda*n*I)^-1 k_c(Q); recalibrated
              prediction = sum_i w_i * delta_{y_valid_i} with w = simplex
              projection of beta  (paper Eq. 20-22).
  PIT       : experiment/recalib_models/PIT/script.jl (Kuleshov et al. 2018) --
              recalibrated CDF = ecdf(Z_valid) o F, Z_valid = F_val(y_val).
  SKCE test : Calibration/src/SKCETest.jl (EmpiricalSKCETest) + eval/
              run_SKCE_test.jl -- unbiased SKCE with tensor kernel
              h_ij = k_Q(q_i,q_j) * ( l(y_i,y_j) - E_{M~q_i} l(M,y_j)
                     - E_{M~q_j} l(M,y_i) + E_{M~q_i,M'~q_j} l(M,M') ),
              l = Laplace-pdf median-heuristic on test observations, k_Q =
              Laplace-pdf median-heuristic on pairwise energy distances of the
              predictive distributions.  The h_ij matrix is handed to
              CalibrationTests.jl v0.6.3 AsymptoticSKCETest (the exact call
              made by SKCETest.jl):
                estimate  = SKCE_uq = mean_{i<j} h_ij            (unbiased)
                statistic = n/(n-1)*SKCE_uq - SKCE_b,  SKCE_b = mean_ij h_ij
                p-value   = degenerate-U-statistic bootstrap (Widmann et al.
                            2021; bootstrap_ccdf, 1000 iters): resample counts
                            C ~ Mult(n, 1/n), T' = (1/n^2) * [ n/(n-1) *
                            (C'HC - sum_i C_i h_ii) - 2 * C'H1 ], p = frac of
                            T' >= statistic.
              Figure 1 of the paper reports the fraction of splits where
              auto-calibration is ACCEPTED at alpha = 5%.  The official
              results CSV (run_SKCE_test.jl) reports `stat` = statistic.
  CRPS      : eval/eval_CRPS.jl -- mean test CRPS, reported RELATIVE to the
              uncalibrated base model (Table 1; base = 1.000).

DOCUMENTED DEVIATIONS (closest well-defined choices; everything else as above):
  D1  Julia/R/TensorFlow toolchain (CUDA-only kernels, R `drf`, TF GPBeta)
      is replaced by NumPy/PyTorch CPU implementations of the same math.
  D2  (REVISED -- root-cause fix of the first revision.)  The first revision
      guessed the train loop (Adam, mean-reduced NLL, <=200 epochs); the actual
      loop IS published in the pinned torch-naut (iclr2025 branch, b1cb5ee9,
      lib/mdn.py train()/bnn_train(), called by the official
      experiment/models/{gdn,mdn,bnn}/script.py with max_patience=50,
      min_std=0.0, and kl_coef=0 for the BNN).  This revision ports it
      verbatim: AdamW(lr=1e-3, weight_decay=l2reg), LinearLR warm-up
      (start_factor=0.1, total_iters=10), loss = -log_likelihood(
      min_log_proba=-20).sum() per batch, grad-norm clip 10, batches by
      tensor_split(randperm, n//bs), max_epoch=1000, early stop when
      patience > 50, best-val-NLL state restored; BNN val NLL by 20-pass
      mixture logsumexp, BNN export by 100 stochastic predict passes in
      chunks of 32 (fresh weight draws per chunk, as in predict()).
  D3  GPBeta (Song et al. 2019; official DistCal needs TensorFlow+GPU) is
      represented by its parametric core: a global Beta distribution-
      calibration map z -> BetaCDF_{a,b}(z), (a,b) fitted by MLE on the
      validation PITs.  Named "beta" below.
  D4  Predictive distributions are handled on a fixed 512-point grid over the
      target range (exact mixture CDF evaluated at grid nodes); pairwise
      E|U-V| terms use the identity E|U-V| = int F_U+F_V-2*F_U*F_V dt on that
      grid.  The official code instead samples ns=1000 points on GPU.
  D5  (REVISED -- root-cause fix of the first revision.)  The first revision
      replaced CalibrationTests.jl's AsymptoticSKCETest with an ad-hoc exact
      conditional Monte-Carlo null (B=200 draws y_i ~ q_i).  That null is far
      more powerful against the n_val-atom empirical output of CKME (Eq. 22)
      than the paper's actual test, and it inverted the Figure-1 ranking
      (U1/U2 False; archived in cells_mcnull/).  This revision ports
      CalibrationTests.jl v0.6.3 AsymptoticSKCETest verbatim (statistic
      n/(n-1)*SKCE_uq - SKCE_b; bootstrap_ccdf with 1000 iterations).  The
      only remaining deviation is the RNG bit-stream of the bootstrap
      resampling (numpy default_rng vs Julia GLOBAL_RNG).
  D6  Julia's Random.seed!(42) shuffle for the CV masks is reproduced with
      numpy default_rng(42) (bit-streams differ; fold sizes and protocol
      identical).
  D7  Budget: 5 of the paper's 9 UCI datasets (the 5 smallest) and the FIRST
      10 of the 20 predefined splits (env UCI_SPLITS overrides), ~45 min CPU.

PREDECLARED ACCEPTANCE RULES (fixed before the full run -- after reading the
paper's own Table 1 / Figure 1 targets, before any full-suite execution;
evaluated by aggregate_uci.py from the produced JSON):
  U1 (Figure 1 / registered auto-calibration claim): the mean SKCE-test
      acceptance fraction (alpha=5%) of CKME across the 20 dataset x model
      cells is >= that of the raw model, >= PIT and >= beta.  Paper: "with the
      exception of our proposed nonparametric recalibration approach, there
      was generally sufficient evidence to reject the hypothesis of
      auto-calibration across most datasets."
  U2 (Prop 5.2 direction): CKME reduces the mean SKCE statistic vs the raw
      model on >= 70% of cells.  "SKCE statistic" is what the official
      evaluation reports as its SKCE result: the `stat` column written by
      run_SKCE_test.jl, i.e. AsymptoticSKCETest's statistic.  The unbiased
      estimate SKCE_uq is additionally recorded and reported per cell.
  U3 (Table 1 / registered CRPS claim): per-cell agreement with the paper's
      reported CKME normalised CRPS (mean +/- sd over its 20 splits):
      |ours_mean - paper_mean| <= 2*(ours_sd + paper_sd) on >= 70% of the 20
      cells.  NOTE the paper itself reports CKME CRPS > 1 on the smallest
      datasets (yacht 1.276/1.326, housing 1.150/1.175 for GDN/MDN) --
      "recalibration hurts CRPS" there -- so the target is AGREEMENT with
      Table 1, not blanket improvement.  Registered example Energy-MDN
      (paper 0.594 +/- 0.178) must satisfy the same agreement rule.
Every number below is measured from executed code; nothing is copied from the
paper into results.  CPU-only, deterministic seeds.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
DATA = HERE / "data_cache"
CELLS = HERE / "cells"
CELLS.mkdir(exist_ok=True)
PARTS = CELLS / "parts"
PARTS.mkdir(exist_ok=True)

DATASETS = ["yacht", "bostonHousing", "energy", "concrete", "wine-quality-red"]
MODELS = ["gdn", "mdn", "bnn", "drf"]
N_SPLITS = int(os.environ.get("UCI_SPLITS", "10"))
GRID_T = 512
SKCE_BOOT = 1000
L2_OVERRIDE = {"bostonHousing": 1e-4, "concrete": 1e-4, "energy": 1e-4, "yacht": 1e-4}


# ----------------------------- data (official split scheme) -----------------
def load_split(ds: str, i: int):
    d = DATA / ds
    data = np.loadtxt(d / "data.txt")
    feat = np.loadtxt(d / "index_features.txt").astype(int)
    targ = int(np.loadtxt(d / "index_target.txt"))
    X, y = data[:, feat], data[:, targ]
    itr = np.loadtxt(d / f"index_train_{i}.txt").astype(int)
    ite = np.loadtxt(d / f"index_test_{i}.txt").astype(int)
    Xtr_all, ytr_all = X[itr], y[itr]
    n80 = int(0.8 * Xtr_all.shape[0])
    return (Xtr_all[:n80], ytr_all[:n80],          # train (80% of train idx)
            Xtr_all[n80:], ytr_all[n80:],          # validation (last 20%)
            X[ite], y[ite])                        # test (predefined 10%)


# ----------------------------- torch models (gdn / mdn / bnn) ---------------
def _torch():
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    return torch


def _mix_loglik(torch, out, y, min_log_proba=None):
    """torchnaut lib/mdn.py transform_output + log_likelihood, verbatim port:
    scale = softplus(clamp(raw,-15)) + min_std (min_std=0.0 per script.py),
    mixture logits clamped to [-15,15]; per-sample log-likelihood, optionally
    clamped below at min_log_proba (train uses -20; eval uses no clamp)."""
    mu = out[:, :, 0]
    scale = torch.nn.functional.softplus(out[:, :, 1].clamp(min=-15.0))
    logit = out[:, :, 2].clamp(min=-15.0, max=15.0)
    logpi = logit - torch.logsumexp(logit, dim=1, keepdim=True)
    z = (y[:, None] - mu) / scale
    logphi = -0.5 * z * z - 0.5 * float(np.log(2 * np.pi)) - torch.log(scale)
    ll = torch.logsumexp(logpi + logphi, dim=1)
    if min_log_proba is not None:
        ll = torch.clamp(ll, min=min_log_proba)
    return ll


def _make_net(torch, d_in, K, bayes: bool):
    nn = torch.nn

    class BayesianParameter(nn.Module):
        """torchnaut lib/bnn.py BayesianParameter, verbatim port."""

        def __init__(self, shape, prior_mu, prior_sigma):
            super().__init__()
            self.shape = shape
            self.mu = nn.Parameter(torch.ones(*shape) * prior_mu
                                   + torch.randn(*shape) / np.sqrt(shape[-1]))
            self.rho = nn.Parameter(torch.ones(*shape) * float(
                np.log(np.exp(prior_sigma) - 1.0)))

        def forward(self):
            eps = torch.randn(*self.shape)
            return self.mu + torch.nn.functional.softplus(self.rho.clamp(min=-30)) * eps

    class BayesianLayer(nn.Module):
        """torchnaut lib/bnn.py BayesianLayer (prior_mu=0, prior_sigma=0.1)."""

        def __init__(self, i, o):
            super().__init__()
            self.weight = BayesianParameter((i, o), 0.0, 0.1)
            self.bias = BayesianParameter((o,), 0.0, 0.1)

        def forward(self, x):
            return x @ self.weight() + self.bias()

    lin = BayesianLayer if bayes else nn.Linear
    net = nn.Sequential(lin(d_in, 50), nn.ReLU(), lin(50, 50), nn.ReLU(),
                        lin(50, 3 * K))
    return net


def fit_predict_nn(ds, model, Xtr, ytr, Xva, yva, Xq_list):
    """Train exactly per the official pipeline: models/{gdn,mdn,bnn}/script.py
    calling torchnaut lib/mdn.py train()/bnn_train() (pinned b1cb5ee9):
    AdamW(lr=1e-3, weight_decay=l2reg), LinearLR warm-up (start_factor=0.1,
    total_iters=10), loss = -log_likelihood(min_log_proba=-20).sum() per batch,
    grad-norm clip 10, batch=int(sqrt(n_train)-5) via tensor_split, max_epoch
    1000, early stop when patience > 50, restore best-val-NLL state.
    BNN: BayesianLayer everywhere, kl_coef=0 (official comment: 'much better
    scores than with kl'), val NLL = 20-pass mixture logsumexp (bnn_eval),
    export = 100 stochastic predict passes concatenated into one mixture.
    Returns list of (mu, var, pi) dicts in RAW y units."""
    torch = _torch()
    np.random.seed(1)
    torch.manual_seed(1)   # torch seed is not set by the official script (D6)
    K = {"gdn": 1, "mdn": 100, "bnn": 1}[model]
    bayes = model == "bnn"
    xs = Xtr.std(0); xs = np.where(xs == 0.0, 1.0, xs)  # sklearn StandardScaler
    xm = Xtr.mean(0)
    ym, ys = ytr.mean(), ytr.std()
    tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    Xt, Yt = tt((Xtr - xm) / xs), tt((ytr - ym) / ys)
    Xv, Yv = tt((Xva - xm) / xs), tt((yva - ym) / ys)
    net = _make_net(torch, Xtr.shape[1], K, bayes)
    l2 = L2_OVERRIDE.get(ds, 1e-6)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=l2)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=10)
    bs = int(np.sqrt(Xtr.shape[0]) - 5)
    n = Xt.shape[0]

    def _forward_chunks(X, chunk=32):
        """predict() semantics: forward in chunks of 32 (fresh weight draws
        per chunk for the BNN, exactly like the official batched predict)."""
        outs = []
        with torch.no_grad():
            for a in range(0, X.shape[0], chunk):
                outs.append(net(X[a:a + chunk]).view(-1, K, 3))
        return torch.cat(outs, dim=0)

    def _val_nll():
        net.eval()
        if not bayes:
            out = _forward_chunks(Xv)
            return float(-_mix_loglik(torch, out, Yv).mean())
        lls = []
        for _ in range(20):                      # bnn_eval(num_evals=20)
            out = _forward_chunks(Xv)
            lls.append(_mix_loglik(torch, out, Yv))
        L = torch.stack(lls, dim=0)
        return float(-(torch.logsumexp(L, dim=0) - float(np.log(20.0))).mean())

    best, best_state, patience = np.inf, None, 0
    epochs_ran = 0
    for _ep in range(1000):                      # max_epoch=1000
        epochs_ran = _ep + 1
        net.train()
        if n <= bs:
            batches = [torch.arange(n)]
        else:
            perm = torch.randperm(n)
            batches = torch.tensor_split(perm, n // bs)   # get_batch_ixs
        for idx in batches:
            out = net(Xt[idx]).view(idx.shape[0], K, 3)
            loss = -_mix_loglik(torch, out, Yt[idx], min_log_proba=-20.0).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            opt.step()
        sched.step()
        vnll = _val_nll()
        if best > vnll:                          # strict improvement
            best, patience = vnll, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience > 50:                    # max_patience=50
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    def _predict(Xq):
        Xq_t = tt((Xq - xm) / xs)
        n_eval = 100 if bayes else 1             # num_evals=100 (bnn script)
        mus, vars_, pis = [], [], []
        for _ in range(n_eval):
            out = _forward_chunks(Xq_t)
            mu = out[:, :, 0]
            scale = torch.nn.functional.softplus(out[:, :, 1].clamp(min=-15.0))
            logit = out[:, :, 2].clamp(min=-15.0, max=15.0)
            pi = torch.softmax(logit, dim=1)
            mus.append(mu.numpy())
            vars_.append((scale.numpy()) ** 2)
            pis.append(pi.numpy())
        mu = np.concatenate(mus, axis=1) * ys + ym
        var = np.concatenate(vars_, axis=1) * ys * ys
        pi = np.concatenate(pis, axis=1)
        pi = pi / pi.sum(1, keepdims=True)       # P ./= sum (official eval code)
        return {"kind": "mixture", "mu": mu, "var": np.maximum(var, 1e-14), "pi": pi}
    return [_predict(Xq) for Xq in Xq_list], {"best_val_nll": best, "epochs": epochs_ran}


def fit_predict_drf(Xtr, ytr, Xq_list):
    """DRF stand-in: RandomForest leaf co-membership weights over y_train atoms."""
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                               random_state=1, n_jobs=1).fit(Xtr, ytr)
    leaves_tr = rf.apply(Xtr)                       # [n_tr, trees]
    n_tr, T = leaves_tr.shape
    outs = []
    for Xq in Xq_list:
        leaves_q = rf.apply(Xq)                     # [n_q, trees]
        W = np.zeros((Xq.shape[0], n_tr))
        for t in range(T):
            lt = leaves_tr[:, t]
            counts = np.bincount(lt, minlength=lt.max() + 1).astype(float)
            W += (lt[None, :] == leaves_q[:, t][:, None]) / counts[lt][None, :]
        W /= T
        W /= W.sum(1, keepdims=True)
        outs.append({"kind": "empirical", "atoms": ytr.copy(), "W": W})
    return outs, None


# ----------------------------- grid representation --------------------------
def make_grid(y_all, reps):
    lo, hi = float(np.min(y_all)), float(np.max(y_all))
    for r in reps:
        if r["kind"] == "mixture":
            sd = np.sqrt(r["var"])
            lo = min(lo, float((r["mu"] - 6 * sd).min()))
            hi = max(hi, float((r["mu"] + 6 * sd).max()))
        else:
            lo = min(lo, float(r["atoms"].min()))
            hi = max(hi, float(r["atoms"].max()))
    pad = 1e-3 * (hi - lo + 1e-12)
    return np.linspace(lo - pad, hi + pad, GRID_T)


def cdf_on_grid(rep, g):
    if rep["kind"] == "mixture":
        mu, var, pi = rep["mu"], rep["var"], rep["pi"]
        sd = np.sqrt(var)
        F = np.zeros((mu.shape[0], g.size))
        chunk = 32  # bound memory: n x K x T
        for a in range(0, mu.shape[0], chunk):
            z = (g[None, None, :] - mu[a:a + chunk, :, None]) / sd[a:a + chunk, :, None]
            F[a:a + chunk] = np.einsum("nk,nkt->nt", pi[a:a + chunk], ndtr(z))
        return np.clip(F, 0.0, 1.0)
    atoms, W = rep["atoms"], rep["W"]
    order = np.argsort(atoms)
    aa, Ws = atoms[order], W[:, order]
    cs = np.cumsum(Ws, axis=1)
    idx = np.searchsorted(aa, g, side="right")
    F = np.concatenate([np.zeros((W.shape[0], 1)), cs], axis=1)[:, idx]
    return np.clip(F, 0.0, 1.0)


def cdf_at_points(rep, y):
    """F_i(y_i) for each row i."""
    if rep["kind"] == "mixture":
        z = (y[:, None] - rep["mu"]) / np.sqrt(rep["var"])
        return np.clip(np.einsum("nk,nk->n", rep["pi"], ndtr(z)), 0.0, 1.0)
    atoms, W = rep["atoms"], rep["W"]
    return np.clip(np.sum(W * (atoms[None, :] <= y[:, None]), axis=1), 0.0, 1.0)


def masses_from_cdf(F):
    Wg = np.diff(F, axis=1, prepend=0.0)
    Wg = np.maximum(Wg, 0.0)
    s = Wg.sum(1, keepdims=True)
    return Wg / np.maximum(s, 1e-300)


def pairwise_mae_from_cdf(F, h):
    """E|U_i - V_j| matrix via  int F_i + F_j - 2 F_i F_j dt  on the grid."""
    u = F.sum(1) * h
    M = (F @ F.T) * h
    return u[:, None] + u[None, :] - 2.0 * M


def crps_from_cdf(F, y, g):
    h = g[1] - g[0]
    step = (y[:, None] <= g[None, :]).astype(float)
    return ((F - step) ** 2).sum(1) * h


# ----------------------------- kernels & helpers ----------------------------
def laplace_pdf(d, b):
    return np.exp(-np.abs(d) / b) / (2.0 * b)


def median_uppertri(D):
    iu = np.triu_indices(D.shape[0], k=1)
    return float(np.median(D[iu]))


def simplex_proj_cols(V):
    """Euclidean projection of every column onto the probability simplex
    (Duchi et al.; mirrors Calibration._batched_euclidean_simplex_proj)."""
    n = V.shape[0]
    M = -np.sort(-V, axis=0)
    cs = np.cumsum(M, axis=0)
    R = M - (cs - 1.0) / np.arange(1, n + 1)[:, None]
    rho = np.maximum((R > 0).cumsum(0).max(0), 1)  # last index with R>0 (1-based)
    theta = (cs[rho - 1, np.arange(V.shape[1])] - 1.0) / rho
    return np.maximum(V - theta[None, :], 0.0)


# ----------------------------- recalibration methods ------------------------
def recal_pit(Z_val):
    zs = np.sort(Z_val)
    return lambda F: np.searchsorted(zs, F, side="right") / zs.size


def recal_beta(Z_val):
    z = np.clip(Z_val, 1e-4, 1 - 1e-4)
    a, b, _, _ = beta_dist.fit(z, floc=0, fscale=1)
    return (lambda F: beta_dist.cdf(np.clip(F, 0.0, 1.0), a, b)), (float(a), float(b))


def ckme_recalibrate(F_val, F_test, y_val, g):
    """Official CKME (ReCalibration.jl): returns weights over y_val atoms for
    each test point, and the CV-selected lambda."""
    h = g[1] - g[0]
    n_c = F_val.shape[0]
    F_ct = np.vstack([F_val, F_test])
    mae = pairwise_mae_from_cdf(F_ct, h)
    ed = mae - 0.5 * (np.diag(mae)[:, None] + np.diag(mae)[None, :])
    dist = np.sqrt(np.maximum(ed, 0.0))
    dist = 0.5 * (dist + dist.T)
    med_q = max(median_uppertri(dist[:n_c, :n_c]), 1e-12)
    Kq = laplace_pdf(dist, med_q)

    D_obs = np.abs(y_val[:, None] - y_val[None, :])
    med_o = max(median_uppertri(D_obs), 1e-12)
    L = laplace_pdf(D_obs, med_o)

    # --- 5-fold CV for lambda (lambda_cross_validation.jl; cv_mask_seed=42) ---
    rng = np.random.default_rng(42)
    n_cv = 5
    ms = np.concatenate([np.full(int(np.ceil(n_c / n_cv)), i + 1) for i in range(n_cv)])
    rng.shuffle(ms)
    masks = [(ms != i + 1)[:n_c] for i in range(n_cv)]  # True = fold-train part
    eigs = []
    Kcc = Kq[:n_c, :n_c]
    for m in masks:
        vals, vecs = np.linalg.eigh(Kcc[np.ix_(m, m)])
        eigs.append((vals, vecs, m))

    def cv_err(lam):
        tot = 0.0
        for vals, vecs, m in eigs:
            nm = int(m.sum())
            B = vecs @ ((vecs.T @ Kcc[np.ix_(m, ~m)]) / (vals + lam * nm)[:, None])
            B = simplex_proj_cols(B)
            obs_self = np.trace(L[np.ix_(~m, ~m)])
            cross = np.trace(B.T @ L[np.ix_(m, ~m)])
            pred_self = np.trace(B.T @ L[np.ix_(m, m)] @ B)
            tot += (obs_self + pred_self - 2.0 * cross) / (~m).sum()
        return tot / n_cv

    lam = float(minimize_scalar(cv_err, bounds=(0.0, 10_000.0), method="bounded").x)

    B = np.linalg.solve(Kcc + lam * n_c * np.eye(n_c), Kq[:n_c, n_c:])
    B = simplex_proj_cols(B)
    return {"kind": "empirical", "atoms": y_val.copy(), "W": B.T}, lam


# ----------------------------- SKCE auto-calibration test -------------------
def skce_test(Wg, F, y_test, g, rng, boot=SKCE_BOOT):
    """Official SKCE auto-calibration test, exactly as executed by the paper's
    pipeline: kernel matrices of Calibration.EmpiricalSKCETest (SKCETest.jl)
    handed to CalibrationTests.jl v0.6.3 AsymptoticSKCETest.
    Wg: [n, T] grid masses; F: [n, T] CDFs (for the energy-distance kernel).
    Returns (estimate SKCE_uq, official statistic, bootstrap p-value)."""
    h = g[1] - g[0]
    n = y_test.size
    D_obs = np.abs(y_test[:, None] - y_test[None, :])
    med_o = max(median_uppertri(D_obs), 1e-12)
    Ly = laplace_pdf(D_obs, med_o)

    mae = pairwise_mae_from_cdf(F, h)
    ed = mae - 0.5 * (np.diag(mae)[:, None] + np.diag(mae)[None, :])
    ed = 0.5 * (ed + ed.T)
    med_q = max(abs(median_uppertri(ed)), 1e-12)
    Kq = laplace_pdf(ed, med_q)

    Kg = laplace_pdf(g[:, None] - g[None, :], med_o)      # kernel on grid atoms
    Lqy = Wg @ laplace_pdf(g[:, None] - y_test[None, :], med_o)   # E_{M~q_i} l(M, y_j)
    A = Wg @ Kg                                           # E_{M~q_i} l(M, g_k)
    Lqq = A @ Wg.T                                        # E l(M_i, M_j)

    H = Kq * (Ly - Lqy - Lqy.T + Lqq)                     # tensor kernel, WITH diagonal
    H = 0.5 * (H + H.T)
    hsum = float(H.sum())
    dsum = float(np.trace(H))
    est = (hsum - dsum) / (n * (n - 1))                   # SKCE_uq (unbiased)
    skce_b = hsum / (n * n)                               # SKCE_b  (biased)
    statistic = n / (n - 1) * est - skce_b                # AsymptoticSKCETest stat
    # ---- bootstrap_ccdf (CalibrationTests.jl v0.6.3), vectorised ----
    alpha = n / (n - 1)
    C = rng.multinomial(n, np.full(n, 1.0 / n), size=boot).astype(np.float64)
    rows = H.sum(1)
    diagH = np.diag(H).copy()
    M = C @ H
    quad = np.einsum("bi,bi->b", M, C)                    # C' H C
    T = (alpha * (quad - C @ diagH) - 2.0 * (C @ rows)) / (n * n)
    pval = float(np.mean(T >= statistic))
    return est, statistic, pval


# ----------------------------- per-cell driver ------------------------------
METHODS = ["none", "pit", "beta", "ckme"]


def run_split(ds: str, model: str, si: int):
    """One (dataset, model, split) unit of work -- fully deterministic and
    independent of execution order (all seeds are set inside; see D6).
    Cached in cells/parts/<ds>__<model>__<si>.json."""
    part_f = PARTS / f"{ds}__{model}__{si}.json"
    if part_f.exists():
        return json.loads(part_f.read_text())
    ts = time.perf_counter()
    Xtr, ytr, Xva, yva, Xte, yte = load_split(ds, si)
    sizes = [len(ytr), len(yva), len(yte)]
    if model == "drf":
        (rep_va, rep_te), aux = fit_predict_drf(Xtr, ytr, [Xva, Xte])
    else:
        (rep_va, rep_te), aux = fit_predict_nn(ds, model, Xtr, ytr, Xva, yva,
                                               [Xva, Xte])
    g = make_grid(np.concatenate([ytr, yva, yte]), [rep_va, rep_te])
    F_va, F_te = cdf_on_grid(rep_va, g), cdf_on_grid(rep_te, g)
    Z_val = cdf_at_points(rep_va, yva)

    pit_map = recal_pit(Z_val)
    beta_map, ab = recal_beta(Z_val)
    rep_ck, lam = ckme_recalibrate(F_va, F_te, yva, g)
    F_ck = cdf_on_grid(rep_ck, g)

    F_methods = {"none": F_te, "pit": pit_map(F_te), "beta": beta_map(F_te),
                 "ckme": F_ck}
    res = {"sizes": sizes, "lam": lam, "beta_ab": list(ab), "aux": aux,
           "methods": {}}
    crps_base = None
    for m in METHODS:
        F = np.clip(F_methods[m], 0.0, 1.0)
        F = np.maximum.accumulate(F, axis=1)  # enforce monotone CDF
        crps = float(crps_from_cdf(F, yte, g).mean())
        if m == "none":
            crps_base = crps
        Wg = masses_from_cdf(F)
        rng = np.random.default_rng(10_000 + 97 * si + 7 * METHODS.index(m))
        est, stat, p = skce_test(Wg, F, yte, g, rng)
        res["methods"][m] = {"crps": crps,
                             "crps_norm": crps / (crps_base + 1e-300),
                             "skce": est, "stat": stat, "pval": p,
                             "accept": bool(p >= 0.05)}
    res["elapsed_s"] = time.perf_counter() - ts
    tmp = part_f.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.rename(part_f)
    print(f"  part {ds}/{model} split {si}: n(tr/va/te)={sizes} "
          f"lam={lam:.4g} crps_norm[ckme]={res['methods']['ckme']['crps_norm']:.3f} "
          f"pval[none,ckme]=[{res['methods']['none']['pval']:.3f},"
          f"{res['methods']['ckme']['pval']:.3f}] "
          f"({res['elapsed_s']:.1f}s)", flush=True)
    return res


def assemble_cell(ds: str, model: str):
    """Merge the N_SPLITS parts into cells/<ds>__<model>.json."""
    parts = []
    for si in range(N_SPLITS):
        f = PARTS / f"{ds}__{model}__{si}.json"
        if not f.exists():
            return False
        parts.append(json.loads(f.read_text()))
    per_split = {m: {k: [pt["methods"][m][k] for pt in parts]
                     for k in ["crps", "crps_norm", "skce", "stat", "pval",
                               "accept"]} for m in METHODS}
    import scipy, sklearn
    out = {"dataset": ds, "model": model, "n_splits": N_SPLITS,
           "methods": METHODS, "sizes_tr_va_te": [pt["sizes"] for pt in parts],
           "per_split": per_split,
           "ckme_lambda": [pt["lam"] for pt in parts],
           "beta_ab": [pt["beta_ab"] for pt in parts],
           "val_nll": [pt["aux"] for pt in parts],
           "grid_T": GRID_T, "skce_boot": SKCE_BOOT,
           "elapsed_s": sum(pt["elapsed_s"] for pt in parts),
           "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                   "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                   "threads": os.environ["OMP_NUM_THREADS"]}}
    if model != "drf":
        import torch
        out["env"]["torch"] = torch.__version__
    (CELLS / f"{ds}__{model}.json").write_text(json.dumps(out, indent=2))
    print(f"ASSEMBLED {ds}/{model} ({out['elapsed_s']:.1f}s total)", flush=True)
    return True


def run_cell(ds: str, model: str):
    for si in range(N_SPLITS):
        run_split(ds, model, si)
    assemble_cell(ds, model)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chunk":
        # chunk <budget_s> [worker_id n_workers] -- run pending split-parts in
        # deterministic order until the time budget is exhausted, then exit 0.
        # Repeated invocations resume from the on-disk parts cache.
        budget = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
        wid = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        nw = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        tasks = [(d, m, si) for m in MODELS for d in DATASETS
                 for si in range(N_SPLITS)]
        t0 = time.perf_counter()
        for ti, (d, m, si) in enumerate(tasks):
            if ti % nw != wid:
                continue
            if (CELLS / f"{d}__{m}.json").exists():
                continue
            if (PARTS / f"{d}__{m}__{si}.json").exists():
                continue
            if time.perf_counter() - t0 > budget:
                print("BUDGET reached", flush=True)
                break
            run_split(d, m, si)
        else:
            print("ALL PARTS DONE (this worker)", flush=True)
        for d in DATASETS:
            for m in MODELS:
                if not (CELLS / f"{d}__{m}.json").exists():
                    try:
                        assemble_cell(d, m)
                    except Exception:
                        pass
        missing = [(d, m) for d in DATASETS for m in MODELS
                   if not (CELLS / f"{d}__{m}.json").exists()]
        print("MISSING CELLS:", len(missing), missing[:5], flush=True)
        sys.exit(0)
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    key2 = sys.argv[2] if len(sys.argv) > 2 else ""
    todo = [(d, m) for d in DATASETS for m in MODELS
            if key in ("", d, m) and key2 in ("", d, m)]
    for d, m in todo:
        f = CELLS / f"{d}__{m}.json"
        if f.exists():
            print(f"skip {d}/{m}", flush=True)
            continue
        print(f"RUN {d}/{m}", flush=True)
        run_cell(d, m)
    print("PRESENT:", sorted(p.name for p in CELLS.glob("*.json")), flush=True)
