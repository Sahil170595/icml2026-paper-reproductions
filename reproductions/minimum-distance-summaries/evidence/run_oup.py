"""
run_oup.py -- Ornstein-Uhlenbeck process (OUP) reproduction.

Second, harder mechanism (time-series, out-of-prior contamination).  Trains one
genuine frozen NPE + one RFF decoder mean-embedding on trajectories, then runs
the paired observed vs MDS-adapted summary sweep.  Evidence for Claims 1, 2, 3
(OUP mechanism).

Outputs: artifacts/oup_results.json, artifacts/oup_npe.pt
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mds_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
os.makedirs(ART, exist_ok=True)

SEED = 20260717
M_TRAIN = 10000
K_RFF = 512
EPS_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]
TRIALS = 50
NPE_HIDDEN = (128, 128)
O = C.OUPModel


def simulate_batch(rng, theta, n_traj):
    """Vectorised OUP simulation over datasets.
    theta: (M,2) -> X: (M, n_traj, T)."""
    Mn = theta.shape[0]
    t1 = theta[:, 0:1]                       # (M,1)
    level = np.exp(theta[:, 1:2])            # (M,1)
    sigma = np.sqrt(O.sigma2)
    sqdt = np.sqrt(O.dt)
    X = np.full((Mn, n_traj), O.X0, dtype=np.float64)
    out = np.empty((Mn, n_traj, O.T), dtype=np.float64)
    for t in range(O.T):
        dW = rng.normal(0.0, 1.0, size=(Mn, n_traj))
        X = X + t1 * (level - X) * O.dt + sigma * sqdt * dW
        out[:, :, t] = X
    return out


def summary_batch(X):
    """X: (M, N, T) -> (M, 3): mean, var, lag-1 autocorr."""
    Mn = X.shape[0]
    s1 = X.mean(axis=(1, 2))
    s2 = ((X - s1[:, None, None]) ** 2).mean(axis=(1, 2))
    a = X[:, :, :-1].reshape(Mn, -1)
    b = X[:, :, 1:].reshape(Mn, -1)
    am = a.mean(1, keepdims=True); bm = b.mean(1, keepdims=True)
    num = ((a - am) * (b - bm)).sum(1)
    den = np.sqrt(((a - am) ** 2).sum(1) * ((b - bm) ** 2).sum(1))
    s3 = np.where(den > 0, num / den, 0.0)
    return np.stack([s1, s2, s3], axis=1)


def embed_datasets(rbf, X, chunk_ds=2000):
    """X: (M, N, T) -> per-dataset mean RFF embedding (M, K) float32."""
    M, N, T = X.shape
    out = np.empty((M, K_RFF), dtype=np.float32)
    for a in range(0, M, chunk_ds):
        b = min(a + chunk_ds, M)
        pts = X[a:b].reshape(-1, T)
        z = rbf.transform(pts).astype(np.float32)
        out[a:b] = z.reshape(b - a, N, K_RFF).mean(axis=1)
    return out


def main():
    t_start = time.perf_counter()
    C.set_determinism(SEED)
    rng = np.random.default_rng(SEED)

    print("== OUP: generate training data (vectorised) ==")
    theta_tr = O.sample_theta(rng, M_TRAIN)
    X_tr = simulate_batch(rng, theta_tr, O.N)             # (M, N, T)
    S_tr = summary_batch(X_tr)                            # (M, 3)

    print("== train frozen NPE q_psi(theta|s) ==")
    npe = C.CondGaussNPE(d_s=3, d_theta=O.d_theta, hidden=NPE_HIDDEN)
    npe_params = C.n_params(npe)
    C.train_npe(npe, S_tr, theta_tr, epochs=250, batch=256, lr=1e-3, seed=SEED, verbose=True)
    npe.eval()
    torch.save(npe.state_dict(), os.path.join(ART, "oup_npe.pt"))
    npe_hash_0 = C.sha256_state_dict(npe)
    print(f"   NPE params={npe_params}  frozen-hash={npe_hash_0[:16]}...")

    print("== fit RFF (median heuristic, K=512) on trajectories + decoder ==")
    pool = X_tr[:500].reshape(-1, O.T)
    gamma, sigma_med = C.median_heuristic_gamma(pool, rng)
    rbf = C.make_rff(gamma, K_RFF, SEED, d_in=O.T)
    t0 = time.perf_counter()
    ZBAR_tr = embed_datasets(rbf, X_tr)
    dec = C.MeanEmbedNet(d_s=3, K=K_RFF, hidden=256)
    dec_mse = C.train_decoder(dec, S_tr, ZBAR_tr, epochs=200, batch=256, lr=1e-3, seed=SEED, verbose=True)
    dec.eval()
    fit_time = time.perf_counter() - t0
    print(f"   gamma={gamma:.6f} sigma_med={sigma_med:.4f}  decoder MSE={dec_mse:.3e}  fit={fit_time:.2f}s")

    _ = C.mds_adapt(dec, ZBAR_tr[0], S_tr[0])             # warmup

    print("== paired eps sweep (frozen NPE, observed vs MDS summary) ==")
    per_level = []
    hash_checks = 0
    hash_all_ok = True
    for eps in EPS_LEVELS:
        rmse_obs, rmse_mds, wins, times, sgap = [], [], 0, [], []
        for tt in range(TRIALS):
            tr_seed = int(rng.integers(0, 2**31 - 1))
            trng = np.random.default_rng(tr_seed)
            theta_star = O.sample_theta(trng, 1)[0]
            X_clean = O._simulate_traj(trng, theta_star, O.N, O.sigma2)   # (N,T)
            s_ref = O.summary(X_clean)
            X_cont = O.contaminate_dataset(trng, X_clean, eps)
            s_obs = O.summary(X_cont)
            zbar_obs = C.embed_points(rbf, X_cont)

            h_before = C.sha256_state_dict(npe)
            s_star, ms = C.mds_adapt(dec, zbar_obs, s_obs)
            h_after = C.sha256_state_dict(npe)
            hash_checks += 1
            if not (h_before == h_after == npe_hash_0):
                hash_all_ok = False

            pm_obs = npe.posterior_mean(s_obs)
            pm_mds = npe.posterior_mean(s_star)
            r_obs = float(np.linalg.norm(pm_obs - theta_star))
            r_mds = float(np.linalg.norm(pm_mds - theta_star))
            rmse_obs.append(r_obs); rmse_mds.append(r_mds); times.append(ms)
            sgap.append(float(np.linalg.norm(s_star - s_ref)))
            if r_mds < r_obs:
                wins += 1
        ro = float(np.sqrt(np.mean(np.square(rmse_obs))))
        rm = float(np.sqrt(np.mean(np.square(rmse_mds))))
        red = 100.0 * (1 - rm / ro) if ro > 0 else 0.0
        med_ms = float(np.median(times)); p95_ms = float(np.percentile(times, 95))
        per_level.append(dict(eps=eps, trials=TRIALS, rmse_obs=ro, rmse_mds=rm,
                              reduction_pct=red, wins=wins, median_ms=med_ms,
                              p95_ms=p95_ms, mean_summary_gap_to_oracle=float(np.mean(sgap))))
        print(f"   eps={eps:.1f}  RMSE {ro:7.4f} -> {rm:7.4f}  red {red:6.2f}%  "
              f"wins {wins}/{TRIALS}  med {med_ms:.3f}ms p95 {p95_ms:.3f}ms")

    nonsevere = [d["reduction_pct"] for d in per_level if 0.1 <= d["eps"] <= 0.4]
    mean_red = float(np.mean(nonsevere))

    print("== RFF-vs-exact MMD approximation gap (trajectory space) ==")
    gaps, ex_list, rf_list = [], [], []
    for _ in range(40):
        trng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        th = O.sample_theta(trng, 1)[0]
        A = O._simulate_traj(trng, th, O.N, O.sigma2)
        B = O.contaminate_dataset(trng, A.copy(), 0.2)
        ex = C.mmd2_exact(A, B, gamma)
        rf = C.mmd2_rff(C.embed_points(rbf, A), C.embed_points(rbf, B))
        gaps.append(abs(ex - rf)); ex_list.append(ex); rf_list.append(rf)
    rff_gap = dict(mean_abs_gap=float(np.mean(gaps)),
                   max_abs_gap=float(np.max(gaps)),
                   corr=float(np.corrcoef(ex_list, rf_list)[0, 1]),
                   mean_exact=float(np.mean(ex_list)), mean_rff=float(np.mean(rf_list)))
    print(f"   mean|exactMMD^2 - rffMMD^2| = {rff_gap['mean_abs_gap']:.5f}  corr={rff_gap['corr']:.4f}")

    total_time = time.perf_counter() - t_start
    result = dict(
        mechanism="oup",
        seed=SEED, N=O.N, T=O.T, dt=O.dt, sigma2=O.sigma2, X0=O.X0,
        theta_c=O.theta_c.tolist(), sigma2_c=O.sigma2_c,
        d_theta=O.d_theta, d_s=3, m_train=M_TRAIN, k_rff=K_RFF,
        eps_levels=EPS_LEVELS, trials_per_level=TRIALS,
        npe_params=npe_params, npe_hidden=list(NPE_HIDDEN),
        npe_frozen_sha256=npe_hash_0,
        rff_gamma=gamma, sigma_med=sigma_med,
        decoder_final_mse=dec_mse, decoder_fit_time_s=fit_time,
        hash_identity_checks=hash_checks, hash_identity_all_ok=hash_all_ok,
        per_level=per_level, mean_reduction_nonsevere_pct=mean_red,
        rff_vs_exact=rff_gap,
        total_runtime_s=total_time,
        env=C.environment_report(),
    )
    out = os.path.join(ART, "oup_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nMean non-severe reduction = {mean_red:.2f}%  | hash-identity {hash_checks}/{hash_checks} OK={hash_all_ok}")
    print(f"Wrote {out}  ({total_time:.1f}s total)")


if __name__ == "__main__":
    main()
