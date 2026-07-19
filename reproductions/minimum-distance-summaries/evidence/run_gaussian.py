"""
run_gaussian.py -- Gaussian conjugate location model reproduction.

Trains one genuine frozen NPE + one RFF decoder mean-embedding, then runs the
paired observed-summary vs MDS-adapted-summary sweep across Huber contamination
levels.  Produces evidence for Claims 1, 2, 3 (Gaussian mechanism).

Outputs: artifacts/gaussian_results.json, artifacts/gaussian_npe.pt
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
M_TRAIN = 20000
K_RFF = 512
DELTA = 8.0                       # outlier magnitude (Figure 1 setting)
EPS_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TRIALS = 50
NPE_HIDDEN = (64, 64)


def embed_datasets(rbf, X, chunk_ds=2000):
    """X: (M, N, d) -> per-dataset mean RFF embedding (M, K) float32."""
    M, N, d = X.shape
    out = np.empty((M, K_RFF), dtype=np.float32)
    for a in range(0, M, chunk_ds):
        b = min(a + chunk_ds, M)
        pts = X[a:b].reshape(-1, d)
        z = rbf.transform(pts).astype(np.float32)         # ((b-a)*N, K)
        out[a:b] = z.reshape(b - a, N, K_RFF).mean(axis=1)
    return out


def main():
    t_start = time.perf_counter()
    C.set_determinism(SEED)
    rng = np.random.default_rng(SEED)
    G = C.GaussianModel

    print("== Gaussian: generate training data ==")
    theta_tr = G.sample_theta(rng, M_TRAIN)
    X_tr = G.simulate(rng, theta_tr)                      # (M, N, d)
    S_tr = G.summary(X_tr)                                # (M, d)

    print("== train frozen NPE q_psi(theta|s) ==")
    npe = C.CondGaussNPE(d_s=G.d, d_theta=G.d, hidden=NPE_HIDDEN)
    npe_params = C.n_params(npe)
    C.train_npe(npe, S_tr, theta_tr, epochs=200, batch=256, lr=1e-3, seed=SEED, verbose=True)
    npe.eval()
    torch.save(npe.state_dict(), os.path.join(ART, "gaussian_npe.pt"))
    npe_hash_0 = C.sha256_state_dict(npe)
    print(f"   NPE params={npe_params}  frozen-hash={npe_hash_0[:16]}...")

    # sanity: learned posterior mean vs analytic conjugate mean (clean)
    s_probe = G.summary(G.simulate(rng, G.sample_theta(rng, 200)))
    learned = np.stack([npe.posterior_mean(s) for s in s_probe])
    analytic = G.analytic_posterior_mean(s_probe)
    npe_vs_analytic = float(np.sqrt(((learned - analytic) ** 2).sum(1).mean()))
    print(f"   NPE-vs-analytic posterior-mean RMSE = {npe_vs_analytic:.5f}")

    print("== fit RFF (median heuristic, K=512) + train decoder mean-embedding ==")
    pool = X_tr[:1000].reshape(-1, G.d)
    gamma, sigma_med = C.median_heuristic_gamma(pool, rng)
    rbf = C.make_rff(gamma, K_RFF, SEED, d_in=G.d)
    t0 = time.perf_counter()
    ZBAR_tr = embed_datasets(rbf, X_tr)
    dec = C.MeanEmbedNet(d_s=G.d, K=K_RFF, hidden=256)
    dec_mse = C.train_decoder(dec, S_tr, ZBAR_tr, epochs=300, batch=256, lr=1e-3, seed=SEED, verbose=True)
    dec.eval()
    fit_time = time.perf_counter() - t0
    print(f"   gamma={gamma:.5f} sigma_med={sigma_med:.4f}  decoder final MSE={dec_mse:.3e}  fit={fit_time:.2f}s")

    # ---- warmup adaptation (exclude lazy-init from timing) ----
    _ = C.mds_adapt(dec, ZBAR_tr[0], S_tr[0])

    print("== paired eps sweep (frozen NPE, observed vs MDS summary) ==")
    per_level = []
    hash_checks = 0
    hash_all_ok = True
    for eps in EPS_LEVELS:
        rmse_obs, rmse_mds, wins, times = [], [], 0, []
        for tt in range(TRIALS):
            tr_seed = int(rng.integers(0, 2**31 - 1))
            trng = np.random.default_rng(tr_seed)
            theta_star = G.sample_theta(trng, 1)
            x_clean = G.simulate(trng, theta_star)                    # (1,N,d)
            x_cont = G.contaminate(trng, x_clean, eps, DELTA)
            s_obs = G.summary(x_cont)[0]                              # observed summary
            zbar_obs = C.embed_points(rbf, x_cont[0])                 # marginal embedding

            h_before = C.sha256_state_dict(npe)
            s_star, ms = C.mds_adapt(dec, zbar_obs, s_obs)
            h_after = C.sha256_state_dict(npe)
            hash_checks += 1
            if not (h_before == h_after == npe_hash_0):
                hash_all_ok = False

            pm_obs = npe.posterior_mean(s_obs)
            pm_mds = npe.posterior_mean(s_star)
            r_obs = float(np.linalg.norm(pm_obs - theta_star[0]))
            r_mds = float(np.linalg.norm(pm_mds - theta_star[0]))
            rmse_obs.append(r_obs); rmse_mds.append(r_mds); times.append(ms)
            if r_mds < r_obs:
                wins += 1
        ro = float(np.sqrt(np.mean(np.square(rmse_obs))))
        rm = float(np.sqrt(np.mean(np.square(rmse_mds))))
        red = 100.0 * (1 - rm / ro) if ro > 0 else 0.0
        med_ms = float(np.median(times)); p95_ms = float(np.percentile(times, 95))
        per_level.append(dict(eps=eps, trials=TRIALS, rmse_obs=ro, rmse_mds=rm,
                              reduction_pct=red, wins=wins, median_ms=med_ms, p95_ms=p95_ms))
        print(f"   eps={eps:.1f}  RMSE {ro:7.4f} -> {rm:7.4f}  red {red:6.2f}%  "
              f"wins {wins}/{TRIALS}  med {med_ms:.3f}ms p95 {p95_ms:.3f}ms")

    # mean reduction over the prespecified non-severe levels (0.1..0.4)
    nonsevere = [d["reduction_pct"] for d in per_level if 0.1 <= d["eps"] <= 0.4]
    mean_red = float(np.mean(nonsevere))

    print("== RFF-vs-exact MMD approximation gap ==")
    gaps, ex_list, rf_list = [], [], []
    for _ in range(60):
        trng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        th = G.sample_theta(trng, 1)
        A = G.simulate(trng, th)[0]                          # clean marginal points
        B = G.contaminate(trng, G.simulate(trng, th), 0.2, DELTA)[0]
        ex = C.mmd2_exact(A, B, gamma)
        rf = C.mmd2_rff(C.embed_points(rbf, A), C.embed_points(rbf, B))
        gaps.append(abs(ex - rf)); ex_list.append(ex); rf_list.append(rf)
    rff_gap = dict(mean_abs_gap=float(np.mean(gaps)),
                   max_abs_gap=float(np.max(gaps)),
                   corr=float(np.corrcoef(ex_list, rf_list)[0, 1]),
                   mean_exact=float(np.mean(ex_list)), mean_rff=float(np.mean(rf_list)))
    print(f"   mean|exactMMD^2 - rffMMD^2| = {rff_gap['mean_abs_gap']:.5f}  "
          f"corr={rff_gap['corr']:.4f}")

    total_time = time.perf_counter() - t_start
    result = dict(
        mechanism="gaussian",
        seed=SEED, delta=DELTA, N=G.N, d_theta=G.d, d_s=G.d,
        m_train=M_TRAIN, k_rff=K_RFF, eps_levels=EPS_LEVELS, trials_per_level=TRIALS,
        npe_params=npe_params, npe_hidden=list(NPE_HIDDEN),
        npe_frozen_sha256=npe_hash_0,
        npe_vs_analytic_rmse=npe_vs_analytic,
        rff_gamma=gamma, sigma_med=sigma_med,
        decoder_final_mse=dec_mse, decoder_fit_time_s=fit_time,
        hash_identity_checks=hash_checks, hash_identity_all_ok=hash_all_ok,
        per_level=per_level, mean_reduction_nonsevere_pct=mean_red,
        rff_vs_exact=rff_gap,
        total_runtime_s=total_time,
        env=C.environment_report(),
    )
    out = os.path.join(ART, "gaussian_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nMean non-severe reduction = {mean_red:.2f}%  | hash-identity {hash_checks}/{hash_checks} OK={hash_all_ok}")
    print(f"Wrote {out}  ({total_time:.1f}s total)")


if __name__ == "__main__":
    main()
