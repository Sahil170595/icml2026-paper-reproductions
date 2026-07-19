# PART B (appended experiment): bimodal mixture -> performance universality HOLDS
# while score universality FAILS (paper Sec.5). Run after Part A; merges into results.json.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time
from pathlib import Path
import numpy as np
from scipy import stats

t0 = time.time()
p, n = 200, 400
lam, sigma2 = 0.50, 0.25
sigma = np.sqrt(sigma2)
RES = Path(__file__).with_name("results.json")

rng_u = np.random.default_rng(2024)
u = rng_u.standard_normal(p); u /= np.linalg.norm(u)
w0, w1, m_scale = 0.3, 0.7, 2.0
mu0 = m_scale * u; mu1 = -(w0 / w1) * m_scale * u
comp = np.stack([mu0, mu1])
spike = m_scale**2 * (w0 / w1)                 # C_x = I + spike u u^T ; E[x]=0
Cx_mix = np.eye(p) + spike * np.outer(u, u)
theta_star_mix = u.copy()                      # aligned signal -> bimodal projection

def sample_mix(K, rng):
    c = rng.choice(2, size=K, p=[w0, w1])
    return rng.standard_normal((K, p)) + comp[c]

def sample_gauss_matched(K, rng):              # N(0, C_x_mix): matched first two moments
    return rng.standard_normal((K, p)) + np.sqrt(spike) * np.outer(rng.standard_normal(K), u)

def fit(X, y):
    return np.linalg.solve(X.T @ X / n + lam * np.eye(p), X.T @ y / n)

M_B = 1500
rngm = np.random.default_rng(555); rngg = np.random.default_rng(556)
risk_true = np.empty(M_B); risk_gauss = np.empty(M_B); Th_mix = np.empty((M_B, p))
for i in range(M_B):
    Xt = sample_mix(n, rngm); yt = Xt @ theta_star_mix + sigma * rngm.standard_normal(n)
    th = fit(Xt, yt); Th_mix[i] = th
    d = th - theta_star_mix; risk_true[i] = d @ (Cx_mix @ d)
    Xg = sample_gauss_matched(n, rngg); yg = Xg @ theta_star_mix + sigma * rngg.standard_normal(n)
    thg = fit(Xg, yg); dg = thg - theta_star_mix; risk_gauss[i] = dg @ (Cx_mix @ dg)

rt, rg = float(risk_true.mean()), float(risk_gauss.mean())
perf_gap_rel = abs(rt - rg) / rg
perf_se = float(np.sqrt(risk_true.var(ddof=1) + risk_gauss.var(ddof=1)) / np.sqrt(M_B))

mu_th = Th_mix.mean(0)
rng_test = np.random.default_rng(999)
Ktest = 40000
s_mix = sample_mix(Ktest, rng_test) @ mu_th
s_gauss = sample_gauss_matched(Ktest, rng_test) @ mu_th
score_exkurt_mix = float(stats.kurtosis(s_mix)); score_skew_mix = float(stats.skew(s_mix))
ks_mix_D, ks_mix_p = stats.kstest(s_mix, 'norm', args=(s_mix.mean(), s_mix.std()))
score_exkurt_gauss = float(stats.kurtosis(s_gauss))
ks_gauss_D, ks_gauss_p = stats.kstest(s_gauss, 'norm', args=(s_gauss.mean(), s_gauss.std()))

print("[PartB] bimodal mixture: performance universality vs score universality")
print(f"  ridge risk true-mixture={rt:.5f} matched-Gaussian={rg:.5f} "
      f"gap={100*perf_gap_rel:.2f}% (+/-{100*perf_se/rg:.2f}%) -> perf-univ HOLDS")
print(f"  score(mixture): skew={score_skew_mix:+.3f} exkurt={score_exkurt_mix:+.3f} "
      f"KS-vs-normal D={ks_mix_D:.4f} p={ks_mix_p:.2e} -> score-univ FAILS")
print(f"  score(matched-Gaussian): exkurt={score_exkurt_gauss:+.3f} "
      f"KS-vs-normal D={ks_gauss_D:.4f} p={ks_gauss_p:.3f} -> Gaussian")
perf_holds = bool(perf_gap_rel < 0.05)
score_fails = bool(ks_mix_p < 1e-3 and score_exkurt_mix < -0.05)
print(f"[Verdict B] perf-univ holds={perf_holds} score-univ fails={score_fails}")

partB = dict(spike=float(spike), M_partB=M_B, risk_true_mixture=rt, risk_matched_gaussian=rg,
             perf_gap_rel=float(perf_gap_rel), perf_gap_se_rel=float(perf_se / rg),
             score_skew_mix=score_skew_mix, score_exkurt_mix=score_exkurt_mix,
             score_ks_mix_D=float(ks_mix_D), score_ks_mix_p=float(ks_mix_p),
             score_exkurt_gauss=score_exkurt_gauss, score_ks_gauss_p=float(ks_gauss_p),
             perf_univ_holds=perf_holds, score_univ_fails=score_fails,
             runtime_s=round(time.time() - t0, 2))
res = json.loads(RES.read_text())
res["partB"] = partB
RES.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"[done] merged partB into results.json  runtime={partB['runtime_s']}s")
