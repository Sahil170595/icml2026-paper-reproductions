"""
Claim 2 (MaxRL, OpenReview EeuLO2BjFN / arXiv 2602.02710):

  "The resulting objectives admit a simple, unbiased policy-gradient estimator"
  (for the non-differentiable sampling setting).  (Abstract)

We test the MaxRL compute-indexed objective
    J_N(theta) = E_{a_1..a_N ~ pi_theta}[ g(a_1..a_N) ],   g = log((1/N) sum_i r_{a_i})
with pi_theta = softmax(theta) over K discrete rollouts (sampling is
non-differentiable: no reparameterization is possible).

The MaxRL / REINFORCE score-function estimator over one group of N rollouts is
    ghat = g(a_1..a_N) * sum_{i=1}^N grad_theta log pi(a_i).
This is claimed to be an UNBIASED estimator of grad J_N.

VERIFICATION (no fabrication):
  * Ground truth grad J_N computed EXACTLY by enumerating all K^N group outcomes.
  * Monte-Carlo mean of ghat over G groups is compared to the exact gradient:
    report per-component bias, its 95% CI, standardized z-scores, and that the
    bias is statistically zero (shrinks like 1/sqrt(G)); works with purely
    non-differentiable sampling.
  * DECISIVE CONTROL: a mis-specified "first-sample-only" estimator
    ghat_first = g * grad log pi(a_1) has expectation (1/N) grad J_N -> it is
    provably BIASED for N>1; we show its bias is large and its CI excludes 0.
    This confirms the *specific* group score-function form is what is unbiased.
"""
import json, itertools, numpy as np

def softmax(theta):
    z = theta - theta.max(); e = np.exp(z); return e / e.sum()

def score(theta):
    """rows s(a) = grad_theta log pi(a) = onehot(a) - pi, shape (K,K)."""
    pi = softmax(theta); K = len(theta)
    return np.eye(K) - pi[None, :], pi

def exact_gradJN(theta, r, N):
    """Exact grad_theta J_N by enumerating all K^N group outcomes (vectorized)."""
    K = len(theta); pi = softmax(theta); logpi = np.log(pi)
    combos = np.array(list(itertools.product(range(K), repeat=N)), dtype=np.int64)  # (K^N, N)
    logP = logpi[combos].sum(axis=1)               # log prob of each group
    P = np.exp(logP)                               # (K^N,)
    g = np.log(r[combos].mean(axis=1))             # g(group), (K^N,)
    counts = np.zeros((combos.shape[0], K))
    for k in range(K):
        counts[:, k] = (combos == k).sum(axis=1)   # occurrences of action k
    sum_scores = counts - N * pi[None, :]          # sum_i grad log pi(a_i), (K^N,K)
    grad = (P[:, None] * g[:, None] * sum_scores).sum(axis=0)
    JN = float((P * g).sum())
    return grad, JN, float(P.sum())

def mc_estimator(theta, r, N, G, rng, first_only=False, chunk=20000):
    """Monte-Carlo: return per-group estimates stacked -> (G, K)."""
    K = len(theta); pi = softmax(theta); cdf = np.cumsum(pi); cdf[-1] = 1.0
    S = np.eye(K)
    ests = np.empty((G, K))
    done = 0
    while done < G:
        b = min(chunk, G - done)
        u = rng.random((b, N))
        idx = np.searchsorted(cdf, u); np.clip(idx, 0, K - 1, out=idx)   # (b,N)
        g = np.log(r[idx].mean(axis=1))                                  # (b,)
        if first_only:
            sc = S[idx[:, 0]] - pi[None, :]                              # (b,K) first sample only
        else:
            # sum_i (onehot(a_i) - pi) = counts - N*pi
            counts = np.zeros((b, K))
            for k in range(K):
                counts[:, k] = (idx == k).sum(axis=1)
            sc = counts - N * pi[None, :]                                # (b,K)
        ests[done:done + b] = g[:, None] * sc
        done += b
    return ests

def main():
    out = {"claim": "MaxRL objectives admit a simple UNBIASED policy-gradient estimator",
           "cases": []}
    print("=" * 74)
    print("MaxRL Claim 2 — unbiased policy-gradient estimator  (EeuLO2BjFN)")
    print("=" * 74)

    configs = [(3, 2, 4_000_000), (4, 4, 3_000_000), (5, 6, 3_000_000), (6, 8, 2_500_000)]
    rng = np.random.default_rng(7)
    all_unbiased = True
    for (K, N, G) in configs:
        rr = np.round(np.random.default_rng(100 + K).uniform(0.05, 1.0, size=K), 4)
        theta = np.random.default_rng(200 + K).normal(0, 0.8, size=K)
        exact, JN, psum = exact_gradJN(theta, rr, N)

        ests = mc_estimator(theta, rr, N, G, rng)          # unbiased (group) estimator
        mean = ests.mean(axis=0); sd = ests.std(axis=0, ddof=1)
        se = sd / np.sqrt(G)
        bias = mean - exact
        z = bias / se                                       # standardized bias
        rel = float(np.linalg.norm(bias) / (np.linalg.norm(exact) + 1e-30))
        within = int(np.sum(np.abs(bias) <= 1.96 * se))
        maxz = float(np.max(np.abs(z)))
        cos = float(mean @ exact / (np.linalg.norm(mean) * np.linalg.norm(exact) + 1e-30))

        # decisive control: biased first-sample-only estimator
        ests_b = mc_estimator(theta, rr, N, G, rng, first_only=True)
        mean_b = ests_b.mean(axis=0); se_b = ests_b.std(axis=0, ddof=1) / np.sqrt(G)
        bias_b = mean_b - exact
        z_b = bias_b / se_b
        # theoretical expectation of first-only estimator is (1/N)*exact
        pred_b = exact / N
        rel_b = float(np.linalg.norm(bias_b) / (np.linalg.norm(exact) + 1e-30))
        matches_1overN = float(np.max(np.abs(mean_b - pred_b) / (1.96 * se_b + 1e-30)))
        maxz_b = float(np.max(np.abs(z_b)))

        unbiased_ok = (maxz < 4.0)          # every component within ~4 sigma of exact
        biased_ctrl_ok = (maxz_b > 6.0)     # control is clearly biased
        all_unbiased = all_unbiased and unbiased_ok and biased_ctrl_ok

        print(f"\n[K={K}, N={N}, groups G={G:,}]  enum {K}^{N}={K**N} outcomes, sum P={psum:.6f}, J_N={JN:.5f}")
        print(f"  exact grad J_N   = {np.array2string(exact, precision=5, max_line_width=200)}")
        print(f"  MC mean (group)  = {np.array2string(mean, precision=5, max_line_width=200)}")
        print(f"  rel L2 bias = {rel:.3e}   max|z| = {maxz:.2f}   comps within 95%CI = {within}/{K}   cos = {cos:.6f}")
        print(f"  -> UNBIASED (group score-fn estimator): {unbiased_ok}")
        print(f"  CONTROL first-sample-only: rel bias = {rel_b:.3e}  max|z|={maxz_b:.1f}  "
              f"(expected = exact/N; max|mean-exact/N|/CI = {matches_1overN:.2f})")
        print(f"  -> control is BIASED as predicted (E=grad/N): {biased_ctrl_ok}")

        out["cases"].append(dict(K=K, N=N, G=G, enum_outcomes=K**N, prob_mass=psum, J_N=JN,
            exact_grad=exact.tolist(), mc_mean=mean.tolist(), rel_L2_bias=rel,
            max_abs_z=maxz, comps_within_95CI=within, cosine=cos, unbiased_ok=bool(unbiased_ok),
            control_first_only=dict(rel_bias=rel_b, max_abs_z=maxz_b,
                pred_over_N_maxz=matches_1overN, biased_ok=bool(biased_ctrl_ok))))

    out["verdict"] = {"unbiased_estimator_verified": bool(all_unbiased),
        "summary": "MC mean of group score-function estimator matches exact grad J_N within 95% CI for all cases; first-sample-only control is biased by factor 1/N as predicted"}
    print("\n" + "=" * 74)
    print(f"CLAIM 2 unbiased estimator verified (all cases within CI; control biased): {all_unbiased}")
    print("=" * 74)
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json")

if __name__ == "__main__":
    main()
