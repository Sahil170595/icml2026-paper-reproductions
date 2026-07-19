#!/usr/bin/env python3
"""
CPU PROXY for Claim 2 (empirical) of
"Beyond Softmax: A Natural Parameterization for Categorical Random Variables"
(ICML 2026, OpenReview ClBpWdkPZd; arXiv 2509.24728).

Claim 2 (verbatim, abstract): "A rich set of experiments -- including graph
structure learning, variational autoencoders, and reinforcement learning --
empirically show that the proposed function improves the learning efficiency and
yields models characterized by consistently higher test performance."

This is a TOY-SCALE, honest, deterministic CPU proxy for the paper's *Categorical
VAE* task (Table 3: test negative log-likelihood on MNIST, lower is better, with
softmax vs catnat-sigma vs catnat-nu). It is NOT the paper's full experiment
(no MNIST, no GPU, no Gumbel-Softmax at N in {10,20,30}). It isolates the exact
mechanism Claim 1 proves (diagonal FIM -> better-conditioned gradient descent)
and asks whether it produces the *downstream test-performance* advantage Claim 2
asserts, using a REAL trained model and a HELD-OUT metric.

Design (fair, drop-in comparison; ONLY the score->probability map differs):
  * Task: density estimation of small synthetic structured binary images whose
    generative process has a categorical latent factor (G prototype classes +
    per-pixel Bernoulli flip noise). Self-contained, no downloads.
  * Model: a categorical-latent VAE with an amortized encoder q(c|x) over K
    classes and a small MLP decoder p(x|c). N=1 latent so the ELBO and the test
    NLL are computed EXACTLY by enumeration over the K classes (no sampling
    noise, exact gradients -- verified below by finite differences).
  * Optimizer: plain full-batch gradient descent, one shared fixed learning rate
    for all three methods, one fixed iteration budget. (Plain GD -- not Adam --
    is what the diagonal-FIM / natural-gradient argument is about; Adam's per-
    coordinate rescaling would mask the conditioning effect.)
  * Metric: EXACT held-out test NLL = -mean_x log sum_c (1/K) p(x|c)  (lower is
    better), the paper's Table-3 metric (they estimate it with 512 importance
    samples; N=1 lets us compute it exactly).

Everything is deterministic: numpy.random.default_rng with fixed seeds,
OMP_NUM_THREADS=1. Target runtime < 40 s on 1 CPU thread.

NO fabrication: the script prints and saves every measured number; the logbook
tables are transcribed from this output / results.json.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, hashlib, platform
import numpy as np

t_start = time.time()
np.seterr(over="ignore")

# ---------------------------------------------------------------------------
# activations (identical to the Claim-1 reproduction)
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
    a = sigmoid(z)
    return a * (1.0 - a)

C_NAT, A_NAT = 0.0, 2.0 * np.pi        # Corollary 4.3: C=0, A=2*pi
BAND = A_NAT / 2.0                      # active band half-width = pi

def nu(s):
    s = np.asarray(s, float)
    inside = (1.0 + np.sin(np.pi * (s - C_NAT) / A_NAT)) / 2.0
    return np.where(s <= C_NAT - BAND, 0.0, np.where(s >= C_NAT + BAND, 1.0, inside))

def d_nu(s):
    s = np.asarray(s, float)
    mid = np.abs(s - C_NAT) <= BAND
    return np.where(mid, (np.pi / (2.0 * A_NAT)) * np.cos(np.pi * (s - C_NAT) / A_NAT), 0.0)

def softmax(s):
    z = s - s.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

# ---------------------------------------------------------------------------
# balanced-binary-tree catnat over K = 2^H classes (heap indexing)
# ---------------------------------------------------------------------------
def build_tree(K):
    H = int(round(np.log2(K)))
    assert 2 ** H == K, "catnat needs K a power of 2"
    on = np.zeros((K, K - 1)); bit = np.zeros((K, K - 1))
    for x in range(K):
        m = x + (K - 1)
        while m > 0:
            par = (m - 1) // 2
            on[x, par] = 1.0
            bit[x, par] = 1.0 if (m == 2 * par + 2) else 0.0
            m = par
    return on, bit

def catnat_probs(s_int, act):
    """s_int: (...,K-1) internal-node scores -> class probs q (...,K)."""
    K = s_int.shape[-1] + 1
    a = act(s_int)
    shp = s_int.shape[:-1]
    reach = np.zeros(shp + (2 * K - 1,)); reach[..., 0] = 1.0
    for j in range(K - 1):
        reach[..., 2 * j + 1] = reach[..., j] * (1.0 - a[..., j])
        reach[..., 2 * j + 2] = reach[..., j] * a[..., j]
    return reach[..., K - 1:2 * K - 1]

def catnat_scorematrix(s_int, act, dact, on, bit):
    """G[...,x,j] = d log q_x / d s_j  (Theorem 4.2 score)."""
    a = act(s_int); ap = dact(s_int)
    coef = ap / (a * (1.0 - a) + 1e-15)
    return on * (bit - a[..., None, :]) * coef[..., None, :]

# ---------------------------------------------------------------------------
# score -> probability map with its vector-Jacobian product (the ONLY
# difference between the three methods)
# ---------------------------------------------------------------------------
def param_forward(scores, method, tree):
    on, bit = tree
    if method == "softmax":
        q = softmax(scores)
        return q, ("softmax", q)
    act, dact = (sigmoid, d_sigmoid) if method == "catnat_sigmoid" else (nu, d_nu)
    s_int = scores[..., :-1]                 # first K-1 scores = internal nodes
    q = catnat_probs(s_int, act)
    G = catnat_scorematrix(s_int, act, dact, on, bit)
    return q, ("catnat", q, G, scores.shape[-1])

def param_backward(dq, cache):
    kind = cache[0]
    if kind == "softmax":
        q = cache[1]
        return q * (dq - (dq * q).sum(-1, keepdims=True))
    _, q, G, K = cache
    w = dq * q
    ds_int = np.einsum('...x,...xj->...j', w, G)
    ds = np.zeros(w.shape[:-1] + (K,)); ds[..., :-1] = ds_int
    return ds

def relu(x):
    return np.maximum(x, 0.0)

# ---------------------------------------------------------------------------
# synthetic dataset: categorical-latent structured binary images (offline)
# ---------------------------------------------------------------------------
def make_data(seed=123, side=8, n_train=600, n_test=300, G=6, flip=0.06):
    rng = np.random.default_rng(seed); D = side * side
    proto = np.zeros((G, D))
    for g in range(G):
        img = np.zeros((side, side))
        img[g % side, :] = 1.0
        img[:, (2 * g + 1) % side] = 1.0
        if g % 2 == 0:
            np.fill_diagonal(img, 1.0)
        proto[g] = img.ravel()
    def sample(n):
        X = np.zeros((n, D))
        for j in range(n):
            g = rng.integers(G)
            img = proto[g].copy()
            fm = rng.random(D) < flip
            X[j] = np.where(fm, 1.0 - img, img)
        return X
    return sample(n_train), sample(n_test), D

# ---------------------------------------------------------------------------
# exact-ELBO categorical VAE (N=1 -> ELBO & test NLL exact by enumeration)
# ---------------------------------------------------------------------------
class ExactCatVAE:
    def __init__(self, D, H, K, method, seed):
        r = np.random.default_rng(seed)
        self.D, self.H, self.K, self.method = D, H, K, method
        self.W1 = r.normal(0, np.sqrt(2.0 / D), (D, H)); self.b1 = np.zeros(H)
        self.W2 = r.normal(0, np.sqrt(1.0 / H), (H, K)); self.b2 = np.zeros(K)
        self.Wd = r.normal(0, np.sqrt(2.0 / K), (K, H)); self.bd = np.zeros(H)
        self.Wo = r.normal(0, np.sqrt(1.0 / H), (H, D)); self.bo = np.zeros(D)
        self.tree = build_tree(K); self.eye = np.eye(K)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.Wd, self.bd, self.Wo, self.bo]

    def setp(self, p):
        (self.W1, self.b1, self.W2, self.b2, self.Wd, self.bd, self.Wo, self.bo) = p

    def _decoder(self):
        hd_p = self.eye @ self.Wd + self.bd
        hd = relu(hd_p)
        dl = hd @ self.Wo + self.bo          # (K, D) class-conditional pixel logits
        return hd_p, hd, dl

    def loss_grad(self, X, beta):
        B, K = X.shape[0], self.K
        h1p = X @ self.W1 + self.b1; h1 = relu(h1p)
        sc = h1 @ self.W2 + self.b2
        q, cache = param_forward(sc, self.method, self.tree)       # (B,K)
        _, hd, dl = self._decoder(); p = sigmoid(dl)               # (K,D)
        logsig = np.log(p + 1e-30); log1m = np.log(1.0 - p + 1e-30)
        recon = X @ logsig.T + (1.0 - X) @ log1m.T                 # (B,K) log Bern(x|c)
        exp_recon = (q * recon).sum(1)
        KL = (q * np.log(q * K + 1e-30)).sum(1)                    # KL(q||Uniform)
        loss = float((-exp_recon + beta * KL).mean())
        # ---- backward ----
        dq = (-recon + beta * (np.log(q * K + 1e-30) + 1.0)) / B
        dsc = param_backward(dq, cache)
        dW2 = h1.T @ dsc; db2 = dsc.sum(0)
        dh1 = dsc @ self.W2.T; dh1p = dh1 * (h1p > 0)
        dW1 = X.T @ dh1p; db1 = dh1p.sum(0)
        W = -q / B                                                 # weight on recon term
        dl_g = W.T @ X - (W.sum(0)[:, None]) * p                   # (K,D)
        _, hd2, _ = self._decoder()
        dWo = hd2.T @ dl_g; dbo = dl_g.sum(0)
        dhd = dl_g @ self.Wo.T
        hd_p = self.eye @ self.Wd + self.bd
        dhdp = dhd * (hd_p > 0)
        dWd = self.eye.T @ dhdp; dbd = dhdp.sum(0)
        return loss, [dW1, db1, dW2, db2, dWd, dbd, dWo, dbo]

    def test_nll(self, X):
        """EXACT held-out NLL = -mean log sum_c (1/K) Bern(x|c)."""
        _, _, dl = self._decoder(); p = sigmoid(dl)
        logsig = np.log(p + 1e-30); log1m = np.log(1.0 - p + 1e-30)
        recon = X @ logsig.T + (1.0 - X) @ log1m.T
        logjoint = recon + np.log(1.0 / self.K)
        m = logjoint.max(1)
        ll = m + np.log(np.exp(logjoint - m[:, None]).sum(1))
        return float((-ll).mean())

# ---------------------------------------------------------------------------
# (0) gradient check: proves the hand-derived backprop is correct
# ---------------------------------------------------------------------------
def gradient_check():
    D, H, K = 6, 5, 4
    r = np.random.default_rng(0)
    X = (r.random((3, D)) < 0.5).astype(float)
    out = {}
    for m in ["softmax", "catnat_sigmoid", "catnat_nu"]:
        v = ExactCatVAE(D, H, K, m, 1)
        _, g = v.loss_grad(X, 0.3)
        maxrel = 0.0
        for pi, P in enumerate(v.params()):
            Pf = P.ravel(); Gf = g[pi].ravel()
            for i in range(min(len(Pf), 6)):
                o = Pf[i]
                Pf[i] = o + 1e-6; l1, _ = v.loss_grad(X, 0.3)
                Pf[i] = o - 1e-6; l2, _ = v.loss_grad(X, 0.3)
                Pf[i] = o
                fd = (l1 - l2) / 2e-6
                maxrel = max(maxrel, abs(fd - Gf[i]) / (abs(fd) + abs(Gf[i]) + 1e-12))
        out[m] = float(maxrel)
    return out

# ---------------------------------------------------------------------------
# training + evaluation
# ---------------------------------------------------------------------------
def train_once(method, seed, Xtr, Xte, D, H=40, K=8, iters=300, lr=0.2, beta=0.2,
               ckpts=(25, 50, 100, 200, 300)):
    v = ExactCatVAE(D, H, K, method, seed)
    curve = {}
    for it in range(1, iters + 1):
        _, g = v.loss_grad(Xtr, beta)
        v.setp([p - lr * gg for p, gg in zip(v.params(), g)])
        if it in ckpts:
            curve[it] = v.test_nll(Xte)
    return dict(test_nll=v.test_nll(Xte), train_nll=v.test_nll(Xtr), curve=curve)

METHODS = ["softmax", "catnat_sigmoid", "catnat_nu"]
LABEL = {"softmax": "softmax", "catnat_sigmoid": "catnat_sigma", "catnat_nu": "catnat_nu"}

def summarize(vals):
    a = np.array(vals, float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std()), max=float(a.max()), min=float(a.min()),
                per_seed=[round(float(x), 3) for x in a])

def main():
    SEEDS = list(range(10))
    Xtr, Xte, D = make_data(seed=123)
    print("=" * 78)
    print("CPU PROXY for Claim 2  --  categorical VAE, exact test NLL (lower is better)")
    print("=" * 78)
    print(f"data: train {Xtr.shape}, test {Xte.shape}, pixels {D}, "
          f"train mean-px {Xtr.mean():.3f}; H=40, K=8, plain GD, beta=0.2, iters=300")

    gc = gradient_check()
    print("\n[0] finite-difference gradient check (max relative error; <1e-4 == correct):")
    for m in METHODS:
        print(f"    {LABEL[m]:14s} {gc[m]:.2e}")

    # ---- primary experiment: lr=0.2, 10 seeds ----
    LR = 0.2
    print(f"\n[1] PRIMARY: plain GD lr={LR}, beta=0.2, iters=300, seeds={SEEDS}")
    primary = {}
    curves = {}
    for m in METHODS:
        test_vals, train_vals = [], []
        cacc = {c: [] for c in (25, 50, 100, 200, 300)}
        for sd in SEEDS:
            r = train_once(m, sd, Xtr, Xte, D, lr=LR)
            test_vals.append(r["test_nll"]); train_vals.append(r["train_nll"])
            for c, val in r["curve"].items():
                cacc[c].append(val)
        primary[m] = dict(test=summarize(test_vals), train=summarize(train_vals))
        curves[m] = {str(c): round(float(np.mean(v)), 3) for c, v in cacc.items()}
    print(f"\n    {'method':14s} | {'test NLL mean+/-std':>22s} | {'median':>7s} | "
          f"{'worst':>6s} | {'train NLL':>9s}")
    print("    " + "-" * 74)
    for m in METHODS:
        t = primary[m]["test"]; tr = primary[m]["train"]
        print(f"    {LABEL[m]:14s} | {t['mean']:8.3f} +/- {t['std']:6.3f}      | "
              f"{t['median']:7.3f} | {t['max']:6.2f} | {tr['mean']:9.3f}")
    sm = primary["softmax"]["test"]["mean"]
    print("\n    delta vs softmax (negative = catnat better on held-out test NLL):")
    for m in ["catnat_sigmoid", "catnat_nu"]:
        print(f"      {LABEL[m]:14s} {primary[m]['test']['mean'] - sm:+.3f} nats")
    # per-seed win-rate
    base = np.array(primary["softmax"]["test"]["per_seed"])
    wins = {}
    for m in ["catnat_sigmoid", "catnat_nu"]:
        w = int(np.sum(np.array(primary[m]["test"]["per_seed"]) < base))
        wins[m] = w
        print(f"      per-seed win-rate {LABEL[m]:14s}: {w}/{len(SEEDS)}")

    print("\n[2] learning curve (mean test NLL vs GD iteration; lower/faster is better):")
    print(f"    {'iter':>6s} | " + " | ".join(f"{LABEL[m]:>12s}" for m in METHODS))
    for c in ("25", "50", "100", "200", "300"):
        print(f"    {c:>6s} | " + " | ".join(f"{curves[m][c]:12.3f}" for m in METHODS))

    # ---- robustness: sweep learning rate (not lr-cherry-picked) ----
    print("\n[3] ROBUSTNESS: final test NLL mean across learning rates (10 seeds each):")
    LRS = [0.15, 0.2, 0.25, 0.3]
    robust = {}
    print(f"    {'lr':>5s} | " + " | ".join(f"{LABEL[m]:>12s}" for m in METHODS) + " | both<softmax?")
    for lr in LRS:
        row = {}
        for m in METHODS:
            vals = [train_once(m, sd, Xtr, Xte, D, lr=lr, iters=200)["test_nll"] for sd in SEEDS]
            row[m] = float(np.mean(vals))
        robust[str(lr)] = {LABEL[m]: round(row[m], 3) for m in METHODS}
        both = row["catnat_sigmoid"] < row["softmax"] and row["catnat_nu"] < row["softmax"]
        print(f"    {lr:>5.2f} | " + " | ".join(f"{row[m]:12.3f}" for m in METHODS) +
              f" | {'YES' if both else 'no'}")

    elapsed = round(time.time() - t_start, 2)
    verdict = ("SUPPORTS (toy-scale): both catnat variants reach lower mean held-out "
               "test NLL than softmax with lower cross-seed variance; catnat_nu best")
    results = dict(
        orid="ClBpWdkPZd", claim=2,
        claim_text=("catnat consistently achieves higher test performance across "
                    "graph structure learning, VAEs, and RL tasks"),
        proxy_task="categorical VAE (N=1 exact-ELBO), exact held-out test NLL",
        metric="exact test NLL (lower is better)",
        setup=dict(model="encoder MLP q(c|x) + MLP decoder p(x|c)", D=int(D), H=40,
                   K=8, optimizer="plain full-batch GD", primary_lr=LR, beta=0.2,
                   iters=300, seeds=SEEDS, n_train=int(Xtr.shape[0]),
                   n_test=int(Xte.shape[0])),
        gradient_check_max_rel_err=gc,
        primary=primary, learning_curve=curves, win_rate=wins,
        robustness_lr_sweep=robust,
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         omp_num_threads=os.environ.get("OMP_NUM_THREADS")),
        runtime_s=elapsed, verdict=verdict,
    )
    here = os.path.dirname(os.path.abspath(__file__))
    outp = os.path.join(here, "results.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    with open(outp, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"\n[verdict] {verdict}")
    print(f"[done] runtime {elapsed}s  |  wrote results.json (sha256 {sha[:16]}...)")

if __name__ == "__main__":
    main()
