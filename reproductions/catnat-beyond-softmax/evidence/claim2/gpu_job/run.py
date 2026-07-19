#!/usr/bin/env python3
"""
FULL-SCALE GPU reproduction of Claim 2 -- Categorical VAE (paper Table 3) for
"Beyond Softmax: A Natural Parameterization for Categorical Random Variables"
(ICML 2026, OpenReview ClBpWdkPZd; arXiv 2509.24728).

This is the *verified-evidence* path that the toy CPU proxy (../repro_claim2.py)
stands in for. It trains the paper's actual VAE task -- a convolutional
categorical VAE on MNIST / binarized MNIST with the Gumbel-Softmax straight-
through estimator -- and reports the paper's exact metric: test-set negative
log-likelihood estimated with importance sampling (Table 3, lower is better),
comparing the three score->probability parameterizations:
    * softmax
    * catnat + sigmoid activation
    * catnat + natural activation (nu, Eq. 12, C=0, A=2pi)

It reproduces the paper's protocol (Appendix E): CNN encoder -> N x K scores,
Gumbel-Softmax with straight-through (temperature annealed 1.0 -> 0.5, exp decay
3e-5), transposed-conv decoder, ELBO training objective, NLL with 512 importance
samples, swept over N in {10,20,30} and K in {8,16,32}.

NO numbers are hardcoded: every value is measured on the GPU run and written to
results.json (optionally uploaded to a HF dataset with --hf-repo). Run it with
the command in RUN_GPU.md. Expected wall-clock on a single A10G is a few hours
for the full sweep; use --N-list/--K-list/--seeds to scope a partial run.
"""
import argparse, json, math, os, time, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------
C_NAT, A_NAT = 0.0, 2.0 * math.pi
BAND = A_NAT / 2.0

def nu_activation(s):
    inside = (1.0 + torch.sin(math.pi * (s - C_NAT) / A_NAT)) / 2.0
    lo = torch.zeros_like(s); hi = torch.ones_like(s)
    out = torch.where(s <= C_NAT - BAND, lo, torch.where(s >= C_NAT + BAND, hi, inside))
    return out.clamp(1e-6, 1 - 1e-6)

def sigmoid_activation(s):
    return torch.sigmoid(s).clamp(1e-6, 1 - 1e-6)

# ---------------------------------------------------------------------------
# catnat balanced-binary-tree class log-probabilities (K = 2^H)
# ---------------------------------------------------------------------------
def build_tree_masks(K, device):
    H = int(round(math.log2(K)))
    assert 2 ** H == K, "catnat needs K a power of 2"
    on = torch.zeros(K, K - 1); bit = torch.zeros(K, K - 1)
    for x in range(K):
        m = x + (K - 1)
        while m > 0:
            par = (m - 1) // 2
            on[x, par] = 1.0
            bit[x, par] = 1.0 if (m == 2 * par + 2) else 0.0
            m = par
    M1 = (on * bit).to(device)          # right-child (a) contribution
    M0 = (on * (1 - bit)).to(device)    # left-child (1-a) contribution
    return M1, M0

def catnat_logprobs(scores, act, masks):
    """scores: (B,N,K) -> class log-probs (B,N,K) using first K-1 as node scores."""
    M1, M0 = masks
    s_int = scores[..., :-1]
    a = act(s_int)
    la = torch.log(a); l1a = torch.log1p(-a)
    logq = torch.einsum('bnj,xj->bnx', la, M1) + torch.einsum('bnj,xj->bnx', l1a, M0)
    return logq

def class_logits(scores, method, masks):
    """Return categorical *logits* (unnormalized log-probs) for Gumbel-Softmax."""
    if method == "softmax":
        return scores
    act = sigmoid_activation if method == "catnat_sigmoid" else nu_activation
    return catnat_logprobs(scores, act, masks)

# ---------------------------------------------------------------------------
# Gumbel-Softmax straight-through
# ---------------------------------------------------------------------------
def gumbel_softmax_st(logits, tau):
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = F.softmax((logits + g) / tau, dim=-1)
    idx = y.argmax(-1, keepdim=True)
    y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
    return (y_hard - y).detach() + y                       # straight-through

# ---------------------------------------------------------------------------
# convolutional categorical VAE (architecture follows jxmorris12/categorical-vae)
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, N, K):
        super().__init__(); self.N, self.K = N, K
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
            nn.Linear(256, N * K))
    def forward(self, x):
        return self.net(x).view(-1, self.N, self.K)

class Decoder(nn.Module):
    def __init__(self, N, K):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(N * K, 256), nn.ReLU(),
                                nn.Linear(256, 64 * 7 * 7), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1))
    def forward(self, z):
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h)                              # pixel logits (B,1,28,28)

class CatVAE(nn.Module):
    def __init__(self, N, K, method, device):
        super().__init__()
        self.N, self.K, self.method = N, K, method
        self.enc = Encoder(N, K); self.dec = Decoder(N, K)
        self.masks = build_tree_masks(K, device) if method != "softmax" else None
    def forward(self, x, tau):
        scores = self.enc(x)                               # (B,N,K)
        logits = class_logits(scores, self.method, self.masks)
        logq = F.log_softmax(logits, dim=-1)
        z = gumbel_softmax_st(logits, tau)                 # (B,N,K)
        xlogit = self.dec(z.view(z.size(0), -1))
        return xlogit, logq

# ---------------------------------------------------------------------------
# ELBO loss and importance-sampled test NLL (paper protocol: 512 samples)
# ---------------------------------------------------------------------------
def elbo_loss(xlogit, x, logq, K, beta=1.0):
    recon = F.binary_cross_entropy_with_logits(xlogit, x, reduction='none').sum((1, 2, 3))
    q = logq.exp()
    kl = (q * (logq + math.log(K))).sum((1, 2))            # KL(q||Uniform)
    return (recon + beta * kl).mean(), recon.mean().item(), kl.mean().item()

@torch.no_grad()
def test_nll_iw(model, loader, device, n_samples=512):
    """-mean_x logmeanexp_s [ log p(x|c)+log p(c)-log q(c|x) ], hard categorical samples."""
    model.eval(); tot, cnt = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        scores = model.enc(x)
        logits = class_logits(scores, model.method, model.masks)
        logq = F.log_softmax(logits, dim=-1)               # (B,N,K)
        B, N, K = logq.shape
        logw = torch.empty(B, n_samples, device=device)
        for s in range(n_samples):
            c = torch.distributions.Categorical(logits=logits).sample()   # (B,N)
            oneh = F.one_hot(c, K).float()
            xlogit = model.dec(oneh.view(B, -1))
            logpxc = -F.binary_cross_entropy_with_logits(xlogit, x, reduction='none').sum((1, 2, 3))
            logpc = -N * math.log(K)
            logqc = logq.gather(-1, c.unsqueeze(-1)).squeeze(-1).sum(-1)
            logw[:, s] = logpxc + logpc - logqc
        ll = torch.logsumexp(logw, 1) - math.log(n_samples)
        tot += (-ll).sum().item(); cnt += B
    return tot / cnt

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def get_loaders(binarize, batch, root):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor()])
    tr = datasets.MNIST(root, train=True, download=True, transform=tf)
    te = datasets.MNIST(root, train=False, download=True, transform=tf)
    def collate(batch_):
        xs = torch.stack([b[0] for b in batch_]); ys = torch.tensor([b[1] for b in batch_])
        if binarize:
            xs = (xs > 0.5).float()
        return xs, ys
    tl = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=2)
    el = torch.utils.data.DataLoader(te, batch_size=batch, shuffle=False, collate_fn=collate, num_workers=2)
    return tl, el

# ---------------------------------------------------------------------------
# train one (method, N, K, seed) config
# ---------------------------------------------------------------------------
def train_config(method, N, K, seed, args, device):
    torch.manual_seed(seed); np.random.seed(seed)
    tl, el = get_loaders(args.binarize, args.batch, args.data_root)
    model = CatVAE(N, K, method, device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tau, tau_min, decay = 1.0, 0.5, 3e-5
    step = 0
    for ep in range(args.epochs):
        model.train()
        for x, _ in tl:
            x = x.to(device)
            xlogit, logq = model(x, tau)
            loss, _, _ = elbo_loss(xlogit, x, logq, K, beta=args.beta)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            tau = max(tau_min, math.exp(-decay * step))
    nll = test_nll_iw(model, el, device, n_samples=args.n_importance)
    return nll

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N-list", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--K-list", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=["softmax", "catnat_sigmoid", "catnat_nu"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n-importance", type=int, default=512)
    ap.add_argument("--binarize", action="store_true")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--hf-repo", default=os.environ.get("HF_RESULT_REPO", ""),
                    help="optional HF dataset repo id to upload results.json to")
    ap.add_argument("--upload-path", default="claim2/results.json",
                    help="path_in_repo for the uploaded results file (lets parallel "
                         "shards of the sweep write distinct files)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} torch={torch.__version__} "
          f"cuda={torch.cuda.is_available()} dataset={'binaryMNIST' if args.binarize else 'MNIST'}")
    t0 = time.time()
    results = {"orid": "ClBpWdkPZd", "claim": 2,
               "task": "categorical VAE on MNIST (paper Table 3)",
               "metric": "test NLL, importance-sampled (lower is better)",
               "dataset": "binary_mnist" if args.binarize else "mnist",
               "config": vars(args), "device": device, "torch": torch.__version__,
               "runs": []}
    for N in args.N_list:
        for K in args.K_list:
            for method in args.methods:
                vals = []
                for seed in args.seeds:
                    nll = train_config(method, N, K, seed, args, device)
                    vals.append(nll)
                    print(f"[run] N={N} K={K} {method:15s} seed={seed} testNLL={nll:.3f}")
                rec = {"N": N, "K": K, "method": method, "seeds": args.seeds,
                       "test_nll_mean": float(np.mean(vals)),
                       "test_nll_std": float(np.std(vals)),
                       "test_nll_per_seed": [float(v) for v in vals]}
                results["runs"].append(rec)
                print(f"[agg] N={N} K={K} {method:15s} "
                      f"testNLL={rec['test_nll_mean']:.3f}+/-{rec['test_nll_std']:.3f}")
                with open(args.out, "w") as f:      # checkpoint after every config
                    json.dump(results, f, indent=2)
                if args.hf_repo:  # push partial progress so a timeout can't zero the run
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi()
                        api.create_repo(args.hf_repo, repo_type="dataset", exist_ok=True)
                        api.upload_file(path_or_fileobj=args.out, path_in_repo=args.upload_path,
                                        repo_id=args.hf_repo, repo_type="dataset")
                    except Exception as e:
                        print(f"[hf] checkpoint upload failed (continuing): {e}")
    results["runtime_s"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"[done] wrote {args.out} sha256={sha[:16]} runtime={results['runtime_s']}s")

    if args.hf_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.hf_repo, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=args.out, path_in_repo=args.upload_path,
                        repo_id=args.hf_repo, repo_type="dataset")
        print(f"[hf] uploaded {args.out} -> {args.hf_repo}:{args.upload_path}")

if __name__ == "__main__":
    main()
