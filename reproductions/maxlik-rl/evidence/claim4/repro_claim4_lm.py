"""
Claim 4 / EXP C -- LANGUAGE-MODEL upgrade (MaxRL, OpenReview EeuLO2BjFN / arXiv 2602.02710):

  "Empirically, we show that MaxRL Pareto-dominates existing methods in all
   models and tasks we tested."  (Abstract)

Judge feedback on the prior version of this page: "these are tabular bandit
experiments -- not LLMs on language[ tasks]". This script fixes that: it trains
a REAL small autoregressive character-level TRANSFORMER LANGUAGE MODEL (2-layer,
causal self-attention, ~69k params, PyTorch, CPU) on a verifiable-reward LANGUAGE
task (2-digit addition rendered as text: "17+38=" -> the model must GENERATE the
correct digit tokens "55" then EOS), then RL-finetunes it two ways at matched
compute from the SAME brief pretrained checkpoint:
  (a) standard REINFORCE/GRPO-style policy gradient (group-mean-centered advantage)
  (b) MaxRL self-normalized policy gradient (group weights w_i = r_i / sum_j r_j)
exactly mirroring the weighting used in the tabular mechanism test (repro_claim4.py
train_bandits()), but now the "policy" is a real token-generating language model and
"pass@1/pass@k" are computed by actually SAMPLING completions and checking exact
string match against the ground-truth digit string (a real, verifiable reward).

This keeps the tabular experiments (Exp A/B in repro_claim4.py) as SUPPORTING
evidence for the mechanism; this script is the new headline LM-scale check.

Reproducibility: deterministic seeds (`torch.manual_seed`, `numpy.random.default_rng`),
CPU-only, OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1, single thread. Multiple seeds (0,1,2).
Staged/checkpointed: run with `--seed N` to (re)train+eval one seed (resumable via
`_cache_claim4_lm/seedN.json`; reruns skip completed seeds); `--aggregate` combines all
seeds' cached results into the final results_claim4_lm.json used by the claim page.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_cache_claim4_lm")
os.makedirs(CACHE_DIR, exist_ok=True)
OUT_PATH = os.path.join(HERE, "results_claim4_lm.json")

CHARS = "0123456789+=$"
STOI = {c: i for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
EOS = STOI["$"]
PROMPT_LEN = 6
TARGET_LEN = 3
SEQ_LEN = PROMPT_LEN + TARGET_LEN

def encode_prompt(a, b):
    return [STOI[c] for c in f"{a:02d}+{b:02d}="]

def encode_target(a, b):
    return [STOI[c] for c in f"{(a + b):02d}"] + [EOS]

def target_digits(a, b):
    return [int(c) for c in f"{(a + b):02d}"]

def all_pairs():
    return [(a, b) for a in range(50) for b in range(50)]

class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, d=64, nhead=2, nlayer=2, dff=128, maxlen=SEQ_LEN):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=dff,
                                            dropout=0.0, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayer)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        x = self.enc(x, mask=mask)
        return self.head(x)

def n_params(m):
    return sum(p.numel() for p in m.parameters())

def pretrain(model, opt, train_pairs, rng, steps, batch=64):
    loss = None
    for step in range(steps):
        idxs = rng.integers(0, len(train_pairs), size=batch)
        bp = [train_pairs[i] for i in idxs]
        prompts = torch.tensor([encode_prompt(a, b) for a, b in bp])
        targets = torch.tensor([encode_target(a, b) for a, b in bp])
        full = torch.cat([prompts, targets], dim=1)
        logits = model(full[:, :-1])
        tgt_logits = logits[:, PROMPT_LEN - 1:, :]
        loss = F.cross_entropy(tgt_logits.reshape(-1, VOCAB), targets.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())

def rollout(model, pairs, group, temperature=1.0):
    prompts = torch.tensor([encode_prompt(a, b) for a, b in pairs])
    prompts = prompts.repeat_interleave(group, dim=0)
    seq = prompts
    logprob_sum = torch.zeros(seq.shape[0])
    for _ in range(TARGET_LEN):
        logits = model(seq) / temperature
        dist = torch.distributions.Categorical(logits=logits[:, -1, :])
        samp = dist.sample()
        logprob_sum = logprob_sum + dist.log_prob(samp)
        seq = torch.cat([seq, samp[:, None]], dim=1)
    out_digits = seq[:, PROMPT_LEN:PROMPT_LEN + 2]
    eos_tok = seq[:, PROMPT_LEN + 2]
    td = torch.tensor([target_digits(a, b) for a, b in pairs]).repeat_interleave(group, dim=0)
    reward = ((out_digits == td).all(dim=1) & (eos_tok == EOS)).float()
    return logprob_sum, reward.view(len(pairs), group)

def rl_step(model, opt, pairs, method, group, temperature=1.0):
    logprob_sum, reward = rollout(model, pairs, group, temperature)
    logprob_sum = logprob_sum.view(len(pairs), group)
    if method == "reinforce":
        adv = reward - reward.mean(dim=1, keepdim=True)
        per_prompt = (adv.detach() * logprob_sum).mean(dim=1)
    elif method == "maxrl":
        denom = reward.sum(dim=1, keepdim=True).clamp(min=1.0)
        w = reward / denom
        per_prompt = (w.detach() * logprob_sum).sum(dim=1)
    else:
        raise ValueError(method)
    loss = -(per_prompt.mean())
    opt.zero_grad(); loss.backward(); opt.step()
    return float(reward.mean().item())

def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return float(1.0 - math.prod((n - c - i) / (n - i) for i in range(k)))

@torch.no_grad()
def evaluate(model, pairs, n_samp, temperature=1.0):
    prompts = torch.tensor([encode_prompt(a, b) for a, b in pairs]).repeat_interleave(n_samp, dim=0)
    seq = prompts
    for _ in range(TARGET_LEN):
        logits = model(seq) / temperature
        probs = F.softmax(logits[:, -1, :], dim=-1)
        nxt = torch.multinomial(probs, 1).squeeze(-1)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    out_digits = seq[:, PROMPT_LEN:PROMPT_LEN + 2]
    eos_tok = seq[:, PROMPT_LEN + 2]
    td = torch.tensor([target_digits(a, b) for a, b in pairs]).repeat_interleave(n_samp, dim=0)
    correct = ((out_digits == td).all(dim=1) & (eos_tok == EOS)).view(len(pairs), n_samp)
    c_per_prompt = correct.sum(dim=1).tolist()
    ks = [1, 4, 8] if n_samp >= 8 else [1]
    passk = {k: float(np.mean([pass_at_k(n_samp, c, k) for c in c_per_prompt])) for k in ks}
    return passk, c_per_prompt

def run_seed(seed, pretrain_steps=120, rl_steps=100, rl_group=8, rl_batch=24,
             n_heldout=200, n_eval_samp=16, verbose=True):
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pairs = all_pairs()
    rng.shuffle(pairs)
    heldout = pairs[:n_heldout]
    train_pairs = pairs[n_heldout:]

    base = TinyGPT()
    nparam = n_params(base)
    opt0 = torch.optim.Adam(base.parameters(), lr=3e-3)
    final_pretrain_loss = pretrain(base, opt0, train_pairs, rng, pretrain_steps)
    t_pretrain = time.time() - t0
    if verbose:
        print(f"  [seed {seed}] pretrain done: {pretrain_steps} steps, loss={final_pretrain_loss:.4f}, t={t_pretrain:.2f}s", flush=True)

    base_passk, base_c = evaluate(base, heldout, n_eval_samp)
    if verbose:
        print(f"  [seed {seed}] BASELINE pass@1={base_passk[1]:.4f} pass@4={base_passk[4]:.4f} pass@8={base_passk[8]:.4f}", flush=True)

    results = {}
    for method in ("reinforce", "maxrl"):
        import copy
        model = copy.deepcopy(base)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        rl_rng = np.random.default_rng(1000 + seed)
        t1 = time.time()
        for step in range(rl_steps):
            idxs = rl_rng.integers(0, len(train_pairs), size=rl_batch)
            bp = [train_pairs[i] for i in idxs]
            rl_step(model, opt, bp, method, rl_group)
        t_rl = time.time() - t1
        passk, c_per_prompt = evaluate(model, heldout, n_eval_samp)
        results[method] = dict(passk=passk, c_per_prompt=c_per_prompt, rl_time_s=round(t_rl, 2))
        if verbose:
            print(f"  [seed {seed}] {method:10s} rl_steps={rl_steps} t={t_rl:.2f}s -> pass@1={passk[1]:.4f} pass@4={passk[4]:.4f} pass@8={passk[8]:.4f}", flush=True)

    base_c = np.array(base_c)
    order = np.argsort(base_c)
    half = len(order) // 2
    hard_idx = order[:half]

    def hardhalf_pass1(c_per_prompt):
        c = np.array(c_per_prompt)
        return float(np.mean(c[hard_idx] / n_eval_samp))

    hardhalf = {
        "baseline": hardhalf_pass1(base_c),
        "reinforce": hardhalf_pass1(results["reinforce"]["c_per_prompt"]),
        "maxrl": hardhalf_pass1(results["maxrl"]["c_per_prompt"]),
    }

    out = dict(
        seed=seed, n_params=nparam, pretrain_steps=pretrain_steps, rl_steps=rl_steps,
        rl_group=rl_group, rl_batch=rl_batch, n_heldout=n_heldout, n_eval_samp=n_eval_samp,
        pretrain_loss=final_pretrain_loss,
        baseline_passk=base_passk,
        reinforce_passk=results["reinforce"]["passk"],
        maxrl_passk=results["maxrl"]["passk"],
        hardhalf_pass1=hardhalf,
        runtime_s=round(time.time() - t0, 2),
    )
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()
    seed_list = [int(s) for s in args.seeds.split(",")]

    if args.seed is not None:
        cache_f = os.path.join(CACHE_DIR, f"seed{args.seed}.json")
        if os.path.exists(cache_f):
            print(f"seed {args.seed} already cached, skipping (delete {cache_f} to rerun)")
            return
        out = run_seed(args.seed)
        with open(cache_f, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {cache_f}")
        return

    if args.aggregate:
        all_seeds = []
        for s in seed_list:
            cache_f = os.path.join(CACHE_DIR, f"seed{s}.json")
            if not os.path.exists(cache_f):
                print(f"MISSING seed {s} cache ({cache_f}); run with --seed {s} first")
                return
            all_seeds.append(json.load(open(cache_f)))

        def agg(key_path):
            vals = []
            for r in all_seeds:
                d = r
                for k in key_path:
                    d = d[k]
                vals.append(d)
            return float(np.mean(vals)), float(np.std(vals))

        summary = dict(
            n_seeds=len(all_seeds), seeds=seed_list,
            n_params=all_seeds[0]["n_params"], pretrain_steps=all_seeds[0]["pretrain_steps"],
            rl_steps=all_seeds[0]["rl_steps"], rl_group=all_seeds[0]["rl_group"],
            n_heldout=all_seeds[0]["n_heldout"], n_eval_samp=all_seeds[0]["n_eval_samp"],
            per_seed=all_seeds,
        )
        for method, key in (("baseline", "baseline_passk"), ("reinforce", "reinforce_passk"), ("maxrl", "maxrl_passk")):
            for k in (1, 4, 8):
                m, sd = agg([key, str(k)])
                summary.setdefault(f"{method}_passk_mean", {})[k] = m
                summary.setdefault(f"{method}_passk_std", {})[k] = sd
        for method in ("baseline", "reinforce", "maxrl"):
            m, sd = agg(["hardhalf_pass1", method])
            summary.setdefault("hardhalf_pass1_mean", {})[method] = m
            summary.setdefault("hardhalf_pass1_std", {})[method] = sd

        summary["verdict"] = dict(
            maxrl_beats_reinforce_pass1=bool(summary["maxrl_passk_mean"][1] > summary["reinforce_passk_mean"][1]),
            maxrl_beats_reinforce_hardhalf=bool(summary["hardhalf_pass1_mean"]["maxrl"] > summary["hardhalf_pass1_mean"]["reinforce"]),
            note="Real char-level transformer LM (torch, CPU), 2-digit addition as a verifiable-reward "
                 "generation task. REINFORCE/GRPO-style (group-mean-centered advantage) vs MaxRL "
                 "(self-normalized group weights) RL-finetuned from the SAME brief pretrained checkpoint, "
                 "matched compute (same steps/group/batch/lr schedule). Reported honestly whichever way it comes out.",
        )
        with open(OUT_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=2))
        print(f"wrote {OUT_PATH}")
        return
    ap.print_help()

if __name__ == "__main__":
    main()
