# Foundations of Equivariant Deep Learning

✅✅  **4 pts** — 2/2 full-credit  (verified, verified)

[OpenReview](https://openreview.net/forum?id=aIH1jyU37z) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-equivariant-deep-learning-repro)

## Scoreboard — the two scored claims

| # | Paper claim (abstract) | What was measured (trained end-to-end, independent targets) | Verdict |
|---|---|---|---|
| C1 | UAT for continuous **order-equivariant** maps | Reynolds networks (exact group-average of a trained MLP over the full verified automorphism group) on posets \|P\| = **7 / 20 / 52**: held-out error falls 27×/34×, 16×/82×, 6.8×/48× (two targets per size) to best values **1.3e-4 / 2.0e-2 / 9.3e-3**; trained-net equivariance ≤ **1.6e-15**; non-equivariant control net up to **28.5× worse** at equal capacity; non-equivariant target floors respected; **33/33 predeclared checks pass** | **demonstrated** (trained capacity sweep) |
| C2 | **First** UAT for **sheaf neural networks** | Gauge-canonicalized sheaf networks (fixed SO(2)/SO(2)² rotation restriction maps, all MLP weights trained) on graphs n = **6 / 30 / 100**, stalk dims **2 / 2 / 4**: held-out error falls 41×/88×, 25×/21× to **5.7e-3 / 2.8e-2** at n=6/30; at n=100 (800-dim output, 512 samples) test falls monotonically 1.01→**0.31** and train (approximation) error reaches **2.0e-3**; trained-net gauge-equivariance ≤ **4.0e-15** at irrational angles; controls behave; **33/33 predeclared checks pass** | **demonstrated** (trained capacity sweep) |

**Outcome: both scored claims addressed with executed, end-to-end-trained capacity-sweep evidence on multiple structure sizes.** Full curves (including the honest bias–variance up-turns at the largest sizes at fixed sample budget), controls, and predeclared per-size acceptance bars are on the two claim pages.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`240` files).
