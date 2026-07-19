# Row-Stochastic vs Doubly-Stochastic Mixing in Decentralized Learning — Reproduction (GAQE4Wr53f)

arXiv 2511.19513 ("Row-Stochastic Matrices Can Provably Outperform Doubly Stochastic Matrices in Decentralized Learning"). Independent NumPy/scipy implementation from scratch (no paper code, no datasets); CPU-only, deterministic (`numpy.random.default_rng`), 18 synthetic graph configs, ~2 s runtime.

## Claim (verified target)

Mechanism + rate. On small graphs the paper predicts: (a) the modified Metropolis-Hastings **row-stochastic** `W` (Eq. 3) satisfies *exact* weighted detailed balance `lambda_i W_ij = lambda_j W_ji` (so `W` is self-adjoint in the weighted inner product `<x,y>_lam = sum_i lambda_i x_i y_i`, Sec. 5 / Lemma B.5), whereas the standard-MH **doubly-stochastic** `W^ds` violates it (O(1) residual) under the same non-uniform `lambda`; (b) as a consequence the `lambda`-weighted consensus-error transient of `W` contracts as `rho_Lambda^t` with prefactor **exactly 1** (Thm 6.5), while `W^ds`'s weighted transient is inflated by up to `kappa_lam = sqrt(lambda_max/lambda_min) > 1`; (c) `lambda^T W = lambda^T`, and on degree-matched `lambda_i propto d_i` (Cor. 7.3 optimal design) Thm 7.1's sufficient condition holds and the row-stochastic weighted error decays strictly faster — a *conditional* advantage that need not hold for adversarial random `lambda`.

Paper target locations: Eq. (3) construction; weighted inner product (Sec. 5); Lemma B.5; Theorem 6.5; Theorem 7.1; Corollary 7.3.

## Result

**OVERALL: PASS** — all headline numbers (claims a, b, and stationarity) match the paper targets; the conditional claim (c) reproduces the paper's conditional statement. Numbers below are aggregated over the 18 configs (n in {10,20,50} x {Erdos-Renyi, ring, star} x {random-lambda, degree-matched}) and are copied verbatim from the run's stdout.

| Quantity | Paper target | Measured | Match |
|---|---|---|---|
| Row-stoch `W` detailed-balance residual `max_ij|lambda_i W_ij - lambda_j W_ji|` (max over cfgs) | ~1e-12 (exact, =0) | **1.11e-16** | yes |
| Doubly-stoch `W^ds` DB residual under same `lambda` (min over non-uniform cfgs) | O(1), > 0 | **2.59e-02** | yes |
| Self-adjointness `norm(D_lam W - W^T D_lam, fro)` (max) | ~1e-12 (=0) | **5.50e-16** | yes |
| Stationarity `max|lambda^T W - lambda^T|` (max) | ~1e-12 (=0) | **7.11e-15** | yes |
| Row-stoch `lambda`-weighted transient prefactor (all 18 cfgs) | = 1 (no `kappa_lam`) | **1 + 5.0e-14** | yes |
| Doubly-stoch `lambda`-weighted transient prefactor (non-uniform cfgs) | > 1, up to `kappa_lam` | **12/15 inflated, max 1.491, all <= kappa_lam** | yes |
| Real executed run from worst-case `x0`: `W` / `W^ds` prefactor | <= 1 / > 1 | **1.000 / 1.456** | yes |
| Thm 7.1 condition holds on degree-matched (non-uniform) cfgs | holds | **6/6** | yes |
| `rho_Lambda < rho_J` (row-stoch decays strictly faster) on degree-matched | holds | **6/6** | yes |
| Thm 7.1 / faster on random-`lambda` cfgs | *conditional* — may fail | **2/9 (does not always hold)** | yes (matches conditional) |

Reading of the mechanism: the row-stochastic `W` has a detailed-balance residual at machine precision (~1e-16) and a weighted transient prefactor pinned to 1 across every graph, exactly because it is self-adjoint in the `lambda`-inner-product. The doubly-stochastic `W^ds` breaks detailed balance by O(1e-2 to 1e-1) under the same weights, and its weighted transient is genuinely inflated (operator-norm prefactor up to 1.491, and a real consensus run started from the worst-case initial error realizes a prefactor of 1.456) — always staying within the theoretical `kappa_lam` envelope. On degree-matched topologies the row-stochastic scheme additionally has the smaller weighted contraction rate (`rho_Lambda < rho_J`) in all 6 non-trivial cases; on random weights it does not, matching the paper's *conditional* superiority claim.

## Scope

Honest statement of what was simplified vs. the full paper:

- **Theory mechanism, not full training.** This reproduces the concrete *linear-algebra* predictions that underpin the paper's theory (mixing-matrix construction, weighted-inner-product self-adjointness, contraction rates and transient prefactors, and the Thm 7.1 / Cor 7.3 spectral conditions). It does **not** run the paper's decentralized deep-learning experiments — no neural networks, no real datasets, no DSGD/gradient-tracking optimization. The claim verified is the "provably" (mechanism + rate) part, on synthetic graphs up to n = 50.
- **Constructions.** Row-stochastic `W` = modified Metropolis-Hastings, Eq. (3), off-diagonal `W_ij = ((1-eps)/d_i)*min(1, (lambda_j d_i)/(lambda_i d_j))`, `eps = 0.1`; doubly-stochastic `W^ds` = standard Metropolis-Hastings `W^ds_ij = 1/(1+max(d_i,d_j))`. The `eps=0.1` laziness is part of the modified-MH construction and applies only to `W`; `W^ds` uses the textbook MH weights. The detailed-balance and prefactor claims (a, b) are exact and independent of this choice; the `rho_Lambda < rho_J` comparison (c) is affected by it, and is reported as the paper's conditional result.
- **Transient prefactor** is the worst-case (sup-over-initialization) constant. It is measured rigorously as `sup_t norm(M^t)_lam / rho^t` (operator norm via SVD of the actual matrix powers), and separately corroborated by a real executed consensus iteration started from the theory's worst-case initial-error direction. A single random `x0` does not excite the worst case and gives prefactor <= 1 for both matrices — expected, not a contradiction.
- **Theorem 7.1 inequality** is evaluated in the exact form `1 - rho_Lambda >= max{(1+eta)*kappa_lam^(-1/3), lambda_max^(-1/2)}*(1 - rho_J)` with `eta = 1.8e-3`, as specified in the vetted plan; "strictly faster" is measured via `rho_Lambda < rho_J` plus the prefactor gap.
- **Degree-matched on regular graphs** (the ring) yields uniform `lambda`, so `W = W^ds` and `kappa_lam = 1` — no gap, correctly flagged `n/a`; consistent with the theory (the row-stochastic advantage requires degree heterogeneity).

## Rerun

```
cd wave-local/out/GAQE4Wr53f && python3 repro.py
```

Requires numpy + scipy only. Deterministic; writes `evidence.json`. Runtime ~2 s.
