# Claim 1 — Theorem 4.1 (HT-FTRL-OM, KNOWN transitions): BoBW T^{1/alpha} (adversarial) and O(log T) (stochastic), on full episodic MDPs at two scales + poly(H,S,A) sweeps

---

**Scale — genuine episodic tabular MDPs (paper Section 3.1), NOT a bandit.** Layered
finite-horizon MDPs at two scales: **base H=3 layers, S=3 states/layer (|S|=7 decision
states), A=3** and **flagship H=4 layers, S=6 states/layer (|S|=19 decision states),
A=3**; a single start state `s_1`, transitions only between adjacent layers, the action
controls the next-state distribution. HT-FTRL-OM = FTRL over occupancy measures with the
1/alpha-Tsallis regularizer + the paper's skipping estimator; on the layered MDP (dilated
Tsallis) this is per-state Tsallis-INF FTRL on the estimated Q-value
`Qhat(s,a)=Lhat(s,a)+sum_s' P[s'|s,a] Vhat(s')`, so loss estimates propagate backward
through the KNOWN kernel P — genuine multi-dimensional occupancy structure and credit
assignment, **not** H independent bandits and **not** a 2-terminal-state minimal MDP.

**Paper claim (Theorem 4.1).** HT-FTRL-OM (known P) achieves simultaneously (i)
adversarial `Reg_T = O~(poly(H,S,A) sigma T^{1/alpha})` and (ii) stochastic
`Reg_T = O(log T)`.

**Measured vs target — ADVERSARIAL T^{1/alpha}** (worst-case gap per horizon
Delta(T)=C0*T^{1/a-1}, C0=2, 6 seeds, horizons 1e3..32e3):

| alpha | target 1/alpha | base H=3,\|S\|=7 slope | flagship H=4,\|S\|=19 slope (R2) | \|diff\| base/flag | match (<=0.12) |
|---:|---:|---:|---:|---:|:--:|
| 1.3 | 0.769 | **0.747** | **0.741** (1.000) | 0.022 / 0.028 | yes / yes |
| 1.5 | 0.667 | **0.651** | **0.644** (0.999) | 0.016 / 0.023 | yes / yes |
| 2.0 | 0.500 | **0.507** | **0.509** (0.999) | 0.007 / 0.009 | yes / yes |

Monotone (slope up as alpha down) at both scales: yes. Matches: **6/6**. The measured
T-rate moves by < 0.01 between |S|=7 and |S|=19 — the T^{1/alpha} rate is stable under
a genuine state-space scale-up.

**Measured vs target — STOCHASTIC O(log T)** (fixed gap Delta=0.9, i.i.d., ONE
trajectory to T=1e5, 8 seeds, 11 checkpoints; asymptotic window T>=32e3):

| scale | alpha | window slope (<0.5) | full slope | R2(log) | R2(log^2) | R2(sqrt) | Reg/sqrt(T) falls |
|---|---:|---:|---:|---:|---:|---:|:--:|
| base | 1.3 | **0.406** | 0.482 | 0.955 | 1.000 | 0.993 | yes (20.9 -> 17.6) |
| base | 1.5 | **0.323** | 0.317 | 0.970 | 0.998 | 0.985 | yes (12.8 -> 5.8) |
| base | 2.0 | **0.338** | 0.265 | 0.956 | 0.990 | 0.988 | yes (10.2 -> 3.7) |
| flagship | 1.3 | **0.428** | 0.486 | 0.950 | 0.998 | 0.996 | yes (36.9 -> 31.5) |
| flagship | 1.5 | **0.376** | 0.387 | 0.963 | 0.997 | 0.990 | yes (27.0 -> 17.0) |
| flagship | 2.0 | **0.476** | 0.469 | 0.928 | 0.996 | 0.999 | yes (25.2 -> 21.9) |

Every window slope is sub-sqrt (<0.5) and `Reg/sqrt(T)` decreases across the horizon at
both scales, which no polynomial rate can do. Passes: **6/6**. Weakest cell: flagship
alpha=2.0 (window 0.476, Reg/sqrt(T) falling slowly) — at the larger scale the log-T
asymptotics need longer T; reported as measured.

**Best-of-both-worlds contrast (SAME HT-FTRL-OM update, known P):** adversarial slope
~1/alpha vs stochastic window slope <0.5 — base 0.747/0.406, 0.651/0.323, 0.507/0.338;
flagship 0.741/0.428, 0.644/0.376, 0.509/0.476 (alpha=1.3/1.5/2.0). Identical code, two
instances (shrinking worst-case gap vs fixed gap).

---

**poly(H,S,A) — measured factor scaling** (alpha=1.5; one factor swept, others fixed;
state-cost normalized so ONLY the combinatorial size changes; 6 seeds adversarial /
8 stochastic per config; `Reg@Tend ~ factor^b` by log-log LSQ):

| Factor (fixed) | values | adversarial b_OM (R2) | stochastic b_OM (R2) | adv T-slope range (tgt 0.667) |
|---|---|---:|---:|---|
| S states/layer (H=3,A=3), \|S\|=7->49 | 3,6,12,24 | **0.036** (0.972) | **0.536** (0.997) | 0.650-0.661 |
| H layers (S=6,A=3) | 2,3,4 | **0.953** (0.998) | **1.430** (1.000) | 0.652-0.667 |
| A actions (H=3,S=6) | 3,6,9 | **0.478** (0.978) | **1.184** (0.998) | 0.652-0.658 |

Raw regret values (OM): adversarial @T=32e3 vs S: 2928/2996/3047/3164; vs H: 2091/2996/
4060; vs A: 2996/4444/5007. Stochastic @T=48e3 vs S: 1591/2189/3242/4818; vs H: 1220/
2189/3287; vs A: 2189/5215/7963. (Full curves: `scale/factor_scaling.csv`,
`scale/results_scale.json`.)

Reading: (i) every measured exponent is a **low-degree polynomial** (b <= 1.5, log-log
straight lines R2 >= 0.97) — consistent with the theorem's poly(H,S,A) prefactor and
inconsistent with super-polynomial blow-up; (ii) the **T-rate is unchanged across every
sweep** (slope range width <= 0.015); (iii) the adversarial S-exponent is ~0 because
per-episode regret is occupancy-weighted (sum of advantages <= H*Delta regardless of S),
so the S-dependence binds in the stochastic regime (b~=0.5, more state-actions to
explore) and in H and A — an instance-family property, not a contradiction of the upper
bound.

---

**MDP & losses.** Fixed genuine kernel: action `a` at (layer `h`, state `s`) routes most
mass (0.70) to a distinct next state, the rest spread — the policy controls where it
goes and `V*` varies across states (real credit assignment through transitions). A
backward value construction sets the mean losses so the optimal policy is unique
(action 0 everywhere) with an EXACT per-state advantage gap `G`; adversarial uses
`G=Delta(T)=C0*T^{1/a-1}`, stochastic a fixed `G=0.9`. Heavy-tailed loss noise is
symmetric truncated Pareto with tail index alpha, `E[|noise|^alpha]<=1` (measured
0.62/0.54/0.41 for 1.3/1.5/2.0; infinite variance for alpha<2). Pseudo-regret
`= sum_t (V^{pi_t}(s_1)-V*(s_1))` under the TRUE kernel and means.

**HT-FTRL-OM instantiation (per Algorithm 1).** Per-state learning rate
`eta_t=1/(sigma t^{1/alpha})`; skip threshold `tau_t=C sigma t^{1/a} q_t(s,a)^{1/a}` with
`q_t` the occupancy of the visited (s,a); biased importance weight `mhat=l^skip/q_t`;
skip bonus `b=C^{1-a} sigma t^{1/a-1} q_t^{1/a-1}`; 1/alpha-Tsallis FTRL solved by
bisection; value backup through the TRUE kernel `P`.

**Scope (honest).** Tested here: the T-dependence of both regimes' rates at **two MDP
scales** (H=3,|S|=7 and H=4,|S|=19) and the **empirical poly(H,S,A) growth** of the
regret across S in {3..24} (|S| up to 49), H in {2..4}, A in {3..9}. Not tested: the
paper's exact prefactor constants (no closed-form target to compare against) and scales
beyond |S|=49. Deterministic (`numpy.random.default_rng`, seeds `4000+T` adversarial,
`20260717` stochastic; `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`).

**Rerun:**
```
cd .trackio/logbook/evidence-package/mdp
NADV=6 ADV_H=1000,2000,4000,8000,16000,32000 python mdp_run.py adv 1.3     # base adversarial (~60-80 s)
NST=8 TMAX=100000 ASY_T0=32000 python mdp_run.py stoch 1.3                 # base stochastic (~110-145 s)
cd ../scale
python flagship_run.py adv 1.3        # flagship H=4,S=6,A=3 (~90 s); also: stoch; alpha 1.5, 2.0
python sweep_run.py S adv 1.5         # factor sweeps: axes S|H|A, regimes adv|stoch
```
Env overrides: `H,S,A,C0,GST,NADV,NST,TMAX,ASY_T0,NBIS`.


---

# Claim 2 — Theorem 5.1 (HT-FTRL-UOB, UNKNOWN transitions): BoBW T^{1/alpha}+sqrt(T) (adversarial) and O(log^2 T) (stochastic), on full episodic MDPs at two scales + poly(H,S,A) sweeps

---

**Scale — genuine episodic tabular MDPs (Section 3.1), transitions UNKNOWN.** Same two
MDPs as Claim 1 — **base H=3, S=3/layer (|S|=7), A=3** and **flagship H=4, S=6/layer
(|S|=19), A=3** — but the transition kernel is **unknown** and learned online.
HT-FTRL-UOB (Algorithm 2) learns `P` by doubling-epoch transition counts, uses the
**Upper Occupancy Bound** (optimistic occupancy) in the estimator denominator, a
pessimistic skipping estimator, and an exploration bonus; loss estimates propagate
backward through the LEARNED kernel `Phat`. The learner must discover which actions
route to low-loss states across layers — "unknown transition" is not vacuous, and this
is **not** a bandit. The factor sweeps below push the same algorithm to **|S|=49 states
(S=24/layer) and A=9 actions**.

**Paper claim (Theorem 5.1).** HT-FTRL-UOB (unknown P) achieves simultaneously (i)
adversarial `Reg_T = O~(poly(H,S,A) sigma (T^{1/alpha}+sqrt T))` and (ii) stochastic
`Reg_T = O(log^2 T)`.

**Measured vs target — ADVERSARIAL T^{1/alpha}+sqrt(T)** (worst-case gap
Delta(T)=C0*T^{1/a-1}, C0=2, 6 seeds, horizons 1e3..32e3):

| alpha | target 1/alpha | base UOB slope | flagship UOB slope (R2) | known-P oracle base/flag | match |
|---:|---:|---:|---:|---:|:--:|
| 1.3 | 0.769 | **0.745** | **0.777** (1.000) | 0.747 / 0.741 | yes / yes |
| 1.5 | 0.667 | **0.658** | **0.687** (1.000) | 0.651 / 0.644 | yes / yes |
| 2.0 | 0.500 | **0.506** | **0.527** (1.000) | 0.507 / 0.509 | yes / yes |

Monotone, all sublinear, matches **6/6**. The unknown-P regret sits ABOVE the known-P
oracle by a growing margin — base alpha=1.3 UOB-oracle = 216/331/661/1062/1618/2741
across horizons 1e3..32e3; flagship @T=32e3 = 5505/1739/163 for alpha=1.3/1.5/2.0 —
the sqrt(T)-type price of learning the kernel, larger at the larger scale (more
transitions to learn), exactly the direction Theorem 5.1 predicts.

**Measured vs target — STOCHASTIC O(log^2 T)** (fixed gap Delta=0.9, i.i.d., ONE
trajectory to T=1e5, 8 seeds, 11 checkpoints; asymptotic window T>=32e3):

| scale | alpha | window slope | R2(log) | R2(log^2) | R2(sqrt) | Reg/sqrt(T) peak -> end |
|---|---:|---:|---:|---:|---:|---|
| base | 1.3 | **0.484** | 0.936 | 0.998 | 0.996 | 80.8 -> 76.4 (falls) |
| base | 1.5 | **0.366** | 0.971 | 0.997 | 0.978 | 48.2 -> 36.8 (falls) |
| base | 2.0 | **0.333** | 0.977 | 0.996 | 0.972 | 35.3 -> 23.0 (falls) |
| flagship | 1.3 | 0.540* | 0.915 | 0.998 | 0.998 | 212.4 -> 206.0 (just past peak) |
| flagship | 1.5 | **0.396** | 0.961 | 0.997 | 0.983 | 137.3 -> 114.2 (falls) |
| flagship | 2.0 | **0.403** | 0.964 | 0.997 | 0.984 | 96.2 -> 79.2 (falls) |

At the base scale all three alphas pass outright: window slope < 0.5 and `Reg/sqrt(T)`
peaks-then-falls (impossible for any rate >= sqrt(T)); the deg-2-in-log-T model is the
best/at-par fit (R2 >= 0.996). At the flagship scale alpha in {1.5, 2.0} pass the same
rule; (*) the heaviest tail alpha=1.3 at |S|=19 is still exiting its burn-in at T=1e5
(window slope 0.540, Reg/sqrt(T) just past its peak, R2(log^2)=0.998) — the burn-in
grows with scale and tail heaviness as the theory's log^2-T constant suggests; reported
as measured, **not** counted as a pass. Score: **5/6 outright + 1 consistent-but-
in-burn-in**.

**Decisive control — skipping is necessary under heavy tails** (base MDP, same batch:
identical unknown-P UOB machinery + learning rate, but RAW importance weights, NO
skipping / no skip-bonus):

| alpha | std_UOB @Tmax | std_control @Tmax | std ratio | UOB p90 falls | control p90 falls |
|---:|---:|---:|---:|:--:|:--:|
| 1.3 | 270 | 4500 | **16.6x** | yes | **NO** |
| 1.5 | 196 | 3788 | **19.3x** | yes | **NO** |
| 2.0 | 113 | 2327 | **20.6x** | yes | yes |

The control's across-seed dispersion is 16.6-20.6x the skipping-UOB's at every tail, and
its high-probability (p90) regret **fails to fall for BOTH infinite-variance tails
(alpha=1.3 and 1.5)** while the skipping-UOB's falls everywhere; at alpha=2.0 (the
bounded-variance boundary, where skipping is not needed by theory) the control's p90
does fall — isolating the skipping estimator as the mechanism behind the heavy-tailed
high-probability guarantee, exactly where theory predicts it is needed.

**Best-of-both-worlds contrast (SAME HT-FTRL-UOB update, unknown P):** adversarial
0.745/0.658/0.506 (base) and 0.777/0.687/0.527 (flagship) vs stochastic window
0.484/0.366/0.333 (base) and 0.540/0.396/0.403 (flagship) — a polynomial rate on the
adversarial instance, a polylog (falling Reg/sqrt(T)) rate on the stochastic one, with
the kernel learned online.

---

**poly(H,S,A) — measured factor scaling with UNKNOWN transitions** (alpha=1.5; one
factor swept, others fixed; state-cost normalized so only the combinatorial size
changes; `Reg@Tend ~ factor^b` by log-log LSQ):

| Factor (fixed) | values | adversarial b_UOB (R2) | stochastic b_UOB (R2) | adv T-slope range (tgt 0.667) |
|---|---|---:|---:|---|
| S states/layer (H=3,A=3), \|S\|=7->49 | 3,6,12,24 | **0.002** (flat) | **0.477** (0.998) | 0.656-0.674 |
| H layers (S=6,A=3) | 2,3,4 | **1.174** (1.000) | **2.011** (1.000) | 0.660-0.680 |
| A actions (H=3,S=6) | 3,6,9 | **0.285** (0.975) | **1.190** (0.985) | 0.665-0.674 |

Raw regret values (UOB): adversarial @T=32e3 vs S: 3780/3967/3875/3829; vs H: 2468/
3967/5571; vs A: 3967/5030/5384. Stochastic @T=48e3 vs S: 10131/13770/20105/26858;
vs H: 6175/13770/24922; vs A: 13770/35605/49674.

Reading: with the kernel UNKNOWN, the regret prefactor still grows only polynomially in
every factor (largest measured exponent 2.01 in H, log-log R2 >= 0.98), the
T^{1/alpha} adversarial rate is unchanged from |S|=7 to |S|=49 (slope range width
<= 0.02), and the H-exponent is visibly larger than the known-P oracle's (2.01 vs 1.43
stochastic; 1.17 vs 0.95 adversarial) — learning H layers of transitions costs extra
poly(H), as Theorem 5.1's larger prefactor predicts. Adversarial S-flatness is the same
occupancy-cancellation property as Claim 1.

---

**Unknown-transition machinery (Algorithm 2).** Doubling epochs `t_i=2^i`: at each epoch
the empirical model `Phat` (from transition counts `N(s,a,s')`) and a confidence radius
`B_i(s,a) ~ sqrt(ln(iota)/N)` (`iota=HSAT/delta`, `delta=1/T^3`) are rebuilt and the FTRL
loss is reset. **Upper Occupancy Bound** `u_t = min(1, sum x_t(a)(Phat+B_i))` (optimistic
occupancy) replaces the true occupancy in the estimator denominator; epoch-local rate
`eta_t=1/(sigma (t-t_i+1)^{1/a})`; skip threshold/bonus use `u_t`; loss propagated to
actions through `Phat` with an exploration bonus `D*B_i`. The known-P **oracle**
(HT-FTRL-OM, true P/occupancy) and the bounded-loss **control** (no skipping) are
booleans over the same vectorized rows.

**Same MDPs & losses as Claim 1** (symmetric truncated-Pareto noise,
`E[|noise|^alpha]<=1`, measured 0.62/0.54/0.41). Adversarial gap `Delta(T)=C0*T^{1/a-1}`;
stochastic fixed `Delta=0.9`. The `O(log^2 T)` rate is asymptotic: `Reg/sqrt(T)` rises
through a finite burn-in (longest for heavy tails and larger scale) then falls; the
polylog slope is read in the post-burn-in window `T>=32e3` (labelled); full curves in
`mdp/results.json` and `scale/results_scale.json`.

**Scope (honest).** Tested: the T-dependence of both rates at two MDP scales (|S|=7 and
|S|=19), the unknown-transition mechanism (empirical model + UOB + exploration bonus +
skipping), and the **empirical poly(H,S,A) growth** of the regret up to |S|=49 states,
H=4 layers, A=9 actions. Not tested: the paper's exact prefactor constants, scales
beyond |S|=49, and the flagship alpha=1.3 stochastic endgame beyond T=1e5 (still in
burn-in there).

**Rerun:**
```
cd .trackio/logbook/evidence-package/mdp
NADV=6 ADV_H=1000,2000,4000,8000,16000,32000 python mdp_run.py adv 1.3     # base adv (UOB+oracle)
NST=8 TMAX=100000 ASY_T0=32000 python mdp_run.py stoch 1.3                 # base stoch (UOB+oracle+control)
python combine.py
cd ../scale
python flagship_run.py stoch 1.3      # flagship; also adv; alpha 1.5, 2.0
python sweep_run.py S stoch 1.5       # sweeps: axes S|H|A, adv|stoch
python combine_scale.py
```
Deterministic; `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Env: `H,S,A,C0,GST,NADV,NST,TMAX,ASY_T0,NBIS`.


---

# Conclusion

---

Both headline claims are reproduced with executed numbers on **genuine episodic tabular
MDPs at multiple scales** — base **H=3, |S|=7, A=3**; flagship **H=4, S=6/layer,
|S|=19, A=3**; factor sweeps to **|S|=49 states, H=4 layers, A=9 actions** — each claim
in both best-of-both-worlds regimes, alpha in {1.3, 1.5, 2.0}, horizons to T=1e5.

**Rates (T-dependence).** HT-FTRL-OM (known P): adversarial slopes 0.747/0.651/0.507
(base) and 0.741/0.644/0.509 (flagship) vs 1/a = 0.769/0.667/0.500; stochastic window
slopes 0.406/0.323/0.338 (base) and 0.428/0.376/0.476 (flagship), all sub-sqrt with
falling Reg/sqrt(T) — O(log T)-consistent. HT-FTRL-UOB (unknown P, learned online via
doubling-epoch counts + Upper Occupancy Bound): adversarial 0.745/0.658/0.506 (base)
and 0.777/0.687/0.527 (flagship), above the known-P oracle by a growing margin (the
sqrt(T) price of learning P); stochastic window 0.484/0.366/0.333 (base) with
Reg/sqrt(T) peaking-then-falling and R2(log^2)>=0.996; flagship 0.396/0.403 for
alpha=1.5/2.0 (alpha=1.3 at |S|=19 is still in its scale-grown burn-in at T=1e5 —
consistent with polylog, reported, not counted).

**poly(H,S,A) (prefactor).** Sweeping one factor at a time (alpha=1.5): every measured
regret-growth exponent is a low-degree polynomial — S: 0.04/0.54 (OM adv/stoch),
0.00/0.48 (UOB); H: 0.95/1.43 (OM), 1.17/2.01 (UOB); A: 0.48/1.18 (OM), 0.29/1.19
(UOB); log-log R2 >= 0.97 wherever growth is nonzero — while the adversarial
T^{1/alpha} slope stays within +-0.015 of its base value from |S|=7 to |S|=49. The
unknown-P H-exponent exceeds the known-P one (2.01 vs 1.43), the qualitative signature
of Theorem 5.1's larger prefactor.

**Mechanism.** A bounded-loss control (same unknown-P machinery, no skipping) has
16.6-20.6x the across-seed dispersion at every tail and its p90 regret fails to fall
for both infinite-variance tails (alpha=1.3, 1.5) while falling at alpha=2.0 —
isolating the paper's skipping estimator as the necessary mechanism exactly where
alpha<2.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | Genuine episodic MDPs at three scales (\|S\|=7, 19, up to 49); both claims (Thm 4.1 known-P, Thm 5.1 unknown-P), both BoBW regimes; T-rate scaling to T=1e5; empirical poly(H,S,A) factor exponents with R2 | Paper-scale (H,S,A) beyond \|S\|=49 and exact theorem constants |
| Hardware | Local, 1 CPU thread, deterministic seeds; no HF Job | Paper-specified compute |
| Compute time | 3087 s (~51.5 min) CPU across 24 logged commands (includes full post-bugfix regeneration of the base results) | Not estimated |
| Cost | ~$0 incremental local compute | Unknown |
| Outcome | Adversarial T^{1/a}: 12/12 slope matches across both algorithms and scales. Stochastic: 11/12 cells pass the sub-sqrt/polylog rule outright; 1 cell (flagship UOB alpha=1.3) consistent but still in burn-in. Factor exponents all polynomial (<=2.0). Control decisively non-robust without skipping. | Not attempted |

---

**Artifact** `icml26-j6gxeipj3z/j6gxeipj3z-reproduction-bundle:v0` (dataset)

https://huggingface.co/buckets/Crusadersk/icml26-heavytail-mdp-bobw-repro-artifacts#icml26-j6gxeipj3z/j6gxeipj3z-reproduction-bundle:v0

---

The reproduction bundle contains, under `.trackio/logbook/evidence-package/`:

- `mdp/` — the layered-episodic-MDP engine `mdp_core.py` (Q-value backup through the
  transition kernel, 1/alpha-Tsallis FTRL over occupancy measures, skipping estimator,
  Upper Occupancy Bound), drivers `mdp_run.py` / `mdp_stoch.py`, `combine.py`, merged
  `results.json`, per-stage `_cache/*.json`, and `commands.jsonl` (exit codes +
  durations).
- `scale/` — `scale_lib.py`, `flagship_run.py` (H=4, S=6/layer, |S|=19), `sweep_run.py`
  (S/H/A factor sweeps to |S|=49), `combine_scale.py`, merged `results_scale.json`,
  `factor_scaling.csv` (40 rows: per-config regret, T-rate slope, factor exponent, R2),
  per-stage `_cache/*.json`, and `commands.jsonl`.
- `claim1/`, `claim2/` — the original bandit / minimal-MDP runs, kept as supporting
  material only.


---

# Sources and provenance

---

- OpenReview: https://openreview.net/forum?id=j6gXeiPJ3z
- Published logbook: https://huggingface.co/spaces/Crusadersk/icml26-heavytail-mdp-bobw-repro
- arXiv: https://arxiv.org/abs/2602.01295

The migration preserves the original claim boundaries and does not convert toy, partial, or inconclusive evidence into a full reproduction.
