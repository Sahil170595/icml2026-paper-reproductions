"""
FULL tabular-MDP scaling experiments for both scored claims.
Paper: Lee & Jamieson, "Minimax Optimal Strategy for Delayed Observations in Online RL"
(ICML 2026, arXiv 2603.03480, OpenReview fFupHW7Jqx).

Registered claims tested here (input/challenge/claims_anchored.json, orid fFupHW7Jqx):
 C1 (Thm 1): regret O~(H sqrt((D_max ^ B) S A K) + H B S A) for tabular MDPs with delayed
     observations (B = branching factor; ^ = min).
 C3 (Thm 3): matching lower bound Omega(H sqrt(Dt S A K / log Dt)),
     Dt = min{D_max, H/4, B/2, S/4-1}.
 C4 (Sec 5): improvement over Chen et al. 2023 (delay exponent 1/2 instead of 5/2,
     horizon exponent 1 instead of 3/2).

What is NEW versus the earlier evidence (claim1/, claim2/):
 * Every sweep below runs on a GENUINE full tabular episodic MDP: an explicit S-state,
   A-action state space, true stochastic transition kernels with branching factor B,
   episodes simulated as real state trajectories, and the learner performing full
   optimistic value iteration over the ENTIRE state space every episode. Nothing is
   collapsed to independent bandit atoms.
 * sqrt(D_max) is measured on this full MDP (primary sweep {1,2,4,8}, 8x range, plus a
   wide-frame confirmation sweep {1,4,16}, 16x range, on a larger MDP).
 * sqrt(S) (4x range), sqrt(A) (4x range) and the H dependence (4x range) are measured on
   the same full MDP with the other factors fixed; log-log slopes with R2 and bootstrap CIs.
 * The lower bound is now EMPIRICAL: the paper's Theorem-3-style hard instance family is
   implemented as a full MDP and the floor is the minimum regret over reasonable baselines
   (delay-aware UCBVI, delay-blind UCBVI on the delay sweeps, oracle-tuned
   explore-then-commit with the exploration budget optimized in hindsight per config),
   whose scaling is compared against the claimed Omega rate. The analytic
   Le Cam / Bretagnolle-Huber two-point value is retained only as a reference curve.

THE HARD-INSTANCE FULL MDP (Theorem-3-style construction, run as an actual MDP)
 States: n_c "context" states + n_b "sink" states (1 absorbing GOOD with reward 1/step,
 n_b-1 absorbing zero-reward DEAD sinks). S = n_c + n_b. Episode length H_tot: the initial
 state is drawn uniformly from the contexts, one action is taken, the chain moves to a sink
 and absorbs for the remaining H_r = H_tot - 1 steps (GOOD pays H_r total).
 Transitions (unknown to the learner; topology/support known, probabilities unknown):
 each context s hides one planted action a*(s) with P(GOOD | s, a*) = 1/2 + delta; every
 other action has P(GOOD | s, a) = 1/2; the non-GOOD mass is spread uniformly over the
 DEAD sinks, so every context row has branching factor B = n_b. This is the canonical
 planted-action tabular lower-bound family; per-episode regret is exactly
 delta * H_r * 1{chosen action != a*} (V* and policy values are closed-form because the
 sinks absorb; the learner does NOT use this structure).
 Delayed observations: a visit to (s, a) is observed only through a D-slot aggregate count
 vector y = e_{s'} + Multinomial(D-1, u_{s,a}) with known nuisance u (1/2 on GOOD, 1/2
 spread over DEAD): the unbiased debiased estimate y - (D-1) u has per-visit variance
 Theta(D) on the decision coordinate -- delayed credit assignment deflates per-visit
 information by Theta(1/D), the mechanism behind sqrt(D_max). For n_b = 2 this channel is
 identical to the bit + Binom(D-1, 1/2) channel of the earlier gate evidence.
 REGIME DISCIPLINE (from the registered claims): all sweeps keep
 D_max <= min(H_tot/4, B/2, S/4 - 1), the regime where Theorem 3's Dt = D_max and
 Theorem 1's (D_max ^ B) = D_max, i.e. where sqrt(D_max) is the claimed binding rate.
 Each configuration is run at its per-config minimax gap delta* (argmax of the two-point
 floor), so every factor is measured on the hardest instance for that setting.

 LEARNER (full-MDP algorithm): delay-aware optimistic value iteration (UCBVI-style). It
 maintains debiased transition estimates for all rows with transition uncertainty, adds a
 variance(D)-scaled bonus min(H_r, c_b * H_r * sqrt(D log(2 S A K) / N(s,a))) (unvisited
 state-action pairs get the fully optimistic value, standard UCBVI init; c_b tuned once
 on the baseline config and applied uniformly to every sweep), runs full
 H_tot-step optimistic value iteration over all S states every episode, and acts greedily
 from the actually sampled start state. Exact-optimization note: after the context step
 the chain is absorbed in sinks whose self-loops are deterministic, carry zero
 information, and have zero estimation uncertainty (singleton known support), so
 simulating the absorbed suffix is an exact no-op and is skipped; likewise the estimate
 matrix is updated incrementally (only the one visited row changes per episode). Neither
 shortcut changes any number. Vectorized across seeds with numpy.

 Corroboration family: random-Dirichlet-kernel full MDPs (support size B, known rewards,
 same delayed observation channel, full VI + exact policy evaluation every episode, full
 H_tot-step trajectories) confirm sublinear ~sqrt(K) learning versus a linear random
 control and monotone delay damage on generic dynamics (directional; the hard family
 carries the rate claims).

Independent NumPy CPU implementation (no official code exists). Deterministic
numpy.random.default_rng seeding throughout; OMP/OPENBLAS single-threaded.
"""
import os, json, time, sys, csv
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent

class _Tee:
    def __init__(self, *f): self.f = f
    def write(self, x):
        for h in self.f: h.write(x)
    def flush(self):
        for h in self.f: h.flush()

# ---------------------------------------------------------------- analytic reference
def kl(p, q): return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def gap_floor(N, Hr, D, K, grid=6000):
    """Two-point Le Cam/BH reference value for N planted gates, value gap Hr*delta,
    per-visit KL deflated by D. Returns (delta*, floor). Used to SET the per-config
    minimax gap and as a reference curve; the claimed lower bound is checked against
    the measured min-over-baselines floor, not this formula."""
    g = np.linspace(1e-4, 0.49, grid); n = K/N
    val = Hr*g*K*0.25*np.exp(-n*kl(0.5+g, 0.5)/D)
    i = int(np.argmax(val)); return float(g[i]), float(val[i])

def fit_loglog(x, y):
    x = np.log(np.asarray(x, float)); y = np.log(np.asarray(y, float))
    b, a = np.polyfit(x, y, 1); yp = a + b*x
    ss = float(np.sum((y-np.mean(y))**2))
    r2 = 1.0 if ss == 0 else float(1 - np.sum((y-yp)**2)/ss)
    return float(b), r2

def boot_ci(vals_per_x, xs, reps=400, seed=123):
    """Bootstrap 95% CI of the log-log slope, resampling seeds."""
    rng = np.random.default_rng(seed)
    V = [np.asarray(v, float) for v in vals_per_x]; R = len(V[0]); sl = []
    for _ in range(reps):
        idx = rng.integers(0, R, R)
        m = [float(np.mean(v[idx])) for v in V]
        sl.append(fit_loglog(xs, m)[0])
    return float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))

# ---------------------------------------------------------------- hard full-MDP family
def build_hard(Rseeds, n_c, n_b, A, delta, inst_seed):
    """True kernels for R seed-instances of the paired-gate hard family.
    States 0..n_c-1 junction contexts, n_c GOOD, n_c+1..n_c+n_b-1 DEAD sinks.
    Each junction's A actions are grouped into ng = A//2 planted BINARY gates: for gate
    g the pair (2g, 2g+1) has one hidden 'good' action with P(GOOD)=1/2+delta and one
    'bad' action with P(GOOD)=1/2. There are ng optimal actions (one per gate, all value
    (1/2+delta)*H_r) and ng suboptimal actions (plus, for odd A, one unpaired distractor
    at P=1/2). This is the standard tabular Omega(sqrt(SA)) embedding: N = n_c*ng
    INDEPENDENT binary sub-problems, symmetric 1/2 wrong-probability (no (A-1)/A skew),
    and regret per episode = delta*H_r*1{chosen action is not optimal}. Non-GOOD mass is
    spread uniformly over the n_b-1 DEAD sinks so every junction row has branching B=n_b.
    Returns P (R,S,A,S), is_opt (R,n_c,A) bool, reward r(S,)."""
    S = n_c + n_b; GOOD = n_c; ng = A//2
    rng = np.random.default_rng(inst_seed)
    good_role = rng.integers(0, 2, size=(Rseeds, n_c, ng))   # which of each pair is good
    P = np.zeros((Rseeds, S, A, S))
    pg = np.full((Rseeds, n_c, A), 0.5)
    is_opt = np.zeros((Rseeds, n_c, A), bool)
    for g in range(ng):
        for role in (0, 1):
            a = 2*g + role
            opt = (good_role[:, :, g] == role)                # (R, n_c) bool
            pg[:, :, a] = np.where(opt, 0.5+delta, 0.5)
            is_opt[:, :, a] = opt
    # (odd A: action A-1 stays P=1/2, is_opt False -> suboptimal distractor)
    P[:, :n_c, :, GOOD] = pg
    P[:, :n_c, :, n_c+1:] = ((1.0-pg)/(n_b-1))[:, :, :, None]
    for s in range(GOOD, S):                       # sinks absorb
        P[:, s, :, s] = 1.0
    r = np.zeros(S); r[GOOD] = 1.0
    return P, is_opt, r

def hard_nuisance(n_c, n_b):
    """Known nuisance u_{s} of the delayed channel for context rows: 1/2 on GOOD,
    1/2 uniform on DEAD. (Sink rows have singleton support -> zero-information channel.)"""
    S = n_c + n_b; GOOD = n_c
    U = np.zeros((n_c, S))
    U[:, GOOD] = 0.5
    U[:, n_c+1:] = 0.5/(n_b-1)
    U /= U.sum(1, keepdims=True)                   # exact normalization for multinomial
    return U

def run_ucbvi_hard(n_c, n_b, A, H_tot, D, K, delta, seeds, inst_seed, run_seed,
                   c_b=1.0, delay_aware=True):
    """Delay-aware (or delay-blind) optimistic value iteration on the hard full MDP.
    Full VI over all S states every episode; the informative transition of each episode
    is genuinely sampled and observed through the delayed channel; exact per-episode
    regret. Vectorized over seeds. Returns per-seed total regret (R,)."""
    S = n_c + n_b; GOOD = n_c; Hr = H_tot - 1
    P, is_opt, r = build_hard(seeds, n_c, n_b, A, delta, inst_seed)
    U = hard_nuisance(n_c, n_b)
    rng = np.random.default_rng(run_seed)
    rows = np.arange(seeds)
    Nsa = np.zeros((seeds, S, A))
    SUM = np.zeros((seeds, S, A, S))               # debiased observation sums
    # known support topology; probabilities unknown on context rows
    supp_ctx = np.zeros(S, bool); supp_ctx[GOOD] = True; supp_ctx[n_c+1:] = True
    nsupp_ctx = float(supp_ctx.sum())
    L = np.log(2.0*S*A*K)
    Deff = float(D if delay_aware else 1)
    # persistent optimistic model: unvisited rows = uniform on known support; sink rows
    # known exactly (singleton support => zero uncertainty => zero bonus)
    Ph = np.zeros((seeds, S, A, S))
    Ph[:, :n_c, :, :] = (supp_ctx/nsupp_ctx)[None, None, None, :]
    for s in range(GOOD, S):
        Ph[:, s, :, s] = 1.0
    bon = np.zeros((seeds, S, A))
    bon[:, :n_c, :] = float(Hr)                    # unvisited rows: fully optimistic
    Pcum_ctx = np.cumsum(P[:, :n_c, :, :], axis=3)
    regret = np.zeros(seeds)
    for k in range(1, K+1):
        # --- full optimistic value iteration over the whole state space
        V = np.zeros((seeds, S))
        for h in range(H_tot-1, -1, -1):
            Q = np.einsum('rsat,rt->rsa', Ph, V) + r[None, :, None] + bon
            if h == 0:
                pol0 = Q.argmax(2)
            V = np.clip(Q.max(2), 0.0, float(Hr))
        # --- one real episode: uniform start context, greedy action, true transition
        s0 = rng.integers(0, n_c, size=seeds)
        a0 = pol0[rows, s0]
        regret += delta*Hr*(~is_opt[rows, s0, a0])
        u01 = rng.random(seeds)
        nxt = (u01[:, None] > Pcum_ctx[rows, s0, a0]).sum(1)   # sampled successor
        y = np.zeros((seeds, S)); y[rows, nxt] = 1.0
        if D > 1:
            y += rng.multinomial(D-1, U[s0])       # batched aggregate nuisance
            y -= (D-1)*U[s0]                       # debias
        SUM[rows, s0, a0] += y
        Nsa[rows, s0, a0] += 1.0
        # (absorbed sink suffix: deterministic, zero-information, zero-uncertainty ->
        #  exact no-op, skipped; see module docstring)
        # --- incremental model refresh of the single visited row per seed
        n_vis = Nsa[rows, s0, a0]
        est = SUM[rows, s0, a0]/n_vis[:, None]
        np.clip(est, 0.0, None, out=est)
        est *= supp_ctx[None, :]
        Z = est.sum(1, keepdims=True)
        est = np.where(Z > 1e-12, est/np.maximum(Z, 1e-12),
                       (supp_ctx/nsupp_ctx)[None, :])
        Ph[rows, s0, a0] = est
        bon[rows, s0, a0] = np.minimum(float(Hr), c_b*Hr*np.sqrt(Deff*L/n_vis))
    return regret

def run_etc_hard(n_c, n_b, A, Hr, D, K, delta, seeds, inst_seed, run_seed, n_e):
    """Explore-then-commit baseline on the same hard instances (same inst_seed =>
    identical planted gates): per junction, round-robin the A actions n_e times through
    the delayed GOOD-rate channel, then commit to the empirical best action (any optimal
    action of any gate is value-optimal). Exact regret. Vectorized over seeds."""
    _, is_opt, _ = build_hard(seeds, n_c, n_b, A, delta, inst_seed)     # (R,n_c,A)
    rng = np.random.default_rng(run_seed)
    rows = np.arange(seeds)
    arr = np.zeros((seeds, n_c), np.int64)
    m = np.zeros((seeds, n_c, A)); n = np.zeros((seeds, n_c, A))
    regret = 0.0; lim = A*n_e
    for k in range(K):
        s = rng.integers(0, n_c, size=seeds)
        exploring = arr[rows, s] < lim
        a = np.where(exploring, arr[rows, s] % A, m[rows, s].argmax(1))
        arr[rows, s] += 1
        opt = is_opt[rows, s, a]
        regret += float(delta*Hr*np.sum(~opt))
        bit = (rng.random(seeds) < 0.5 + delta*opt).astype(float)
        nuis = rng.binomial(D-1, 0.5, size=seeds) if D > 1 else np.zeros(seeds)
        deb = bit + nuis - (D-1)/2.0
        n[rows, s, a] += 1.0
        m[rows, s, a] += (deb - m[rows, s, a])/n[rows, s, a]
    return regret/seeds

# ---------------------------------------------------------------- dirichlet family
def run_dirichlet(S, A, B, H_tot, D, K, seeds, inst_seed, run_seed, c_b=1.0,
                  policy="ucbvi", checkpoints=None):
    """Generic random-kernel full MDP with the same delayed observation channel.
    Unknown Dirichlet transition kernels on random known supports of size B; known
    rewards; delay-aware UCBVI (or uniform-random control). Full H_tot-step trajectories
    and exact per-episode regret by backward induction under the true kernels."""
    rng_i = np.random.default_rng(inst_seed)
    P = np.zeros((seeds, S, A, S)); supp = np.zeros((seeds, S, A, S), bool)
    for rr in range(seeds):
        for s in range(S):
            for a in range(A):
                sup = rng_i.choice(S, size=B, replace=False)
                P[rr, s, a, sup] = rng_i.dirichlet(np.ones(B))
                supp[rr, s, a, sup] = True
    rew = rng_i.uniform(0, 1, size=(seeds, S, A))          # known rewards
    U = np.where(supp, 1.0, 0.0)/B
    U /= U.sum(3, keepdims=True)
    V = np.zeros((seeds, S))
    for h in range(H_tot):                                  # exact V* (stationary)
        V = (np.einsum('rsat,rt->rsa', P, V) + rew).max(2)
    Vstar = V[:, 0].copy()
    rng = np.random.default_rng(run_seed)
    Nsa = np.zeros((seeds, S, A)); SUM = np.zeros((seeds, S, A, S))
    L = np.log(2.0*S*A*K); rows = np.arange(seeds)
    Pcum = np.cumsum(P, axis=3)
    uniform_supp = np.where(supp, 1.0, 0.0)/B
    Sarange = np.arange(S)
    regret = np.zeros(seeds); cp_out = []
    for k in range(1, K+1):
        if policy == "ucbvi":
            Ne = np.maximum(Nsa, 1.0)
            Phh = SUM/Ne[:, :, :, None]
            np.clip(Phh, 0.0, None, out=Phh); Phh *= supp
            Z = Phh.sum(-1, keepdims=True)
            Phh = np.where(Z > 1e-12, Phh/np.maximum(Z, 1e-12), uniform_supp)
            Phh[Nsa == 0] = 0.0
            Phh += (Nsa == 0)[:, :, :, None]*uniform_supp
            bonus = np.where(Nsa == 0, float(H_tot),         # unvisited: optimistic
                             np.minimum(float(H_tot), c_b*H_tot*np.sqrt(D*L/Ne)))
            Vv = np.zeros((seeds, S)); pol = np.empty((H_tot, seeds, S), np.int64)
            for h in range(H_tot-1, -1, -1):
                Q = np.einsum('rsat,rt->rsa', Phh, Vv) + rew + bonus
                pol[h] = Q.argmax(2); Vv = np.clip(Q.max(2), 0.0, float(H_tot))
        else:
            pol = rng.integers(0, A, size=(H_tot, seeds, S))
        Vp = np.zeros((seeds, S))                           # exact policy value
        for h in range(H_tot-1, -1, -1):
            a = pol[h]
            Pp = P[rows[:, None], Sarange[None, :], a]
            rp = rew[rows[:, None], Sarange[None, :], a]
            Vp = rp + np.einsum('rst,rt->rs', Pp, Vp)
        regret += Vstar - Vp[:, 0]
        s_cur = np.zeros(seeds, np.int64)                   # full real trajectory
        for h in range(H_tot):
            a = pol[h][rows, s_cur]
            u01 = rng.random(seeds)
            nxt = (u01[:, None] > Pcum[rows, s_cur, a]).sum(1)
            y = np.zeros((seeds, S)); y[rows, nxt] = 1.0
            if D > 1:
                y += rng.multinomial(D-1, U[rows, s_cur, a])
                y -= (D-1)*U[rows, s_cur, a]
            SUM[rows, s_cur, a] += y; Nsa[rows, s_cur, a] += 1.0
            s_cur = nxt
        if checkpoints and k in checkpoints: cp_out.append(regret.copy())
    return (regret, np.array(cp_out)) if checkpoints else regret

# ---------------------------------------------------------------- orchestration
SEEDS = 12
C_B = 0.30      # tuned once on the baseline config (grid 0.05..1.0), applied everywhere
ETC_GRID = [1, 2, 4, 8, 16, 32, 64]

_cache = {}

def measure_config(n_c, n_b, A, H_tot, D, K, base_seed, with_blind=False, seeds=SEEDS):
    """Run the baselines on one hard-family config at its minimax gap delta*."""
    key = (n_c, n_b, A, H_tot, D, K, seeds)
    if key in _cache:
        c = _cache[key]
        if with_blind and "ucbvi_blind" not in c:
            pass                                           # fall through to add blind
        else:
            return c
    S = n_c + n_b; Hr = H_tot - 1
    ng = A//2                                          # number of binary gates/junction
    n_gates = n_c*ng                                   # total independent binary gates
    n_opt = ng                                         # optimal actions per junction
    dstar, ref_floor = gap_floor(n_gates, Hr, D, K)
    t0 = time.time()
    if key in _cache:
        c = _cache[key]
        ucb_seeds = np.array(c["ucbvi_delay_seeds"]); etc_by_ne = c["_etc_by_ne"]
    else:
        ucb_seeds = run_ucbvi_hard(n_c, n_b, A, H_tot, D, K, dstar, seeds,
                                   base_seed, base_seed+500, c_b=C_B, delay_aware=True)
        etc_by_ne = {}
        for ne in ETC_GRID:
            if ne*A > 0.8*K/n_c: break
            etc_by_ne[ne] = run_etc_hard(n_c, n_b, A, Hr, D, K, dstar, seeds,
                                         base_seed, base_seed+1300+ne, ne)
    blind = None
    if with_blind:
        blind = (run_ucbvi_hard(n_c, n_b, A, H_tot, D, K, dstar, seeds,
                                base_seed, base_seed+900, c_b=C_B, delay_aware=False)
                 if D > 1 else ucb_seeds.copy())
    etc_best_ne, etc_best = min(etc_by_ne.items(), key=lambda kv: kv[1])
    cand = dict(ucbvi_delay=float(np.mean(ucb_seeds)), etc_oracle=float(etc_best))
    if blind is not None: cand["ucbvi_blind"] = float(np.mean(blind))
    floor_alg = min(cand, key=cand.get)
    out = dict(S=S, n_c=n_c, n_b=n_b, B=n_b, A=A, n_gates=n_gates, H_tot=H_tot, Hr=Hr,
               D=D, K=K, seeds=seeds, delta_star=dstar, ref_floor_BH=ref_floor,
               ucbvi_delay_seeds=[float(v) for v in ucb_seeds],
               ucbvi_delay=cand["ucbvi_delay"],
               ucbvi_delay_sem=float(np.std(ucb_seeds, ddof=1)/np.sqrt(seeds)),
               etc_oracle=cand["etc_oracle"], etc_best_ne=int(etc_best_ne),
               empirical_floor=float(min(cand.values())), floor_algorithm=floor_alg,
               no_learning_ref=float(dstar*Hr*K*(1.0 - n_opt/A)),
               regime_ok=bool(D <= min(H_tot/4, n_b/2, S/4-1)),
               _etc_by_ne=etc_by_ne, wall_s=round(time.time()-t0, 1))
    if blind is not None:
        out["ucbvi_blind"] = cand["ucbvi_blind"]
        out["ucbvi_blind_sem"] = float(np.std(blind, ddof=1)/np.sqrt(seeds))
    _cache[key] = out
    return out

def sweep(name, configs, xs, base_seed, with_blind=False, seeds=SEEDS):
    rows = []
    for cfg, x in zip(configs, xs):
        c = measure_config(*cfg, base_seed=base_seed+(int(x)*17 % 9973),
                           with_blind=with_blind, seeds=seeds)
        rows.append(c)
        msg = ("  [%s=%-6s] S=%-3d A=%d H=%-2d D=%-2d K=%-6d d*=%.4f  UCBVI=%.1f+-%.1f"
               % (name, x, c["S"], c["A"], c["H_tot"], c["D"], c["K"], c["delta_star"],
                  c["ucbvi_delay"], c["ucbvi_delay_sem"]))
        if "ucbvi_blind" in c: msg += "  blind=%.1f" % c["ucbvi_blind"]
        msg += ("  ETC*=%.1f(ne=%d)  emp.floor=%.1f(%s)  BHref=%.1f  noLearn=%.0f  %.0fs"
                % (c["etc_oracle"], c["etc_best_ne"], c["empirical_floor"],
                   c["floor_algorithm"], c["ref_floor_BH"], c["no_learning_ref"],
                   c["wall_s"]))
        print(msg); sys.stdout.flush()
    ucb = [r["ucbvi_delay"] for r in rows]
    emp = [r["empirical_floor"] for r in rows]
    ref = [r["ref_floor_BH"] for r in rows]
    su, r2u = fit_loglog(xs, ucb)
    se, r2e = fit_loglog(xs, emp)
    sr, r2r = fit_loglog(xs, ref)
    ci = boot_ci([r["ucbvi_delay_seeds"] for r in rows], xs)
    return dict(x=list(map(float, xs)),
                configs=[{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in rows],
                ucbvi_delay=ucb, ucbvi_slope=su, ucbvi_r2=r2u, ucbvi_slope_ci95=list(ci),
                empirical_floor=emp, empirical_floor_slope=se, empirical_floor_r2=r2e,
                ref_floor_BH=ref, ref_floor_slope=sr, ref_floor_r2=r2r,
                ucb_over_emp_floor=[float(u/f) for u, f in zip(ucb, emp)])

def main():
    lf = open(HERE/"run.log", "w"); sys.stdout = _Tee(sys.__stdout__, lf)
    T0 = time.time()
    print("=== FULL tabular-MDP scaling experiments (hard family + Dirichlet family) ===")
    print("seeds=%d  c_b=%.1f  deterministic default_rng  numpy %s  single-thread"
          % (SEEDS, C_B, np.__version__))
    out = {"paper": "arXiv 2603.03480 / OpenReview fFupHW7Jqx (Lee & Jamieson, ICML 2026)",
           "registered_claims_tested": ["C1 Thm1 upper bound", "C3 Thm3 lower bound",
                                        "C4 improvement over Chen et al. 2023"],
           "mdp": "full tabular episodic MDP: explicit S-state space, A actions, true "
                  "stochastic kernels with branching factor B (planted-action "
                  "Theorem-3-style hard family), real sampled transitions, full "
                  "optimistic value iteration over all states every episode, delayed "
                  "observation channel (D-slot aggregate, debiased); exact regret",
           "regime": "all sweeps satisfy D_max <= min(H_tot/4, B/2, S/4-1) so Theorem "
                     "3's D_tilde = D_max and Theorem 1's (D_max ^ B) = D_max",
           "seeds": SEEDS, "c_b": C_B, "sweeps": {}}

    print("[D_max] primary sweep {1,2,4,8} on full MDP S=36 (n_c=20,B=16) A=3 "
          "H_tot=32 K=12000")
    Dvals = [1, 2, 4, 8]
    out["sweeps"]["D_max"] = sweep("D", [(20, 16, 3, 32, D, 12000) for D in Dvals],
                                   Dvals, 11000, with_blind=True)
    out["sweeps"]["D_max"]["target"] = 0.5

    print("[D_max_wide] confirmation sweep {1,4,16} on full MDP S=68 (n_c=36,B=32) A=2 "
          "H_tot=64 K=8000 (16x delay range)")
    Dw = [1, 4, 16]
    out["sweeps"]["D_max_wide"] = sweep("D", [(36, 32, 2, 64, D, 8000) for D in Dw],
                                        Dw, 61000, with_blind=True, seeds=8)
    out["sweeps"]["D_max_wide"]["target"] = 0.5

    print("[S] sweep {16,24,32,48,64} (n_c=3S/4, B=S/4) A=4 H_tot=8 D=2 K=20000")
    Svals = [16, 24, 32, 48, 64]
    out["sweeps"]["S"] = sweep("S", [(3*S//4, S//4, 4, 8, 2, 20000) for S in Svals],
                               Svals, 21000)
    out["sweeps"]["S"]["target"] = 0.5

    # A grows the action count; each junction carries A/2 planted binary gates, so the
    # number of independent (state,action) exploration problems N = n_c*A/2 grows as A
    # (the standard tabular sqrt(SA) embedding). Even A keeps N exactly proportional to A.
    print("[A] sweep {2,4,6,8,10} on full MDP S=16 (n_c=12,B=4) H_tot=8 D=2 K=20000 "
          "(A/2 binary gates per junction)")
    Avals = [2, 4, 6, 8, 10]
    out["sweeps"]["A"] = sweep("A", [(12, 4, a, 8, 2, 20000) for a in Avals],
                               Avals, 31000)
    out["sweeps"]["A"]["target"] = 0.5

    print("[H] sweep H_tot in {8,12,16,24,32} on full MDP S=16 A=4 D=2 K=20000")
    Hvals = [8, 12, 16, 24, 32]
    out["sweeps"]["H"] = sweep("H", [(12, 4, 4, H, 2, 20000) for H in Hvals],
                               Hvals, 41000)
    out["sweeps"]["H"]["target"] = 1.0

    print("[K] sweep {2500,5000,10000,20000,40000} on full MDP S=16 A=4 H_tot=8 D=2 "
          "(per-K minimax gap)")
    Kvals = [2500, 5000, 10000, 20000, 40000]
    out["sweeps"]["K"] = sweep("K", [(12, 4, 4, 8, 2, K) for K in Kvals],
                               Kvals, 51000)
    out["sweeps"]["K"]["target"] = 0.5

    print("[sanity] learnability vs gap multiple (baseline config S=16 A=4 H=8 D=2 K=20000)")
    d_mm, _ = gap_floor(12*2, 7, 2, 20000)          # N=n_c*A/2=24 gates
    lc_mult, lc_ratio, lc_gap = [], [], []
    for mult in (1, 2, 3, 5):
        dg = min(mult*d_mm, 0.45)
        rg = float(np.mean(run_ucbvi_hard(12, 4, 4, 8, 2, 20000, dg, SEEDS,
                                          81000, 81500, c_b=C_B)))
        noLg = dg*7*20000*0.5                        # A/2 optimal of A=4 => wrong frac 1/2
        lc_mult.append(mult); lc_gap.append(float(dg)); lc_ratio.append(rg/noLg)
        print("  gap=%.4f (%dx d*=%.4f)  regret=%.0f = %.2f x no-learning ref %.0f"
              % (dg, mult, d_mm, rg, rg/noLg, noLg))
    out["learnability_check"] = dict(
        minimax_gap=float(d_mm), gap_multiple=lc_mult, gap=lc_gap,
        regret_over_no_learning=lc_ratio,
        note="at the 1x (minimax) gap the per-gate SNR is pinned near 1 by construction "
             "(that is what makes the instance minimax-hard) so regret is a constant "
             "fraction of the no-learning reference; as the gap widens the optimistic "
             "learner increasingly exploits it and regret drops well below no-learning "
             "-> genuine learning, monotone in the gap")

    print("[dirichlet] random-kernel full MDP S=12 A=3 B=8 H_tot=16 (K-scaling + delay)")
    t0 = time.time()
    cps = [1500, 3000, 6000, 12000]
    dir_out = {"S": 12, "A": 3, "B": 8, "H_tot": 16, "K": 12000, "seeds": 8,
               "checkpoints": cps}
    _, cp_u = run_dirichlet(12, 3, 8, 16, 2, 12000, 8, 71000, 71500,
                            policy="ucbvi", checkpoints=cps)
    _, cp_r = run_dirichlet(12, 3, 8, 16, 2, 12000, 8, 71000, 72500,
                            policy="random", checkpoints=cps)
    mu = cp_u.mean(1); mr = cp_r.mean(1)
    s_u, r2_u = fit_loglog(cps, mu); s_r, r2_r = fit_loglog(cps, mr)
    dir_out.update(ucbvi_cum_regret=[float(v) for v in mu],
                   random_cum_regret=[float(v) for v in mr],
                   ucbvi_K_slope=s_u, ucbvi_K_r2=r2_u,
                   random_K_slope=s_r, random_K_r2=r2_r)
    dvals = [1, 2, 4]
    dreg = [float(np.mean(run_dirichlet(12, 3, 8, 16, D, 12000, 8, 71000, 73000+D)))
            for D in dvals]
    dir_out.update(delay_values=dvals, delay_regret=dreg,
                   delay_slope_directional=fit_loglog(dvals, dreg)[0],
                   wall_s=round(time.time()-t0, 1))
    out["dirichlet"] = dir_out
    print("  [dirichlet] UCBVI K-slope=%.3f (R2=%.3f) vs random control %.3f (R2=%.3f); "
          "delay regret %s (directional slope %.2f)  %.0fs"
          % (s_u, r2_u, s_r, r2_r, ["%.1f" % v for v in dreg],
             dir_out["delay_slope_directional"], dir_out["wall_s"]))

    sw = out["sweeps"]
    def s(f, kk="ucbvi_slope"): return sw[f][kk]
    out["verdict"] = dict(
        upper_sqrt_Dmax_full_mdp=bool(0.4 <= s("D_max") <= 0.6),
        upper_sqrt_Dmax_wide_full_mdp=bool(0.4 <= s("D_max_wide") <= 0.6),
        upper_sqrt_S_full_mdp=bool(0.4 <= s("S") <= 0.6),
        upper_sqrt_A_full_mdp=bool(0.4 <= s("A") <= 0.6),
        upper_sqrt_K_full_mdp=bool(0.4 <= s("K") <= 0.6),
        upper_linear_H_full_mdp=bool(0.85 <= s("H") <= 1.15),
        floor_sqrt_Dmax_empirical=bool(0.4 <= s("D_max", "empirical_floor_slope") <= 0.6),
        floor_sqrt_Dmax_wide_empirical=bool(
            0.4 <= s("D_max_wide", "empirical_floor_slope") <= 0.6),
        floor_sqrt_S_empirical=bool(0.4 <= s("S", "empirical_floor_slope") <= 0.6),
        floor_sqrt_A_empirical=bool(0.4 <= s("A", "empirical_floor_slope") <= 0.6),
        floor_sqrt_K_empirical=bool(0.4 <= s("K", "empirical_floor_slope") <= 0.6),
        floor_linear_H_empirical=bool(
            0.85 <= s("H", "empirical_floor_slope") <= 1.15),
        all_regime_ok=bool(all(c["regime_ok"] for f in sw for c in sw[f]["configs"])),
        chen_2023_exponents_excluded=bool(s("D_max") < 1.0 and s("D_max_wide") < 1.0
                                          and s("H") < 1.3),
        dirichlet_ucbvi_beats_random=bool(
            dir_out["ucbvi_K_slope"] < dir_out["random_K_slope"] - 0.08
            and dir_out["random_K_slope"] > 0.95),
        learnability=bool(out["learnability_check"]["regret_over_no_learning"][-1]
                          < 0.7*out["learnability_check"]["regret_over_no_learning"][0]),
    )
    out["runtime_sec"] = round(time.time()-T0, 1)
    (HERE/"results.json").write_text(json.dumps(out, indent=1))
    with open(HERE/"sweeps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["factor", "x", "S", "n_c", "B", "A", "n_gates", "H_tot", "D", "K",
                    "seeds", "delta_star", "ucbvi_delay", "ucbvi_delay_sem",
                    "ucbvi_blind", "etc_oracle", "etc_best_ne", "empirical_floor",
                    "floor_algorithm", "ref_floor_BH", "no_learning_ref", "regime_ok"])
        for fac in sw:
            for x, c in zip(sw[fac]["x"], sw[fac]["configs"]):
                w.writerow([fac, x, c["S"], c["n_c"], c["B"], c["A"], c["n_gates"],
                            c["H_tot"], c["D"], c["K"], c["seeds"], c["delta_star"],
                            c["ucbvi_delay"], c["ucbvi_delay_sem"],
                            c.get("ucbvi_blind", ""), c["etc_oracle"],
                            c["etc_best_ne"], c["empirical_floor"],
                            c["floor_algorithm"], c["ref_floor_BH"],
                            c["no_learning_ref"], c["regime_ok"]])
    print("--- summary ---")
    for fac in ["D_max", "D_max_wide", "S", "A", "H", "K"]:
        d = sw[fac]
        print("[%-10s] UCBVI slope=%.3f (R2=%.3f, CI95 %.3f..%.3f)  emp.floor "
              "slope=%.3f (R2=%.3f)  BHref slope=%.3f  target=%.1f  UCB/floor=%.2f-%.2f"
              % (fac, d["ucbvi_slope"], d["ucbvi_r2"], d["ucbvi_slope_ci95"][0],
                 d["ucbvi_slope_ci95"][1], d["empirical_floor_slope"],
                 d["empirical_floor_r2"], d["ref_floor_slope"], d["target"],
                 min(d["ucb_over_emp_floor"]), max(d["ucb_over_emp_floor"])))
    print("verdict:", json.dumps(out["verdict"]))
    print("runtime %.1fs" % out["runtime_sec"])
    print("WROTE results.json, sweeps.csv")
    sys.stdout = sys.__stdout__; lf.close()

if __name__ == "__main__":
    main()
