#!/usr/bin/env python3
"""
mdp_core.py -- genuine episodic tabular MDP core for reproducing
  "Best-of-Both-Worlds for Heavy-Tailed Markov Decision Processes"
  OpenReview j6gXeiPJ3z / arXiv 2602.01295 (ICML 2026).

This is the ACTUAL layered episodic MDP of the paper's Section 3.1, NOT the H=1
bandit reduction:
  * H decision layers.  Layer 1 = single initial state s_1 (paper: S_1={s_1}).
    Layers 2..H each have S states.  After layer H -> terminal s_{H+1}.
    (Configurable; defaults H=3, S=4 states/multi-state layer, A=4 actions.)
  * Transitions happen ONLY between adjacent layers (paper).  The action taken
    genuinely controls the next-state DISTRIBUTION, so the occupancy measure is
    multi-dimensional and credit assignment runs through the transition kernel.
  * Heavy-tailed losses with bounded alpha-moment E[|l|^alpha] <= sigma^alpha.

Algorithms (paper): HT-FTRL-OM (known P) and HT-FTRL-UOB (unknown P).  Both are
FTRL over occupancy measures with the 1/alpha-Tsallis-entropy regularizer and the
paper's skipping loss estimator; on the layered MDP with the (standard) dilated
Tsallis regularizer this FTRL-over-occupancy decomposes into per-state Tsallis-INF
FTRL whose LOCAL loss is the estimated Q-value
    Qhat(s_h,a) = Lhat(s_h,a) + sum_{s'} P[s'|s_h,a] Vhat(s'),          (paper Eq. Q)
    Vhat(s_h)   = < pi(.|s_h), Qhat(s_h,.) >,
i.e. the loss estimates are propagated backward through the transition kernel
(true P for OM, learned Phat for UOB) -- this is what makes it a real MDP, not a
bandit.  For UNKNOWN P we additionally learn P by doubling-epoch transition counts,
use the Upper Occupancy Bound (optimistic occupancy) in the estimator denominator,
and add an exploration bonus, exactly as HT-FTRL-UOB (Algorithm 2) prescribes.

Everything is vectorized over ROWS = (algorithm-variant x seed); one call simulates
many seeds/variants for one alpha at once.  Deterministic RNG.  No hand numbers.
"""
import numpy as np

MCAP = 1.0e6            # truncation cap for the Pareto magnitude (finite sim support)

# --------------------------------------------------------------------------- #
#  Heavy-tailed noise: symmetric truncated Pareto, E[|noise|^alpha] <= 1.
# --------------------------------------------------------------------------- #
def noise_scale(alpha):
    # E|X|^a for X=U^{-1/a}, U~Unif(0,1], truncated at MCAP, then divide by s so
    # that E[|noise|^a] ~ 1.  E[(U^{-1/a})^a]=E[1/U] diverges; truncation gives
    # ~ a*ln(MCAP)+1.  s = (a*ln(MCAP)+1)^{1/a}.
    return (alpha*np.log(MCAP) + 1.0)**(1.0/alpha)

def make_noise(rng, alpha, s, R):
    U = rng.random(R)
    mag = U**(-1.0/alpha)
    np.minimum(mag, MCAP, out=mag)
    sgn = np.where(rng.random(R) < 0.5, -1.0, 1.0)
    return sgn * mag / s

def empirical_moment(alpha, n=300000, seed=7):
    rng = np.random.default_rng(seed)
    s = noise_scale(alpha)
    nz = make_noise(rng, alpha, s, n)
    return float(np.mean(np.abs(nz)**alpha)), float(np.var(nz))

# --------------------------------------------------------------------------- #
#  Tsallis-INF / (1/alpha)-Tsallis FTRL simplex solve (batched over leading dim).
#  x_a = ( c / (eta*(L_a-Lmin) + mu) )^{pexp},  pexp=1/(1-q), c=q/(1-q), q=1/alpha,
#  with the Lagrange multiplier mu found by bisection so sum_a x_a = 1.
# --------------------------------------------------------------------------- #
def solve_ftrl(w, pexp, c, A, n_iter=6):
    # w = eta * Lcum, shape (..., A)
    wmin = w.min(axis=-1, keepdims=True)
    d    = w - wmin
    logc = np.log(c)
    lo   = np.full(w.shape[:-1] + (1,), logc)
    hi   = logc + (1.0/pexp)*np.log(A) + np.zeros_like(lo)
    for _ in range(n_iter):
        mid = 0.5*(lo+hi)
        x   = np.exp(pexp*(logc - np.log(d + np.exp(mid))))
        too = x.sum(axis=-1, keepdims=True) > 1.0
        lo  = np.where(too, mid, lo)
        hi  = np.where(too, hi, mid)
    x = np.exp(pexp*(logc - np.log(d + np.exp(0.5*(lo+hi)))))
    return x / x.sum(axis=-1, keepdims=True)

# --------------------------------------------------------------------------- #
#  Build a genuine layered MDP instance (transition kernel + mean losses) with a
#  UNIQUE optimal policy (action 0 everywhere) and an EXACT per-state advantage
#  gap G for every suboptimal action, via a backward value construction:
#     V*(terminal)=0;  Q*(s,0)=lbar(s,0)+P[s,0].V*next;  V*(s)=Q*(s,0);
#     lbar(s,a!=0) := V*(s)+G - P[s,a].V*next   ->  Q*(s,a)=V*(s)+G  (adv=G).
#  Transitions are genuine: action a routes mass toward a distinct next-state, and
#  a state-dependent base loss makes V* vary across states, so the transition kernel
#  genuinely affects value (real credit assignment, not H independent bandits).
# --------------------------------------------------------------------------- #
def build_mdp(H, S, A, G, seed=12345, p_route=0.70, c_state=0.25):
    """Returns P_true[H,S,A,S] (row h: from a layer-h state via a to a layer-(h+1)
    state; last layer -> terminal, encoded as all-zero next), lbar[H,S,A] mean loss,
    Vstar[H,S], adv[H,S,A]=G*1[a!=0], nS[H] = #active states per layer (layer0=1).
    """
    rng = np.random.default_rng(seed)
    nS = np.array([1] + [S]*(H-1), dtype=int)      # layer 0 has the single start state
    P = np.zeros((H, S, A, S))
    for h in range(H-1):
        ns_next = nS[h+1]
        for s in range(nS[h]):
            for a in range(A):
                base = np.full(ns_next, (1.0 - p_route)/ns_next)
                tgt = (a + s) % ns_next            # action routes to a distinct next state
                base[tgt] += p_route
                base /= base.sum()
                P[h, s, a, :ns_next] = base
    # state-dependent base loss for the optimal action -> V* varies across states
    lbar = np.zeros((H, S, A))
    Vstar = np.zeros((H, S))
    adv = np.zeros((H, S, A))
    # base loss level of the optimal action at (h,s): higher-index states cost more
    lbar0 = np.zeros((H, S))
    for h in range(H):
        for s in range(nS[h]):
            lbar0[h, s] = c_state * s
    # backward construction
    for h in range(H-1, -1, -1):
        for s in range(nS[h]):
            if h == H-1:
                Vnext_dot0 = 0.0
                lbar[h, s, 0] = lbar0[h, s]
                Vstar[h, s] = lbar[h, s, 0]
                for a in range(1, A):
                    lbar[h, s, a] = Vstar[h, s] + G   # no future
                    adv[h, s, a] = G
            else:
                Vnext = Vstar[h+1]                     # shape (S,)
                d0 = float(P[h, s, 0, :] @ Vnext)
                lbar[h, s, 0] = lbar0[h, s]
                Vstar[h, s] = lbar[h, s, 0] + d0
                for a in range(1, A):
                    da = float(P[h, s, a, :] @ Vnext)
                    lbar[h, s, a] = Vstar[h, s] + G - da
                    adv[h, s, a] = G
    return P, lbar, Vstar, adv, nS
def run_episodes(alpha, P_true, lbar, adv, nS, is_uob, is_skip, gap, seed, tmax, ckpts,
                 sigma=1.0, C=1.0, cconf=0.25, dexp=1.0, occ_floor=1.0e-3, nbis=6, state=None):
    H, S, A, _ = P_true.shape; R = is_uob.size
    q = 1.0/alpha; pexp = 1.0/(1.0-q); c = q/(1.0-q); inv_a = 1.0/alpha; Cpow = C**(1.0-alpha)
    s_noise = noise_scale(alpha); ridx = np.arange(R); uob_c = is_uob[:, None, None]; uob4 = is_uob[:, None, None, None]; uob5 = is_uob[:, None, None, None, None]; skip_c = is_skip[:, None, None]
    ck_set = {int(x): i for i, x in enumerate(ckpts)}; out = np.zeros((R, len(ckpts)))
    smask = np.zeros((H, S))
    for h in range(H): smask[h, :nS[h]] = 1.0
    P_true_R = np.broadcast_to(P_true, (R, H, S, A, S))
    if state is None:
        rng = np.random.default_rng(seed); Lhat = np.zeros((R, H, S, A)); N = np.zeros((R, H, S, A))
        Nz = np.zeros((R, H, S, A, S)); Phat = np.zeros((R, H, S, A, S))
        for h in range(H-1): Phat[:, h, :, :, :nS[h+1]] = 1.0/nS[h+1]
        Bconf = np.zeros((R, H, S, A)); cumreg = np.zeros(R); t_i = 1; t0 = 1
    else:
        rng, Lhat, N, Nz, Phat, Bconf, cumreg, t_i, t0 = state
    iota0 = H*S*A; Vterm = np.zeros((R, S))
    for t in range(t0, tmax+1):
        if (t & (t-1)) == 0:
            t_i = t; Ntot = np.maximum(N, 1.0); newP = Nz / Ntot[..., None]; unseen = (N <= 0)[..., None]
            Phat = np.where(unseen, Phat, newP); iota = iota0*(float(t)**4)
            Bconf = np.sqrt(cconf*np.log(max(iota, 3.0))/np.maximum(N, 1.0)) * smask[None, :, :, None]
            Lhat = np.where(uob4, 0.0, Lhat)
        tau_row = np.where(is_uob, float(t - t_i + 1), float(t)); eta = (1.0/(sigma*tau_row**inv_a))[:, None, None]
        P_model = np.where(uob5, Phat, P_true_R); pi = np.zeros((R, H, S, A)); Vnext = Vterm.copy()
        for h in range(H-1, -1, -1):
            backup = np.einsum('rsam,rm->rsa', P_model[:, h], Vnext); Qh = Lhat[:, h] + backup
            xh = solve_ftrl(eta*Qh, pexp, c, A, nbis); xh = xh*smask[h][None, :, None]
            ssum = xh.sum(axis=-1, keepdims=True); xh = np.where(ssum > 0, xh/np.maximum(ssum, 1e-12), 1.0/A)
            pi[:, h] = xh; Vnext = np.sum(xh*Qh, axis=-1)
        q_true = np.zeros((R, H, S)); q_true[:, 0, 0] = 1.0; occ_e = np.zeros((R, H, S)); occ_e[:, 0, 0] = 1.0
        Popt = np.minimum(Phat + Bconf[..., None], 1.0)
        for h in range(H-1):
            qta = q_true[:, h][:, :, None]*pi[:, h]; q_true[:, h+1] = np.einsum('rsa,rsam->rm', qta, P_true_R[:, h])
            Pest = np.where(uob_c[..., None], Popt[:, h], P_true_R[:, h]); qea = occ_e[:, h][:, :, None]*pi[:, h]
            occ_e[:, h+1] = np.minimum(1.0, np.einsum('rsa,rsam->rm', qea, Pest))
        cumreg += np.einsum('rhs,rhsa,hsa->r', q_true, pi, adv)
        cur = np.zeros(R, dtype=int); vis_s = np.zeros((R, H), dtype=int); vis_a = np.zeros((R, H), dtype=int); losses = np.zeros((R, H))
        for h in range(H):
            pcur = pi[ridx, h, cur, :]; cdf = np.cumsum(pcur, axis=1)
            a_h = np.minimum((cdf < rng.random((R, 1))).sum(axis=1), A-1)
            vis_s[:, h] = cur; vis_a[:, h] = a_h; losses[:, h] = lbar[h, cur, a_h] + make_noise(rng, alpha, s_noise, R)
            if h < H-1:
                Pnext = P_true[h, cur, a_h, :]; cdfn = np.cumsum(Pnext, axis=1); cur = np.minimum((cdfn < rng.random((R, 1))).sum(axis=1), S-1)
        for h in range(H):
            s_h = vis_s[:, h]; a_h = vis_a[:, h]; occ_s = np.maximum(occ_e[ridx, h, s_h], occ_floor)
            pi_sa = pi[ridx, h, s_h, a_h]; occ_sa = np.maximum(occ_s*pi_sa, occ_floor); l = losses[:, h]
            tau_sk = C*sigma*(tau_row**inv_a)*(occ_sa**inv_a); l_sk = np.where(np.abs(l) <= tau_sk, l, 0.0)
            l_used = np.where(is_skip, l_sk, l); iw = l_used/occ_sa
            occ_all = np.maximum(occ_e[:, h][:, :, None]*pi[:, h], occ_floor)
            bonus_full = Cpow*sigma*(tau_row[:, None, None]**(inv_a-1.0))*(occ_all**(inv_a-1.0))
            bonus = np.where(skip_c, bonus_full, 0.0)*smask[h][None, :, None]; lhat = -bonus
            lhat[ridx, s_h, a_h] += iw; lhat = lhat - np.where(uob_c, dexp*Bconf[:, h], 0.0); Lhat[:, h] += lhat
            if h < H-1:
                s_next = vis_s[:, h+1]; N[ridx, h, s_h, a_h] += 1.0; Nz[ridx, h, s_h, a_h, s_next] += 1.0
            else:
                N[ridx, h, s_h, a_h] += 1.0
        if t in ck_set: out[:, ck_set[t]] = cumreg
    new_state = (rng, Lhat, N, Nz, Phat, Bconf, cumreg, t_i, tmax+1); return out, new_state
