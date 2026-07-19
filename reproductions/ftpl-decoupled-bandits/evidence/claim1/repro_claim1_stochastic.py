#!/usr/bin/env python3
"""
CLAIM 1 (stochastic half) -- FTPL for Decoupled Bandits (arXiv:2510.12152, q1KhliMwKP)
Theorem 2 / Corollary 3: in the stochastically-constrained-adversarial (SCA)
regime with a unique best arm, Algorithm 1 (alpha in (1,3], eta_t=cK^{1/a-1/2}/sqrt(t))
attains CONSTANT (T-INDEPENDENT) pseudo-regret  Reg(T) <= O(K/Delta_min).

Independent NumPy re-implementation of Algorithm 1 (decoupled protocol):
  exploit i_t suffers l_{t,i_t} (unobserved -> regret); explore j_t observed (IW est).
  i_t = argmin_i { Lgap_i - r_i/eta_t },  r ~ Pareto(alpha)  (U^{-1/a}, >=1)     [Eq 5]
  q_i = min(1/(1+eta*Lgap_i), 1/rank_i^{1/a})^{(a+1)/2}, p=q/sum(q)              [Eq 7]
  l_hat_{t,i}=l_{t,i}1[j_t=i]/p_i ; eta_t=c*K^{1/a-1/2}/sqrt(t)  (c=2, alpha=3)
Deterministic (default_rng). Prints ONLY measured numbers. Writes results.json.
Reduced to 500 seeds so a single run stays <40 s on CPU; the archived
artifacts/evidence.json holds the full paper-scale 1000-seed run (same conclusion).
"""
import json, time, os
import numpy as np

MU    = np.array([0.4, 0.45, 0.55, 0.7, 0.8])   # Bernoulli mean-LOSS (paper Sec 4 / Jourdan 2023)
K     = MU.size
ALPHA = 3.0
C     = 2.0
SEEDS = int(os.environ.get("REPRO_SEEDS", 500))
T_MAX = int(os.environ.get("REPRO_TMAX", 80_000))
SEED0 = 20260716
DELTA = MU - MU.min()
DMIN  = float(DELTA[DELTA > 0].min())
ISTAR = int(MU.argmin())
CPS = np.array([1000, 2000, 5000, 10_000, 20_000, 50_000, 80_000])
CPS = CPS[CPS <= T_MAX]
if CPS[-1] != T_MAX:
    CPS = np.append(CPS, T_MAX)


def run(seeds, T, alpha, c, seed0):
    rng  = np.random.default_rng(seed0)
    L    = np.zeros((seeds, K)); creg = np.zeros(seeds); idx = np.arange(seeds)
    Kpow = K ** (1.0/alpha - 0.5); apow = (alpha+1.0)/2.0; inva = 1.0/alpha
    cset = set(int(x) for x in CPS); rm, rs = {}, {}
    for t in range(1, T+1):
        eta  = c*Kpow/np.sqrt(t)
        Lg   = L - L.min(axis=1, keepdims=True)
        pert = rng.random((seeds, K)) ** (-inva)
        it   = (Lg - pert/eta).argmin(axis=1)
        creg += DELTA[it]
        ranks = L.argsort(axis=1).argsort(axis=1) + 1
        qq = np.minimum(1.0/(1.0+eta*Lg), 1.0/ranks**inva) ** apow
        p  = qq/qq.sum(axis=1, keepdims=True)
        u  = rng.random((seeds, 1))
        jt = (p.cumsum(axis=1) > u).argmax(axis=1)
        ell = (rng.random(seeds) < MU[jt]).astype(float)
        L[idx, jt] += ell/p[idx, jt]
        if t in cset:
            rm[t] = float(creg.mean()); rs[t] = float(creg.std(ddof=1))
    m = np.array([rm[int(t)] for t in CPS]); s = np.array([rs[int(t)] for t in CPS])
    return m, s


def slope(T, R, mask=None):
    T = np.asarray(T, float); R = np.asarray(R, float)
    if mask is not None: T, R = T[mask], R[mask]
    return float(np.polyfit(np.log10(T), np.log10(R), 1)[0])


def main():
    t0 = time.time(); m, s = run(SEEDS, T_MAX, ALPHA, C, SEED0); wall = time.time()-t0
    sem = s/np.sqrt(SEEDS); ref = m[0]*np.sqrt(CPS/CPS[0])
    sf = slope(CPS, m); tail = CPS >= 10_000; st = slope(CPS, m, tail)
    ratio = float(m[-1]/m[tail][0])
    sub = DELTA[DELTA > 0]
    border = float(np.sqrt(K/DMIN)*np.sum(1.0/sub) + K/DMIN)   # Cor-3 O(1) order (const omitted)
    print("K=%d mu=%s best=%d Delta=%s Dmin=%.3f alpha=%s c=%s seeds=%d Tmax=%d"
          % (K, list(MU), ISTAR, list(np.round(DELTA,3)), DMIN, ALPHA, C, SEEDS, T_MAX))
    print("%9s %10s %8s %12s %9s" % ("T","Reg","SEM","sqrtT_ref","Reg/ref"))
    for T, mm, ss, rr in zip(CPS, m, sem, ref):
        print("%9d %10.2f %8.2f %12.2f %9.3f" % (T, mm, ss, rr, mm/rr))
    print("loglog_slope_full=%.3f loglog_slope_tail(T>=1e4)=%.3f sqrtT_slope=0.500"%(sf,st))
    print("plateau_ratio_Reg(Tmax)/Reg(1e4)=%.3f"%ratio)
    print("Reg_at_T1e4=%.2f Reg_at_Tmax=%.2f Cor3_O1_bound_order=~%.0f"%(m[tail][0],m[-1],border))
    c1 = st < 0.15; c2 = st < 0.25; c3 = ratio < 1.35; c4 = m[-1] < 5*border
    passed = bool(c1 and c2 and c3 and c4)
    print("cond tail_slope<0.15=%s | slope<<0.5=%s | plateau_ratio<1.35=%s | within_O1_order=%s"
          % (c1,c2,c3,c4))
    print("wall=%.1fs VERDICT=%s"%(wall, "PASS_constant_T_independent_regret" if passed else "FAIL"))
    ev = {"orid":"q1KhliMwKP","claim":"1_stochastic","regime":"SCA/stochastic",
          "setup":{"K":K,"mu_loss":list(MU),"best_arm":ISTAR,"Delta":list(np.round(DELTA,4)),
                   "Delta_min":DMIN,"alpha":ALPHA,"c":C,"seeds":SEEDS,"T_max":T_MAX,"seed0":SEED0},
          "checkpoints_T":[int(x) for x in CPS],
          "regret_mean":[round(float(x),3) for x in m],
          "regret_sem":[round(float(x),3) for x in sem],
          "sqrtT_reference":[round(float(x),3) for x in ref],
          "loglog_slope_full":round(sf,4),"loglog_slope_tail":round(st,4),
          "sqrtT_reference_slope":0.5,"plateau_ratio_tail_over_1e4":round(ratio,4),
          "regret_at_T1e4":round(float(m[tail][0]),3),"regret_at_Tmax":round(float(m[-1]),3),
          "corollary3_O1_bound_order":round(border,2),"wall_seconds":round(wall,1),
          "passed":passed,"verdict":"verified" if passed else "failed"}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_stochastic.json")
    with open(out, "w") as f: json.dump(ev, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
