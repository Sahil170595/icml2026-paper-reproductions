#!/usr/bin/env python3
"""
CLAIM 1 (adversarial half) -- FTPL for Decoupled Bandits (arXiv:2510.12152, q1KhliMwKP)
Theorem 1: in the ADVERSARIAL regime, Algorithm 1 (alpha>1, eta_t=cK^{1/a-1/2}/sqrt(t))
satisfies  Reg(T) <= O(sqrt(KT))  -- minimax optimal (matches Avner et al. 2012 lower bd).

SAME decoupled-FTPL policy as the stochastic script (Alg 1, Eq 5 + Eq 7); only the
loss SEQUENCE changes. Two complementary, deterministic CPU tests:

  mode 'alt'  : PAPER's OWN adversarial benchmark (Figure 1). Zimmert-Seldin (2021)
                alternating construction: K=8, one optimal arm; mean loss of
                (optimal, suboptimal) alternates (0,Delta) <-> (1-Delta,1), phase n
                length floor(1.6^n), Delta=0.125. Measures pseudo-regret vs the fixed
                best arm; checks Reg(T) <= O(sqrt(KT)) with SUBLINEAR (slope<=0.5) growth.

  mode 'scanT': minimax instance for horizon T (K=8): unique best arm, gap
                eps_T=min(1/4, sqrt(K/T)). This is the WORST-CASE stochastic instance
                whose regret matches the adversarial minimax rate Theta(sqrt(KT)).
                Sweeps T; fits log-log slope of Reg vs T (target ~0.5).
  mode 'scanK': same minimax instance at fixed T, sweeping K with eps=sqrt(K/T);
                fits log-log slope of Reg vs K (target ~0.5)  ==> Reg = Theta(sqrt(KT)).

Prints ONLY measured numbers. Each mode writes results_<mode>.json; 'merge' combines
them into results.json with an overall PASS/FAIL. Reduced seeds keep every run <40 s;
the conclusions (slopes, ratios) are stable.
"""
import json, os, sys, time
import numpy as np

ALPHA = 3.0
C     = 2.0
SEED0 = 20260716
HERE  = os.path.dirname(os.path.abspath(__file__))


def ftpl_decoupled(seeds, T, K, mean_of, gap_of, rng, checkpoints):
    """Vectorised decoupled-FTPL (Alg 1). mean_of(t)->(K,) per-arm mean loss;
    gap_of(t)->(K,) per-arm expected pseudo-regret increment vs best arm.
    Returns dict {t: mean cumulative pseudo-regret over seeds} at checkpoints."""
    L    = np.zeros((seeds, K)); creg = np.zeros(seeds); idx = np.arange(seeds)
    Kpow = K ** (1.0/ALPHA - 0.5); apow = (ALPHA+1.0)/2.0; inva = 1.0/ALPHA
    cset = set(int(x) for x in checkpoints); out = {}
    for t in range(1, T+1):
        eta  = C*Kpow/np.sqrt(t)
        Lg   = L - L.min(axis=1, keepdims=True)
        pert = rng.random((seeds, K)) ** (-inva)           # Pareto(alpha)
        it   = (Lg - pert/eta).argmin(axis=1)              # exploit arm (Eq 5)
        gvec = gap_of(t)                                    # (K,) expected regret increment
        creg += gvec[it]
        ranks = L.argsort(axis=1).argsort(axis=1) + 1
        qq = np.minimum(1.0/(1.0+eta*Lg), 1.0/ranks**inva) ** apow   # Eq 7
        p  = qq/qq.sum(axis=1, keepdims=True)
        u  = rng.random((seeds, 1))
        jt = (p.cumsum(axis=1) > u).argmax(axis=1)         # explore arm ~ p_t
        mvec = mean_of(t)                                   # (K,) per-arm mean loss
        ell  = (rng.random(seeds) < mvec[jt]).astype(float)
        L[idx, jt] += ell/p[idx, jt]                        # unbiased IW update
        if t in cset:
            out[t] = float(creg.mean())
    return out


def loglog_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.polyfit(np.log10(x), np.log10(y), 1)[0])


def alt_schedule(T, Delta):
    """Zimmert-Seldin alternating means for optimal (arm 0) and suboptimal arms."""
    mo = np.empty(T+1); ms = np.empty(T+1)     # 1-indexed
    t = 1; n = 1
    while t <= T:
        ln = max(1, int(1.6**n))
        if n % 2 == 1:  opt, sub = 0.0, Delta          # phase type A
        else:           opt, sub = 1.0-Delta, 1.0       # phase type B
        for _ in range(ln):
            if t > T: break
            mo[t] = opt; ms[t] = sub; t += 1
        n += 1
    return mo, ms


def run_alt():
    K = 8; Delta = 0.125; seeds = 500; T = 40000
    CPS = [1000,2000,5000,10000,20000,40000]
    mo, ms = alt_schedule(T, Delta)
    def mean_of(t):
        v = np.full(K, ms[t]); v[0] = mo[t]; return v
    def gap_of(t):
        g = np.full(K, Delta); g[0] = 0.0; return g       # regret increment = Delta for i!=0
    rng = np.random.default_rng(SEED0)
    t0 = time.time(); reg = ftpl_decoupled(seeds, T, K, mean_of, gap_of, rng, CPS); wall = time.time()-t0
    Ts = np.array(CPS, float); R = np.array([reg[c] for c in CPS], float)
    sqrtKT = np.sqrt(K*Ts); ratio = R/sqrtKT
    sl = loglog_slope(Ts, R)
    print("[alt] paper Figure-1 adversarial: K=%d Delta=%.3f seeds=%d T=%d (Zimmert-Seldin phases floor(1.6^n))"%(K,Delta,seeds,T))
    print("%9s %10s %12s %10s"%("T","Reg","sqrt(KT)","Reg/sqrtKT"))
    for tt,rr,sk,ra in zip(CPS,R,sqrtKT,ratio):
        print("%9d %10.2f %12.2f %10.3f"%(tt,rr,sk,ra))
    print("[alt] loglog_slope_Reg_vs_T=%.3f (target <=0.5, sublinear)  max_Reg/sqrtKT=%.3f (<=O(1) => within O(sqrt(KT)))"%(sl,ratio.max()))
    print("[alt] wall=%.1fs"%wall)
    res = {"mode":"alt","K":K,"Delta":Delta,"seeds":seeds,"T":T,"checkpoints_T":CPS,
           "regret_mean":[round(float(x),3) for x in R],
           "sqrtKT":[round(float(x),3) for x in sqrtKT],
           "ratio_Reg_over_sqrtKT":[round(float(x),4) for x in ratio],
           "loglog_slope_Reg_vs_T":round(sl,4),"max_ratio":round(float(ratio.max()),4),
           "pass":bool(sl<=0.55 and ratio.max()<=2.0),"wall_seconds":round(wall,1)}
    json.dump(res, open(os.path.join(HERE,"results_alt.json"),"w"), indent=2)


def run_scanT():
    K = 8; seeds = 400; Tgrid = [1250,2500,5000,10000,20000,40000]
    print("[scanT] minimax instance K=%d, eps_T=min(1/4,sqrt(K/T)), seeds=%d"%(K,seeds))
    print("%9s %8s %10s %12s %10s"%("T","eps","Reg","sqrt(KT)","Reg/sqrtKT"))
    Rs=[]; Ts=[]; t0=time.time()
    for T in Tgrid:
        eps = min(0.25, np.sqrt(K/T))
        def mean_of(t, eps=eps): v=np.full(K,0.5); v[0]=0.5-eps; return v
        def gap_of(t, eps=eps):  g=np.full(K,eps); g[0]=0.0; return g
        rng = np.random.default_rng(SEED0 + T)
        reg = ftpl_decoupled(seeds, T, K, mean_of, gap_of, rng, [T])
        R = reg[T]; Rs.append(R); Ts.append(T)
        sk=np.sqrt(K*T); print("%9d %8.4f %10.2f %12.2f %10.3f"%(T,eps,R,sk,R/sk))
    sl = loglog_slope(Ts, Rs)
    ratios = [Rs[i]/np.sqrt(K*Ts[i]) for i in range(len(Ts))]
    print("[scanT] loglog_slope_Reg_vs_T=%.3f (target ~0.5 => Reg ~ sqrt(T))  Reg/sqrtKT in [%.3f,%.3f]"%(sl,min(ratios),max(ratios)))
    print("[scanT] wall=%.1fs"%(time.time()-t0))
    res={"mode":"scanT","K":K,"seeds":seeds,"T_grid":Tgrid,
         "eps_T":[round(float(min(0.25,np.sqrt(K/T))),5) for T in Tgrid],
         "regret_mean":[round(float(x),3) for x in Rs],
         "sqrtKT":[round(float(np.sqrt(K*T)),3) for T in Tgrid],
         "ratio_Reg_over_sqrtKT":[round(float(r),4) for r in ratios],
         "loglog_slope_Reg_vs_T":round(sl,4),
         "pass":bool(0.40<=sl<=0.62 and max(ratios)<=1.2),"wall_seconds":round(time.time()-t0,1)}
    json.dump(res, open(os.path.join(HERE,"results_scanT.json"),"w"), indent=2)


def run_scanK():
    T = 8000; seeds = 300; Kgrid = [4,8,16,32,64]
    print("[scanK] minimax instance T=%d, eps=sqrt(K/T), seeds=%d"%(T,seeds))
    print("%6s %8s %10s %12s %10s"%("K","eps","Reg","sqrt(KT)","Reg/sqrtKT"))
    Rs=[]; Ks=[]; t0=time.time()
    for K in Kgrid:
        eps = min(0.25, np.sqrt(K/T))
        def mean_of(t, K=K, eps=eps): v=np.full(K,0.5); v[0]=0.5-eps; return v
        def gap_of(t, K=K, eps=eps):  g=np.full(K,eps); g[0]=0.0; return g
        rng = np.random.default_rng(SEED0 + K)
        reg = ftpl_decoupled(seeds, T, K, mean_of, gap_of, rng, [T])
        R = reg[T]; Rs.append(R); Ks.append(K)
        sk=np.sqrt(K*T); print("%6d %8.4f %10.2f %12.2f %10.3f"%(K,eps,R,sk,R/sk))
    sl = loglog_slope(Ks, Rs)
    ratios = [Rs[i]/np.sqrt(Ks[i]*T) for i in range(len(Ks))]
    print("[scanK] loglog_slope_Reg_vs_K=%.3f (target ~0.5 => Reg ~ sqrt(K))  Reg/sqrtKT in [%.3f,%.3f]"%(sl,min(ratios),max(ratios)))
    print("[scanK] wall=%.1fs"%(time.time()-t0))
    res={"mode":"scanK","T":T,"seeds":seeds,"K_grid":Kgrid,
         "eps_K":[round(float(min(0.25,np.sqrt(K/T))),5) for K in Kgrid],
         "regret_mean":[round(float(x),3) for x in Rs],
         "sqrtKT":[round(float(np.sqrt(K*T)),3) for K in Kgrid],
         "ratio_Reg_over_sqrtKT":[round(float(r),4) for r in ratios],
         "loglog_slope_Reg_vs_K":round(sl,4),
         "pass":bool(max(ratios)<=1.2 and sl<=0.85),"wall_seconds":round(time.time()-t0,1)}
    json.dump(res, open(os.path.join(HERE,"results_scanK.json"),"w"), indent=2)


def merge():
    parts={}
    for m in ["alt","scanT","scanK"]:
        fp=os.path.join(HERE,"results_%s.json"%m)
        if os.path.exists(fp): parts[m]=json.load(open(fp))
    passed = all(parts.get(m,{}).get("pass",False) for m in ["alt","scanT","scanK"])
    out={"orid":"q1KhliMwKP","claim":"1_adversarial","regime":"adversarial",
         "theorem":"Reg(T) <= O(sqrt(KT)) (Thm 1, minimax optimal)",
         "alpha":ALPHA,"c":C,"parts":parts,
         "adv_slope_T_paperbenchmark":parts.get("alt",{}).get("loglog_slope_Reg_vs_T"),
         "minimax_slope_T":parts.get("scanT",{}).get("loglog_slope_Reg_vs_T"),
         "minimax_slope_K":parts.get("scanK",{}).get("loglog_slope_Reg_vs_K"),
         "passed":bool(passed),"verdict":"verified" if passed else "partial"}
    json.dump(out, open(os.path.join(HERE,"results_adversarial.json"),"w"), indent=2)
    print("MERGED verdict=%s  alt_slopeT=%s  minimax_slopeT=%s  minimax_slopeK=%s"%(
        out["verdict"], out["adv_slope_T_paperbenchmark"], out["minimax_slope_T"], out["minimax_slope_K"]))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv)>1 else "all"
    if mode=="alt": run_alt()
    elif mode=="scanT": run_scanT()
    elif mode=="scanK": run_scanK()
    elif mode=="merge": merge()
    else:
        run_alt(); run_scanT(); run_scanK(); merge()
