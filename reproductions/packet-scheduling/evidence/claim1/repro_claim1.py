#!/usr/bin/env python3
"""
Claim 1 (Theorems 4.2 & 4.3): the learning-augmented algorithm ALG^{theta,U}
attains an alpha=theta_K-regret upper-bounded as O~(sqrt(K T)), matching the
standard multi-armed-bandit minimax lower bound Omega(sqrt(K T)).

Paper: "Online Packet Scheduling with Deadlines and Learning" (arXiv 2606.00835,
OpenReview rZTiFcDihH). Theorem 4.2: E[G_OPT] <= theta_K E[G_ALG] + O~(sqrt(KT)).
Theorem 4.3: E[G_OPT] - theta_K E[G_ALG] >= Omega(sqrt(T)).  Lemma 2.1: 1-bounded
K-OPSD == K-armed (sleeping) bandit, where the minimax rate is Theta(sqrt(KT)).

Two fully-deterministic (seeded) CPU experiments:
 (A) Standard K-armed bandit reduction (Lemma 2.1): UCB1 at the worst-case gap
     Delta_T = sqrt(K/T). Measured pseudo-regret must scale as sqrt(KT):
     log-log slope in T ~ 0.5, doubling ratio R(2T)/R(T) ~ sqrt(2), and R/sqrt(KT)
     bounded (upper bound) and bounded away from 0 (Omega, Thm 4.3).
 (B) 2-bounded stochastic K-OPSD: the learning overhead of ALG^{theta,U}
     (Algorithm 3, UCB thresholds) versus the mean-aware oracle ALG^theta, which
     equals the theta_K-scaled second term of Theorem 4.2, must also scale sqrt(KT).
"""
import json, math, os, time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

OUT = Path(__file__).with_name("results.json")
PHI = (1.0 + 5.0**0.5) / 2.0
t0 = time.time()

def theta_K(K):
    if K < 2: return 1.0
    def resid(t):
        x = np.zeros(K); x[0] = 1.0; x[1] = 1.0/(t-1.0); r = (t+1.0)/(t-1.0)
        for j in range(2, K): x[j] = r*(x[j-1]-x[j-2])
        return x[K-1] - (t+1.0)*x[K-2]
    return brentq(resid, 1+1e-9, PHI-1e-12, xtol=1e-15, rtol=1e-15)

# ---------- (A) UCB1 on K-armed bandit (1-bounded K-OPSD, Lemma 2.1) ----------
def ucb_regret(K, T, delta, seed, c=2.0):
    """Expected pseudo-regret of UCB1 on Bernoulli arms: mu0=0.5+d/2, rest 0.5-d/2."""
    rng = np.random.default_rng(seed)
    mu = np.full(K, 0.5 - delta/2.0); mu[0] = 0.5 + delta/2.0
    n = np.zeros(K); s = np.zeros(K); reg = 0.0
    for t in range(1, T+1):
        if t <= K:
            a = t-1
        else:
            a = int(np.argmax(s/np.maximum(n,1) + np.sqrt(c*np.log(t)/np.maximum(n,1))))
        reg += mu[0] - mu[a]
        n[a] += 1; s[a] += (rng.random() < mu[a])
    return reg

def expA():
    Ks = [2, 4, 8, 16]; Ts = [500, 1000, 2000, 4000, 8000]; S = 24
    R = {}
    for K in Ks:
        for T in Ts:
            delta = min(0.45, math.sqrt(K/T))       # worst-case (minimax) gap
            vals = [ucb_regret(K, T, delta, 1000*K+i) for i in range(S)]
            R[(K,T)] = float(np.mean(vals))
    # slope of log R vs log T for each K
    slopes = {}
    for K in Ks:
        xs = np.log(np.array(Ts, float)); ys = np.log(np.array([R[(K,T)] for T in Ts]))
        slopes[K] = float(np.polyfit(xs, ys, 1)[0])
    # R / sqrt(KT) table
    norm = {f"K{K}_T{T}": R[(K,T)]/math.sqrt(K*T) for K in Ks for T in Ts}
    doub = {f"K{K}": R[(K,Ts[-1])]/R[(K,Ts[-2])] for K in Ks}   # R(2T)/R(T)
    kslope = None
    xs = np.log(np.array(Ks,float)); ys = np.log(np.array([R[(K,2000)] for K in Ks]))
    kslope = float(np.polyfit(xs, ys, 1)[0])
    return {"Ks":Ks,"Ts":Ts,"seeds":S,"regret":{f"K{K}_T{T}":R[(K,T)] for K in Ks for T in Ts},
            "logT_slopes":slopes,"logK_slope_atT2000":kslope,
            "R_over_sqrtKT":norm,"doubling_R2T_over_RT":doub}

# ---------- (B) 2-bounded stochastic K-OPSD: ALG^{theta,U} vs oracle ALG^theta ----------
def run_2bounded(K, T, learn, seed, delta):
    """Repeated 2-slot gadget arrivals with K Bernoulli types.
    learn=False -> oracle uses true means; learn=True -> ALG^{theta,U} uses UCBs.
    Returns realized gain G_ALG."""
    rng = np.random.default_rng(seed)
    th = theta_K(K)
    x = np.zeros(K+1); x[0]=1.0; x[1]=1.0/(th-1.0); r=(th+1.0)/(th-1.0)
    for j in range(2,K): x[j]=r*(x[j-1]-x[j-2])
    x[K]=x[K-1]
    # K type means spread in (0,1]
    mu = 0.5 + (delta/2.0)*np.linspace(1.0, -1.0, K)   # gaps ~ delta (minimax regime)
    n=np.zeros(K); ssum=np.zeros(K)
    def ucb(i,t):
        if n[i]==0: return 1.0
        return min(1.0, ssum[i]/n[i] + math.sqrt(math.log(K*T*T+1)/(2*n[i])))
    j=0; G=0.0; buf=[]   # buf: list of (deadline, type)
    for t in range(1,T+1):
        # arrivals: tight packet of a "low" type (deadline t), slack packet of a "high" type (deadline t+1)
        lo = (t*7) % K; hi = (t*3) % K
        buf.append((t, lo)); buf.append((t+1, hi))
        buf=[p for p in buf if p[0]>=t]
        V=[p for p in buf if p[0]==t]; B=[p for p in buf if p[0]==t+1]
        def wt(i):
            return mu[i] if not learn else ucb(i,t)
        v = max(V, key=lambda p: wt(p[1])) if V else None
        b = max(B, key=lambda p: wt(p[1])) if B else None
        vw = wt(v[1]) if v else 0.0; bw = wt(b[1]) if b else 0.0
        if vw < (x[j]/x[j+1])*bw and b is not None:
            ch=b; sb=True
        else:
            ch=v if v is not None else b; sb=(v is None)
        # realize reward of scheduled type (bandit feedback)
        typ = ch[1]; rew = 1.0 if rng.random() < mu[typ] else 0.0
        G += rew
        if learn:
            n[typ]+=1; ssum[typ]+=rew
        buf.remove(ch)
        j = 0 if sb else (min(j+1,K-1) if (b is not None and v is not None and vw < bw) else 0)
    return G

def expB():
    Ks=[2,4,8]; Ts=[1000,2000,4000,8000]; S=16
    out={}
    for K in Ks:
        for T in Ts:
            delta=min(0.4, math.sqrt(K/T))          # worst-case (minimax) type-gap
            g_or=np.mean([run_2bounded(K,T,False,55+i,delta) for i in range(S)])
            g_le=np.mean([run_2bounded(K,T,True, 55+i,delta) for i in range(S)])
            out[(K,T)]=float(g_or-g_le)     # learning regret (oracle - learner) = O~(sqrt(KT)) term of Thm 4.2
    slopes={}
    for K in Ks:
        xs=np.log(np.array(Ts,float)); ys=np.log(np.maximum(np.array([out[(K,T)] for T in Ts]),1e-9))
        slopes[K]=float(np.polyfit(xs,ys,1)[0])
    norm={f"K{K}_T{T}": out[(K,T)]/math.sqrt(K*T) for K in Ks for T in Ts}
    doub={f"K{K}": out[(K,Ts[-1])]/out[(K,Ts[-2])] for K in Ks}
    return {"Ks":Ks,"Ts":Ts,"seeds":S,
            "learning_regret":{f"K{K}_T{T}":out[(K,T)] for K in Ks for T in Ts},
            "logT_slopes":slopes,"LR_over_sqrtKT":norm,"doubling_R2T_over_RT":doub}

def main():
    res={"paper":"arXiv 2606.00835 (rZTiFcDihH)","phi":PHI,
         "theta_K":{K:theta_K(K) for K in [2,4,8,16]}}
    print("=== (A) K-armed bandit reduction (Lemma 2.1): UCB1 regret ~ sqrt(KT) ===")
    A=expA(); res["experimentA_bandit"]=A
    print("log-log slope of regret vs T (target ~0.5):")
    for K,sl in A["logT_slopes"].items(): print(f"   K={K}: slope={sl:.3f}")
    print(f"log-log slope of regret vs K at T=2000 (target ~0.5): {A['logK_slope_atT2000']:.3f}")
    print("R / sqrt(KT) (should be ~constant, i.e. Theta(sqrt(KT))):")
    for k,v in A["R_over_sqrtKT"].items(): print(f"   {k}: {v:.3f}")
    print("doubling R(2T)/R(T) (target ~1.414):")
    for k,v in A["doubling_R2T_over_RT"].items(): print(f"   {k}: {v:.3f}")
    print("\n=== (B) 2-bounded stochastic K-OPSD: ALG^{theta,U} learning regret ~ sqrt(KT) ===")
    B=expB(); res["experimentB_2bounded"]=B
    print("log-log slope of learning-regret vs T (target ~0.5):")
    for K,sl in B["logT_slopes"].items(): print(f"   K={K}: slope={sl:.3f}")
    print("learning-regret / sqrt(KT):")
    for k,v in B["LR_over_sqrtKT"].items(): print(f"   {k}: {v:.3f}")
    res["runtime_s"]=round(time.time()-t0,2)
    OUT.write_text(json.dumps(res,indent=2))
    print(f"\nWrote {OUT}  ({res['runtime_s']}s)")

if __name__=="__main__":
    main()
