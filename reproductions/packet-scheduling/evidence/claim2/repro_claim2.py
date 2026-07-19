#!/usr/bin/env python3
"""Claim 2 (Prop 4.1 + Hajek 2001): ALG^theta achieves the tightest competitive ratio
theta_K in 2-bounded K-OPSD. G_OPT<=theta_K G_ALG (upper bound); no det. algorithm beats
theta_K (lower bound = Hajek Eq.1, which defines theta_K). arXiv 2606.00835 (rZTiFcDihH).
ALG^theta (Alg 2): epoch j, threshold x_j/x_{j+1}; schedule slack b_t iff w(v_t)<(x_j/x_{j+1})w(b_t);
epoch continues iff v_t scheduled with w(v_t)<w(b_t)."""
import json, math, itertools, time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from collections import defaultdict
OUT = Path(__file__).with_name("results.json")
PHI = (1.0 + 5.0**0.5) / 2.0; SQRT2 = math.sqrt(2.0); t0 = time.time()
def solve(K):
    def resid(t):
        x = np.zeros(K); x[0]=1.0; x[1]=1.0/(t-1.0); r=(t+1.0)/(t-1.0)
        for j in range(2,K): x[j]=r*(x[j-1]-x[j-2])
        return x[K-1]-(t+1.0)*x[K-2]
    th = brentq(resid, 1.35, PHI-1e-13, xtol=1e-15, rtol=1e-15)
    x = np.zeros(K); x[0]=1.0; x[1]=1.0/(th-1.0); r=(th+1.0)/(th-1.0)
    for j in range(2,K): x[j]=r*(x[j-1]-x[j-2])
    return th, x
def opt_2b(pk, T):
    free=[False]*(T+2); tot=0.0
    for (r,d,w) in sorted(pk,key=lambda p:-p[2]):
        s=min(d,T)
        while s>=r:
            if not free[s]: free[s]=True; tot+=w; break
            s-=1
    return tot
def alg_theta(pk, x):
    K=len(x); T=max(d for (_,d,_) in pk)
    rel=defaultdict(list)
    for (r,d,w) in pk: rel[r].append([d,w])
    def rt(j): return x[j]/(x[j+1] if j+1<K else x[K-1])
    buf=[]; j=0; tot=0.0
    for t in range(1,T+1):
        buf+=rel.get(t,[]); buf=[p for p in buf if p[0]>=t]
        if not buf: continue
        V=[p for p in buf if p[0]==t]; B=[p for p in buf if p[0]==t+1]
        v=max(V,key=lambda p:p[1]) if V else None
        b=max(B,key=lambda p:p[1]) if B else None
        vw=v[1] if v else 0.0; bw=b[1] if b else 0.0
        if vw<rt(j)*bw and b is not None: ch=b; sb=True
        else: ch=v if v is not None else b; sb=(v is None)
        tot+=ch[1]; buf.remove(ch)
        j=0 if sb else (min(j+1,K-1) if (b is not None and v is not None and vw<bw) else 0)
    return tot, T
def exhaustive_worst(K, S):
    th,x=solve(K); wopt=[None]+list(x); choices=list(itertools.product(wopt,wopt))
    best=[0.0]; count=[0]
    def rec(t, pk):
        if t>S:
            if not pk: return
            T=max(d for (_,d,_) in pk); a,_=alg_theta(pk,x)
            if a>1e-12:
                r=opt_2b(pk,T)/a; count[0]+=1
                if r>best[0]: best[0]=r
            return
        for (tw,sw) in choices:
            add=[]
            if tw is not None: add.append((t,t,tw))
            if sw is not None: add.append((t,t+1,sw))
            rec(t+1, pk+add)
    rec(1,[]); return th, best[0], count[0]
def main():
    res={"paper":"arXiv 2606.00835 (rZTiFcDihH)","phi":PHI,"sqrt2":SQRT2}
    print("=== (A) UPPER BOUND (Prop 4.1): exhaustive worst-case ratio <= theta_K ===")
    A={}
    for K,S in [(2,5),(3,4),(4,3)]:
        th,worst,cnt=exhaustive_worst(K,S)
        A[K]={"theta_K":th,"slots":S,"instances_enumerated":cnt,"worst_ratio":worst,"exceeds_theta_K":bool(worst>th+1e-9)}
        print(f"K={K} (S={S}, {cnt} instances): worst ratio={worst:.6f}  theta_K={th:.6f}  exceeds: {worst>th+1e-9}")
    res["experimentA_upperbound_exhaustive"]=A
    print("\n=== (B) TIGHTNESS: 2-type drop gadget -> competitive ratio = theta_K ===")
    B={}
    for K in range(2,11):
        th,x=solve(K); pk=[(1,1,th-1.0-1e-9),(1,2,1.0)]
        a,T=alg_theta(pk,x); r=opt_2b(pk,T)/a
        B[K]={"theta_K":th,"gadget_ratio":r,"gap_to_theta_K":th-r}
        print(f"K={K}: theta_K={th:.9f}  gadget OPT/ALG={r:.9f}  gap={th-r:.1e}")
    res["experimentB_tightness_gadget"]=B
    print("\n=== (B') sup = theta_K: gadget ratio -> theta_K as eps -> 0 (K=3) ===")
    K=3; th,x=solve(K); conv={}
    for eps in [1e-1,1e-2,1e-3,1e-6,1e-9]:
        pk=[(1,1,th-1.0-eps),(1,2,1.0)]; a,T=alg_theta(pk,x); r=opt_2b(pk,T)/a
        conv[eps]=r; print(f"   eps={eps:.0e}: ratio={r:.9f}  (theta_3={th:.9f})")
    res["experimentBp_convergence_K3"]={"theta_K":th,"ratio_by_eps":{f"{e:.0e}":v for e,v in conv.items()}}
    res["experimentC_tight_targets"]={"theta_2":solve(2)[0],"theta_2_target_sqrt2":SQRT2,"theta_2_err":abs(solve(2)[0]-SQRT2),
        "theta_3":solve(3)[0],"theta_3_target_1.5":1.5,"theta_3_err":abs(solve(3)[0]-1.5)}
    print("\n=== (C) achieved ratio == Hajek(2001) lower bound ===")
    print(f"   theta_2={solve(2)[0]:.12f} vs sqrt2={SQRT2:.12f} (err {abs(solve(2)[0]-SQRT2):.1e})")
    print(f"   theta_3={solve(3)[0]:.12f} vs 3/2 (err {abs(solve(3)[0]-1.5):.1e})")
    res["runtime_s"]=round(time.time()-t0,2)
    OUT.write_text(json.dumps(res,indent=2,default=float))
    print(f"\nWrote {OUT}  ({res['runtime_s']}s)")
if __name__=="__main__": main()
