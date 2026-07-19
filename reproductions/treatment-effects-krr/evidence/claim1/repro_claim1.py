"""
Claim 1 (adaptivity to the induced effect) -- independent NumPy/scipy repro of
"Estimating Continuous Treatment Effects with Two-Stage Kernel Ridge Regression"
(ICML 2026, arXiv 2604.13410, OpenReview ziqS4yXFQX).

Thm 4.1 + Example 4.1:
  The two-stage KRR estimator of the induced treatment-effect function
  h*(a) = E_X[ f*(X,a) ] converges at a rate governed by the SIMPLE 1-D target
  RKHS H, NOT the (d+1)-dim nuisance space F.  For a Sobolev pair the two-stage
  MISE is O~((gamma n)^{-2b/(1+2b)}) (dimension-free in d), whereas learning
  f* directly is O(n^{-2b/(d+2b)}) (degrades with d).  For a smooth (RBF) kernel
  b is effectively large so 2b/(1+2b) -> 1 and the two-stage log-log slope -> -1.

Comparison rule (declared BEFORE running):
  (i)  two-stage h_hat log-log slope within +/-0.15 of -1 (smooth kernel), AND
  (ii) at least 0.2 more negative than the direct-f* slope, for d in {3,5},
       in the asymptotic regime n >= 1000.
Falsification: if slope(h) not steeper than slope(f) (gap <= 0), or MISE(h)
  grows with d as fast as MISE(f), the adaptivity claim fails.

Self-contained, CPU-only, deterministic (fixed seeds). NumPy + scipy only.
"""
import time, json, os
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

OUT = Path(__file__).with_name("results.json")

def g(a):            # smooth 1-D dose-response
    return np.sin(2.0*np.pi*a) + a**2
def s(x):            # covariate fn with E[s(U[0,1])] = 0
    return np.cos(2.0*np.pi*x)
def f_star(X, A, c):
    return g(A) + s(X).dot(c)
def h_star(a):       # E_X[f*] = g  (since E[cos 2pi U]=0)
    return g(a)
def sample(rng, n, d, c, sigma):
    X = rng.uniform(0,1,(n,d)); A = rng.uniform(0,1,n)
    Y = f_star(X,A,c) + rng.normal(0,sigma,n)
    return X, A, Y
def sqdist(U, V):
    uu=(U*U).sum(1)[:,None]; vv=(V*V).sum(1)[None,:]
    return np.maximum(uu+vv-2.0*U.dot(V.T), 0.0)
def median_bw(D2):
    iu=np.triu_indices(D2.shape[0],1); med=np.median(D2[iu])
    if med<=0: med=np.mean(D2)+1e-9
    return np.sqrt(med/2.0)+1e-9

def run_once(rng, n, d, c, sigma, a_grid, h_true, n_test):
    X, A, Y = sample(rng, n, d, c, sigma); Am=A[:,None]
    D2x=sqdist(X,X); D2a=sqdist(Am,Am)
    lx=median_bw(D2x); la=median_bw(D2a)
    Kx=np.exp(-D2x/(2*lx*lx)); Ka=np.exp(-D2a/(2*la*la)); G=Kx*Ka
    reg=np.log(n)                              # n*lambda0, lambda0 ~ log n / n
    cf=cho_factor(G+reg*np.eye(n), lower=True, check_finite=False)
    alpha=cho_solve(cf, Y, check_finite=False)
    # Stage 2: induced effect via empirical covariate average (closed form)
    mbar=Kx.mean(0); w=alpha*mbar
    Ka_g=np.exp(-sqdist(a_grid[:,None],Am)/(2*la*la))
    h_hat=Ka_g.dot(w)
    mise_h=float(np.mean((h_hat-h_true)**2))
    # Direct reference: MISE of stage-1 f_hat over (x,a) (fresh test pts)
    Xt=rng.uniform(0,1,(n_test,d)); At=rng.uniform(0,1,n_test)
    Kxt=np.exp(-sqdist(Xt,X)/(2*lx*lx)); Kat=np.exp(-sqdist(At[:,None],Am)/(2*la*la))
    f_hat=(Kxt*Kat).dot(alpha); f_true=f_star(Xt,At,c)
    mise_f=float(np.mean((f_hat-f_true)**2))
    return mise_h, mise_f

def slope(ns, vals):
    x=np.log(np.asarray(ns,float)); y=np.log(np.asarray(vals,float))
    A=np.vstack([x,np.ones_like(x)]).T
    return float(np.linalg.lstsq(A,y,rcond=None)[0][0])

def main():
    t0=time.time()
    ns=[250,500,1000,2000,4000]; ns_asym=[1000,2000,4000]
    ds=[3,5]; R=10; sigma=0.3; n_test=600; cval=0.4
    a_grid=np.linspace(0,1,200); h_true=h_star(a_grid)
    results={}
    print("="*72); print("Claim 1 | Two-stage KRR adaptivity | arXiv 2604.13410"); print("="*72)
    for d in ds:
        c=cval*np.ones(d); mh=[]; mf=[]
        print(f"\n--- d={d} (nuisance dim d+1={d+1}) ---")
        print(f"{'n':>6} {'MISE_h(2stage)':>16} {'MISE_f(direct)':>16}")
        for n in ns:
            hr=[]; fr=[]
            for r in range(R):
                rng=np.random.default_rng(1000*d+7*n+r)
                a,b=run_once(rng,n,d,c,sigma,a_grid,h_true,n_test)
                hr.append(a); fr.append(b)
            mh.append(float(np.mean(hr))); mf.append(float(np.mean(fr)))
            print(f"{n:>6} {mh[-1]:>16.3e} {mf[-1]:>16.3e}")
        idx=[ns.index(k) for k in ns_asym]
        sh=slope(ns,mh); sf=slope(ns,mf)
        sha=slope(ns_asym,[mh[i] for i in idx]); sfa=slope(ns_asym,[mf[i] for i in idx])
        print(f"  full slope h={sh:+.3f} f={sf:+.3f} gap={sf-sh:+.3f}")
        print(f"  asym(n>=1000) slope h={sha:+.3f} f={sfa:+.3f} gap={sfa-sha:+.3f}")
        results[f"d={d}"]={"n":ns,"mise_h":mh,"mise_f":mf,
            "slope_h_full":sh,"slope_f_full":sf,
            "slope_h_asym":sha,"slope_f_asym":sfa,"gap_asym":sfa-sha}
    print("\n--- dimension-free sanity: MISE at largest n vs d ---")
    for d in ds:
        print(f"  d={d}: MISE_h={results[f'd={d}']['mise_h'][-1]:.3e}  MISE_f={results[f'd={d}']['mise_f'][-1]:.3e}")
    print("\n"+"="*72); ok=True
    for d in ds:
        sh=results[f"d={d}"]["slope_h_asym"]; sf=results[f"d={d}"]["slope_f_asym"]
        within=abs(sh+1.0)<=0.15; steeper=(sf-sh)>=0.2; dok=within and steeper; ok=ok and dok
        print(f"d={d}: slope_h={sh:+.3f} (within .15 of -1: {within}) | steeper by {sf-sh:+.3f} (>=.2: {steeper}) -> {dok}")
    print(f"\nOVERALL comparison-rule pass (asymptotic n>=1000): {ok}")
    print(f"elapsed: {time.time()-t0:.1f}s")
    OUT.write_text(json.dumps({"claim":"Claim 1 adaptivity (Thm 4.1/Ex 4.1)",
        "arxiv":"2604.13410","id":"ziqS4yXFQX","sigma":sigma,"R":R,
        "rule":"slope_h within 0.15 of -1 AND >=0.2 steeper than f, d in {3,5}, n>=1000",
        "results":results,"overall_pass":bool(ok)}, indent=2))
    print("wrote results.json")

if __name__=="__main__":
    main()
