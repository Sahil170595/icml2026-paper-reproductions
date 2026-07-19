"""
Claim 2 reproduction --- "Experimental validation demonstrates the novel analysis
more accurately describes the effect of topology on the convergence rate than prior work."

Paper: Takezawa, Koloskova, Stich, arXiv 2606.09154v1 (ICML 2026), Sec 6.2/6.3.
OpenReview pYI0WjV5iM.

We run ACTUAL Decentralized SGD (paper Eq. 2):
    x_i^{(r+1)} = sum_j W_ij ( x_j^{(r)} - eta grad F_j(x_j^{(r)}) )
on CPU and check which topology metric predicts the measured convergence behaviour:
    prior (spectral gap)  : T_gap = (1-p)/p,  p = 1 - max_{i>=2} lambda_i^2      (Proposition 1)
    paper (full spectrum) : T_new = (1/n) sum_{i>=2} lambda_i^2/(1-lambda_i^2)    (Theorem 1)

Part 1  (homogeneous, at optimum, grad = stochastic noise -- paper's sigma_* term):
  measured steady-state consensus Omega_ss of real D-SGD. Exact target
  Omega_ss = eta^2 sigma^2 T_new. Regress measured Omega_ss vs T_new (expect
  slope ~ eta^2 sigma^2, R^2>=0.99) and vs T_gap (prior; expect worse + wrong ranking).
Part 1b (homogeneous robustness): averaged-model suboptimality f(xbar)-f* floor is
  topology-INDEPENDENT for fixed n (equal across topologies whose spectral gaps span >10x)
  -- the behaviour the prior 1/p^2 analysis cannot explain but the full spectrum does.
Part 2  (heterogeneous, deterministic): measured steady-state suboptimality f(xbar)-f* of
  real D-SGD across topologies. Regress vs T_new and vs T_gap; report which metric predicts
  the measured optimization error better, with the concrete rank inversion.

Everything below is MEASURED from the D-SGD trajectory. Deterministic, CPU, 1 thread, <40 s.
"""
import json, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def det_seed(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)

def metropolis_W(A):
    A=(A!=0).astype(float); np.fill_diagonal(A,0.0)
    deg=A.sum(1); n=A.shape[0]; W=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j and A[i,j]>0: W[i,j]=1.0/(1.0+max(deg[i],deg[j]))
    for i in range(n): W[i,i]=1.0-W[i].sum()
    return W
def ring(n,k=1):
    A=np.zeros((n,n))
    for i in range(n):
        for s in range(1,k+1): A[i,(i+s)%n]=1; A[i,(i-s)%n]=1
    return A
def torus2d(r,c):
    n=r*c; A=np.zeros((n,n)); idx=lambda a,b:(a%r)*c+(b%c)
    for a in range(r):
        for b in range(c):
            u=idx(a,b); A[u,idx(a+1,b)]=1;A[idx(a+1,b),u]=1;A[u,idx(a,b+1)]=1;A[idx(a,b+1),u]=1
    return A
def grid2d(r,c):
    n=r*c; A=np.zeros((n,n)); idx=lambda a,b:a*c+b
    for a in range(r):
        for b in range(c):
            u=idx(a,b)
            if a+1<r:A[u,idx(a+1,b)]=1;A[idx(a+1,b),u]=1
            if b+1<c:A[u,idx(a,b+1)]=1;A[idx(a,b+1),u]=1
    return A
def hypercube(d):
    n=1<<d; A=np.zeros((n,n))
    for u in range(n):
        for b in range(d): A[u,u^(1<<b)]=1
    return A
def star(n):
    A=np.zeros((n,n)); A[0,1:]=1; A[1:,0]=1; return A
def complete(n):
    A=np.ones((n,n)); np.fill_diagonal(A,0); return A

def spectral_metrics(W):
    n=W.shape[0]; lam=np.sort(np.linalg.eigvalsh(W))[::-1]; l=lam[1:]
    T_new=float(np.sum(l**2/(1.0-l**2))/n); p=1.0-float(np.max(l**2)); T_gap=(1.0-p)/p
    return lam,T_new,T_gap,p

def linfit(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    A=np.vstack([x,np.ones_like(x)]).T
    (a,b),*_=np.linalg.lstsq(A,y,rcond=None)
    yh=a*x+b; r2=1-np.sum((y-yh)**2)/np.sum((y-y.mean())**2)
    return float(a),float(b),float(r2)
def spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
    return float(np.corrcoef(rx,ry)[0,1])

TOPOS=[("ring16",metropolis_W(ring(16,1))),("ring24",metropolis_W(ring(24,1))),
       ("ring32",metropolis_W(ring(32,1))),("ring24_k2",metropolis_W(ring(24,2))),
       ("torus4x4",metropolis_W(torus2d(4,4))),("torus4x6",metropolis_W(torus2d(4,6))),
       ("grid4x4",metropolis_W(grid2d(4,4))),("grid5x5",metropolis_W(grid2d(5,5))),
       ("hypercube4",metropolis_W(hypercube(4))),("hypercube5",metropolis_W(hypercube(5))),
       ("star16",metropolis_W(star(16)))]
# fixed-n=16 family for the robustness comparison (spectral gaps span a wide range)
N16=[("ring16",metropolis_W(ring(16,1))),("torus4x4",metropolis_W(torus2d(4,4))),
     ("grid4x4",metropolis_W(grid2d(4,4))),("hypercube4",metropolis_W(hypercube(4))),
     ("star16",metropolis_W(star(16))),("complete16",metropolis_W(complete(16)))]

def dsgd_consensus_floor(W, eta=0.1, sigma2=1.0, D=1500, R=3000, burn=1400, seed=0):
    n=W.shape[0]; rng=np.random.default_rng(seed)
    X=np.zeros((n,D)); sq=np.sqrt(sigma2); accO=0.0; cnt=0
    for r in range(R):
        G=rng.standard_normal((n,D))*sq                # grad at optimum = stochastic noise
        X=W@(X-eta*G)                                   # paper Eq.2
        if r>=burn:
            X=X-X.mean(0,keepdims=True)                 # deviation (node-mean is a free random walk)
            accO+=float(np.mean(np.sum(X**2,axis=0)/n)) # (1/n)||X-Xbar||^2
            cnt+=1
    return accO/cnt

def dsgd_homogeneous_opt(W, h=0.3, eta=0.2, sigma2=1.0, D=900, R=2500, burn=1200, seed=0):
    n=W.shape[0]; rng=np.random.default_rng(seed)
    X=np.ones((n,D)); sq=np.sqrt(sigma2); accF=0.0; cnt=0
    for r in range(R):
        G=h*X + rng.standard_normal((n,D))*sq
        X=W@(X-eta*G)
        if r>=burn:
            xbar=X.mean(0,keepdims=True); accF+=float(np.mean(0.5*h*xbar**2)); cnt+=1
    return accF/cnt   # f(xbar)-f* ; x*=0, f*=0

def dsgd_heterogeneous(W, eta=0.08, D=600, R=4000, burn=2500, seed=0):
    n=W.shape[0]; rng=np.random.default_rng(seed)
    H=rng.uniform(0.5,1.5,(n,D)); B=rng.uniform(-1.0,1.0,(n,D))
    xstar=(H*B).sum(0)/H.sum(0); fstar=(0.5*H*(xstar[None,:]-B)**2).mean(0)
    X=np.zeros((n,D)); accF=0.0; cnt=0
    for r in range(R):
        G=H*(X-B); X=W@(X-eta*G)
        if r>=burn:
            xbar=X.mean(0); accF+=float(np.mean((0.5*H*(xbar[None,:]-B)**2).mean(0)-fstar)); cnt+=1
    return accF/cnt

def main():
    out={"paper":"arXiv:2606.09154v1",
         "claim":"novel (full-spectrum) analysis describes topology effect on D-SGD convergence better than prior (spectral-gap)"}
    ETA=0.1; SIG2=1.0

    print("=== Part 1: real D-SGD steady-state consensus (grad = noise at optimum) ===")
    print(f"eta={ETA} sigma^2={SIG2}. Exact target Omega_ss = eta^2 sigma^2 T_new = {ETA**2*SIG2:.4f}*T_new")
    print(f"{'topo':11s}{'T_new':>8s}{'T_gap':>9s}{'Omega_meas':>12s}{'Om/(e2Tnew)':>13s}")
    rows=[]
    for name,W in TOPOS:
        lam,T_new,T_gap,p=spectral_metrics(W)
        Om=dsgd_consensus_floor(W,eta=ETA,sigma2=SIG2,seed=det_seed("c_"+name))
        rows.append(dict(name=name,T_new=T_new,T_gap=T_gap,p=p,Omega=Om,
                         Om_over_pred=Om/(ETA**2*SIG2*T_new)))
        print(f"{name:11s}{T_new:8.4f}{T_gap:9.4f}{Om:12.6f}{Om/(ETA**2*SIG2*T_new):13.4f}")
    Tn=[r['T_new'] for r in rows]; Tg=[r['T_gap'] for r in rows]; Om=[r['Omega'] for r in rows]
    aN,bN,r2N=linfit(Tn,Om); aG,bG,r2G=linfit(Tg,Om)
    print(f"\nOmega_ss ~ T_new : slope={aN:.5f} (target eta^2 sigma^2={ETA**2*SIG2:.4f})  R^2={r2N:.6f}")
    print(f"Omega_ss ~ T_gap : slope={aG:.6f}  R^2={r2G:.6f}   (prior spectral-gap metric)")
    out["part1"]=dict(rows=rows,slope_Tnew=aN,R2_Tnew=r2N,slope_Tgap=aG,R2_Tgap=r2G,
                      target_slope=ETA**2*SIG2,
                      Tnew_ok=bool(abs(aN-ETA**2*SIG2)<0.05*ETA**2*SIG2 and r2N>=0.99),
                      Tgap_worse=bool(r2G<0.95))
    star=[r for r in rows if r['name']=='star16'][0]; r32=[r for r in rows if r['name']=='ring32'][0]
    inv=(r32['T_gap']>star['T_gap']) and (r32['Omega']<star['Omega'])
    print(f"\nRank inversion (prior metric vs measured D-SGD consensus):")
    print(f"  ring32: T_gap={r32['T_gap']:.1f} (prior:WORST) measured Omega={r32['Omega']:.4f}")
    print(f"  star16: T_gap={star['T_gap']:.1f} (prior:better) measured Omega={star['Omega']:.4f}")
    print(f"  -> prior ranks ring32 << star16 but D-SGD is BETTER on ring32: inversion={inv};"
          f" T_new ranks correctly ({r32['T_new']:.2f}<{star['T_new']:.2f}).")
    out["part1"]["rank_inversion_gap_vs_measured"]=bool(inv)

    print("\n=== Part 1b: homogeneous robustness (n=16 family, f(xbar)-f* floor) ===")
    print(f"{'topo':11s}{'p(gap)':>8s}{'T_gap':>8s}{'f(xbar)-f*':>13s}")
    rob=[]
    for name,W in N16:
        lam,T_new,T_gap,p=spectral_metrics(W)
        Fx=dsgd_homogeneous_opt(W,seed=det_seed("r_"+name))
        rob.append(dict(name=name,p=p,T_gap=T_gap,f_xbar=Fx))
        print(f"{name:11s}{p:8.4f}{T_gap:8.4f}{Fx:13.6e}")
    fx=[r['f_xbar'] for r in rob]; ps=[r['p'] for r in rob if r['p']>1e-9]
    fx_nz=[r['f_xbar'] for r in rob if r['p']>1e-9]
    gap_span=max(ps)/min(ps); f_span=max(fx_nz)/min(fx_nz)
    print(f"\nspectral gap p spans {gap_span:.0f}x across this family, yet f(xbar)-f* spans only "
          f"{f_span:.2f}x -> homogeneous optimization is robust to topology (prior 1/p^2 predicts "
          f"~{gap_span**2:.0f}x transient-time blow-up).")
    out["part1b"]=dict(rows=rob,gap_span=float(gap_span),fxbar_span=float(f_span),
                       robust=bool(f_span<2.0 and gap_span>8))

    print("\n=== Part 2: real D-SGD (heterogeneous) steady-state suboptimality f(xbar)-f* ===")
    print(f"{'topo':11s}{'T_new':>8s}{'T_gap':>9s}{'subopt_meas':>13s}")
    rows2=[]
    for name,W in TOPOS:
        lam,T_new,T_gap,p=spectral_metrics(W)
        sub=dsgd_heterogeneous(W,seed=det_seed("het_"+name))
        rows2.append(dict(name=name,T_new=T_new,T_gap=T_gap,subopt=sub))
        print(f"{name:11s}{T_new:8.4f}{T_gap:9.4f}{sub:13.6e}")
    Tn2=[r['T_new'] for r in rows2]; Tg2=[r['T_gap'] for r in rows2]; S=[r['subopt'] for r in rows2]
    _,_,r2N2=linfit(Tn2,S); _,_,r2G2=linfit(Tg2,S); rhoN=spearman(Tn2,S); rhoG=spearman(Tg2,S)
    st=[r for r in rows2 if r['name']=='star16'][0]; rr=[r for r in rows2 if r['name']=='ring32'][0]
    inv2=(rr['T_gap']>st['T_gap']) and (rr['subopt']<st['subopt'])
    print(f"\nsubopt ~ T_new : R^2={r2N2:.4f}  Spearman={rhoN:.4f}   (paper full-spectrum)")
    print(f"subopt ~ T_gap : R^2={r2G2:.4f}  Spearman={rhoG:.4f}   (prior spectral-gap)")
    print(f"rank inversion (optimization): ring32 T_gap={rr['T_gap']:.1f} subopt={rr['subopt']:.2e} vs "
          f"star16 T_gap={st['T_gap']:.1f} subopt={st['subopt']:.2e} -> inversion={inv2}")
    out["part2"]=dict(rows=rows2,R2_Tnew=r2N2,R2_Tgap=r2G2,spearman_Tnew=rhoN,spearman_Tgap=rhoG,
                      rank_inversion=bool(inv2),
                      Tnew_better=bool(r2N2>r2G2 and rhoN>=rhoG))

    verdict=(out["part1"]["Tnew_ok"] and out["part1"]["Tgap_worse"]
             and out["part1"]["rank_inversion_gap_vs_measured"]
             and out["part1b"]["robust"] and out["part2"]["Tnew_better"])
    out["verdict"]="VERIFIED" if verdict else "PARTIAL"
    print("\nVERDICT:", out["verdict"])
    (HERE/"results.json").write_text(json.dumps(out,indent=2))
    print("wrote", HERE/"results.json")

if __name__=="__main__":
    main()
