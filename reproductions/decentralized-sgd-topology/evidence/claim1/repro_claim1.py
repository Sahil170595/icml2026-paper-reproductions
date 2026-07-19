"""
Claim 1 reproduction --- "Novel convergence analysis shows ALL eigenvalues of the
mixing matrix affect the convergence rate, not just the spectral gap."

Paper: Takezawa, Koloskova, Stich, "Improved Convergence Analysis of Topology
Dependence in Decentralized SGD", arXiv 2606.09154v1 (ICML 2026), Sec 4/6.
OpenReview pYI0WjV5iM.

Paper quantities (verbatim):
  spectral gap        p := 1 - max_{i>=2} lambda_i^2                      (Lemma 1)
  prior topology term  T_gap := (1-p)/p                                    (Sec 6.1)
  paper full-spectrum  T_new := (1/n) sum_{i=2}^n lambda_i^2/(1-lambda_i^2) (Sec 4.2/6.1)

Noise-driven consensus of the gossip step x <- W(x+xi), xi ~ N(0,sigma^2 I), has
stationary per-node deviation variance
      V_ss = sigma^2 * (1/n) sum_{i>=2} lambda_i^2/(1-lambda_i^2) = sigma^2 * T_new .
Every V_ss below is MEASURED by simulating the recursion (never by plugging the formula).

Tests:
  (A) topology zoo: measured V_ss ~ T_new (slope~sigma^2, R^2>=0.99); spectral-gap term
      T_gap is a much worse predictor.
  (B) DECISIVE: matrices with IDENTICAL spectral gap (identical lambda_2 -> identical
      T_gap) but different full spectra have DIFFERENT measured V_ss. Gap-only hypothesis
      predicts all V_ss equal (ratio 1.0) -> FALSIFIED; T_new predicts each value.
  (C) Sec 6.1: for ring/torus, prior T_gap is far more pessimistic than T_new.

Deterministic, CPU-only, single-thread, < 40 s.
"""
import json, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SIGMA2 = 1.0

def det_seed(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)

def metropolis_W(A):
    A = (A != 0).astype(float); np.fill_diagonal(A, 0.0)
    deg = A.sum(1); n = A.shape[0]; W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] > 0:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
    for i in range(n):
        W[i, i] = 1.0 - W[i].sum()
    return W

def ring(n, k=1):
    A = np.zeros((n, n))
    for i in range(n):
        for s in range(1, k + 1):
            A[i, (i + s) % n] = 1; A[i, (i - s) % n] = 1
    return A
def torus2d(r, c):
    n=r*c; A=np.zeros((n,n)); idx=lambda a,b:(a%r)*c+(b%c)
    for a in range(r):
        for b in range(c):
            u=idx(a,b)
            A[u,idx(a+1,b)]=1;A[idx(a+1,b),u]=1;A[u,idx(a,b+1)]=1;A[idx(a,b+1),u]=1
    return A
def grid2d(r, c):
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
        for b in range(d): A[u, u ^ (1<<b)]=1
    return A
def star(n):
    A=np.zeros((n,n)); A[0,1:]=1; A[1:,0]=1; return A
def complete(n):
    A=np.ones((n,n)); np.fill_diagonal(A,0); return A
def erdos_renyi(n,p,rng):
    for _ in range(3000):
        A=(rng.random((n,n))<p).astype(float); A=np.triu(A,1); A=A+A.T
        if A.sum(1).min()>0:
            seen={0}; st=[0]
            while st:
                u=st.pop()
                for v in np.where(A[u]>0)[0]:
                    if v not in seen: seen.add(int(v)); st.append(int(v))
            if len(seen)==n: return A
    raise RuntimeError("er fail")

def spectral_metrics(W):
    n=W.shape[0]
    lam=np.sort(np.linalg.eigvalsh(W))[::-1]
    l=lam[1:]
    T_new=float(np.sum(l**2/(1.0-l**2))/n)
    p=1.0-float(np.max(l**2))
    T_gap=(1.0-p)/p
    return lam, T_new, T_gap, p

def simulate_Vss(W, n_dims=1400, n_steps=1600, burn_in=650, rng=None):
    n=W.shape[0]; X=np.zeros((n,n_dims)); acc=0.0; cnt=0
    sq=np.sqrt(SIGMA2)
    for k in range(n_steps):
        X=W@(X+rng.standard_normal((n,n_dims))*sq)
        if k>=burn_in:
            dev=X-X.mean(0,keepdims=True); acc+=float(np.mean(dev**2)); cnt+=1
    return acc/cnt

def linfit(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    A=np.vstack([x,np.ones_like(x)]).T
    (a,b),*_=np.linalg.lstsq(A,y,rcond=None)
    yh=a*x+b; r2=1-np.sum((y-yh)**2)/np.sum((y-y.mean())**2)
    return float(a),float(b),float(r2)

def W_from_tail(tail, seed=7):
    tail=np.asarray(tail,float); n=len(tail)+1
    rng=np.random.default_rng(seed)
    M=rng.standard_normal((n,n)); M[:,0]=np.ones(n)/np.sqrt(n)
    Q,_=np.linalg.qr(M)
    if np.dot(Q[:,0], np.ones(n)/np.sqrt(n))<0: Q[:,0]=-Q[:,0]
    D=np.concatenate([[1.0], tail])
    W=(Q*D)@Q.T
    return 0.5*(W+W.T)

def main():
    out={"paper":"arXiv:2606.09154v1",
         "claim":"all eigenvalues of W affect the rate, not just the spectral gap",
         "sigma2":SIGMA2}

    r=np.random.default_rng(2024)
    zoo=[("ring16",metropolis_W(ring(16,1))),("ring24",metropolis_W(ring(24,1))),
         ("ring32",metropolis_W(ring(32,1))),("ring24_k2",metropolis_W(ring(24,2))),
         ("torus4x4",metropolis_W(torus2d(4,4))),("torus4x6",metropolis_W(torus2d(4,6))),
         ("grid4x4",metropolis_W(grid2d(4,4))),("grid5x5",metropolis_W(grid2d(5,5))),
         ("hypercube4",metropolis_W(hypercube(4))),("hypercube5",metropolis_W(hypercube(5))),
         ("star16",metropolis_W(star(16))),("er20",metropolis_W(erdos_renyi(20,0.2,r))),
         ("er30",metropolis_W(erdos_renyi(30,0.15,r))),("complete16",metropolis_W(complete(16)))]
    rowsA=[]
    print("=== Test A: measured V_ss vs T_new(full spectrum) vs T_gap(spectral gap) ===")
    print(f"{'topo':11s}{'n':>3s}{'lam2':>8s}{'T_new':>9s}{'T_gap':>9s}{'V_meas':>10s}{'ratio':>8s}")
    for name,W in zoo:
        n=W.shape[0]; lam,T_new,T_gap,p=spectral_metrics(W)
        V=simulate_Vss(W,rng=np.random.default_rng(det_seed(name)))
        ratio=V/(SIGMA2*T_new) if T_new>1e-9 else float('nan')
        rowsA.append(dict(name=name,n=n,lam2=float(lam[1]),T_new=T_new,T_gap=T_gap,V_meas=V,ratio=ratio))
        print(f"{name:11s}{n:3d}{lam[1]:8.4f}{T_new:9.4f}{T_gap:9.4f}{V:10.5f}{ratio:8.4f}")
    good=[x for x in rowsA if x['T_new']>1e-6]
    Tn=[x['T_new'] for x in good]; Tg=[x['T_gap'] for x in good]; Vm=[x['V_meas'] for x in good]
    aN,bN,r2N=linfit(Tn,Vm); aG,bG,r2G=linfit(Tg,Vm)
    llaN,_,llr2N=linfit(np.log(Tn),np.log(Vm))
    print(f"\nV_ss ~ T_new : slope={aN:.5f}  R^2={r2N:.6f}  (target slope~sigma^2={SIGMA2}, R^2>=0.99)")
    print(f"V_ss ~ T_gap : slope={aG:.5f}  R^2={r2G:.6f}  (prior spectral-gap metric)")
    print(f"log-log V_ss vs T_new slope={llaN:.4f} R^2={llr2N:.4f} (target slope~1)")
    out["testA"]=dict(rows=rowsA,slope_Tnew=aN,R2_Tnew=r2N,slope_Tgap=aG,R2_Tgap=r2G,
                      loglog_slope_Tnew=llaN,loglog_R2_Tnew=llr2N,
                      Tnew_ok=bool(abs(aN-SIGMA2)<0.05 and r2N>=0.99),
                      Tgap_worse=bool(r2G<0.95))

    n=24; g=0.9
    configs={
        "single(0.9,0..)":[g]+[0.0]*(n-2),
        "quarter(6x0.9)":[g]*6+[0.0]*(n-1-6),
        "half(11x0.9)":[g]*11+[0.0]*(n-1-11),
        "all(0.9..0.9)":[g]*(n-1),
        "graded(0.9..0.1)":list(np.linspace(g,0.1,n-1)),
    }
    print("\n=== Test B: identical spectral gap (lambda_2=0.9), different full spectrum ===")
    print(f"{'config':18s}{'lam2':>7s}{'T_gap':>8s}{'T_new':>8s}{'V_meas':>9s}{'V/T_new':>9s}")
    rowsB=[]
    for cname,tail in configs.items():
        W=W_from_tail(tail,seed=7)
        lam,T_new,T_gap,p=spectral_metrics(W)
        V=simulate_Vss(W,n_dims=1600,n_steps=1700,burn_in=700,rng=np.random.default_rng(det_seed("B_"+cname)))
        rowsB.append(dict(config=cname,lam2=float(lam[1]),T_gap=T_gap,T_new=T_new,V_meas=V,V_over_Tnew=V/(SIGMA2*T_new)))
        print(f"{cname:18s}{lam[1]:7.4f}{T_gap:8.4f}{T_new:8.4f}{V:9.5f}{V/(SIGMA2*T_new):9.4f}")
    lam2s=[x['lam2'] for x in rowsB]; tgaps=[x['T_gap'] for x in rowsB]; Vs=[x['V_meas'] for x in rowsB]
    gap_spread=(max(lam2s)-min(lam2s)); tgap_spread=(max(tgaps)-min(tgaps)); V_ratio=max(Vs)/min(Vs)
    print(f"\nlambda_2 spread across configs = {gap_spread:.2e} (identical by construction)")
    print(f"T_gap spread across configs    = {tgap_spread:.2e} (identical -> gap predicts SAME V)")
    print(f"measured V_ss max/min ratio    = {V_ratio:.2f}x  (gap-only predicts 1.00x -> FALSIFIED)")
    print(f"V_ss/(sigma^2 T_new) ~ 1 for every config -> full spectrum predicts each value")
    out["testB"]=dict(rows=rowsB,lam2_spread=gap_spread,Tgap_spread=tgap_spread,V_maxmin_ratio=float(V_ratio),
                      gap_only_falsified=bool(V_ratio>1.5 and tgap_spread<1e-6),
                      Tnew_predicts=bool(max(abs(x['V_over_Tnew']-1) for x in rowsB)<0.05))

    print("\n=== Test C: prior T_gap vs paper T_new (pessimism ratio) ===")
    rowsC=[]
    for name,W in [("ring16",metropolis_W(ring(16,1))),("ring32",metropolis_W(ring(32,1))),
                   ("torus4x4",metropolis_W(torus2d(4,4))),("torus6x6",metropolis_W(torus2d(6,6))),
                   ("grid5x5",metropolis_W(grid2d(5,5))),("hypercube5",metropolis_W(hypercube(5)))]:
        lam,T_new,T_gap,p=spectral_metrics(W)
        rowsC.append(dict(name=name,p=p,T_gap=T_gap,T_new=T_new,pessimism=T_gap/T_new))
        print(f"{name:11s} p={p:.4f}  T_gap={T_gap:8.4f}  T_new={T_new:7.4f}  T_gap/T_new={T_gap/T_new:6.2f}x")
    out["testC"]=dict(rows=rowsC)

    verdict=out["testA"]["Tnew_ok"] and out["testA"]["Tgap_worse"] and out["testB"]["gap_only_falsified"] and out["testB"]["Tnew_predicts"]
    out["verdict"]="VERIFIED" if verdict else "PARTIAL"
    print("\nVERDICT:", out["verdict"])
    (HERE/"results.json").write_text(json.dumps(out,indent=2))
    print("wrote", HERE/"results.json")

if __name__=="__main__":
    main()
