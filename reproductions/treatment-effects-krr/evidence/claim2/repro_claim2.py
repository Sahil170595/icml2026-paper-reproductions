"""
Claim 2 (fully data-driven model selection with provable adaptivity to overlap
and kernel regularity) -- independent NumPy/scipy repro of
"Estimating Continuous Treatment Effects with Two-Stage Kernel Ridge Regression"
(ICML 2026, arXiv 2604.13410, OpenReview ziqS4yXFQX).

Algorithm 2 + Theorem 4.2 (verbatim mechanism):
  Split D into train/val (n/2). Train candidates h_lambda on D_train over the
  geometric grid Lambda = {2^{k-1} log n / n : k=1..L}, L=ceil(log2 n)+1. Train a
  low-regularization PROXY h_tilde on D_val (lam_tilde ~ log n / n; M=n MC pts).
  Select  lam_hat = argmin_lambda sum_k ( h_lambda(a_k) - h_tilde(a_k) )^2, a_k~P_ref.
  Thm 4.2: h_{lam_hat} attains the SAME minimax-optimal MISE rate as the ORACLE
  regularizer of Thm 4.1, ADAPTING to unknown overlap gamma (n_eff=gamma n) and
  unknown kernel spectral decay -- without knowing either.

Declared BEFORE running.  Oracle-inequality ratio
      rho = MISE(h_{lam_hat}) / MISE(h_{lam_star}),  lam_star = argmin TRUE-MISE,
  over 6 cells:  overlap gamma in {1.0,0.5,0.25} x kernel in {RBF (exp decay,
  Case b), Laplace (poly decay, Case a)}.  Low-SNR regime (sigma=6) places the
  bias-variance optimum in the grid interior so model selection is non-trivial;
  in high-SNR the optimum sits on the grid floor and any tiny lambda ties oracle.
  PASS conditions:
    (a) mean rho <= 1.5 in EVERY cell (constant-factor oracle inequality);
    (b) rho bounded in n (rate sweep at weak overlap gamma=0.5);
    (c) selected-estimator log-log slope within 0.15 of ORACLE slope, BOTH kernels
        (minimax-rate adaptivity, no knowledge of decay);
    (d) structural adaptivity is real: oracle-optimal lambda index MOVES across
        cells (range >= 2 grid steps) AND the theory-default fixed lambda
        (floor = log n/n, i.e. Case-(b)-optimal) has a larger worst-cell ratio
        than the data-driven rule (a fixed a-priori lambda is not adaptive).
  We also report the best oracle-tuned single fixed lambda as an honest caveat.
  Falsification: rho grows with n, lam_hat far from lam_star, or a fixed lambda
  matches the data-driven rule in every cell.

Self-contained, CPU-only, deterministic (fixed seeds). NumPy + scipy only.
"""
import time, json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

OUT = Path(__file__).with_name("results.json")

def g(a):  return np.sin(2.0*np.pi*a) + a**2
def s(x):  return np.cos(2.0*np.pi*x)

def sample_A(rng, n, gamma):
    u=rng.uniform(0,1,n)
    if abs(gamma-1.0)<1e-12: return u
    return (-gamma+np.sqrt(gamma*gamma+4.0*(1.0-gamma)*u))/(2.0*(1.0-gamma))

def sqd(u,v):
    if u.ndim==1: u=u[:,None]
    if v.ndim==1: v=v[:,None]
    uu=(u*u).sum(1)[:,None]; vv=(v*v).sum(1)[None,:]
    return np.maximum(uu+vv-2.0*u.dot(v.T),0.0)
def kA(U,V,l,fam):
    if fam=="rbf": return np.exp(-sqd(U,V)/(2*l*l))
    return np.exp(-np.sqrt(sqd(U,V))/l)
def bwA(U,fam):
    D2=sqd(U,U); iu=np.triu_indices(D2.shape[0],1)
    if fam=="rbf":
        m=np.median(D2[iu]); m=m if m>0 else np.mean(D2)+1e-9; return np.sqrt(m/2.0)+1e-9
    m=np.median(np.sqrt(D2[iu])); return (m if m>0 else 1e-3)+1e-9
def bwX(X):
    D2=sqd(X,X); iu=np.triu_indices(D2.shape[0],1); m=np.median(D2[iu])
    m=m if m>0 else np.mean(D2)+1e-9; return np.sqrt(m/2.0)+1e-9

def stage12(rng,X,A,Y,fam):
    n=len(Y); lx=bwX(X); la=bwA(A,fam)
    Kx=np.exp(-sqd(X,X)/(2*lx*lx)); G=Kx*kA(A,A,la,fam)
    cf=cho_factor(G+np.log(n)*np.eye(n),lower=True,check_finite=False)
    alpha=cho_solve(cf,Y,check_finite=False)
    aq=rng.uniform(0,1,n); w=alpha*Kx.mean(0)
    m=kA(aq,A,la,fam).dot(w)
    return aq,m,bwA(aq,fam)

def one_fit(rng,X,A,Y,fam,adense,gtrue,atil):
    n=len(Y); h=n//2
    L=int(np.ceil(np.log2(n)))+1
    grid=np.array([2.0**(k-1)*np.log(n)/n for k in range(1,L+1)])
    aq,m,la3=stage12(rng,X[:h],A[:h],Y[:h],fam)
    KH=kA(aq,aq,la3,fam); ew,ev=np.linalg.eigh(KH); ew=np.maximum(ew,0.0)
    Vt_m=ev.T.dot(m)
    C=ev.dot(Vt_m[:,None]/(ew[:,None]+h*grid[None,:]))
    Hd=kA(adense,aq,la3,fam).dot(C); Ha=kA(atil,aq,la3,fam).dot(C)
    aqv,mv,la3v=stage12(rng,X[h:2*h],A[h:2*h],Y[h:2*h],fam)
    lamt=np.log(2*h)/(2*h); KHv=kA(aqv,aqv,la3v,fam)
    cfv=cho_factor(KHv+2*h*lamt*np.eye(len(aqv)),lower=True,check_finite=False)
    proxy_atil=kA(atil,aqv,la3v,fam).dot(cho_solve(cfv,mv,check_finite=False))
    mise=np.mean((Hd-gtrue[:,None])**2,axis=0)
    lam_star=int(np.argmin(mise))
    sel=np.sum((Ha-proxy_atil[:,None])**2,axis=0)
    lam_hat=int(np.argmin(sel))
    return mise, lam_star, lam_hat, L

def main():
    t0=time.time(); d=3; cval=0.4; sigma=6.0; R=10
    adense=np.linspace(0,1,300); gtrue=g(adense)
    gammas=[1.0,0.5,0.25]; fams=["rbf","lap"]; n_main=1000
    print("="*82); print("Claim 2 | Data-driven model selection adaptivity (Alg 2 / Thm 4.2) | 2604.13410"); print("="*82)
    print("\n[A] Oracle-inequality table  n=%d, d=%d, R=%d, sigma=%.1f, M=n   (rho=MISE_sel/MISE_oracle)"%(n_main,d,R,sigma))
    print("%7s %6s %6s %10s %10s %6s %5s %7s"%("kernel","gamma","n_eff","MISE_orac","MISE_sel","rho","idx*","idxhat"))
    table={}; worst_dd=0.0; cellkeys=[]; meancurve={}; oracle_cell={}; idxstars=[]
    for fam in fams:
        for gm in gammas:
            key="%s_g%s"%(fam,gm); cellkeys.append(key)
            miserows=[]; ls=[]; lh=[]; orc=[]; sel=[]
            for r in range(R):
                rng=np.random.default_rng(20260+1000*(fam=="lap")+int(100*gm)+r)
                X=rng.uniform(0,1,(n_main,d)); A=sample_A(rng,n_main,gm)
                Y=g(A)+s(X).dot(cval*np.ones(d))+rng.normal(0,sigma,n_main)
                atil=rng.uniform(0,1,n_main)
                mise,i_star,i_hat,L=one_fit(rng,X,A,Y,fam,adense,gtrue,atil)
                miserows.append(mise); ls.append(i_star); lh.append(i_hat)
                orc.append(mise[i_star]); sel.append(mise[i_hat])
            mo=float(np.mean(orc)); ms=float(np.mean(sel)); rho=ms/mo
            worst_dd=max(worst_dd,rho); imed=float(np.median(ls)); idxstars.append(imed)
            meancurve[key]=np.mean(np.array(miserows),axis=0); oracle_cell[key]=mo
            print("%7s %6.2f %6.0f %10.3e %10.3e %6.3f %5.1f %7.1f"%(fam,gm,gm*n_main,mo,ms,rho,imed,float(np.median(lh))))
            table[key]={"mise_oracle":mo,"mise_sel":ms,"rho":rho,
                "med_lam_star":float(np.median(ls)),"med_lam_hat":float(np.median(lh)),"n_eff":gm*n_main}
    Lc=len(next(iter(meancurve.values())))
    floor_worst=float(max(meancurve[k][0]/oracle_cell[k] for k in cellkeys))
    large_worst=float(max(meancurve[k][min(Lc-1,8)]/oracle_cell[k] for k in cellkeys))
    fixed_worst=[max(meancurve[k][j]/oracle_cell[k] for k in cellkeys) for j in range(Lc)]
    best_fixed_idx=int(np.argmin(fixed_worst)); best_fixed_worst=float(fixed_worst[best_fixed_idx])
    idx_range=float(max(idxstars)-min(idxstars))
    print("\n  oracle-optimal idx* per cell = %s  (range=%.1f grid steps -> lambda* is regime-dependent)"%([round(x,1) for x in idxstars],idx_range))
    print("  worst-cell ratio vs oracle:")
    print("    data-driven (Algorithm 2, no oracle) ........ %.3f"%worst_dd)
    print("    fixed lambda = theory floor log n / n (idx0) . %.3f   <- a-priori default, Case-(b) optimal"%floor_worst)
    print("    fixed lambda = large (idx %d) ................ %.3f"%(min(Lc-1,8),large_worst))
    print("    best oracle-tuned single fixed lambda (idx %d)  %.3f   (needs ground truth; honest caveat)"%(best_fixed_idx,best_fixed_worst))

    print("\n[B] Rate adaptivity at weak overlap gamma=0.5: selected slope vs oracle slope")
    ns=[500,1000,2000]; Rr=8; rate={}
    print("%7s "%"kernel"+" ".join("n=%5d"%n for n in ns)+"   slope_sel slope_orac")
    for fam in fams:
        mo_n=[]; ms_n=[]
        for n in ns:
            orc=[]; sel=[]
            for r in range(Rr):
                rng=np.random.default_rng(77000+1000*(fam=="lap")+n+r)
                X=rng.uniform(0,1,(n,d)); A=sample_A(rng,n,0.5)
                Y=g(A)+s(X).dot(cval*np.ones(d))+rng.normal(0,sigma,n)
                atil=rng.uniform(0,1,n)
                mise,i_star,i_hat,L=one_fit(rng,X,A,Y,fam,adense,gtrue,atil)
                orc.append(mise[i_star]); sel.append(mise[i_hat])
            mo_n.append(float(np.mean(orc))); ms_n.append(float(np.mean(sel)))
        def sl(v):
            x=np.log(ns); y=np.log(v); return float(np.linalg.lstsq(np.vstack([x,np.ones_like(x)]).T,y,rcond=None)[0][0])
        ss=sl(ms_n); so=sl(mo_n)
        print("%7s "%fam+" ".join("%7.2e"%v for v in ms_n)+"   %+8.3f %+8.3f"%(ss,so))
        rate[fam]={"n":ns,"mise_sel":ms_n,"mise_oracle":mo_n,"slope_sel":ss,"slope_oracle":so}

    print("\n"+"="*82)
    cond_a=worst_dd<=1.5
    cond_c=all(abs(rate[f]["slope_sel"]-rate[f]["slope_oracle"])<=0.15 for f in fams)
    cond_d=(idx_range>=2) and (floor_worst>worst_dd)
    ratios_bigN={f:rate[f]["mise_sel"][-1]/rate[f]["mise_oracle"][-1] for f in fams}
    cond_b=all(v<=1.5 for v in ratios_bigN.values())
    print("(a) worst-cell oracle ratio %.3f <= 1.5 : %s"%(worst_dd,cond_a))
    print("(b) selected/oracle at largest n %s <=1.5 : %s"%({k:round(v,3) for k,v in ratios_bigN.items()},cond_b))
    print("(c) selected rate within 0.15 of oracle rate (both kernels): %s"%cond_c)
    print("(d) lambda* moves (range %.1f>=2) AND theory-floor fixed %.3f > data-driven %.3f : %s"%(idx_range,floor_worst,worst_dd,cond_d))
    verified=cond_a and cond_b and cond_c and cond_d
    print("\nOVERALL Claim-2 adaptivity PASS: %s"%verified)
    print("elapsed: %.1fs"%(time.time()-t0))
    OUT.write_text(json.dumps({"claim":"Claim 2 data-driven model selection adaptivity (Alg 2 / Thm 4.2)",
        "arxiv":"2604.13410","id":"ziqS4yXFQX","n_main":n_main,"d":d,"R":R,"sigma":sigma,"M":"n",
        "rule":"worst-cell oracle-ratio<=1.5 AND bounded in n AND selected rate within 0.15 of oracle (both kernels) AND lambda* range>=2 with theory-floor fixed worse than data-driven",
        "table":table,"worst_dd_rho":worst_dd,"floor_worst_rho":floor_worst,"large_worst_rho":large_worst,
        "best_fixed_idx":best_fixed_idx,"best_fixed_worst_rho":best_fixed_worst,"idx_star_per_cell":idxstars,"idx_star_range":idx_range,
        "rate":rate,"ratios_bigN":ratios_bigN,"cond_a":cond_a,"cond_b":cond_b,"cond_c":cond_c,"cond_d":cond_d,
        "overall_pass":bool(verified)},indent=2))
    print("wrote results.json")

if __name__=="__main__":
    main()
