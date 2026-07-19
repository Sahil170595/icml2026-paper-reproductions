"""
Independent CPU reproduction of CLAIM 1 of
  "Stochastic Linear Bandits with Parameter Noise" (arXiv 2601.23164, ICML 2026).

CLAIM 1 (verbatim from scored set):
  "Stochastic linear bandits with parameter noise achieve regret upper bound of
   O(sqrt(dT log(K/delta) sigma^2_max)) with matching lower bound O(d sqrt(T) sigma^2_max)
   tight up to logarithmic factors."
Paper anchors: Theorem 3.1 (VASE upper bound, general finite action set):
   R_T = Otilde(d^2 + sqrt(dT log(K/delta) M_sigma)),  M_sigma <= max_a sigma^2(a) = sigma^2_max.
Lower bound Theorem 4.4 (l_p unit ball p>2 / general):  R_T = Omegatilde(d sqrt(T sigma^2_max)).
On the p>2 / hypercube family the UB (Corollary 3.3: d^2 + d sqrt(T sigma^2_max)) and LB coincide
=> Theta(d sqrt(T) sigma_max): "matching, tight up to log factors".

Reward model (parameter noise): X_t = a_t^T theta_t,  theta_t ~ nu iid, E[theta_t]=theta*,
Cov(theta_t)=Sigma  =>  Var(X_t|a_t)=a_t^T Sigma a_t = sigma^2(a).  Regret vs a*=argmax_a a^T theta*.

Algorithm (independent implementation of VASE / Algorithm 2 = G-optimal-design phased
elimination). In each phase l: compute a G-optimal (D-optimal, Kiefer-Wolfowitz) design over the
ACTIVE arms via Frank-Wolfe; draw n_l(a) ~ pi_l(a)*(2 d /eps_l^2) log(K/delta_l)*sigma^2 samples;
form the (optionally inverse-variance weighted) least-squares estimate; eliminate arms whose
estimate is > 2 eps_l suboptimal.  The sufficient statistic of the aggregated rewards is simulated
exactly for speed (a sum of n rewards ~ Normal(n a^T theta*, n sigma^2(a))).

Checkable predictions and ACCEPT rules (numbers must come from real stdout):
  A  T-scaling : minimax/critical-gap family, log-log slope of R_T vs T in [0.40,0.60]  (sqrt T)
  B  sigma     : doubling sigma_max doubles R_T (ratios in [1.70,2.30])                 (sqrt sigma^2)
  C  log K     : 256x more arms -> <1.6x regret (poly-log in K; rules out any K^c)       (log(K/delta))
  D  d (fix K) : log-log slope of R_T vs d in [0.40,0.75]                                (sqrt d)
  E  FORM FIT  : R / sqrt(dT log(K/delta) sigma^2_max) ~ const over a joint (d,K,T,sigma)
                 sweep; coefficient of variation CV < 0.20                               (whole UB form)
  F  LOWER BND : hypercube action set K=2^d (log K ~ d): d-slope climbs to ~ d sqrt(T),
                 clearly steeper than the fixed-K sqrt(d) case (slope_hyper >= slope_fixedK+0.15);
                 hypercube also shows T-slope ~0.5 and sigma-ratio ~2  => d sqrt(T) sigma_max rate.
CPU-only, deterministic (numpy.random.default_rng, fixed seeds), OMP/OPENBLAS threads = 1.
"""
import json, os, sys, time, hashlib, itertools
from pathlib import Path
import numpy as np

DELTA = 0.05
OUT = Path(__file__).with_name("results.json")

def frank_wolfe_gopt(A, iters=150, tol=0.02):
    K, d = A.shape
    pi = np.full(K, 1.0/K)
    V = (A*pi[:,None]).T @ A + 1e-9*np.eye(d)
    for _ in range(iters):
        Vinv = np.linalg.inv(V)
        g = np.einsum('ki,ij,kj->k', A, Vinv, A)
        k = int(np.argmax(g)); gk = g[k]
        if gk <= d*(1.0+tol): break
        gamma = (gk/d - 1.0)/(gk - 1.0)
        pi *= (1.0-gamma); pi[k] += gamma
        V = (1.0-gamma)*V + gamma*np.outer(A[k], A[k])
    return pi

def run_phased(A, theta_star, sig2a, Tmax, checkpoints, aware, rng, fw_iters=150):
    K, d = A.shape
    mu = A @ theta_star; astar = int(np.argmax(mu)); gaps = mu[astar]-mu
    sigma_max2 = float(sig2a.max())
    active = np.arange(K); ell = 0; plays = 0.0; R = 0.0
    ckpts = sorted(checkpoints); ci = 0; out = {}
    while plays < Tmax and len(active) > 1 and ell < 40:
        ell += 1; eps = 0.5*(0.5**(ell-1))
        Aa = A[active]; pi = frank_wolfe_gopt(Aa, iters=fw_iters)
        delta_ell = DELTA/(ell*(ell+1)); logf = np.log(max(K,2)/delta_ell)
        if aware: weff = 1.0/np.maximum(sig2a[active],1e-12); sig_scale = 1.0
        else:     weff = np.ones(len(active));                sig_scale = sigma_max2
        n_total = 2.0*d/eps**2 * logf * sig_scale
        supp = pi > 1e-10; n_a = np.ceil(pi[supp]*n_total).astype(float)
        idx = active[supp]; w = weff[supp]
        V = (Aa[supp]*(n_a*w)[:,None]).T @ Aa[supp] + 1e-9*np.eye(d)
        means = n_a*(Aa[supp]@theta_star)
        noise = np.sqrt(n_a*np.maximum(sig2a[idx],1e-12))*rng.standard_normal(len(idx))
        b = (Aa[supp]*(w*(means+noise))[:,None]).sum(0)
        theta_hat = np.linalg.solve(V, b)
        phase_plays = float(n_a.sum()); phase_reg = float((n_a*gaps[idx]).sum())
        avg_gap = phase_reg/max(phase_plays,1e-9)
        while ci < len(ckpts) and plays+phase_plays >= ckpts[ci]:
            tc = ckpts[ci]; out[tc] = R + (tc-plays)*avg_gap; ci += 1
        plays += phase_plays; R += phase_reg
        est = Aa @ theta_hat; best = est.max(); active = active[est >= best-2.0*eps]
    resid = gaps[active].min() if len(active)>0 else 0.0
    while ci < len(ckpts):
        tc = ckpts[ci]; out[tc] = R + max(tc-plays,0.0)*resid; ci += 1
    return out

def make_arms(d, K, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((K,d)); A /= np.linalg.norm(A,axis=1,keepdims=True); return A

def crit_theta(A, u, T, d, logK, sigma, c=1.5):
    base = A@u; g = base.max()-base; typ = np.median(g[g>1e-9])
    return (c*sigma*np.sqrt(d*logK/T)/typ)*u

def hypercube(d):
    C = np.array(list(itertools.product([-1.0,1.0], repeat=d)))/np.sqrt(d); return C

def mean_over_seeds(fn, nseeds):
    return float(np.mean([fn(sd) for sd in range(nseeds)]))

def expA_T(nseeds=8):
    d,K = 6,24; A = make_arms(d,K,1); u = np.zeros(d); u[0]=1.0; logK = np.log(K/DELTA)
    Ts = [1000,2000,4000,8000,16000]; Rs=[]
    for T in Ts:
        th = crit_theta(A,u,T,d,logK,0.5)
        Sig=(0.25)*np.eye(d); s2=np.einsum('ki,ij,kj->k',A,Sig,A)
        Rs.append(mean_over_seeds(lambda sd: run_phased(A,th,s2,T,[T],False,np.random.default_rng(100*sd+T))[T], nseeds))
    slope=float(np.polyfit(np.log(Ts),np.log(Rs),1)[0])
    return {"Ts":Ts,"R_T":[round(x,2) for x in Rs],"loglog_slope":round(slope,4),
            "rule":"slope in [0.40,0.60]","accept":0.40<=slope<=0.60}

def expB_sigma(nseeds=8):
    d,K,T = 6,24,12000; A=make_arms(d,K,1); u=np.zeros(d);u[0]=1.0; logK=np.log(K/DELTA)
    sgs=[0.25,0.5,1.0,2.0]; Rs=[]
    for sg in sgs:
        th=crit_theta(A,u,T,d,logK,sg); Sig=(sg**2)*np.eye(d); s2=np.einsum('ki,ij,kj->k',A,Sig,A)
        Rs.append(mean_over_seeds(lambda sd: run_phased(A,th,s2,T,[T],False,np.random.default_rng(700*sd+int(sg*1000)))[T], nseeds))
    ratios=[Rs[i+1]/Rs[i] for i in range(3)]
    return {"sigmas":sgs,"R_T":[round(x,2) for x in Rs],"doubling_ratios":[round(r,3) for r in ratios],
            "rule":"each doubling ratio in [1.70,2.30]","accept":all(1.70<=r<=2.30 for r in ratios)}

def expC_logK(nseeds=8):
    d,T,sg=6,12000,0.5; Ks=[16,64,256,1024,4096]; logKref=np.log(256/DELTA); Rs=[]
    for K in Ks:
        A=make_arms(d,K,5); u=np.zeros(d);u[0]=1.0
        th=crit_theta(A,u,T,d,logKref,sg); Sig=(sg**2)*np.eye(d); s2=np.einsum('ki,ij,kj->k',A,Sig,A)
        Rs.append(mean_over_seeds(lambda sd: run_phased(A,th,s2,T,[T],False,np.random.default_rng(800*sd+K))[T], nseeds))
    growth=Rs[-1]/Rs[0]; slope=float(np.polyfit(np.log([np.log(K/DELTA) for K in Ks]),np.log(Rs),1)[0])
    return {"Ks":Ks,"R_T":[round(x,2) for x in Rs],"growth_256x_K":round(growth,3),
            "loglog_slope_vs_logK":round(slope,3),
            "rule":"256x arms -> <1.6x regret (poly-log; no K^c)","accept":growth<1.6}

def expD_d(nseeds=8):
    K,T,sg=24,12000,0.5; ds=[2,4,8,16]; Rs=[]
    for d in ds:
        A=make_arms(d,K,4+d); u=np.zeros(d);u[0]=1.0; logK=np.log(K/DELTA)
        th=crit_theta(A,u,T,d,logK,sg); Sig=(sg**2)*np.eye(d); s2=np.einsum('ki,ij,kj->k',A,Sig,A)
        Rs.append(mean_over_seeds(lambda sd: run_phased(A,th,s2,T,[T],False,np.random.default_rng(900*sd+d))[T], nseeds))
    slope=float(np.polyfit(np.log(ds),np.log(Rs),1)[0])
    return {"ds":ds,"R_T":[round(x,2) for x in Rs],"loglog_slope_d":round(slope,4),
            "rule":"slope in [0.40,0.75] (sqrt d, fixed K)","accept":0.40<=slope<=0.75}

def expE_formfit(nseeds=8):
    cfgs=[(4,20,8000,0.5),(6,24,12000,0.5),(8,40,16000,0.5),(5,32,10000,1.0),
          (6,24,8000,0.25),(10,50,16000,0.5),(6,64,12000,0.5),(7,30,14000,0.75)]
    rows=[]; ratios=[]
    for (d,K,T,sg) in cfgs:
        A=make_arms(d,K,3+d+K); u=np.zeros(d);u[0]=1.0; logK=np.log(K/DELTA)
        th=crit_theta(A,u,T,d,logK,sg); Sig=(sg**2)*np.eye(d); s2=np.einsum('ki,ij,kj->k',A,Sig,A); smax2=float(s2.max())
        R=mean_over_seeds(lambda sd: run_phased(A,th,s2,T,[T],False,np.random.default_rng(1200*sd+d*K+T))[T], nseeds)
        pred=np.sqrt(d*T*logK*smax2); ratios.append(R/pred)
        rows.append({"d":d,"K":K,"T":T,"sigma":sg,"R":round(R,2),"pred":round(float(pred),2),"ratio":round(R/pred,3)})
    rr=np.array(ratios); cv=float(rr.std()/rr.mean())
    return {"configs":rows,"mean_ratio":round(float(rr.mean()),3),"CV":round(cv,3),
            "rule":"R/sqrt(dT log(K/delta) sigma^2_max) ~ const, CV<0.20","accept":cv<0.20}

def expF_lb(nseeds=4):
    # d-slope on hypercube K=2^d  (want steeper than fixed-K sqrt d, -> d sqrt T)
    T,sg=12000,0.5; ds=[4,5,6,7,8,9,10]; Rs=[]
    for d in ds:
        A=hypercube(d); K=len(A); u=np.zeros(d);u[0]=1.0; logK=np.log(K/DELTA)
        th=crit_theta(A,u,T,d,logK,sg); Sig=(sg**2)*np.eye(d); s2=np.einsum('ki,ij,kj->k',A,Sig,A)
        Rs.append(mean_over_seeds(lambda sd: run_phased(A,th,s2,T,[T],False,np.random.default_rng(1100*sd+d),fw_iters=100)[T], nseeds))
    slope=float(np.polyfit(np.log(ds),np.log(Rs),1)[0])
    # T-scaling and sigma-scaling on a fixed hypercube d=8 (LB rate = d sqrt(T) sigma_max)
    d=8; A=hypercube(d); K=len(A); u=np.zeros(d);u[0]=1.0; logK=np.log(K/DELTA)
    Ts=[2000,4000,8000,16000]; RT=[]
    for T2 in Ts:
        th=crit_theta(A,u,T2,d,logK,sg); Sig=(sg**2)*np.eye(d); s2=np.einsum('ki,ij,kj->k',A,Sig,A)
        RT.append(mean_over_seeds(lambda sd: run_phased(A,th,s2,T2,[T2],False,np.random.default_rng(1300*sd+T2),fw_iters=100)[T2], nseeds))
    tslope=float(np.polyfit(np.log(Ts),np.log(RT),1)[0])
    T3=12000; Rsig=[]
    for sg2 in [0.5,1.0,2.0]:
        th=crit_theta(A,u,T3,d,logK,sg2); Sig=(sg2**2)*np.eye(d); s2=np.einsum('ki,ij,kj->k',A,Sig,A)
        Rsig.append(mean_over_seeds(lambda sd: run_phased(A,th,s2,T3,[T3],False,np.random.default_rng(1400*sd+int(sg2*10)),fw_iters=100)[T3], nseeds))
    sig_ratios=[Rsig[i+1]/Rsig[i] for i in range(2)]
    return {"ds":ds,"R_hyper":[round(x,2) for x in Rs],"hyper_d_slope":round(slope,4),
            "hyper_T_slope":round(tslope,4),"hyper_sigma_ratios":[round(r,3) for r in sig_ratios],
            "rule":"hyper d-slope >= fixedK sqrt-d slope + 0.15 (climbs to d sqrt T); T-slope~0.5; sigma-ratio~2",
            "accept": None}  # filled in main vs expD

def main():
    t0=time.time(); os.environ.setdefault("OMP_NUM_THREADS","1")
    stage = sys.argv[1] if len(sys.argv)>1 else "all"
    res={}
    print("="*72); print("CLAIM 1  VASE upper bound sqrt(dT log(K/delta) sigma^2_max) + matching LB d sqrt(T) sigma_max")
    res["A_Tscaling"]=expA_T()
    print("[A] T-scaling: T=",res["A_Tscaling"]["Ts"]); print("    R_T=",res["A_Tscaling"]["R_T"],
          " slope=",res["A_Tscaling"]["loglog_slope"]," accept=",res["A_Tscaling"]["accept"])
    res["B_sigma"]=expB_sigma()
    print("[B] sigma:",res["B_sigma"]["sigmas"]," R_T=",res["B_sigma"]["R_T"],
          " ratios=",res["B_sigma"]["doubling_ratios"]," accept=",res["B_sigma"]["accept"])
    res["C_logK"]=expC_logK()
    print("[C] logK: K=",res["C_logK"]["Ks"]," R_T=",res["C_logK"]["R_T"],
          " growth256x=",res["C_logK"]["growth_256x_K"]," accept=",res["C_logK"]["accept"])
    res["D_dscaling"]=expD_d()
    print("[D] d(fixK): d=",res["D_dscaling"]["ds"]," R_T=",res["D_dscaling"]["R_T"],
          " slope=",res["D_dscaling"]["loglog_slope_d"]," accept=",res["D_dscaling"]["accept"])
    res["E_formfit"]=expE_formfit()
    print("[E] FORM FIT mean R/pred=",res["E_formfit"]["mean_ratio"]," CV=",res["E_formfit"]["CV"],
          " accept=",res["E_formfit"]["accept"])
    res["F_lowerbound"]=expF_lb()
    fixedK_slope=res["D_dscaling"]["loglog_slope_d"]
    hyper_slope=res["F_lowerbound"]["hyper_d_slope"]
    res["F_lowerbound"]["accept"]= (hyper_slope>=fixedK_slope+0.15) and (0.40<=res["F_lowerbound"]["hyper_T_slope"]<=0.60) \
                                   and all(1.6<=r<=2.4 for r in res["F_lowerbound"]["hyper_sigma_ratios"])
    print("[F] LB hyper d-slope=",hyper_slope," (fixedK",fixedK_slope,") T-slope=",res["F_lowerbound"]["hyper_T_slope"],
          " sigma-ratios=",res["F_lowerbound"]["hyper_sigma_ratios"]," accept=",res["F_lowerbound"]["accept"])
    res["all_accept"]=all(res[k]["accept"] for k in ["A_Tscaling","B_sigma","C_logK","D_dscaling","E_formfit","F_lowerbound"])
    res["runtime_sec"]=round(time.time()-t0,1)
    res["env"]={"python":sys.version.split()[0],"numpy":np.__version__,"seeds_upper":8,"seeds_lb":4,"delta":DELTA}
    print("="*72); print("ALL ACCEPT =",res["all_accept"]," runtime=",res["runtime_sec"],"s")
    OUT.write_text(json.dumps(res,indent=2)); print("wrote",OUT)

if __name__=="__main__":
    main()
