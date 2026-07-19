"""
Independent CPU reproduction of CLAIM 2 of
  "Stochastic Linear Bandits with Parameter Noise" (arXiv 2601.23164, ICML 2026).

CLAIM 2 (verbatim from scored set):
  "For l_p unit ball action sets with p<=2, minimax regret is Theta(sqrt(dT sigma_e^2)),
   substantially better than the d sqrt(T) regret in the classic additive noise model."
Paper anchors: Theorem 3.7 (VALEE upper bound, l_p unit ball, p in (1,2], dual q>=2, KNOWN Sigma):
   R_T = Otilde(d + sqrt(dT q log(1/delta) sigma_q^2)),  sigma_q^2 = (sum_i Sigma_ii^{q/2})^{2/q}.
Theorem 4.1 (lower bound, same setting): R_T = Omegatilde(sqrt(dT sigma_q^2)).
   => minimax regret Theta(sqrt(dT sigma_q^2))  (here sigma_e^2 == sigma_q^2).
Classic additive-noise l_2-ball minimax regret is Theta(d sqrt(T)) (Dani et al.; Lattimore &
Szepesvari Ch.24).  Improvement factor sqrt(d) when sigma_q^2 = Theta(1).

Reward model.  l_2 unit ball action set.
  * Parameter noise (paper): X_t = a^T theta_t, theta_t~nu, Cov=Sigma.  We take the isotropic
    parameter perturbation Sigma = (sigma^2/d) I, so sigma_q^2 = tr(Sigma) = sigma^2 (q=2) and the
    per-ACTION reward variance is sigma^2(a) = a^T Sigma a = sigma^2/d  (the parameter perturbation
    is diluted across the d coordinates of a unit action -- this is the model's defining feature).
  * Classic additive noise: X_t = a^T theta* + eta,  Var(eta) = sigma^2 (constant, action-independent).
  Both are compared at the SAME model-natural unit noise level: sigma_q^2(param) = Var(eta)(additive)
  = sigma^2.  (Disclosed normalisation; the resulting per-reward-variance gap sigma^2/d vs sigma^2 is
  exactly the paper's thesis, not an artefact -- see the estimation-error control below.)

Algorithm.  ONE identical explore-then-commit routine (VALEE / Algorithm 3 style, known covariance):
  explore each standard basis direction e_i with an optimally-tuned budget, form the coordinate-wise
  least-squares estimate theta_hat, then commit to the ball optimum a_hat = theta_hat/||theta_hat||.

Checkable predictions and ACCEPT rules (numbers must come from real stdout):
  P1 param d-slope    : log-log slope of R_T vs d in [0.40,0.70]   (sqrt(dT sigma_q^2))
  P2 additive d-slope : log-log slope of R_T vs d in [0.85,1.15]   (d sqrt(T))
  P3 param T-slope    : log-log slope of R_T vs T in [0.40,0.60]   (sqrt T)
  P4 improvement      : (R_additive/R_param) grows with d; log-slope in [0.30,0.65] (-> sqrt d),
                        and the ratio at the largest d is >= 3x.
  P5 mechanism        : with EQUAL per-coordinate exploration budget, ||theta_hat-theta*|| grows as
                        sqrt(d) in additive (slope ~0.5) but is ~d-independent in param (slope < 0.30):
                        the sqrt(tr Sigma) vs sqrt(d) estimation-error gap of Lemma 3.11.
Falsification: param d-slope >= 0.85 (param also scales like d, no improvement) OR additive d-slope
  <= 0.65 OR the additive/param ratio does not grow with d.
CPU-only, deterministic (numpy.random.default_rng, fixed seeds), OMP/OPENBLAS threads = 1.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

OUT = Path(__file__).with_name("results.json")

def valee_etc(theta_star, noise_var_coord, T, rng):
    """Explore-then-commit on the l2 unit ball. noise_var_coord:(d,) = reward variance when
    playing e_i (known-covariance tuning). Returns (cumulative_regret, ||theta_hat-theta*||)."""
    d = len(theta_star); nts = float(np.linalg.norm(theta_star))
    S = d*float(np.sum(noise_var_coord))
    Texp_opt = np.sqrt(T*S)/(2*max(nts,1e-6))
    n = int(max(1, min(Texp_opt/d, T/(2*d)))); Texp = n*d
    theta_hat = theta_star + np.sqrt(noise_var_coord/n)*rng.standard_normal(d)
    reg_expl = n*float(np.sum(nts - theta_star))          # gap of each basis pull on the ball
    ahat = theta_hat/np.linalg.norm(theta_hat)
    exploit_gap = nts - float(ahat@theta_star)
    return reg_expl + (T-Texp)*exploit_gap, float(np.linalg.norm(theta_hat-theta_star))

def noise_coord(model, d, sigma2):
    return np.full(d, sigma2/d) if model=="param" else np.full(d, sigma2)

def d_sweep(model, ds, T, sigma2, nseeds):
    Rs=[]; Es=[]
    for d in ds:
        u = np.random.default_rng(1000+d).standard_normal(d); u/=np.linalg.norm(u)
        nvc = noise_coord(model,d,sigma2)
        vv=[valee_etc(u,nvc,T,np.random.default_rng(5000*sd+d)) for sd in range(nseeds)]
        Rs.append(float(np.mean([x[0] for x in vv]))); Es.append(float(np.mean([x[1] for x in vv])))
    return np.array(Rs), np.array(Es)

def mech_equal_budget(ds, n_per_coord, sigma2, nseeds):
    """Equal per-coordinate exploration budget n; measure ||theta_hat-theta*|| in both models."""
    ep=[]; ea=[]
    for d in ds:
        u=np.random.default_rng(3000+d).standard_normal(d); u/=np.linalg.norm(u)
        vp=[]; va=[]
        for sd in range(nseeds):
            r=np.random.default_rng(6000*sd+d)
            th_p=u+np.sqrt(noise_coord("param",d,sigma2)/n_per_coord)*r.standard_normal(d)
            r2=np.random.default_rng(6000*sd+d+77)
            th_a=u+np.sqrt(noise_coord("additive",d,sigma2)/n_per_coord)*r2.standard_normal(d)
            vp.append(np.linalg.norm(th_p-u)); va.append(np.linalg.norm(th_a-u))
        ep.append(float(np.mean(vp))); ea.append(float(np.mean(va)))
    return np.array(ep), np.array(ea)

def main():
    t0=time.time(); os.environ.setdefault("OMP_NUM_THREADS","1")
    sigma2=1.0; nseeds=8; res={}
    print("="*72); print("CLAIM 2  l2-ball param-noise Theta(sqrt(dT sigma_q^2)) vs additive d sqrt(T)")
    ds=[2,4,8,16,32,64]; T=40000
    Rp,Ep=d_sweep("param",ds,T,sigma2,nseeds)
    Ra,Ea=d_sweep("additive",ds,T,sigma2,nseeds)
    slp=float(np.polyfit(np.log(ds),np.log(Rp),1)[0]); sla=float(np.polyfit(np.log(ds),np.log(Ra),1)[0])
    ratio=Ra/Rp; rslope=float(np.polyfit(np.log(ds),np.log(ratio),1)[0])
    res["P1_param_dslope"]={"ds":ds,"R_param":[round(x,1) for x in Rp],"slope":round(slp,4),
        "rule":"[0.40,0.70] -> sqrt(dT sigma_q^2)","accept":0.40<=slp<=0.70}
    res["P2_additive_dslope"]={"R_additive":[round(x,1) for x in Ra],"slope":round(sla,4),
        "rule":"[0.85,1.15] -> d sqrt(T)","accept":0.85<=sla<=1.15}
    res["P4_improvement"]={"ratio_add_over_param":[round(x,2) for x in ratio],"loglog_slope":round(rslope,4),
        "ratio_at_max_d":round(float(ratio[-1]),2),"rule":"ratio grows ~sqrt(d): slope in [0.30,0.65] and max ratio>=3",
        "accept":bool(0.30<=rslope<=0.65 and ratio[-1]>=3.0)}
    print("[P1] param   d-slope=",round(slp,3)," R=",[round(x,1) for x in Rp]," accept=",res["P1_param_dslope"]["accept"])
    print("[P2] additive d-slope=",round(sla,3)," R=",[round(x,1) for x in Ra]," accept=",res["P2_additive_dslope"]["accept"])
    print("[P4] additive/param ratio=",[round(x,2) for x in ratio]," slope=",round(rslope,3),
          " max=",round(float(ratio[-1]),2)," accept=",res["P4_improvement"]["accept"])
    # P3 param T-scaling, fixed d
    Ts=[5000,10000,20000,40000,80000]; d=10
    u=np.random.default_rng(7).standard_normal(d); u/=np.linalg.norm(u)
    Rt=[float(np.mean([valee_etc(u,noise_coord("param",d,sigma2),T2,np.random.default_rng(9*sd+T2))[0] for sd in range(nseeds)])) for T2 in Ts]
    tsl=float(np.polyfit(np.log(Ts),np.log(Rt),1)[0])
    res["P3_param_Tslope"]={"Ts":Ts,"R_T":[round(x,1) for x in Rt],"slope":round(tsl,4),
        "rule":"[0.40,0.60] -> sqrt T","accept":0.40<=tsl<=0.60}
    print("[P3] param T-slope=",round(tsl,3)," accept=",res["P3_param_Tslope"]["accept"])
    # P5 mechanism: equal per-coord budget, estimation-error scaling in d
    Ep2,Ea2=mech_equal_budget(ds,200,sigma2,64)
    mp=float(np.polyfit(np.log(ds),np.log(Ep2),1)[0]); ma=float(np.polyfit(np.log(ds),np.log(Ea2),1)[0])
    res["P5_mechanism"]={"ds":ds,"err_param":[round(x,4) for x in Ep2],"err_additive":[round(x,4) for x in Ea2],
        "param_slope":round(mp,4),"additive_slope":round(ma,4),
        "rule":"additive err ~sqrt(d) (slope~0.5); param err ~const (slope<0.30): sqrt(trSigma) vs sqrt(d)",
        "accept":bool(ma>=0.40 and mp<0.30)}
    print("[P5] est-err slope param=",round(mp,3)," additive=",round(ma,3)," accept=",res["P5_mechanism"]["accept"])
    res["all_accept"]=bool(all(bool(res[k]["accept"]) for k in res if k.startswith("P")))
    res["runtime_sec"]=round(time.time()-t0,2)
    res["env"]={"python":sys.version.split()[0],"numpy":np.__version__,"seeds":nseeds,"sigma2":sigma2,
                "normalisation":"param sigma_q^2 = additive Var(eta) = sigma^2"}
    print("="*72); print("ALL ACCEPT =",res["all_accept"]," runtime=",res["runtime_sec"],"s")
    OUT.write_text(json.dumps(res,indent=2,default=lambda o:o.item() if hasattr(o,"item") else str(o))); print("wrote",OUT)

if __name__=="__main__":
    main()
