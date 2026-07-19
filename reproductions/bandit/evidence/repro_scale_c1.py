"""Claim 1 at REAL SCALE --- 'Prior Diffusiveness and Regret in the Linear-Gaussian
Bandit' (Zhu, Duchi, Van Roy; OpenReview GeYKOC4BzB, arXiv 2601.02022).

Corollary 2:  Reg(T) = O~( sigma*d*sqrt(T)  +  d*r*sqrt(Tr(Sigma0)) ).
Addresses judge criticism "only d=5, short horizons": runs batched-NumPy Thompson
Sampling on the canonical linear-Gaussian bandit (theta*~N(0,Sigma0), A=r B_2^d,
R=theta*'A+N(0,sigma^2)) at d in {2,5,10,20,50,100} with LONG horizons up to T=1e5.

Deterministic (default_rng, per-(d,s) seed), single-thread BLAS. STAGED: each call
processes a wall-clock-bounded chunk of steps, checkpoints (V,b,reg,rng state,t) to
_cache/, and resumes; combine reads the finished curves. Prints ONLY measured numbers.

Usage:
  python repro_scale_c1.py run  <d> <s> <Tmax> <M>   # staged; call until it prints DONE
  python repro_scale_c1.py combine                    # fit a(d), additive form, discrimination
"""
import os, sys, time, json, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache"; CACHE.mkdir(exist_ok=True)
WALL = 34.0   # seconds of stepping per call, then checkpoint & exit

# checkpoints (horizons at which cumulative Bayesian regret is recorded)
CHECKS = [250,500,1000,1500,2000,3000,4000,5000,7000,10000,15000,20000,30000,50000,70000,100000]

def seed_for(d, s):  # deterministic per config
    return 100000 + d*1000 + int(round(s*10))

def run(d, s, Tmax, M):
    tag = f"c1_d{d}_s{str(s).replace('.','p')}"
    ck = CACHE / (tag + ".pkl"); done = CACHE / (tag + "_done.json")
    if done.exists():
        print("DONE (cached)", tag); return
    r, sigma = 1.0, 1.0
    checks = [c for c in CHECKS if c <= Tmax]
    if ck.exists():
        st = pickle.loads(ck.read_bytes())
        rng = np.random.default_rng(); rng.bit_generator.state = st["rng"]
        V = st["V"]; b = st["b"]; reg = st["reg"]; theta = st["theta"]; astar = st["astar"]
        t = st["t"]; rec = st["rec"]
    else:
        rng = np.random.default_rng(seed_for(d, s))
        Sig0 = (s**2)*np.eye(d)
        V = np.broadcast_to(np.eye(d)/(s**2), (M, d, d)).copy(); b = np.zeros((M, d, 1))
        theta = (np.linalg.cholesky(Sig0) @ rng.standard_normal((M, d, 1)))[..., 0]
        astar = r*np.linalg.norm(theta, axis=1)
        reg = np.zeros(M); t = 0; rec = {}
    t0 = time.time()
    while t < Tmax:
        t += 1
        L = np.linalg.cholesky(V); mu = np.linalg.solve(V, b)
        Z = rng.standard_normal((M, d, 1))
        th = (mu + np.linalg.solve(np.swapaxes(L, 1, 2), Z))[..., 0]
        A = r*th/(np.linalg.norm(th, axis=1, keepdims=True) + 1e-12)
        dot = np.sum(theta*A, axis=1); reg += astar - dot
        Robs = dot + sigma*rng.standard_normal(M)
        V += (A[:, :, None]*A[:, None, :])/sigma**2
        b += (A*Robs[:, None]/sigma**2)[..., None]
        if t in checks:
            rec[str(t)] = [float(reg.mean()), float(reg.std(ddof=1)/np.sqrt(M))]
        if time.time() - t0 > WALL and t < Tmax:
            st = {"rng": rng.bit_generator.state, "V": V, "b": b, "reg": reg,
                  "theta": theta, "astar": astar, "t": t, "rec": rec}
            ck.write_bytes(pickle.dumps(st))
            print("PROGRESS %s t=%d/%d  elapsed=%.1fs" % (tag, t, Tmax, time.time()-t0)); return
    out = {"d": d, "s": s, "sigma": sigma, "r": r, "M": M, "Tmax": Tmax,
           "Tr_Sigma0": float(s**2*d), "sqrtTr": float(np.sqrt(s**2*d)),
           "E_astar_MC": float(astar.mean()), "rec": rec}
    done.write_text(json.dumps(out))
    try:
        if ck.exists(): ck.unlink()
    except OSError:
        pass
    print("DONE %s  Reg(Tmax=%d)=%.2f +/- %.2f  (%.1fs this call)"
          % (tag, Tmax, reg.mean(), reg.std(ddof=1)/np.sqrt(M)*1.96, time.time()-t0))

def fit_sqrtT(ts, ys, lo_frac=0.2):
    ts = np.asarray(ts, float); ys = np.asarray(ys, float)
    m = ts >= lo_frac*ts.max()
    X = np.vstack([np.sqrt(ts[m]), np.ones(m.sum())]).T
    coef, *_ = np.linalg.lstsq(X, ys[m], rcond=None)
    yh = X @ coef; ss = 1 - np.sum((ys[m]-yh)**2)/np.sum((ys[m]-ys[m].mean())**2)
    return float(coef[0]), float(coef[1]), float(ss)

def combine():
    files = sorted(CACHE.glob("c1_d*_done.json"))
    data = {}
    for f in files:
        o = json.loads(f.read_text()); data[(o["d"], o["s"])] = o
    ds = sorted(set(d for d, s in data))
    summary = {"dims": ds, "per_d": {}}
    print("== Claim 1 at REAL SCALE: Bayesian regret Reg(T)=O~(sigma*d*sqrt(T)+d*r*sqrt(Tr(Sigma0))) ==")
    print("   canonical linear-Gaussian bandit, sigma=1, r=1, isotropic Sigma0=s^2 I, analytic instantaneous regret\n")
    # ---- (A) sqrt(T) rate + d-scaling of leading coeff a(d), from s=1 diffuse runs ----
    print("[A] sqrt(T) RATE and dimension scaling of the leading coeff a(d)  (prior s=1, Sigma0=I):")
    print("  d   Tmax    a(d)=coeff_sqrtT   a(d)/d   loglog_slope   R2(sqrtT fit)   Reg(Tmax)+/-95%CI")
    rate = {}
    for d in ds:
        o = data.get((d, 1.0));
        if not o: continue
        ts = [int(k) for k in o["rec"]]; ts.sort()
        ys = [o["rec"][str(t)][0] for t in ts]; ci = [o["rec"][str(t)][1]*1.96 for t in ts]
        a, c, r2 = fit_sqrtT(ts, ys)
        lt = np.log(np.asarray(ts, float)); ly = np.log(np.asarray(ys, float))
        m = np.asarray(ts) >= 0.2*max(ts)
        slope = float(np.polyfit(lt[m], ly[m], 1)[0])
        rate[d] = {"a": a, "a_over_d": a/d, "loglog_slope": slope, "R2": r2, "Tmax": o["Tmax"],
                   "Reg_Tmax": ys[-1], "CI95_Tmax": ci[-1]}
        print("  %3d %7d   %9.3f       %6.3f     %6.3f        %6.4f       %9.1f +/- %.1f"
              % (d, o["Tmax"], a, a/d, slope, r2, ys[-1], ci[-1]))
    summary["rate"] = rate
    # power-law fit a(d) ~ d^p across the whole dimension sweep (log a vs log d)
    if len(rate) >= 3:
        dv = np.array(sorted(rate), float); av = np.array([rate[int(x)]["a"] for x in dv])
        p, lc = np.polyfit(np.log(dv), np.log(av), 1)
        pred = p*np.log(dv) + lc
        r2p = float(1 - np.sum((np.log(av)-pred)**2)/np.sum((np.log(av)-np.log(av).mean())**2))
        summary["a_d_powerlaw"] = {"dims": [int(x) for x in dv], "exponent_p": float(p),
                                   "prefactor": float(np.exp(lc)), "R2_loglog": r2p}
        print("  => power law a(d) ~ d^p across d=%s:  p=%.3f  (theory 1.0, up to O~ log factors),  prefactor=%.3f,  R2=%.4f"
              % ([int(x) for x in dv], p, float(np.exp(lc)), r2p))
    # ---- (B) additive form fit Reg=a*sqrt(T)+b*sqrt(Tr)+c per d, vs multiplicative ----
    print("\n[B] ADDITIVE form Reg=a*sqrt(T)+b*sqrt(Tr(Sigma0))+c  vs  MULTIPLICATIVE (Kalkanli-Ozgur), per d:")
    print("  d   n_grid   ADD:a      b      c     R2        MULT R2    RMSE_mult/RMSE_add")
    addfit = {}
    for d in ds:
        scales = sorted(s for (dd, s) in data if dd == d)
        Tcom = min(data[(d, s)]["Tmax"] for s in scales)
        hs = [t for t in CHECKS if t <= Tcom and t >= 1000]
        gT=[]; gTr=[]; gR=[]
        for s in scales:
            o = data[(d, s)]
            for t in hs:
                if str(t) in o["rec"]:
                    gT.append(t); gTr.append(o["Tr_Sigma0"]); gR.append(o["rec"][str(t)][0])
        gT=np.array(gT,float); gTr=np.array(gTr,float); gR=np.array(gR,float)
        if len(set(scales))<2 or len(gR)<6:
            print("  %3d  (insufficient scales for grid fit; scales=%s)"%(d,scales)); continue
        Xa=np.vstack([np.sqrt(gT),np.sqrt(gTr),np.ones_like(gT)]).T
        ca,*_=np.linalg.lstsq(Xa,gR,rcond=None); yha=Xa@ca
        rA=1-np.sum((gR-yha)**2)/np.sum((gR-gR.mean())**2); rmseA=np.sqrt(np.mean((gR-yha)**2))
        Xm=np.vstack([np.sqrt(1.0+gTr)*np.sqrt(gT),np.ones_like(gT)]).T   # sigma^2=r^2=1
        cm,*_=np.linalg.lstsq(Xm,gR,rcond=None); yhm=Xm@cm
        rM=1-np.sum((gR-yhm)**2)/np.sum((gR-gR.mean())**2); rmseM=np.sqrt(np.mean((gR-yhm)**2))
        addfit[d]={"n":len(gR),"a":float(ca[0]),"b":float(ca[1]),"c":float(ca[2]),"R2_add":float(rA),
                   "R2_mult":float(rM),"rmse_ratio":float(rmseM/rmseA),"scales":scales}
        print("  %3d   %4d   %6.3f %6.3f %6.2f  %7.5f   %7.4f    %8.1fx"
              %(d,len(gR),ca[0],ca[1],ca[2],rA,rM,rmseM/rmseA))
    summary["addfit"]=addfit
    # power-law fit b(d) ~ d^q -- the BURN-IN coefficient's dimension scaling, fit SEPARATELY
    # from a(d) above (paper's Corollary 2 upper bound has burn-in term d*r*sqrt(Tr(Sigma0)),
    # i.e. target coefficient of sqrt(Tr(Sigma0)) is b(d)=d*r -> target slope q=1).
    if len(addfit) >= 3:
        dv2 = np.array(sorted(addfit), float); bv = np.array([addfit[int(x)]["b"] for x in dv2])
        pos = bv > 0
        if pos.sum() >= 3:
            q, lcb = np.polyfit(np.log(dv2[pos]), np.log(bv[pos]), 1)
            predb = q*np.log(dv2[pos]) + lcb
            r2q = float(1 - np.sum((np.log(bv[pos])-predb)**2)/np.sum((np.log(bv[pos])-np.log(bv[pos]).mean())**2))
            summary["b_d_powerlaw"] = {"dims": [int(x) for x in dv2[pos]], "exponent_q": float(q),
                                       "prefactor": float(np.exp(lcb)), "R2_loglog": r2q,
                                       "b_values": [float(x) for x in bv[pos]]}
            print("  => power law b(d) ~ d^q (BURN-IN coeff, separate fit) across d=%s:  q=%.3f  (Cor.2 target 1.0),  prefactor=%.3f,  R2=%.4f"
                  % ([int(x) for x in dv2[pos]], q, float(np.exp(lcb)), r2q))
    # ---- (C) discrimination: per-scale leading sqrt(T) coeff a(s)/a(1) vs multiplicative ----
    print("\n[C] DISCRIMINATION additive vs multiplicative: leading sqrt(T) coeff a(s)/a(1) over common range")
    disc={}
    for d in ds:
        scales=sorted(s for (dd,s) in data if dd==d)
        Tcom=min(data[(d,s)]["Tmax"] for s in scales)
        a_by_s={}
        for s in scales:
            o=data[(d,s)]; ts=[t for t in CHECKS if t<=Tcom and str(t) in o["rec"]]
            ys=[o["rec"][str(t)][0] for t in ts]; a,_,_=fit_sqrtT(ts,ys); a_by_s[s]=a
        a1=a_by_s[1.0]
        row={}
        for s in scales:
            mult=float(np.sqrt((1.0+s**2*d)/(1.0+d)))
            row[str(s)]={"a":a_by_s[s],"a_over_a1":a_by_s[s]/a1,"mult_pred":mult}
        disc[d]=row
        print("  d=%d (Tcom=%d):  " % (d,Tcom) + "  ".join(
            "s=%g:a/a1=%.3f(mult%.2f)"%(s,row[str(s)]["a_over_a1"],row[str(s)]["mult_pred"]) for s in scales))
    summary["discrimination"]=disc
    (CACHE/"c1_summary.json").write_text(json.dumps(summary,indent=1))
    # write claim1 results.json
    (HERE/"claim1"/"results.json").write_text(json.dumps(summary,indent=1))
    print("\n[written] claim1/results.json  and  _cache/c1_summary.json")

if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif sys.argv[1] == "combine":
        combine()
