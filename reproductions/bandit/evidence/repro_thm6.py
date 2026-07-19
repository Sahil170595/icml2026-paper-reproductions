"""Theorem 6 (Section 4) burn-in LOWER BOUND --- the registered-anchor construction,
for 'Prior Diffusiveness and Regret in the Linear-Gaussian Bandit' (Zhu, Duchi,
Van Roy; OpenReview GeYKOC4BzB, arXiv 2601.02022).

Registered claim (claims_anchored.json, GeYKOC4BzB #4):
    "Theorem 6 proves a matching lower bound showing the burn-in term dr sqrt(Tr(Sigma0))
     is unavoidable for non-pathological priors, via a bound of the form
        Reg^p(T) >= (r / (pi ||tau||_2)) * sum_{i=2}^{min{T,d}} (i-1) tau_i^2
     (Section 4, Theorem 6)."
with Sigma0 = diag(tau_1^2..tau_d^2).  Note the identity
    sum_{i=2}^{m} (i-1) tau_i^2  =  sum_{t=1}^{m-1} sum_{i>t} tau_i^2   (tau sorted descending),
i.e. Theorem 6 is a SUM over time of unexplored-tail-variance floors: after t rounds any
policy has learned at most a t-dimensional linear sketch of theta*, so the residual prior
variance R_t = sum_{i>t} tau_i^2 forces per-step Bayes regret >= (r/pi) R_t/||tau||_2.
The old 'first-step floor' r E||theta*|| ~ r sqrt(Tr(Sigma0)) is exactly the t=0 term;
summing the WHOLE schedule gives the d r sqrt(Tr(Sigma0))-order bound (isotropic:
F(d) = r s sqrt(d) (d-1) / (2 pi), i.e. ~ d^{3/2} -- a factor d stronger than the floor).

This script produces three measured objects and the sandwich between them:
 (1) F_thm6(d):    the registered Theorem-6 formula, evaluated exactly.
 (2) ORACLE(d):    the construction, simulated.  Sequential-revelation oracle in the
                   sigma->0 limit: at step t the policy knows P_t theta* exactly
                   (projection on the top-t prior eigendirections -- the information-
                   optimal schedule, one linear functional of theta* per round, which is
                   all the model's observation R = theta*'A + noise can reveal) and plays
                   the best action a_t = r P_t theta*/||P_t theta*||.  Its regret
                       Reg_oracle = r sum_{t=0}^{d-1} E[||theta*|| - ||P_t theta*||]
                   lower-bounds this best-case schedule by Monte Carlo; we also check the
                   paper's per-step form  E[||theta|| - ||P_t theta||] >= (1/pi) R_t/||tau||_2
                   at every t.
 (3) B_TS(d):      the measured Thompson-sampling burn-in (low-noise sigma=0.02 isolation,
                   the same protocol as claim1's small-scale isolation), across the full
                   dimension sweep -- the UPPER side of the sandwich.
 Sandwich + scaling: F_thm6(d) <= Reg_oracle(d) <= B_TS(d) at every d, and all three scale
 as d^{3/2} at fixed prior scale s (= the d r sqrt(Tr(Sigma0)) law), against the old
 first-step floor's d^{1/2}.

Honest scope: the paper PDF is not in the challenge bundle; the formula is taken verbatim
from the registered claim anchor, and the revelation oracle is our simulable realization of
the Section-4 information argument (one linear functional per round), not a transcription
of the paper's proof.  Deterministic seeds; single-thread BLAS; prints ONLY measured numbers.

Usage:  python repro_thm6.py         (single run, ~2 min; writes claim2/results_thm6.json)
"""
import json, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
from scipy.special import gammaln

HERE = Path(__file__).resolve().parent
DIMS = [2, 5, 10, 20, 50, 100]
S = 2.0                    # prior scale (Sigma0 = s^2 I), same as floor/EPL sections
R_ACT, SIGMA_LO = 1.0, 0.02
# low-noise TS configs (T chosen past saturation; quartiles recorded to prove flatness)
TS_CFG = {2: (250, 512), 5: (250, 384), 10: (400, 384), 20: (400, 256), 50: (800, 192), 100: (1000, 128)}

def Enorm_closed(d, s):
    return s*np.sqrt(2.0)*np.exp(gammaln((d+1)/2) - gammaln(d/2))

def thm6_formula(tau, T, r=R_ACT):
    """Registered Theorem-6 value; tau sorted descending (worst residual ordering)."""
    tau = np.sort(np.asarray(tau, float))[::-1]
    m = min(T, len(tau))
    i = np.arange(1, m+1)
    return float(r/(np.pi*np.linalg.norm(tau)) * np.sum((i-1)*tau[:m]**2))

def oracle(d, s, M, seed):
    """Sequential-revelation construction, MC.  Returns total regret and per-step check."""
    rng = np.random.default_rng(seed)
    th = s*rng.standard_normal((M, d))
    nrm = np.linalg.norm(th, axis=1)
    csum = np.cumsum(th**2, axis=1)                     # ||P_t theta||^2 for t=1..d
    pnorm = np.hstack([np.zeros((M, 1)), np.sqrt(csum)])  # t=0..d
    per_step = R_ACT*(nrm.mean() - pnorm.mean(axis=0))  # E[||th|| - ||P_t th||], t=0..d
    total = float(np.sum(per_step[:d]))                 # steps t=0..d-1 (0 afterwards)
    tail = np.array([(d-t)*s**2 for t in range(d)])     # R_t = sum_{i>t} tau_i^2, isotropic
    paper_per_step = (1.0/np.pi)*tail/(s*np.sqrt(d))    # (1/pi) R_t/||tau||_2
    ok = bool(np.all(per_step[:d] >= paper_per_step - 3e-3*max(1.0, per_step[0])))
    return total, per_step, paper_per_step, ok

def ts_burnin(d, s, sig, T, M, seed):
    """Low-noise TS burn-in isolation (claim1 protocol), analytic instantaneous regret."""
    rng = np.random.default_rng(seed)
    V = np.broadcast_to(np.eye(d)/(s**2), (M, d, d)).copy(); b = np.zeros((M, d, 1))
    theta = s*rng.standard_normal((M, d)); astar = R_ACT*np.linalg.norm(theta, axis=1)
    reg = np.zeros(M); q = {}
    for t in range(1, T+1):
        Lc = np.linalg.cholesky(V); mu = np.linalg.solve(V, b)
        Z = rng.standard_normal((M, d, 1))
        th = (mu + np.linalg.solve(np.swapaxes(Lc, 1, 2), Z))[..., 0]
        A = R_ACT*th/(np.linalg.norm(th, axis=1, keepdims=True) + 1e-12)
        dot = np.sum(theta*A, axis=1); reg += astar - dot
        Robs = dot + sig*rng.standard_normal(M)
        V += (A[:, :, None]*A[:, None, :])/sig**2
        b += (A*Robs[:, None]/sig**2)[..., None]
        if t in (T//4, T//2, T):
            q[t] = [float(reg.mean()), float(reg.std(ddof=1)/np.sqrt(M))]
    return q

def powerfit(ds, ys):
    ds = np.asarray(ds, float); ys = np.asarray(ys, float)
    p, lc = np.polyfit(np.log(ds), np.log(ys), 1)
    pred = p*np.log(ds) + lc
    r2 = float(1 - np.sum((np.log(ys)-pred)**2)/np.sum((np.log(ys)-np.log(ys).mean())**2))
    return float(p), r2

def main():
    t0 = time.time()
    res = {"s": S, "r": R_ACT, "sigma_lownoise": SIGMA_LO, "per_d": {}, "anisotropic_example": {}}
    print("== Theorem 6 (Section 4): the d*r*sqrt(Tr(Sigma0)) burn-in LOWER BOUND, constructed and measured ==")
    print("   Sigma0 = s^2 I, s=%g, r=%g.  F_thm6 = registered formula; ORACLE = sequential-revelation" % (S, R_ACT))
    print("   construction (sigma->0, one linear functional/round, optimal schedule); B_TS = measured TS")
    print("   burn-in (sigma=%g isolation).  Old weak floor r E||theta*|| shown for contrast.\n" % SIGMA_LO)
    print("  d   F_thm6   oracle   B_TS(sat)+/-95%CI   old_floor   per-step form holds   sandwich F<=orc<=B_TS   B_TS flat? [T/4,T/2,T]")
    Fv, Ov, Bv, OldV = [], [], [], []
    for d in DIMS:
        tau = S*np.ones(d)
        T_ts, M_ts = TS_CFG[d]
        F = thm6_formula(tau, T_ts)
        tot, per, paper_ps, ok_ps = oracle(d, S, 200000, seed=6000+d)
        q = ts_burnin(d, S, SIGMA_LO, T_ts, M_ts, seed=4000+d)
        B, Bse = q[T_ts]
        old = R_ACT*float(Enorm_closed(d, S))
        sandwich = bool(F <= tot <= B + 1.96*Bse)
        qs = [q[T_ts//4][0], q[T_ts//2][0], q[T_ts][0]]
        flat = (qs[2]-qs[1])/qs[2]
        res["per_d"][str(d)] = {"F_thm6": F, "oracle": tot, "B_TS": B, "B_TS_se": Bse,
                                "old_floor_rEnorm": old, "per_step_form_holds": ok_ps,
                                "sandwich_ok": sandwich, "T": T_ts, "M": M_ts,
                                "B_TS_quartiles": qs, "flatness_lasthalf_rel": flat,
                                "oracle_per_step_first3": [float(x) for x in per[:3]],
                                "paper_per_step_first3": [float(x) for x in paper_ps[:3]]}
        Fv.append(F); Ov.append(tot); Bv.append(B); OldV.append(old)
        print("%3d  %7.2f  %7.2f   %8.2f+/-%5.2f    %7.3f        %5s                %5s            %.1f%% [%s]"
              % (d, F, tot, B, 1.96*Bse, old, ok_ps, sandwich, 100*flat,
                 " ".join("%.1f" % x for x in qs)))
    # dimension scaling (target: d^{3/2} at fixed s, = the d r sqrt(Tr) law; old floor is d^{1/2})
    fits = {}
    for name, ys in [("F_thm6", Fv), ("oracle", Ov), ("B_TS", Bv), ("old_floor", OldV)]:
        p_all, r2_all = powerfit(DIMS, ys)
        p_10, r2_10 = powerfit(DIMS[2:], ys[2:])
        fits[name] = {"p_all_d": p_all, "R2_all_d": r2_all, "p_d_ge_10": p_10, "R2_d_ge_10": r2_10}
    res["dimension_scaling"] = fits
    print("\n  dimension scaling exponent p in  X(d) ~ d^p   (fixed s: theory 1.5 for Thm-6 objects, 0.5 for old floor):")
    for name, f in fits.items():
        print("    %-9s p(all d)=%.3f (R2=%.4f)   p(d>=10)=%.3f (R2=%.4f)"
              % (name, f["p_all_d"], f["R2_all_d"], f["p_d_ge_10"], f["R2_d_ge_10"]))
    # ratio of B_TS to F_thm6 (constant across d => matching order)
    ratios = [b/f for b, f in zip(Bv, Fv)]
    res["B_TS_over_F_thm6"] = {str(d): float(x) for d, x in zip(DIMS, ratios)}
    print("  B_TS / F_thm6 across d:  " + "  ".join("d=%d:%.2f" % (d, x) for d, x in zip(DIMS, ratios)))
    # anisotropic example: formula ordering ambiguity disclosed
    tau_an = np.array([4.0, 2.0, 1.0, 1.0, 0.5])
    res["anisotropic_example"] = {"tau": tau_an.tolist(),
                                 "F_desc": thm6_formula(tau_an, 400),
                                 "F_asc": float(R_ACT/(np.pi*np.linalg.norm(tau_an))
                                                * np.sum((np.arange(1, 6)-1)*np.sort(tau_an)**2))}
    res["runtime_s"] = round(time.time()-t0, 1)
    (HERE/"claim2"/"results_thm6.json").write_text(json.dumps(res, indent=1))
    print("\n[written] claim2/results_thm6.json   (runtime %.1fs)" % res["runtime_s"])

if __name__ == "__main__":
    main()
