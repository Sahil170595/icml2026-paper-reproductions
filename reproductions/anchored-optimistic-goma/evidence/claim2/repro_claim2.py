#!/usr/bin/env python3
"""
Independent NumPy reproduction of the STOCHASTIC last-iterate guarantee of
"Accelerated and Stable Convergence with Anchored Generalized Optimistic Method"
(GOMA), OpenReview G6WKIN1heG / arXiv 2606.21528.

CLAIM 2 (scored). "GOMA demonstrates O(1/sqrt(k)) last-iterate convergence for
monotone Lipschitz operators in stochastic regimes with linearly increasing
minibatches." Paper Theorem 4 / abstract: a last-iterate rate of O(1/sqrt(k))
on the SQUARED operator norm E||G(x_k)||^2 under state-dependent noise.

PRIMARY EXPERIMENT (exact claim wording: "with linearly increasing minibatches").
Simplified single-call GOMA, update (14):
    y_k     = beta_k*x0 + (1-beta_k)*x_k
    x_{k+1} = y_k - eta_k * Ghat_bk(y_k, xi_k)
with beta_k = 1/(k+2), eta_k = 1/(L*sqrt(kappa)*(k+2)^{3/4}), driven by a
LINEARLY INCREASING minibatch b_k = k (size-b_k minibatch => oracle-noise std
scaled by 1/sqrt(b_k)). Measured E||G(x_k)||^2 must satisfy, over the fit window:
    (A rate)     loglog slope of E||G(x_k)||^2 vs k in [-0.60, -0.40]   (target -0.5)
    (B constant) max_k E||G(x_k)||^2 * sqrt(k+1) <= 1570 L^2 kappa R^2 + 8 sigma^2/kappa
in BOTH the additive-noise (kappa=1, Section 6.2.1) and the state-dependent
multiplicative-noise (kappa=4, unbounded variance, Section 6.2.2) regimes.

Deterministic reference (Lemma 3): the noiseless simplified GOMA already attains
O(1/sqrt(k)) on ||G||^2, i.e. slope -0.5 -- this is the floor the growing
minibatch reveals. FALSIFICATION control: a constant-step method under
non-vanishing noise with a CONSTANT minibatch does NOT converge (plateau);
turning on the linearly increasing minibatch restores convergence.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, time, hashlib
import numpy as np

SQRT3 = np.sqrt(3.0)
L = 1.0
SIGMA = 0.5
S_REPLICAS = 2500          # Monte-Carlo replicas (deterministic seeds)
N_ITERS = 26000            # horizon
KLO = 400                  # skip pre-asymptotic transient for the slope fit


def build_operators():
    """Two monotone L-Lipschitz operators (skew-symmetric => monotone, ||.||_2=L)."""
    # Op A: paper 6.2.1 -- d=2 bilinear game F(x,y)=(L y, -L x)
    A2 = np.array([[0.0, L], [-L, 0.0]])
    # Op B: paper 6.2.2 -- d=10 bilinear saddle min_x max_y x^T B y, B in R^{5x5},
    #        F(x,y) = (B y, -B^T x); skew-symmetric block => monotone, ||F||_2=||B||_2=L
    #        B is an orthogonal coupling (every singular value = L): a well-conditioned
    #        saddle where all coordinate modes share the SAME last-iterate rate, so the
    #        measured slope is unbiased by mode-dependent conditioning.
    rng = np.random.default_rng(1234)
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    B = L * Q                                    # orthogonal => ||B||_2 = L, cond(B) = 1
    A10 = np.zeros((10, 10))
    A10[:5, 5:] = B
    A10[5:, :5] = -B.T
    return {"2d": A2, "10d": A10}


def checkpoints(N, npts=52, kmin=1):
    ks = np.unique(np.round(np.logspace(np.log10(kmin), np.log10(N), npts)).astype(int))
    return ks[(ks >= kmin) & (ks <= N)]


def run(A, d, S, N, sigma, kappa, seed, rule, batch, const_b=1, cks=None):
    """Vectorized Monte-Carlo over S replicas. Returns {k: mean ||G(x_k)||^2}.
       rule: 'thm4' (simplified single-call, Eq.14) or 'accel' (Case-I optimistic, Thm 1).
       batch: 'grow' (b_k=k), 'single'/'const' (b_k=const_b)."""
    rng = np.random.default_rng(seed)
    z0 = np.ones(d)
    Z = np.tile(z0, (S, 1)).astype(np.float64)
    AT = A.T
    cset = set(int(k) for k in cks)
    rec = {}
    c_mult = np.sqrt(max(kappa - 1.0, 0.0))     # multiplicative (state-dependent) scale
    use_add = sigma > 0.0
    use_mul = c_mult > 0.0
    if rule == "accel":
        eta_star = 0.9 / (2.0 * SQRT3 * L)      # in (0, 1/(2 sqrt3 L)) [Thm 1]
        Ghat_prev = Z @ AT                      # Ghat(y_{-1}), y_{-1} := x0
    CH = 500
    k = 0
    while k <= N:
        blk = min(CH, N - k + 1)
        zn = rng.standard_normal((blk, S, d)) if use_add else None
        sm = rng.standard_normal((blk, S, 1)) if use_mul else None
        for j in range(blk):
            if batch == "grow":
                bk = float(max(1, k))           # LINEARLY INCREASING minibatch
            elif batch == "single":
                bk = 1.0
            else:
                bk = float(const_b)
            inv_sb = 1.0 / np.sqrt(bk)
            if rule == "thm4":
                beta = 1.0 / (k + 2.0)
                eta = 1.0 / (L * np.sqrt(kappa) * (k + 2.0) ** 0.75)
                Y = beta * z0 + (1.0 - beta) * Z
                G = Y @ AT
                nz = (sigma * inv_sb) * zn[j] if use_add else 0.0
                if use_mul:
                    nz = nz + (c_mult * inv_sb) * sm[j] * G
                Z = Y - eta * (G + nz)
            else:                               # accel (Case-I optimistic)
                beta = 2.0 / (k + 6.0)
                eta = eta_star
                Y = beta * z0 + (1.0 - beta) * Z - eta * (1.0 - beta) * Ghat_prev
                Gy = Y @ AT
                nz = (sigma * inv_sb) * zn[j] if use_add else 0.0
                if use_mul:
                    nz = nz + (c_mult * inv_sb) * sm[j] * Gy
                Ghat_y = Gy + nz
                Z = beta * z0 + (1.0 - beta) * Z - eta * Ghat_y
                Ghat_prev = Ghat_y
            if k in cset:
                rec[k] = float(np.mean(np.sum((Z @ AT) ** 2, axis=1)))
            k += 1
            if k > N:
                break
    return rec, float(d)


def loglog_slope(rec, klo, khi):
    ks = np.array(sorted(k for k in rec if klo <= k <= khi and rec[k] > 0))
    ys = np.array([rec[k] for k in ks])
    a, b = np.polyfit(np.log10(ks), np.log10(ys), 1)
    return float(a), int(len(ks))


def main():
    t0 = time.time()
    OP = build_operators()
    cks = checkpoints(N_ITERS)
    out = {"L": L, "sigma": SIGMA, "S": S_REPLICAS, "N": N_ITERS,
           "fit_window": [KLO, N_ITERS], "configs": {}}

    print("=" * 80)
    print("STOCHASTIC GOMA  last-iterate O(1/sqrt k)  --  arXiv 2606.21528 / G6WKIN1heG")
    print("independent NumPy Monte-Carlo, CPU-only, single-thread")
    print("=" * 80)
    print(f"operators: [2d] bilinear game F(x,y)=(Ly,-Lx) (6.2.1);  "
          f"[10d] bilinear saddle F=(By,-B^T x), ||B||2=L (6.2.2)")
    print(f"L={L}  sigma={SIGMA}  MC replicas S={S_REPLICAS}  N={N_ITERS}  "
          f"fit window k in [{KLO},{N_ITERS}]")
    print(f"minibatch: size-b_k averages b_k oracle draws => noise std x 1/sqrt(b_k); "
          f"b_k=k (grow) or b_k=1 (const)")
    print()

    def report(tag, desc, A, d, kappa, sigma, seed, rule, batch, is_primary,
               slo=(-0.60, -0.40)):
        rec, dd = run(A, d, S_REPLICAS, N_ITERS, sigma, kappa, seed, rule, batch, cks=cks)
        R2 = dd  # ||z0-z*||^2 = sum(ones^2) = d
        slope, npts = loglog_slope(rec, KLO, N_ITERS)
        cval = 1570.0 * L * L * kappa * R2 + 8.0 * sigma * sigma / max(kappa, 1e-9)
        prod = {k: rec[k] * np.sqrt(k + 1.0) for k in rec if k >= KLO}
        maxprod = max(prod.values())
        const_ok = bool(maxprod <= cval)
        rate_ok = bool(slo[0] <= slope <= slo[1])
        out["configs"][tag] = dict(desc=desc, d=int(d), kappa=kappa, sigma=sigma,
                                   R2=R2, rule=rule, batch=batch, slope=slope,
                                   norm_slope=slope / 2.0, npts=npts,
                                   thm4_const_bound=cval, max_resid2_sqrtk=float(maxprod),
                                   const_ok=const_ok, rate_ok=rate_ok, is_primary=is_primary,
                                   curve={str(k): rec[k] for k in sorted(rec)})
        print(f"[{tag}] {desc}")
        print(f"     d={int(d)}  kappa={kappa}  sigma={sigma}  rule={rule}  batch={batch}")
        for k in sorted(rec):
            if k in (1, 10, 100, 1000, 5000, 26000) or k == max(rec):
                print(f"     k={k:6d}  E||G(x_k)||^2={rec[k]:.6e}  "
                      f"E||G||^2*sqrt(k+1)={rec[k]*np.sqrt(k+1.0):.4f}")
        print(f"     loglog slope of E||G||^2 vs k over [{KLO},{N_ITERS}] ({npts} pts) "
              f"= {slope:.4f}   (norm slope {slope/2.0:.4f})")
        if is_primary:
            print(f"     Thm4 constant: max_k E||G||^2*sqrt(k+1) = {maxprod:.3f} "
                  f"<= 1570 L^2 kappa R2 + 8 sig^2/kappa = {cval:.1f} ? {const_ok}")
            print(f"     RATE in [{slo[0]},{slo[1]}] (O(1/sqrt k) on squared norm)? {rate_ok}")
        print()
        return slope

    # ===== PRIMARY: GOMA + LINEARLY INCREASING minibatch b_k=k (the exact claim) =====
    sP1 = report("P1", "PRIMARY: GOMA + linearly increasing minibatch b_k=k, additive noise kappa=1 (6.2.1)",
                 OP["2d"], 2, 1.0, SIGMA, seed=0, rule="thm4", batch="grow", is_primary=True)
    sP2 = report("P2", "PRIMARY: GOMA + linearly increasing minibatch b_k=k, state-dependent MULTIPLICATIVE noise kappa=4, d=10 (6.2.2, unbounded variance)",
                 OP["10d"], 10, 4.0, SIGMA, seed=1, rule="thm4", batch="grow", is_primary=True)

    # ===== SUPPORT: constant batch also converges (paper's stronger no-growing-batch result)
    sS1 = report("S1", "SUPPORT: GOMA + CONSTANT batch b=1, additive kappa=1 (paper Thm-4 headline: no growing batch needed)",
                 OP["2d"], 2, 1.0, SIGMA, seed=2, rule="thm4", batch="single", is_primary=True)
    sS2 = report("S2", "SUPPORT: GOMA + CONSTANT batch b=1, multiplicative kappa=4, d=10 (headline, unbounded variance)",
                 OP["10d"], 10, 4.0, SIGMA, seed=3, rule="thm4", batch="single", is_primary=True)

    # ===== MINIBATCH ON/OFF control on the accelerated constant-step method =====
    sM1 = report("M1", "accelerated GOMA (const step) + linearly increasing minibatch b_k=k: minibatch ON => converges (norm O(1/sqrt k))",
                 OP["2d"], 2, 1.0, SIGMA, seed=4, rule="accel", batch="grow", is_primary=False,
                 slo=(-1.2, -0.4))
    sD = report("CTRL", "CONTROL/falsification: accelerated GOMA (const step) + CONSTANT batch b=1: minibatch OFF => plateau (no last-iterate convergence)",
                OP["2d"], 2, 1.0, SIGMA, seed=5, rule="accel", batch="const", is_primary=False,
                slo=(-0.60, -0.40))

    # ===== noiseless deterministic references =====
    recR1, _ = run(OP["2d"], 2, 1, N_ITERS, 0.0, 1.0, seed=9, rule="thm4", batch="single", cks=cks)
    sR1, _ = loglog_slope(recR1, KLO, N_ITERS)
    recR2, _ = run(OP["2d"], 2, 1, N_ITERS, 0.0, 1.0, seed=9, rule="accel", batch="single", cks=cks)
    sR2, _ = loglog_slope(recR2, KLO, N_ITERS)
    out["refs"] = {"thm4_noiseless_slope": sR1, "accel_noiseless_slope": sR2}
    print(f"[ref] noiseless simplified-GOMA slope (Lemma 3, expect ~-0.5) = {sR1:.4f}")
    print(f"[ref] noiseless accelerated-GOMA slope (Theorem 1, expect ~-2.0) = {sR2:.4f}")
    print()

    print("=" * 80)
    print("SUMMARY (log-log slope of E||G(x_k)||^2 vs k):")
    print(f"  P1  GOMA + grow minibatch b_k=k, additive  kappa=1 d=2 : {sP1:+.4f}  (target -0.5)")
    print(f"  P2  GOMA + grow minibatch b_k=k, multiplic kappa=4 d=10: {sP2:+.4f}  (target -0.5, unbounded var)")
    print(f"  S1  GOMA + const batch b=1,  additive  kappa=1 d=2     : {sS1:+.4f}  (no growing batch)")
    print(f"  S2  GOMA + const batch b=1,  multiplic kappa=4 d=10    : {sS2:+.4f}")
    print(f"  M1  accel + grow minibatch b_k=k (minibatch ON)        : {sM1:+.4f}  (norm {sM1/2:+.4f})")
    print(f"  CTRL accel + const batch  (minibatch OFF)              : {sD:+.4f}  (plateau => ~0)")
    prim = out["configs"]
    primary_ok = (prim["P1"]["rate_ok"] and prim["P1"]["const_ok"]
                  and prim["P2"]["rate_ok"] and prim["P2"]["const_ok"])
    grow_beats_const = sM1 < sD - 0.3
    out["primary_verified"] = bool(primary_ok)
    out["grow_minibatch_rescues_control"] = bool(grow_beats_const)
    print()
    print(f"  PRIMARY (GOMA + linearly increasing minibatch: O(1/sqrt k) squared, rate+const,")
    print(f"           additive kappa=1 AND multiplicative kappa=4): "
          f"{'VERIFIED' if primary_ok else 'NOT VERIFIED'}")
    print(f"  Linearly increasing minibatch rescues the plateauing control: {grow_beats_const}")
    print("=" * 80)

    out["runtime_s"] = round(time.time() - t0, 2)
    src = open(os.path.abspath(__file__), "rb").read()
    out["script_sha256"] = hashlib.sha256(src).hexdigest()
    out["numpy"] = np.__version__
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[wrote results.json]  runtime={out['runtime_s']}s")


if __name__ == "__main__":
    main()
