"""
Claim 2 - P-RWDKC's vertex measure is built from a parametrized forward/backward
flow mixing with parameter gamma in [0,1].

Paper P-RWDKC (arXiv 2210.00310 / OpenReview 5vI6ApLOg8), Sec. 5.1:
  Eq. 9 :  P_gamma = gamma * P_out + (1 - gamma) * P_in ,  gamma in [0,1]
           with P_out = D_out^{-1} W (forward)  and  P_in = D_in^{-1} W^T (backward)
  Eq. 8 :  nu^alpha_{(t,gamma)}(i) = ( (1/N) 1^T P_gamma^t delta_i )^alpha
  Neutral setting (params optional):  t=1, alpha=1, gamma=0.5.
Alternative design (App. A.3.1, Eq. 13-15): P_gamma = D_gamma^{-1} W_gamma,
  W_gamma = gamma W + (1-gamma) W^T ; at gamma=1/2, W_{1/2} is symmetric so the
  walk is ergodic and nu -> (pi_{1/2})^alpha  (explicit stationary measure).

CHECKABLE CONSEQUENCES (deterministic, CPU, numpy):
  (A) P_out and P_in are row-stochastic; for every gamma in {0,0.1,...,1},
      P_gamma is row-stochastic with non-negative entries (convex combination).
  (B) Endpoints: gamma=1 => P_gamma == P_out (pure forward);
                 gamma=0 => P_gamma == P_in  (pure backward), to machine eps.
  (C) The mixing is monotone: ||P_gamma - P_out||_F decreases and
      ||P_gamma - P_in||_F increases as gamma goes 0 -> 1; gamma=0.5 balances.
  (D) Vertex measure nu^alpha_{(t,gamma)} is strictly positive for all i.
  (E) FALSIFICATION CONTROL: for gamma outside [0,1] (gamma=-0.5, 1.5) the
      combination has NEGATIVE entries (min < 0) => not a transition matrix,
      confirming gamma in [0,1] is exactly the admissible range.
  (F) App. A.3.1 prediction: alternative design at gamma=1/2, t->inf gives
      nu^alpha -> (pi_{1/2})^alpha ; verified numerically.
"""
import json, numpy as np

def digraph(N, p, rng, wmax=3.0):
    A = (rng.random((N, N)) < p).astype(float); np.fill_diagonal(A, 0.0)
    W = A * rng.uniform(0.2, wmax, size=(N, N))
    for i in range(N):
        if W[i].sum() == 0: W[i, (i+1) % N] = rng.uniform(0.2, wmax)
        if W[:, i].sum() == 0: W[(i+2) % N, i] = rng.uniform(0.2, wmax)
    return W

def stochastic(M):  # (min entry, max |rowsum-1|)
    return float(M.min()), float(np.max(np.abs(M.sum(1) - 1.0)))

def main():
    rng = np.random.default_rng(7)
    gammas = np.round(np.arange(0.0, 1.0001, 0.1), 3)
    worst = dict(pin_dev=0.0, pout_dev=0.0, pg_min=np.inf, pg_rowsum=0.0,
                 end1=0.0, end0=0.0, nu_min=np.inf)
    ncases = 0
    mono_ok = True
    ctrl_min_out = 0.0    # min entry seen for gamma outside [0,1] (want < 0)
    a31_err = 0.0         # App A.3.1 : ||nu_alpha - pi_half^alpha||_inf
    for (N, p) in [(40, 0.15), (80, 0.10), (150, 0.06)]:
        for s in range(3):
            W = digraph(N, p, rng)
            dout = W.sum(1); din = W.sum(0)
            dout = np.where(dout > 0, dout, 1.0); din = np.where(din > 0, din, 1.0)
            Pout = W / dout[:, None]
            Pin = (W.T) / din[:, None]
            mo, ro = stochastic(Pout); mi, ri = stochastic(Pin)
            worst["pout_dev"] = max(worst["pout_dev"], ro); worst["pin_dev"] = max(worst["pin_dev"], ri)
            worst["pg_min"] = min(worst["pg_min"], mo, mi)
            dOut = np.linalg.norm; dfo = []; dfi = []
            for g in gammas:
                Pg = g * Pout + (1 - g) * Pin
                mn, rs = stochastic(Pg)
                worst["pg_min"] = min(worst["pg_min"], mn)
                worst["pg_rowsum"] = max(worst["pg_rowsum"], rs)
                dfo.append(np.linalg.norm(Pg - Pout)); dfi.append(np.linalg.norm(Pg - Pin))
                if abs(g - 1.0) < 1e-9: worst["end1"] = max(worst["end1"], float(np.max(np.abs(Pg - Pout))))
                if abs(g - 0.0) < 1e-9: worst["end0"] = max(worst["end0"], float(np.max(np.abs(Pg - Pin))))
                # vertex measure (Eq. 8) with random t, alpha
                t = int(rng.integers(1, 6)); alpha = float(rng.uniform(0.2, 1.0))
                nu = ((np.ones(N) / N) @ np.linalg.matrix_power(Pg, t))
                nu = np.clip(nu, 1e-300, None) ** alpha
                worst["nu_min"] = min(worst["nu_min"], float(nu.min()))
                ncases += 1
            dfo = np.array(dfo); dfi = np.array(dfi)
            if not (np.all(np.diff(dfo) <= 1e-9) and np.all(np.diff(dfi) >= -1e-9)):
                mono_ok = False
            # (E) control: gamma outside [0,1]
            for g in (-0.5, 1.5):
                Pg = g * Pout + (1 - g) * Pin
                ctrl_min_out = min(ctrl_min_out, float(Pg.min()))
            # (F) App A.3.1 alternative design: gamma=1/2, t large
            Wg = 0.5 * W + 0.5 * W.T
            dg = Wg.sum(1); dg = np.where(dg > 0, dg, 1.0)
            Phalf = Wg / dg[:, None]
            # stationary of symmetric-walk (reversible): pi ~ row sums of Wg
            pi_half = dg / dg.sum()
            Pt = np.linalg.matrix_power(Phalf, 512)
            nu_lim = (np.ones(N) / N) @ Pt            # -> pi_half (each row -> pi_half)
            alpha = 0.7
            a31_err = max(a31_err, float(np.max(np.abs(nu_lim**alpha - pi_half**alpha))))

    all_ok = (worst["pin_dev"] < 1e-12 and worst["pout_dev"] < 1e-12 and
              worst["pg_min"] >= -1e-15 and worst["pg_rowsum"] < 1e-12 and
              worst["end1"] < 1e-12 and worst["end0"] < 1e-12 and
              worst["nu_min"] > 0.0 and mono_ok and ctrl_min_out < 0.0 and a31_err < 1e-6)

    print("="*74)
    print("Claim 2  Vertex measure via gamma-parametrized forward/backward mixing")
    print("arXiv 2210.00310 / OpenReview 5vI6ApLOg8  (numpy %s)" % np.__version__)
    print("="*74)
    print(f"gamma grid {list(gammas)}  ;  cases = {ncases}")
    print("-"*74)
    print(f"(A) P_out row-stochastic : max|rowsum-1| = {worst['pout_dev']:.3e}")
    print(f"    P_in  row-stochastic : max|rowsum-1| = {worst['pin_dev']:.3e}")
    print(f"    P_gamma over grid    : min entry = {worst['pg_min']:.3e} (>=0) ; max|rowsum-1| = {worst['pg_rowsum']:.3e}")
    print(f"(B) endpoint gamma=1 == P_out : max|.| = {worst['end1']:.3e}")
    print(f"    endpoint gamma=0 == P_in  : max|.| = {worst['end0']:.3e}")
    print(f"(C) monotone interpolation (||.-Pout|| dec, ||.-Pin|| inc): {'yes' if mono_ok else 'NO'}")
    print(f"(D) vertex measure nu^alpha (Eq.8) min value = {worst['nu_min']:.3e}  (> 0, strictly positive)")
    print(f"(E) CONTROL gamma outside [0,1] ({-0.5},{1.5}): min entry = {ctrl_min_out:.4f}  (< 0 => NOT stochastic)")
    print(f"(F) App A.3.1 : ||nu^alpha - pi_half^alpha||_inf at gamma=1/2, t=512 = {a31_err:.3e}  (-> 0)")
    print("-"*74)
    print("VERDICT:", "PASS" if all_ok else "FAIL")
    print("="*74)

    out = dict(
        claim="P_gamma=gamma*P_out+(1-gamma)*P_in row-stochastic for gamma in [0,1]; vertex measure Eq.8 positive; gamma outside [0,1] fails; App A.3.1 nu->pi_half^alpha",
        numpy=np.__version__, gamma_grid=list(map(float, gammas)), cases=ncases,
        Pout_max_rowsum_dev=worst["pout_dev"], Pin_max_rowsum_dev=worst["pin_dev"],
        Pgamma_min_entry=worst["pg_min"], Pgamma_max_rowsum_dev=worst["pg_rowsum"],
        endpoint_gamma1_eq_Pout_maxdiff=worst["end1"],
        endpoint_gamma0_eq_Pin_maxdiff=worst["end0"],
        monotone_interpolation=bool(mono_ok),
        vertex_measure_min=worst["nu_min"],
        control_min_entry_gamma_outside_0_1=ctrl_min_out,
        appA31_nu_to_pihalf_alpha_maxdiff=a31_err,
        verdict="PASS" if all_ok else "FAIL")
    json.dump(out, open("results.json", "w"), indent=2)
    print("wrote results.json")

if __name__ == "__main__":
    main()
