"""
Claim 1 reproduction -- provable SUB-LINEAR regret of the OFD-UCB contextual-bandit
mechanism for online fair division.

Paper: Verma, Saha, Yokoo, Low, "Keep Everyone Happy: Online Fair Division of
Numerous Items with Few Copies" (ICML 2026 / arXiv 2408.12845). Theorem 1:

    R_T(OFD-UCB) <= 2 * alpha_T * w_max * sqrt(2 d T log(lambda + T L / d))
    alpha_T      =  R * sqrt(d * log((1 + T L^2/lambda)/delta)) + lambda^{1/2} * S
    => R_T = O(sqrt(d T log T)),  hence  lim_{T->inf} R_T / T = 0   (sub-linear).

Independent NumPy implementation of OFD-UCB (ridge least squares + the OFUL/LinUCB
optimism bonus alpha_t * ||m||_{M_t^-1} with the paper's confidence radius). No
official code used. Deterministic: numpy.random.default_rng(fixed seed), CPU-only,
single BLAS thread.

TARGET / COMPARISON RULE (falsifiable):
  * OFD-UCB regret must be SUB-LINEAR: log-log growth exponent b (fit
    log R_T = a + b log T over the 2nd half of the horizon) must be < 1, and in
    particular OFD-UCB average regret R_T/T must fall to << the no-learning
    baseline (>=10x below OFD-Uniform).
  * On the tight canonical instance (Theorem 1's O(sqrt(dT log T)) regime) the
    exponent must land in the sqrt band [0.45, 0.65].
  * The no-learning baseline (OFD-Uniform / random allocation) must be LINEAR:
    exponent >= 0.9, R_T/T -> const.
  FALSIFIED if OFD-UCB exponent >= 0.9 (linear) or OFD-UCB does not beat OFD-Uniform.
"""
import json, os, time
from pathlib import Path
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
OUT = Path(__file__).with_name("results.json")


def loglog_exponent(reg, T):
    """Fit log R_T = a + b log T over the second half of the horizon; return b."""
    t = np.arange(1, T + 1)
    idx = np.arange(T // 2, T)
    R, tt = reg[idx], t[idx]
    m = R > 0
    b, a = np.polyfit(np.log(tt[m]), np.log(R[m]), 1)
    return float(b)


# ----------------------------------------------------------------------------
# EXP A -- faithful OFD process: item arrives, N agents, concatenated
#          item-agent features, OFD-UCB allocates; regret vs myopic optimum.
# ----------------------------------------------------------------------------
def run_ofd(seed, T, dm, dn, N, sigma=0.1, lam=1.0, algo="ucb"):
    rng = np.random.default_rng(seed)
    d = dm + dn
    theta = rng.standard_normal(d); theta /= np.linalg.norm(theta)      # ||theta*||=S=1
    agent_feats = rng.uniform(0, 10, size=(N, dn))                      # fixed agent features
    Ainv = (1.0 / lam) * np.eye(d)
    bvec = np.zeros(d); th = np.zeros(d)
    reg = np.empty(T); cum = 0.0; delta = 0.05
    for t in range(1, T + 1):
        item = rng.uniform(0, 10, size=dm)                             # item features ~ U(0,10)
        M = np.concatenate([np.broadcast_to(item, (N, dm)), agent_feats], axis=1)
        M = M / np.linalg.norm(M, axis=1, keepdims=True)               # ||m||<=L=1
        u = M @ theta                                                   # true utilities
        astar = int(np.argmax(u))
        if algo == "uniform":
            a = int(rng.integers(N))
        else:
            alpha = sigma * np.sqrt(d * np.log((1.0 + t / lam) / delta)) + np.sqrt(lam)
            quad = np.einsum("ij,jk,ik->i", M, Ainv, M)
            a = int(np.argmax(M @ th + alpha * np.sqrt(np.maximum(quad, 0.0))))
        x = M[a]; y = u[a] + sigma * rng.standard_normal()
        cum += u[astar] - u[a]; reg[t - 1] = cum
        Ax = Ainv @ x; Ainv -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        bvec += y * x; th = Ainv @ bvec
    return reg


# ----------------------------------------------------------------------------
# EXP B -- tight O(sqrt(dT log T)) rate: FIXED action set spanning R^d (no gap),
#          the canonical instance where the OFUL/Theorem-1 bound is tight.
# ----------------------------------------------------------------------------
def run_fixed(seed, T, d, Mn, sigma=0.1, lam=1.0, algo="ucb"):
    g = np.random.default_rng(1000 + seed)
    X = g.standard_normal((Mn, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    theta = g.standard_normal(d); theta /= np.linalg.norm(theta)
    rng = np.random.default_rng(seed)
    u = X @ theta; astar = int(np.argmax(u))
    Ainv = (1.0 / lam) * np.eye(d); bvec = np.zeros(d); th = np.zeros(d)
    reg = np.empty(T); cum = 0.0; delta = 0.05
    for t in range(1, T + 1):
        if algo == "uniform":
            a = int(rng.integers(Mn))
        else:
            alpha = sigma * np.sqrt(d * np.log((1.0 + t / lam) / delta)) + np.sqrt(lam)
            quad = np.einsum("ij,jk,ik->i", X, Ainv, X)
            a = int(np.argmax(X @ th + alpha * np.sqrt(np.maximum(quad, 0.0))))
        x = X[a]; y = u[a] + sigma * rng.standard_normal()
        cum += u[astar] - u[a]; reg[t - 1] = cum
        Ax = Ainv @ x; Ainv -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        bvec += y * x; th = Ainv @ bvec
    return reg


def avg(fn, seeds, *a):
    acc = None
    for s in seeds:
        r = fn(s, *a)
        acc = r if acc is None else acc + r
    return acc / len(seeds)


def main():
    t0 = time.time(); seeds = [0, 1, 2, 3, 4]

    # ---------------- EXP A ----------------
    TA, dmA, dnA, NA = 12000, 3, 3, 10
    print(f"[EXP A] OFD process  T={TA} d={dmA+dnA} (dm={dmA},dn={dnA}) N={NA} seeds={seeds}")
    a_ucb = avg(run_ofd, seeds, TA, dmA, dnA, NA, 0.1, 1.0, "ucb")
    a_uni = avg(run_ofd, seeds, TA, dmA, dnA, NA, 0.1, 1.0, "uniform")
    bA_ucb, bA_uni = loglog_exponent(a_ucb, TA), loglog_exponent(a_uni, TA)
    print(f"  OFD-UCB    : final R_T={a_ucb[-1]:.3f}  R_T/T={a_ucb[-1]/TA:.3e}  exponent={bA_ucb:.3f}")
    print(f"  OFD-Uniform: final R_T={a_uni[-1]:.3f}  R_T/T={a_uni[-1]/TA:.3e}  exponent={bA_uni:.3f}")
    ratio = (a_uni[-1] / TA) / (a_ucb[-1] / TA)
    print(f"  R_T/T ratio (Uniform/UCB) = {ratio:.1f}x  (rule: >=10x)")

    # ---------------- EXP B ----------------
    TB, dB, MB = 6000, 30, 500
    print(f"\n[EXP B] tight sqrt(dT logT) instance  T={TB} d={dB} M={MB} seeds={seeds}")
    b_ucb = avg(run_fixed, seeds, TB, dB, MB, 0.1, 1.0, "ucb")
    b_uni = avg(run_fixed, seeds, TB, dB, MB, 0.1, 1.0, "uniform")
    bB_ucb, bB_uni = loglog_exponent(b_ucb, TB), loglog_exponent(b_uni, TB)
    print("  t        R(UCB)   R/sqrt(t)  R/t      | R(UNI)    R/sqrt(t)  R/t")
    ck = {}
    for c in [TB // 8, TB // 4, TB // 2, TB - 1]:
        tt = c + 1; ru, rr = b_ucb[c], b_uni[c]
        print(f"  {tt:<7d}  {ru:7.2f}  {ru/np.sqrt(tt):8.4f}  {ru/tt:7.5f}  | {rr:8.2f}  {rr/np.sqrt(tt):8.4f}  {rr/tt:7.5f}")
        ck[str(tt)] = {"R_ucb": float(ru), "ucb_over_sqrt": float(ru/np.sqrt(tt)), "ucb_over_t": float(ru/tt),
                       "R_uni": float(rr), "uni_over_sqrt": float(rr/np.sqrt(tt)), "uni_over_t": float(rr/tt)}
    print(f"  OFD-UCB    : final R_T={b_ucb[-1]:.2f}  exponent(2nd half)={bB_ucb:.3f}  R/sqrt(T) {b_ucb[TB//2]/np.sqrt(TB//2+1):.2f}->{b_ucb[-1]/np.sqrt(TB):.2f}")
    print(f"  OFD-Uniform: final R_T={b_uni[-1]:.2f}  exponent(2nd half)={bB_uni:.3f}")

    ucb_band = 0.45 <= bB_ucb <= 0.65
    uni_lin = bB_uni >= 0.9 and bA_uni >= 0.9
    ucb_sub = bA_ucb < 0.9 and ratio >= 10.0
    verified = ucb_band and uni_lin and ucb_sub
    print(f"\n  EXP B UCB exponent in [0.45,0.65]? {ucb_band}   Uniform linear (>=0.9)? {uni_lin}")
    print(f"  EXP A UCB sub-linear & >=10x below Uniform? {ucb_sub}")
    print(f"  CLAIM 1 VERIFIED? {verified}")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    res = {
        "claim": "The contextual-bandit approach (OFD-UCB) for online fair division provides a provable sub-linear regret upper bound (Theorem 1).",
        "target_rule": "OFD-UCB log-log regret exponent < 1 (sub-linear) and >=10x below no-learning baseline in R_T/T; tight-instance exponent in [0.45,0.65]; OFD-Uniform linear (exponent>=0.9). Falsified if OFD-UCB exponent>=0.9 or does not beat OFD-Uniform.",
        "config": {"seeds": seeds, "sigma_R": 0.1, "lambda": 1.0, "delta": 0.05,
                    "expA": {"T": TA, "dm": dmA, "dn": dnA, "d": dmA+dnA, "N": NA},
                    "expB": {"T": TB, "d": dB, "M": MB}},
        "expA_ofd_process": {
            "ofd_ucb": {"final_R_T": float(a_ucb[-1]), "R_T_over_T": float(a_ucb[-1]/TA), "loglog_exponent": bA_ucb},
            "ofd_uniform": {"final_R_T": float(a_uni[-1]), "R_T_over_T": float(a_uni[-1]/TA), "loglog_exponent": bA_uni},
            "R_over_T_ratio_uni_over_ucb": float(ratio)},
        "expB_tight_rate": {
            "ofd_ucb": {"final_R_T": float(b_ucb[-1]), "loglog_exponent": bB_ucb},
            "ofd_uniform": {"final_R_T": float(b_uni[-1]), "loglog_exponent": bB_uni},
            "checkpoints": ck},
        "verdict": {"expB_ucb_exponent_in_band": bool(ucb_band), "baseline_linear": bool(uni_lin),
                     "expA_ucb_sublinear_beats_baseline": bool(ucb_sub), "verified": bool(verified)},
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
