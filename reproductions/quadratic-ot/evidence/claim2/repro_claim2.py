"""
Claim 2 (Theorem 3.3): GENERAL lower bound with optimal exponent.
"The support of the QOT optimizer cannot concentrate around the Monge graph
faster than order eps^(1/(d+2)) in directed Hausdorff distance."

Paper: arXiv 2605.24644 (kcnuX4xEpL), Theorem 3.3 / Corollary 3.4, Lemma 3.1.
   r := dist(spt pi_eps ; gr T)            (directed Hausdorff distance)
   b := sup_{(x,y) in spt pi_eps} ||y-T(x)||   (vertical bias)
   Lemma 3.1 (T is L-Lipschitz):   r <= b <= sqrt(1+L^2) r
   Corollary 3.4 (eps in (0,1]):   r >= c_sm eps^(1/(d+2)),  b >= c_sm eps^(1/(d+2))
So the support CANNOT concentrate onto gr T faster than eps^(1/(d+2)):
the fitted exponent of r,b must NOT exceed 1/(d+2) (a faster/steeper decay
would falsify the theorem).

Robust lower witness (ties to the value-gap proof, Lemma 3.5 / Thm 3.6):
the mean-squared bias  m_eps = \int ||y-T(x)||^2 dpi_eps  obeys
m_eps = Theta(eps^(2/(d+2)))  ==>  RMS := sqrt(m_eps) ~ eps^(1/(d+2)),
and since b = sup >= RMS, RMS is a clean (integral, not edge-dominated)
certificate of the eps^(1/(d+2)) lower bound WITH the correct exponent.

Comparison rule (fit log(.) vs log(eps) over eps in [1e-3,1e-1]):
  * RMS bias slope  ~= 1/(d+2)   (d=1: 0.333 in [0.27,0.40]; d=2: 0.25 in [0.19,0.31])
  * mean-sq slope   ~= 2/(d+2)   (d=1: 0.667; d=2: 0.500)
  * directed Hausdorff r & vertical bias b: slope <= 1/(d+2)+0.05 (ANTI-concentration:
    support does not concentrate faster than eps^(1/(d+2)))
FALSIFICATION: any of these slopes clearly ABOVE 1/(d+2) (e.g. ~0.5), i.e. the
support concentrating faster than eps^(1/(d+2)); or RMS/eps^(1/(d+2)) -> 0.

Solver: paper's product-reference QOT (line 407): pi_ij = P_ij [f_i+g_j-c_ij]_+/eps,
P = a (x) b, marginals a=mu, b=nu enforced by alternating safeguarded Newton.
CPU, deterministic grids. Staged by argv (d1 | d1na | d2 | reduce) for <40s/run.
"""
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EPS_GRID = np.array([1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3])


def gauss_marg(sig, n, k=3.0):
    lo, hi = -k * sig, k * sig
    p = np.linspace(lo, hi, n)
    d = np.exp(-0.5 * (p / sig) ** 2); d = d / d.sum()
    return p, d


def mixture_marg(n, means, sig, lo, hi):
    """Bimodal Gaussian-mixture density on [lo,hi] (non-Gaussian target)."""
    y = np.linspace(lo, hi, n)
    d = np.zeros_like(y)
    for m in means:
        d += np.exp(-0.5 * ((y - m) / sig) ** 2)
    d = d / d.sum()
    return y, d


def solve_qot(mu, nu, C, P, eps, n_outer=6000, final_tol=1e-9):
    n, m = C.shape
    f = np.zeros(n); g = np.zeros(m)
    def plan(f, g):
        return P * np.maximum(f[:, None] + g[None, :] - C, 0.0) / eps
    def newton_rows(f, g, target):
        base = g[None, :] - C; var = f.copy()
        for _ in range(80):
            arg = var[:, None] + base; act = arg > 0.0
            val = np.sum(P * np.where(act, arg, 0.0), axis=1) / eps
            deriv = np.sum(np.where(act, P, 0.0), axis=1) / eps
            resid = val - target
            if np.max(np.abs(resid)) < 1e-12: break
            var = var - np.where(deriv > 0, resid / np.maximum(deriv, 1e-300), -resid - 1.0)
        return var
    def newton_cols(f, g, target):
        base = f[:, None] - C; var = g.copy()
        for _ in range(80):
            arg = var[None, :] + base; act = arg > 0.0
            val = np.sum(P * np.where(act, arg, 0.0), axis=0) / eps
            deriv = np.sum(np.where(act, P, 0.0), axis=0) / eps
            resid = val - target
            if np.max(np.abs(resid)) < 1e-12: break
            var = var - np.where(deriv > 0, resid / np.maximum(deriv, 1e-300), -resid - 1.0)
        return var
    err = np.inf
    for _ in range(n_outer):
        f = newton_rows(f, g, mu); g = newton_cols(f, g, nu)
        pi = plan(f, g)
        err = max(np.max(np.abs(pi.sum(1) - mu)), np.max(np.abs(pi.sum(0) - nu)))
        if err < final_tol: break
    return plan(f, g), err


def fit(eps, vals):
    return float(np.polyfit(np.log(eps), np.log(np.asarray(vals)), 1)[0])


def run_d1_affine(nx=81, ny=601, s=1.5):
    A = s; L = A
    x, mu = gauss_marg(1.0, nx); y, nu = gauss_marg(s, ny)
    C = 0.5 * (x[:, None] - y[None, :]) ** 2; P = np.outer(mu, nu)
    Tx = A * x
    i_mode = int(np.argmin(np.abs(x - 0.0)))  # densest source point
    rows = []
    for eps in EPS_GRID:
        pi, err = solve_qot(mu, nu, C, P, eps)
        tol = 1e-6 * pi.max()
        S = pi > tol
        dev = np.abs(y[None, :] - Tx[:, None])          # |y - T(x)|
        b = float(dev[S].max())
        r = float((dev[S] / np.sqrt(1 + L * L)).max())   # perp dist to line y=Ax
        m = float(np.sum(pi * dev ** 2))                 # mean-squared bias
        rms = float(np.sqrt(m))
        Jm = np.where(pi[i_mode] > tol)[0]
        wmode = float(np.max(np.abs(y[Jm] - Tx[i_mode]))) # pointwise width at mode
        rows.append(dict(eps=float(eps), r=r, b=b, rms=rms, mse=m, wmode=wmode,
                         merr=float(err), npts=int(S.sum())))
    return dict(dim=1, case="affine_gaussian", A=A, L=L, nx=nx, ny=ny, rows=rows)


def run_d1_nonaffine(nx=81, ny=601):
    # mu = N(0,1)|[-3,3]; nu = bimodal mixture => Monge map T=F_nu^{-1}oF_mu is NON-affine.
    x, mu = gauss_marg(1.0, nx)
    y, nu = mixture_marg(ny, means=(-1.0, 1.0), sig=0.8, lo=-4.0, hi=4.0)
    # Monge map on the x-grid via monotone rearrangement (quadratic-cost OT in 1D).
    Fmu = np.cumsum(mu) - 0.5 * mu
    Fnu = np.cumsum(nu) - 0.5 * nu
    Tx = np.interp(Fmu, Fnu, y)                          # T(x_i)
    L = float(np.max(np.abs(np.diff(Tx) / np.diff(x)))) # empirical Lipschitz const
    C = 0.5 * (x[:, None] - y[None, :]) ** 2; P = np.outer(mu, nu)
    # graph polyline for directed Hausdorff distance
    xg = np.linspace(x[0], x[-1], 1200); Tg = np.interp(xg, x, Tx)
    i_mode = int(np.argmax(mu))
    rows = []
    for eps in EPS_GRID:
        pi, err = solve_qot(mu, nu, C, P, eps)
        tol = 1e-6 * pi.max()
        S = pi > tol
        dev = np.abs(y[None, :] - Tx[:, None]); b = float(dev[S].max())
        m = float(np.sum(pi * dev ** 2)); rms = float(np.sqrt(m))
        ii, jj = np.where(S)
        d2 = (x[ii][:, None] - xg[None, :]) ** 2 + (y[jj][:, None] - Tg[None, :]) ** 2
        r = float(np.sqrt(d2.min(axis=1)).max())        # true directed Hausdorff
        Jm = np.where(pi[i_mode] > tol)[0]
        wmode = float(np.max(np.abs(y[Jm] - Tx[i_mode])))
        rows.append(dict(eps=float(eps), r=r, b=b, rms=rms, mse=m, wmode=wmode,
                         merr=float(err), npts=int(S.sum())))
    return dict(dim=1, case="nonaffine_mixture", A=None, L=L, nx=nx, ny=ny, rows=rows)


def run_d2_affine(nx1=21, ny1=45, s=1.5):
    A = s; L = A
    x1, m1 = gauss_marg(1.0, nx1); y1, n1 = gauss_marg(s, ny1)
    X = np.array(np.meshgrid(x1, x1, indexing="ij")).reshape(2, -1).T
    muw = np.outer(m1, m1).reshape(-1)
    Y = np.array(np.meshgrid(y1, y1, indexing="ij")).reshape(2, -1).T
    nuw = np.outer(n1, n1).reshape(-1)
    C = 0.5 * ((X[:, None, :] - Y[None, :, :]) ** 2).sum(2); P = np.outer(muw, nuw)
    TX = A * X
    i_mode = int(np.argmin((X ** 2).sum(1)))
    rows = []
    for eps in EPS_GRID:
        pi, err = solve_qot(muw, nuw, C, P, eps)
        tol = 1e-6 * pi.max(); S = pi > tol
        dif = Y[None, :, :] - TX[:, None, :]
        dn = np.sqrt((dif ** 2).sum(2))                  # ||y - T(x)||
        b = float(dn[S].max())
        r = float((dn[S] / np.sqrt(1 + L * L)).max())    # perp dist (A=sI => factor sqrt(1+L^2))
        m = float(np.sum(pi * dn ** 2)); rms = float(np.sqrt(m))
        Jm = np.where(pi[i_mode] > tol)[0]
        wmode = float(np.max(np.sqrt(((Y[Jm] - TX[i_mode]) ** 2).sum(1))))
        rows.append(dict(eps=float(eps), r=r, b=b, rms=rms, mse=m, wmode=wmode,
                         merr=float(err), npts=int(S.sum())))
    return dict(dim=2, case="affine_gaussian", A=A, L=L, nx=nx1**2, ny=ny1**2, rows=rows)


def summarize(res):
    e = np.array([row["eps"] for row in res["rows"]])
    d = res["dim"]
    for key in ("r", "b", "rms", "mse", "wmode"):
        res["slope_" + key] = fit(e, [row[key] for row in res["rows"]])
    res["target_rms"] = 1.0 / (d + 2); res["target_mse"] = 2.0 / (d + 2)
    # anti-concentration diagnostics
    b = np.array([row["b"] for row in res["rows"]])
    res["b_over_epsHalf"] = [float(bi / (ei ** 0.5)) for bi, ei in zip(b, e)]
    res["rms_over_target"] = [float(row["rms"] / (row["eps"] ** (1.0/(d+2)))) for row in res["rows"]]
    return res


def pretty(res):
    d = res["dim"]
    print(f"\n[d={d} | {res['case']} | L={res['L']:.3f} | nx={res['nx']} ny={res['ny']}]")
    print("   eps        r          b        RMS       mse       wmode   b<=s(1+L2)r  merr    npts")
    s = np.sqrt(1 + res["L"] ** 2)
    for row in res["rows"]:
        lem = row["r"] <= row["b"] + 1e-9 and row["b"] <= s * row["r"] + 1e-9
        print(f"  {row['eps']:.1e}  {row['r']:.5f}  {row['b']:.5f}  {row['rms']:.5f}  "
              f"{row['mse']:.5f}  {row['wmode']:.5f}   {str(lem):5s}    {row['merr']:.1e} {row['npts']}")
    print(f"   slopes: r={res['slope_r']:.4f}  b={res['slope_b']:.4f}  "
          f"RMS={res['slope_rms']:.4f} (tgt {res['target_rms']:.4f})  "
          f"mse={res['slope_mse']:.4f} (tgt {res['target_mse']:.4f})  wmode={res['slope_wmode']:.4f}")
    print(f"   anti-conc: b/eps^0.5 over eps = "
          f"[{res['b_over_epsHalf'][0]:.2f} .. {res['b_over_epsHalf'][-1]:.2f}] (grows => slower than eps^0.5)")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    np.random.default_rng(0)  # determinism marker (grids are deterministic)
    if mode in ("d1", "all"):
        r = summarize(run_d1_affine()); pretty(r)
        json.dump(r, open(HERE / "part_d1.json", "w"), indent=1)
    if mode in ("d1na", "all"):
        r = summarize(run_d1_nonaffine()); pretty(r)
        json.dump(r, open(HERE / "part_d1na.json", "w"), indent=1)
    if mode in ("d2", "all"):
        r = summarize(run_d2_affine()); pretty(r)
        json.dump(r, open(HERE / "part_d2.json", "w"), indent=1)
    if mode == "reduce":
        out = {}
        for tag, fn in (("d1_affine", "part_d1.json"), ("d1_nonaffine", "part_d1na.json"),
                        ("d2_affine", "part_d2.json")):
            p = HERE / fn
            if p.exists(): out[tag] = json.load(open(p))
        out["meta"] = dict(paper="arXiv 2605.24644 (kcnuX4xEpL)", theorem="3.3 / Cor 3.4 / Lemma 3.1",
                           eps_grid=EPS_GRID.tolist(),
                           numpy=np.__version__)
        json.dump(out, open(HERE / "results.json", "w"), indent=1)
        print("wrote results.json with keys:", list(out.keys()))


if __name__ == "__main__":
    main()
