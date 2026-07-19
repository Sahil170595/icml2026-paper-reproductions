#!/usr/bin/env python3
"""
tail_check.py -- confirm the heavy-tailed moment ASSUMPTION of the paper
  "Best-of-Both-Worlds for Heavy-Tailed Markov Decision Processes"
  OpenReview j6gXeiPJ3z (bounded alpha-moment E[|loss noise|^alpha] <= sigma^alpha
  for alpha in (1,2], UNBOUNDED variance for alpha<2) is actually exercised by the
  noise generator used in every mdp_core.py / mdp_run.py / mdp_stoch.py run, not
  merely asserted.

Two independent, falsifiable checks on the SAME noise family used everywhere else
in this reproduction (mdp_core.make_noise / mdp_core.noise_scale, symmetric
truncated Pareto with tail index alpha):

  (1) TAIL-INDEX FIT: the noise is Pareto-tailed by construction (mag = U^{-1/a},
      U~Unif(0,1)), so its survival function obeys P(|X|>x) = x^{-alpha} exactly
      (up to the finite truncation cap and the sigma-normalizing scale). We
      estimate the tail index from a log-log regression of the empirical survival
      function against x and compare it to the alpha the noise was DRAWN with --
      an estimated exponent of ~alpha (not ~2, not ~infinity) is direct evidence
      the simulator is generating actual power-law heavy tails, not e.g. a
      clipped/light-tailed stand-in.

  (2) VARIANCE DIVERGENCE WITH THE TRUNCATION CAP: E[|X|^alpha] is normalized to
      ~1 (sigma=1) at every truncation cap M by construction (that is what
      noise_scale(alpha, M) solves for). But the SECOND moment (variance) of a
      Pareto(alpha) truncated at M grows like M^{2-alpha} for alpha<2 (diverges as
      M->infinity: unbounded variance) and only like log(M) for alpha=2 (the
      classical finite-variance boundary). We recompute noise_scale/make_noise at
      a widening sequence of truncation caps M and show empirically that variance
      grows near-polynomially in M for alpha in {1.25,1.3,1.5,1.75} (fitted
      log-log slope ~= 2-alpha) while it grows only logarithmically for alpha=2.0
      (fitted slope ~= 0, i.e. NOT power-law) -- i.e. the alpha<2 runs are
      genuinely exercising unbounded-variance losses, not a finite-variance
      distribution relabeled "alpha".

No hand-entered numbers; every value below is computed by this script.
"""
import numpy as np

ALPHAS = [1.25, 1.3, 1.5, 1.75, 2.0]
N = 2_000_000
MCAP_MAIN = 1.0e6          # the exact truncation cap used by mdp_core.py
MCAPS = [1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8]
XGRID = [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]


def noise_scale(alpha, M):
    """Identical formula to mdp_core.noise_scale, parameterized by the
    truncation cap M (mdp_core.py hardcodes M=1e6)."""
    return (alpha * np.log(M) + 1.0) ** (1.0 / alpha)


def make_noise(rng, alpha, s, M, n):
    """Identical formula to mdp_core.make_noise, parameterized by M."""
    U = rng.random(n)
    mag = U ** (-1.0 / alpha)
    np.minimum(mag, M, out=mag)
    sgn = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    return sgn * mag / s


def tail_index_fit(alpha, seed=1000):
    """Check (1): draw N samples at the MDP's actual MCAP=1e6, estimate the
    tail index from the empirical survival function P(|X|>x), x in XGRID."""
    rng = np.random.default_rng(seed)
    s = noise_scale(alpha, MCAP_MAIN)
    x = make_noise(rng, alpha, s, MCAP_MAIN, N)
    ax = np.abs(x)
    surv = np.array([np.mean(ax > xv) for xv in XGRID])
    ok = surv > 0
    lx = np.log(np.asarray(XGRID)[ok])
    ls = np.log(surv[ok])
    slope, intercept = np.polyfit(lx, ls, 1)
    yh = np.polyval([slope, intercept], lx)
    ss_res = np.sum((ls - yh) ** 2)
    ss_tot = np.sum((ls - ls.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mom = float(np.mean(ax ** alpha))
    var = float(np.var(x))
    return dict(alpha=alpha, tail_index_fit=float(-slope), r2=float(r2),
                survival=[float(v) for v in surv], moment_alpha=mom, variance=var,
                xgrid=list(XGRID))


def variance_growth(alpha, seed=2000):
    """Check (2): variance vs truncation cap M; fit log(Var) ~ b*log(M)."""
    var_list = []
    mom_list = []
    for M in MCAPS:
        rng = np.random.default_rng(seed)  # same seed at every M: only M changes
        s = noise_scale(alpha, M)
        x = make_noise(rng, alpha, s, M, N)
        var_list.append(float(np.var(x)))
        mom_list.append(float(np.mean(np.abs(x) ** alpha)))
    lM = np.log(np.asarray(MCAPS))
    lV = np.log(np.asarray(var_list))
    slope, intercept = np.polyfit(lM, lV, 1)
    yh = np.polyval([slope, intercept], lM)
    ss_res = np.sum((lV - yh) ** 2)
    ss_tot = np.sum((lV - lV.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    target = max(2.0 - alpha, 0.0)
    return dict(alpha=alpha, mcaps=list(MCAPS), variance=var_list, moment_alpha=mom_list,
                fitted_var_slope=float(slope), target_slope_2_minus_alpha=float(target),
                r2=float(r2), diverges=bool(slope > 0.25))


def main():
    print("=== (1) TAIL-INDEX FIT: empirical P(|noise|>x) ~ x^{-alpha} at the MDP's actual MCAP=1e6, N=%d ===" % N)
    print("%7s %14s %8s %10s %10s" % ("alpha", "fit_tail_idx", "R2", "E|X|^a", "Var(X)"))
    fits = {}
    for a in ALPHAS:
        f = tail_index_fit(a)
        fits[str(a)] = f
        print("%7s %14.3f %8.4f %10.3f %10.1f" % (a, f['tail_index_fit'], f['r2'], f['moment_alpha'], f['variance']))
    print("  (fit_tail_idx should be close to alpha: the noise really is Pareto(alpha)-tailed,")
    print("   not merely alpha-labeled; E|X|^a ~ 1 by construction (sigma=1) for every alpha)")

    print("\n=== (2) VARIANCE DIVERGENCE vs truncation cap M (fixed alpha-moment ~1 at every M) ===")
    print("%7s %16s %20s %8s %10s" % ("alpha", "fit_slope(logVar/logM)", "target(2-alpha)", "R2", "diverges?"))
    grows = {}
    for a in ALPHAS:
        g = variance_growth(a)
        grows[str(a)] = g
        print("%7s %16.3f %20.3f %8.4f %10s" % (a, g['fitted_var_slope'], g['target_slope_2_minus_alpha'], g['r2'], g['diverges']))
    print("  variance(M) for M in %s:" % [ "%.0e" % m for m in MCAPS])
    for a in ALPHAS:
        print("    alpha=%s: " % a + ", ".join("%.1f" % v for v in grows[str(a)]['variance']))
    print("\n  RULE: for alpha<2 the fitted log(Var)/log(M) slope should track (2-alpha)>0 (variance")
    print("  GROWS POLYNOMIALLY with the cap -> truly unbounded as cap->infinity); for alpha=2.0 the")
    print("  slope should be ~0 (variance is asymptotically POLYLOG in the cap, i.e. genuinely finite).")

    n_pass_tail = sum(1 for a in ALPHAS if abs(fits[str(a)]['tail_index_fit'] - a) <= 0.15)
    n_pass_var = sum(1 for a in ALPHAS if a < 2.0 and grows[str(a)]['diverges'])
    bounded_at_2 = not grows['2.0']['diverges']
    print("\nRESULT: tail-index matches alpha for %d/%d alphas (tol 0.15); variance diverges with cap for" %
          (n_pass_tail, len(ALPHAS)))
    print("        %d/%d of the alpha<2 cases; alpha=2.0 variance NON-divergent (bounded/log growth): %s" %
          (n_pass_var, sum(1 for a in ALPHAS if a < 2.0), bounded_at_2))
    verdict = (n_pass_tail >= 4) and (n_pass_var >= 3) and bounded_at_2
    print("VERDICT: heavy-tailed (alpha in (1,2], unbounded-variance) assumption genuinely exercised: %s" %
          ("PASS" if verdict else "PARTIAL"))

    import json, os
    ev = dict(n_samples=N, mcap_main=MCAP_MAIN, xgrid=XGRID, mcaps=MCAPS, alphas=ALPHAS,
              tail_index_fits=fits, variance_growth=grows,
              n_pass_tail=n_pass_tail, n_pass_var=n_pass_var, bounded_at_alpha2=bool(bounded_at_2),
              verdict_pass=bool(verdict), numpy=np.__version__)
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(ev, open(os.path.join(here, "tail_check_results.json"), "w"), indent=1)
    print("\n[wrote tail_check_results.json]")


if __name__ == "__main__":
    main()
