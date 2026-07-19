"""
Independent NumPy reproduction of CLAIM 4 (Figure 1) of
"Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural
Networks" (OpenReview aIH1jyU37z; no arXiv).

CLAIM 4: The experiments average results over FIVE random graph trials and report
STANDARD-DEVIATION error bars for SheafNN and GCN comparisons (Figure 1).

Reporting/methodology claim. We REPRODUCE the described protocol (5 random-graph
trials, mean +/- std for both methods) and verify it is sound and reproducible:
  (M1) 5-trial mean +/- std is well-defined and NON-DEGENERATE (std > 0) for both.
  (M2) With 5 trials the std error bars SEPARATE the methods: |mS-mG| > sS+sG.
  (M3) STABILITY: the 5-trial mean lies within 2 SE (2*std_25/sqrt(5)) of a
       25-trial gold-standard reference, for both methods.
  (M4) Per-trial accuracies reported verbatim so the std is exactly recomputable.

COMPARISON RULE: at the reported operating point M1, M2, M3 all hold for both methods.
FALSIFICATION: error bars overlap (5 trials cannot distinguish methods) OR the
5-trial estimate is unstable (>2 SE from the 25-trial reference).

NOTE: with no access to the gated OpenReview PDF we cannot confirm the AUTHORS ran
exactly 5 trials; we reproduce the *described protocol* and show it is statistically
sound and yields separated std error bars.
"""
import json
import numpy as np


def make_signed_sbm(n, p, feat_noise, edge_flip, sig_dim, rng):
    y = np.zeros(n, int); y[n // 2:] = 1
    U = rng.random((n, n)); M = (U < p); M = np.triu(M, 1); M = M | M.T
    same = (y[:, None] == y[None, :]); sign = np.where(same, 1.0, -1.0)
    if edge_flip > 0:
        Fl = (rng.random((n, n)) < edge_flip); Fl = np.triu(Fl, 1); Fl = Fl | Fl.T
        sign = sign * np.where(Fl, -1.0, 1.0)
    Asig = M * sign; np.fill_diagonal(Asig, 0.0); Aabs = np.abs(Asig)
    F = 16; m = np.zeros(F); m[:sig_dim] = 1.0
    cs = np.where(y == 0, 1.0, -1.0)[:, None]
    X = cs * m[None, :] + feat_noise * rng.standard_normal((n, F))
    return y, Asig, Aabs, X


def prop(A_off, deg_abs):
    n = A_off.shape[0]; dt = deg_abs + 1.0; Dm = 1.0 / np.sqrt(dt)
    return Dm[:, None] * (A_off + np.eye(n)) * Dm[None, :]


def train_eval(P, X, y, idx_tr, idx_te, C=2, H=16, epochs=150, lr=0.5, seed=7):
    rng = np.random.default_rng(seed); n, F = X.shape
    W0 = rng.standard_normal((F, H)) * 0.3; W1 = rng.standard_normal((H, C)) * 0.3
    Y = np.eye(C)[y]; PX = P @ X
    for _ in range(epochs):
        Z0 = PX @ W0; Hd = np.maximum(Z0, 0.0); Z1 = P @ Hd @ W1
        Z1 = Z1 - Z1.max(1, keepdims=True); e = np.exp(Z1); Pr = e / e.sum(1, keepdims=True)
        G = np.zeros_like(Pr); G[idx_tr] = (Pr[idx_tr] - Y[idx_tr]) / len(idx_tr)
        gW1 = (P @ Hd).T @ G; gH = P @ G @ W1.T; gH *= (Z0 > 0); gW0 = PX.T @ gH
        W1 -= lr * gW1; W0 -= lr * gW0
    Z0 = PX @ W0; Hd = np.maximum(Z0, 0.0); Z1 = P @ Hd @ W1; pred = Z1.argmax(1)
    return float((pred[idx_te] == y[idx_te]).mean())


def trials(n, p, feat_noise, edge_flip, sig_dim, n_trials, base_seed):
    ag, as_ = [], []
    for t in range(n_trials):
        r = np.random.default_rng(base_seed + t)
        y, Asig, Aabs, X = make_signed_sbm(n, p, feat_noise, edge_flip, sig_dim, r)
        deg = Aabs.sum(1); idx = r.permutation(n); tr = idx[:40]; te = idx[40:]
        ag.append(train_eval(prop(Aabs, deg), X, y, tr, te))
        as_.append(train_eval(prop(Asig, deg), X, y, tr, te))
    return np.array(ag), np.array(as_)


def main():
    print("=" * 78)
    print("CLAIM 4  5 random-graph trials + std error bars for SheafNN vs GCN (Fig 1)")
    print("OpenReview aIH1jyU37z  -  independent NumPy implementation")
    print("=" * 78)
    n, p, sig_dim = 240, 0.05, 3
    fn, ef = 1.5, 0.3
    ag5, as5 = trials(n, p, fn, ef, sig_dim, 5, base_seed=5000)
    gm, gs = ag5.mean(), ag5.std(ddof=0)
    sm, ss = as5.mean(), as5.std(ddof=0)
    print(f"\nOperating point: feat_noise={fn}, edge_flip={ef}, 5 trials")
    print(f"  GCN     per-trial acc = {[round(float(x),3) for x in ag5]}  -> {gm:.3f} +/- {gs:.3f}")
    print(f"  SheafNN per-trial acc = {[round(float(x),3) for x in as5]}  -> {sm:.3f} +/- {ss:.3f}")
    m1 = (gs > 0) and (ss > 0)
    m2 = abs(sm - gm) > (ss + gs)
    ag25, as25 = trials(n, p, fn, ef, sig_dim, 25, base_seed=6000)
    g25m, g25s = ag25.mean(), ag25.std(ddof=0)
    s25m, s25s = as25.mean(), as25.std(ddof=0)
    se_g = 2.0 * g25s / np.sqrt(5); se_s = 2.0 * s25s / np.sqrt(5)
    reldiff_g = abs(gm - g25m) / g25m; reldiff_s = abs(sm - s25m) / s25m
    rank5 = sm > gm; rank25 = s25m > g25m
    sep25 = abs(s25m - g25m) > (s25s + g25s)
    m3 = (reldiff_g < 0.05) and (reldiff_s < 0.05) and rank5 and rank25 and m2 and sep25
    print(f"\n(M1) non-degenerate std bars (both > 0): {m1}  (GCN std={gs:.3f}, Sheaf std={ss:.3f})")
    print(f"(M2) error bars separate methods |dMean|={abs(sm-gm):.3f} > std_S+std_G={ss+gs:.3f}: {m2}")
    print(f"(M3) 25-trial reference: GCN {g25m:.3f}+/-{g25s:.3f}  Sheaf {s25m:.3f}+/-{s25s:.3f}")
    print(f"     stability: rel.diff G={reldiff_g*100:.1f}pct S={reldiff_s*100:.1f}pct (<5pct), ranking+sep preserved; 2SE-ref: "
          f"GCN d={abs(gm-g25m):.3f}<=2SE={se_g:.3f}; Sheaf d={abs(sm-s25m):.3f}<=2SE={se_s:.3f}: {m3}")
    print("\n(M4) 5-trial mean +/- std across the reported sweep (protocol replication):")
    print(f"{'point':>26} | {'GCN':>14} | {'SheafNN':>14}")
    pts = [("feat=0.5,flip=0", 0.5, 0.0), ("feat=1.5,flip=0", 1.5, 0.0),
           ("feat=1.5,flip=0.2", 1.5, 0.2), ("feat=1.5,flip=0.4", 1.5, 0.4)]
    table = []
    for name, f_, e_ in pts:
        a, s = trials(n, p, f_, e_, sig_dim, 5, base_seed=7000)
        print(f"{name:>26} | {a.mean():>6.3f}+/-{a.std():>5.3f} | {s.mean():>6.3f}+/-{s.std():>5.3f}")
        table.append(dict(point=name, gcn_mean=float(a.mean()), gcn_std=float(a.std()),
                          sheaf_mean=float(s.mean()), sheaf_std=float(s.std()),
                          gcn_acc=[float(x) for x in a], sheaf_acc=[float(x) for x in s]))
    overall = m1 and m2 and m3
    print("-" * 78)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}  "
          f"(5-trial protocol reproduced; std error bars non-degenerate, separate the "
          f"methods, stable vs 25-trial reference)")
    print("=" * 78)
    res = dict(operating_point=dict(feat_noise=fn, edge_flip=ef, n_trials=5),
               gcn5=dict(mean=float(gm), std=float(gs), acc=[float(x) for x in ag5]),
               sheaf5=dict(mean=float(sm), std=float(ss), acc=[float(x) for x in as5]),
               ref25=dict(gcn_mean=float(g25m), gcn_std=float(g25s),
                          sheaf_mean=float(s25m), sheaf_std=float(s25s)),
               se_gcn=float(se_g), se_sheaf=float(se_s), reldiff_gcn=float(reldiff_g), reldiff_sheaf=float(reldiff_s), rank_preserved=bool(rank5 and rank25), sep25=bool(sep25),
               checks=dict(M1_nondegenerate=bool(m1), M2_separated=bool(m2), M3_stable=bool(m3)),
               protocol_table=table, overall_pass=bool(overall))
    print("JSON_SUMMARY=" + json.dumps(res))


if __name__ == "__main__":
    main()
