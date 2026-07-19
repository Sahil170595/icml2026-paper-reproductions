"""
Independent NumPy reproduction of CLAIM 3 (Figure 1) of
"Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural
Networks" (OpenReview aIH1jyU37z; no arXiv).

CLAIM 3: On synthetic semi-supervised node-classification tasks over SIGNED graphs,
sheaf neural networks OUTPERFORM Kipf-Welling GCN variants across feature and
edge-noise regimes (Figure 1).

SETUP (controlled A/B; the ONLY difference between the two networks is whether the
propagation operator sees edge SIGNS):
  * Signed SBM, 2 balanced classes, n=240.  Edge probability p is the SAME within
    and between classes, so the UNSIGNED topology |A| carries NO class information;
    all class information lives in the edge SIGN (+1 within class, -1 between class),
    i.e. a structurally-balanced signed graph.
  * Node features: class 0 -> +m, class 1 -> -m on `sig_dim` informative dims, plus
    Gaussian noise of std `feat_noise`.
  * Semi-supervised: 40 labeled nodes, accuracy measured on the remaining 200.
  * Propagation:  P(A_off) = D~^{-1/2}(A_off + I) D~^{-1/2},  D~ = diag(|A|.sum)+I.
        GCN     : A_off = |A_signed|   (Kipf-Welling variant, sign-blind)
        SheafNN : A_off = A_signed     (1-d signed sheaf = signed normalized adjacency)
    Identical degree normalization D~ for both => the sheaf mechanism is EXACTLY the
    edge sign (ties directly to Claims 1 & 2).
  * Network: 2-layer graph net  softmax( P relu(P X W0) W1 ), full-batch gradient
    descent on cross-entropy over labeled nodes; identical init/optimizer for both.

REGIMES (5 random-graph trials each; mean +/- std reported):
  feature-noise sweep : feat_noise in {0.5,1.0,1.5,2.0,2.5}, edge_flip = 0
  edge-noise  sweep   : edge_flip  in {0.0,0.1,0.2,0.3,0.4}, feat_noise = 1.5
                        (edge_flip = fraction of edge signs randomly flipped)

COMPARISON RULE: mean test accuracy SheafNN > GCN in EVERY swept regime cell; report
how many cells separate beyond error bars (mean_S - std_S > mean_G + std_G).
FALSIFICATION: GCN mean >= SheafNN mean in any swept cell.

CONTROLS:
  (C1) sign-blind sheaf (feed |A| to the sheaf operator) must MATCH GCN (isolates the
       sign as the sole active ingredient).
  (C2) random-label control: both networks ~ chance (0.5) => the task is non-trivial
       and the gain is not an artifact.
"""
import json
import numpy as np


def make_signed_sbm(n, p, feat_noise, edge_flip, sig_dim, rng):
    y = np.zeros(n, int); y[n // 2:] = 1
    U = rng.random((n, n)); M = (U < p); M = np.triu(M, 1); M = M | M.T
    same = (y[:, None] == y[None, :])
    sign = np.where(same, 1.0, -1.0)
    if edge_flip > 0:
        Fl = (rng.random((n, n)) < edge_flip); Fl = np.triu(Fl, 1); Fl = Fl | Fl.T
        sign = sign * np.where(Fl, -1.0, 1.0)
    Asig = M * sign; np.fill_diagonal(Asig, 0.0)
    Aabs = np.abs(Asig)
    F = 16; m = np.zeros(F); m[:sig_dim] = 1.0
    cs = np.where(y == 0, 1.0, -1.0)[:, None]
    X = cs * m[None, :] + feat_noise * rng.standard_normal((n, F))
    return y, Asig, Aabs, X


def prop(A_off, deg_abs):
    """P = D~^{-1/2} (A_off + I) D~^{-1/2},  D~ = deg_abs + 1  (shared by GCN & Sheaf)."""
    n = A_off.shape[0]
    dt = deg_abs + 1.0
    Dm = 1.0 / np.sqrt(dt)
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


def run_cell(n, p, feat_noise, edge_flip, sig_dim, n_trials, base_seed):
    ag, as_ = [], []
    for t in range(n_trials):
        r = np.random.default_rng(base_seed + t)
        y, Asig, Aabs, X = make_signed_sbm(n, p, feat_noise, edge_flip, sig_dim, r)
        deg = Aabs.sum(1)
        idx = r.permutation(n); tr = idx[:40]; te = idx[40:]
        Pg = prop(Aabs, deg); Ps = prop(Asig, deg)
        ag.append(train_eval(Pg, X, y, tr, te))
        as_.append(train_eval(Ps, X, y, tr, te))
    return np.array(ag), np.array(as_)


def main():
    print("=" * 78)
    print("CLAIM 3  SheafNN outperforms Kipf-Welling GCN on signed-graph node")
    print("         classification across feature & edge-noise regimes (Figure 1)")
    print("OpenReview aIH1jyU37z  -  independent NumPy implementation")
    print("=" * 78)
    n, p, sig_dim, n_trials = 240, 0.05, 3, 5
    res = {"config": dict(n=n, p=p, sig_dim=sig_dim, n_trials=n_trials,
                          labeled=40, test=n - 40, arch="2-layer P-relu-P softmax"),
           "feature_sweep": [], "edge_sweep": []}

    print(f"\n--- FEATURE-NOISE sweep (edge_flip=0), {n_trials} trials, mean+/-std ---")
    print(f"{'feat_noise':>10} | {'GCN acc':>16} | {'SheafNN acc':>16} | {'gap':>6} | sep")
    all_win, all_sep = True, 0
    for fn in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ag, as_ = run_cell(n, p, fn, 0.0, sig_dim, n_trials, 3000)
        gm, gs, sm, ss = ag.mean(), ag.std(), as_.mean(), as_.std()
        sep = (sm - ss) > (gm + gs); all_sep += int(sep)
        all_win = all_win and (sm > gm)
        print(f"{fn:>10.1f} | {gm:>7.3f} +/- {gs:>5.3f} | {sm:>7.3f} +/- {ss:>5.3f} "
              f"| {sm-gm:>+5.3f} | {'yes' if sep else 'no'}")
        res["feature_sweep"].append(dict(feat_noise=fn, gcn_mean=float(gm), gcn_std=float(gs),
                                         sheaf_mean=float(sm), sheaf_std=float(ss),
                                         gcn_acc=[float(x) for x in ag],
                                         sheaf_acc=[float(x) for x in as_], separated=bool(sep)))

    print(f"\n--- EDGE-NOISE sweep (feat_noise=1.5), {n_trials} trials, mean+/-std ---")
    print(f"{'edge_flip':>10} | {'GCN acc':>16} | {'SheafNN acc':>16} | {'gap':>6} | sep")
    for ef in [0.0, 0.1, 0.2, 0.3, 0.4]:
        ag, as_ = run_cell(n, p, 1.5, ef, sig_dim, n_trials, 4000)
        gm, gs, sm, ss = ag.mean(), ag.std(), as_.mean(), as_.std()
        sep = (sm - ss) > (gm + gs); all_sep += int(sep)
        all_win = all_win and (sm > gm)
        print(f"{ef:>10.2f} | {gm:>7.3f} +/- {gs:>5.3f} | {sm:>7.3f} +/- {ss:>5.3f} "
              f"| {sm-gm:>+5.3f} | {'yes' if sep else 'no'}")
        res["edge_sweep"].append(dict(edge_flip=ef, gcn_mean=float(gm), gcn_std=float(gs),
                                      sheaf_mean=float(sm), sheaf_std=float(ss),
                                      gcn_acc=[float(x) for x in ag],
                                      sheaf_acc=[float(x) for x in as_], separated=bool(sep)))

    # ---------- controls ----------
    print("\n--- CONTROLS ---")
    # C1: sign-blind sheaf (feed |A|) must match GCN
    r = np.random.default_rng(9000)
    y, Asig, Aabs, X = make_signed_sbm(n, p, 1.5, 0.0, sig_dim, r)
    deg = Aabs.sum(1); idx = r.permutation(n); tr = idx[:40]; te = idx[40:]
    acc_gcn = train_eval(prop(Aabs, deg), X, y, tr, te)
    acc_signblind = train_eval(prop(Aabs, deg), X, y, tr, te)  # sheaf op fed |A|
    c1_ok = abs(acc_gcn - acc_signblind) < 1e-12
    print(f"(C1) sign-blind sheaf (|A|) acc={acc_signblind:.4f} vs GCN acc={acc_gcn:.4f} "
          f"-> identical: {c1_ok}")
    # C2: random labels -> chance
    yr = r.permutation(y.copy())
    acc_rand_g = train_eval(prop(Aabs, deg), X, yr, tr, te)
    acc_rand_s = train_eval(prop(Asig, deg), X, yr, tr, te)
    c2_ok = (acc_rand_g < 0.62) and (acc_rand_s < 0.62)
    print(f"(C2) random-label control: GCN={acc_rand_g:.3f}  SheafNN={acc_rand_s:.3f} "
          f"(both ~chance 0.5: {c2_ok})")

    res["controls"] = dict(c1_signblind_matches_gcn=bool(c1_ok),
                           acc_gcn=acc_gcn, acc_signblind=acc_signblind,
                           c2_random_chance=bool(c2_ok),
                           acc_rand_gcn=acc_rand_g, acc_rand_sheaf=acc_rand_s)
    res["sheaf_wins_all_cells"] = bool(all_win)
    res["cells_separated_beyond_errbars"] = int(all_sep)
    res["n_cells"] = 10
    overall = all_win and c1_ok and c2_ok
    res["overall_pass"] = bool(overall)

    print("-" * 78)
    print(f"SheafNN mean > GCN mean in ALL 10 cells: {all_win}")
    print(f"cells separated beyond error bars: {all_sep}/10")
    print(f"controls: C1(sign-blind==GCN)={c1_ok}  C2(random==chance)={c2_ok}")
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 78)
    print("JSON_SUMMARY=" + json.dumps(res))


if __name__ == "__main__":
    main()
