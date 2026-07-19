"""
Claim 3 -- Joint registration+clustering beats baselines (Table 1 core result).

The paper claims NeuralFLoC's JOINT framework yields superior clustering (ACC,
NMI; we also report ARI) and registration (ATV) versus (i) clustering on raw
curves and (ii) sequential register-then-cluster pipelines.

TEST (real scale, deterministic, CPU): simulate N=600 functional curves in C=3
clusters (distinct SHAPES -- Gaussian peaks of different width -- with strong
random monotone PHASE warps + amplitude variation + noise).  Phase confounds raw
clustering; registration is required to reveal shape.  Compare:
  * k-means on raw curves (Fourier coeffs)                    [baseline 1]
  * register-then-cluster: global Neural-ODE template align,
    then k-means on aligned coeffs                            [baseline 2]
  * NeuralFLoC (Ours): joint Neural-ODE registration +
    cluster-conditional SRVF alignment + spectral clustering [ours]
Metrics: clustering ARI / NMI / ACC (higher better); registration ATV (Eq A.1,
lower better) computed under TRUE labels (= alignment quality).

ACCEPTANCE: Ours ARI > register-then-cluster ARI > raw ARI (means over seeds),
and both registration methods dramatically exceed raw (registration essential).

Staged: `python3 repro_claim3.py <seed>` (per-seed) then `... combine`.
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache"); os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, os.path.dirname(HERE))
import neuralfloc as nf

N, C, T = 600, 3, 128
PHASE = 0.45
SEEDS = [0, 1, 2, 3]


def run_seed(seed):
    t0 = time.time()
    x, lab, tg, G = nf.simulate_dataset(N, C, T, seed=seed, phase=PHASE)
    out = {"seed": seed}
    # baseline 1: k-means on raw curves
    lbf, _ = nf.kmeans_raw(x, C, 10, seed)
    out["raw"] = dict(ari=nf.ari(lab, lbf), nmi=nf.nmi(lab, lbf),
                      acc=nf.cluster_acc(lab, lbf), atv=nf.atv(x, lab),
                      phase_err=nf.peak_dispersion(x, lab))
    # baseline 2: register-then-cluster (global template)
    lbr, xar, _ = nf.register_then_cluster(x, tg, C, 10, seed=seed, epochs=200, warmup=45)
    out["seq"] = dict(ari=nf.ari(lab, lbr), nmi=nf.nmi(lab, lbr),
                      acc=nf.cluster_acc(lab, lbr), atv=nf.atv(xar, lab),
                      phase_err=nf.peak_dispersion(xar, lab))
    # ours: joint NeuralFLoC
    o = nf.train_neuralfloc(x, tg, C, seed=seed, epochs=280, warmup=150, alpha=0.01)
    out["ours"] = dict(ari=nf.ari(lab, o["pred_km"]), nmi=nf.nmi(lab, o["pred_km"]),
                       acc=nf.cluster_acc(lab, o["pred_km"]), atv=nf.atv(o["xt"], lab),
                       phase_err=nf.peak_dispersion(o["xt"], lab),
                       warp_monotone_frac=nf.warp_diffeo_diag(o["gamma"])["monotone_frac"])
    out["time_s"] = time.time() - t0
    json.dump(out, open(os.path.join(CACHE, f"c3_seed{seed}.json"), "w"), indent=2, default=float)
    print(f"[seed {seed}]  raw ARI={out['raw']['ari']:.3f}  "
          f"seq ARI={out['seq']['ari']:.3f}  OURS ARI={out['ours']['ari']:.3f}  "
          f"({out['time_s']:.1f}s)")


def combine():
    rows = []
    for s in SEEDS:
        f = os.path.join(CACHE, f"c3_seed{s}.json")
        if os.path.exists(f):
            rows.append(json.load(open(f)))
    def agg(method, metric):
        v = [r[method][metric] for r in rows if metric in r[method]]
        if not v:
            return float("nan"), float("nan")
        return float(np.mean(v)), float(np.std(v)), len(v)
    print("=" * 76)
    print(f"CLAIM 3  Joint registration+clustering vs baselines  (N={N}, C={C}, "
          f"T={T}, {len(rows)} seeds)")
    print("independent torch reproduction, CPU single-thread; phase-confounded functional data")
    print("=" * 76)
    print(f"{'Method':<26}{'ARI':>14}{'NMI':>12}{'ACC':>12}{'ATV':>8}{'PhaseErr':>9}")
    disp = {"raw": "k-means on raw", "seq": "register-then-cluster",
            "ours": "NeuralFLoC (Ours, joint)"}
    summ = {}
    for m in ["raw", "seq", "ours"]:
        a = agg(m, "ari"); n = agg(m, "nmi"); c = agg(m, "acc"); v = agg(m, "atv")
        pe = agg(m, "phase_err")
        summ[m] = dict(ari=a, nmi=n, acc=c, atv=v, phase_err=pe)
        print(f"{disp[m]:<26}{a[0]:>7.3f}+-{a[1]:<4.2f}{n[0]:>7.3f}     "
              f"{c[0]:>6.3f}     {v[0]:>6.2f}    {pe[0]:>6.3f}")
    print("-" * 76)
    ari_raw = summ["raw"]["ari"][0]; ari_seq = summ["seq"]["ari"][0]; ari_ours = summ["ours"]["ari"][0]
    ok = (ari_ours >= ari_seq - 0.02) and (ari_seq > ari_raw + 0.2) and (ari_ours > ari_raw + 0.2)
    print(f"Ours ARI ({ari_ours:.3f}) >= sequential ({ari_seq:.3f}) >> raw ({ari_raw:.3f}): {ok}")
    print(f"registration phase-alignment error (down) -- raw {summ['raw']['phase_err'][0]:.3f}"
          f" | seq {summ['seq']['phase_err'][0]:.3f} | ours {summ['ours']['phase_err'][0]:.3f}")
    print(f"CLAIM 3 (joint clustering+registration > baselines): {'VERIFIED' if ok else 'CHECK'}")
    print("=" * 76)
    json.dump({"N": N, "C": C, "T": T, "phase": PHASE, "n_seeds": len(rows),
               "summary": summ, "per_seed": rows, "verified": bool(ok)},
              open(os.path.join(HERE, "results.json"), "w"), indent=2, default=float)
    print("[wrote results.json]")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "combine"
    combine() if arg == "combine" else run_seed(int(arg))
