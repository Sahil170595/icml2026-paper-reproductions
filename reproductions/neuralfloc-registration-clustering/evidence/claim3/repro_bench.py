"""
Claim 3 (C2) -- paper benchmark PROTOCOL: the paper's own named simulation
scenarios (their class counts C and phase+amplitude+noise DGP) AND an attempt
at a real functional dataset loadable OFFLINE.

The UCR archive used by the paper (Shapes / Wave / Symbols) is not fetchable in
this offline sandbox, so we (1) reproduce the paper's simulation benchmark under
its exact class counts, comparing NeuralFLoC (Ours) vs the paper's structural
baselines (k-means-on-raw, register-then-cluster) on the paper's metrics
(ACC / NMI / ARI / ATV), and (2) attempt to load a REAL functional series that
ships offline (scipy / statsmodels / sklearn) and report registration + cluster
metrics on it, or state honestly that none is loadable.

Staged: `python3 repro_bench.py <scenario_index>`, `python3 repro_bench.py real`,
then `python3 repro_bench.py combine`.
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache"); os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, os.path.dirname(HERE))
import neuralfloc as nf
import nfloc_ext as ext

SCEN = list(ext.SCENARIOS.keys())     # Shapes, Wave(d=1), Symbols(2), Symbols(3)
SEEDS = [0]
EPOCHS, WARMUP = 160, 90
BENCH_PHASE = 0.55                    # moderate phase regime (model converges; registration helps)
BENCH_T = 128


def _cluster_metrics(lab, pred):
    return dict(ari=float(nf.ari(lab, pred)), nmi=float(nf.nmi(lab, pred)),
                acc=float(nf.cluster_acc(lab, pred)))


def run_scenario(idx):
    name = SCEN[idx]
    C = ext.SCENARIOS[name][0]
    N = ext.SCENARIOS[name][1]
    rows = []
    for s in SEEDS:
        x, lab, t, _G = nf.simulate_dataset(N, C, BENCH_T, seed=s, phase=BENCH_PHASE)
        raw, _ = nf.kmeans_raw(x, C, 10, s)
        mr = _cluster_metrics(lab, raw); mr["atv"] = float(nf.atv(x, lab))
        slab, sxt, _ = nf.register_then_cluster(x, t, C, 10, s, epochs=140, warmup=40)
        ms = _cluster_metrics(lab, slab); ms["atv"] = float(nf.atv(sxt, lab))
        o = nf.train_neuralfloc(x, t, C, seed=s, epochs=EPOCHS, warmup=WARMUP, alpha=0.01)
        mo = _cluster_metrics(lab, o["pred_km"]); mo["atv"] = float(nf.atv(o["xt"], lab))
        rows.append({"seed": s, "raw": mr, "seq": ms, "ours": mo})
    def agg(meth, met):
        v = [r[meth][met] for r in rows]
        return [float(np.mean(v)), float(np.std(v)), len(v)]
    out = {"name": name, "C": C, "paper_N": ext.SCENARIOS[name][5],
           "paper_T": ext.SCENARIOS[name][6], "n_seeds": len(rows),
           "raw": {m: agg("raw", m) for m in ("ari", "nmi", "acc", "atv")},
           "seq": {m: agg("seq", m) for m in ("ari", "nmi", "acc", "atv")},
           "ours": {m: agg("ours", m) for m in ("ari", "nmi", "acc", "atv")}}
    json.dump(out, open(os.path.join(CACHE, f"scen_{idx}.json"), "w"), indent=2)
    print(f"[{name}] Ours ARI={out['ours']['ari'][0]:.3f} seq={out['seq']['ari'][0]:.3f} "
          f"raw={out['raw']['ari'][0]:.3f}")


def load_real():
    """Real, labelled dataset that ships OFFLINE.  The UCR functional archive
       and statsmodels/scipy series need network or optional deps that are
       absent here; sklearn's digits is bundled (real handwritten-digit scans).
       We take two classes and read each 8x8 image row-major as a length-64
       real observed 1-D signal.  Returns (name, X, labels, note)."""
    try:
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        keep = np.isin(y, (3, 8))                     # two visually-confusable classes
        Xk = X[keep].astype(float); yk = (y[keep] == 8).astype(int)
        return ("digits_3v8", Xk, yk,
                "sklearn load_digits (real handwritten digits, offline); classes 3 vs 8, "
                "each 8x8 scan read row-major as a length-64 signal (raster, not phase-warped FDA)")
    except Exception as e:
        return (None, None, None, f"no real dataset loadable offline: {e!r}")


def run_real():
    name, X, lab, note = load_real()
    if X is None:
        out = {"loaded": False, "note": note}
        json.dump(out, open(os.path.join(CACHE, "real.json"), "w"), indent=2)
        print("real data: NONE loadable offline —", note)
        return
    X = np.asarray(X, dtype=np.float32)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    T = X.shape[1]; t = np.linspace(0, 1, T).astype(np.float32); K = 2
    # baseline clustering vs registered clustering, scored against TRUE labels
    raw, _ = nf.kmeans_raw(X, K, 10, 0)
    slab, sxt, sg = nf.register_then_cluster(X, t, K, 10, 0, epochs=150, warmup=45)
    diffe = nf.warp_diffeo_diag(sg)
    out = {"loaded": True, "name": name, "note": note,
           "n_curves": int(X.shape[0]), "T": int(T),
           "raw_ari": float(nf.ari(lab, raw)), "raw_acc": float(nf.cluster_acc(lab, raw)),
           "reg_ari": float(nf.ari(lab, slab)), "reg_acc": float(nf.cluster_acc(lab, slab)),
           "warp_monotone_frac": diffe["monotone_frac"],
           "warp_boundary_err": diffe["max_boundary_err"]}
    json.dump(out, open(os.path.join(CACHE, "real.json"), "w"), indent=2)
    print(f"real [{name}] n={X.shape[0]} T={T} raw ARI={out['raw_ari']:.3f} "
          f"reg ARI={out['reg_ari']:.3f}")


def combine():
    scens = []
    for i in range(len(SCEN)):
        p = os.path.join(CACHE, f"scen_{i}.json")
        if os.path.exists(p):
            scens.append(json.load(open(p)))
    real = None
    if os.path.exists(os.path.join(CACHE, "real.json")):
        real = json.load(open(os.path.join(CACHE, "real.json")))
    print("=" * 70)
    print("C2  Paper benchmark protocol: named scenarios, Ours vs baselines")
    print("=" * 70)
    for sc in scens:
        print(f"{sc['name']:12s} C={sc['C']} | Ours ARI {sc['ours']['ari'][0]:.3f} "
              f"NMI {sc['ours']['nmi'][0]:.3f} ACC {sc['ours']['acc'][0]:.3f} | "
              f"seq {sc['seq']['ari'][0]:.3f} | raw {sc['raw']['ari'][0]:.3f}")
    ours_beats_raw = all(sc["ours"]["ari"][0] > sc["raw"]["ari"][0] + 0.2 for sc in scens) if scens else False
    out = {"scenarios": scens, "real": real,
           "ours_beats_raw_all": bool(ours_beats_raw), "n_scenarios": len(scens)}
    json.dump(out, open(os.path.join(HERE, "results_bench.json"), "w"), indent=2)
    if real is not None:
        print("real:", real.get("name", real.get("note")))
    print("[wrote results_bench.json]")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "combine"
    if a == "combine":
        combine()
    elif a == "real":
        run_real()
    else:
        run_scenario(int(a))
