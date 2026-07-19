"""
Claim C2 (paper-protocol benchmark) -- state-of-the-art registration + clustering
on the paper's OWN named functional-benchmark scenarios (Section 5, Table 1).

The paper's benchmarks are UCR real datasets (Shapes / Wave / Symbols), which are
offline here, so we build matched-dimension synthetic analogues at the paper's
EXACT class counts C (and comparable N, T -- tabulated vs the paper) using the
paper's generative model (phase + amplitude + noise), and evaluate with the
paper's EXACT metrics: ATV (Eq A.1, lower better), ACC, NMI (higher better),
under the paper's hyperparameters (K=10, alpha=0.01, encoder+MLP, mean over
seeds).  Baselines mirror Table 1: sequential register-then-cluster (SrvfRegNet+
clustering analogue) and Ours w/o Reg (clustering without registration).

Staged/cache-resumable:
  python3 repro_claim3b.py run <scenario> <seed>
  python3 repro_claim3b.py batch
  python3 repro_claim3b.py combine
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache"); os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import neuralfloc as nf
import nfloc_ext as ext

# repro dims (downscaled T,N vs paper -- tabulated); C is EXACT to the paper.
# name: (C, N_repro, T_repro, phase, paper_N, paper_T)
SCN = {
    "Shapes":     (2, 300, 96, 0.9, 1095, 1024),
    "Wave(d=1)":  (2, 300, 96, 0.6, 1120, 315),
    "Symbols(2)": (2, 300, 96, 0.7, 343, 398),
    "Symbols(3)": (3, 300, 96, 0.8, 510, 398),
}
# paper Table 1 "Ours" reference (ATV, ACC, NMI) for side-by-side context
PAPER_OURS = {
    "Shapes": (8.1, 0.937, 0.651), "Wave(d=1)": (4.1, 0.993, 0.941),
    "Symbols(2)": (3.1, 0.930, 0.639), "Symbols(3)": (1.6, 0.951, 0.835),
}
SEEDS = [0, 1, 2]
K = 10


def run(scn, seed):
    tag = f"c3b_{scn.replace('(', '').replace(')', '').replace('=', '')}_s{seed}"
    p = os.path.join(CACHE, tag + ".json")
    if os.path.exists(p):
        print("cached", tag); return
    C, N, T, phase, pN, pT = SCN[scn]
    x, lab, t, G = nf.simulate_dataset(N, C, T, seed=seed, phase=phase)
    t0 = time.time()
    # Ours (joint NeuralFLoC), k-means-on-aligned readout
    o = nf.train_neuralfloc(x, t, C, seed=seed, epochs=220, warmup=120, K=K,
                            alpha=0.01, Sode=30)
    ours = {"acc": float(nf.cluster_acc(lab, o["pred_km"])),
            "nmi": float(nf.nmi(lab, o["pred_km"])),
            "atv": float(nf.atv(o["xt"], lab)),
            "mono": float(nf.warp_diffeo_diag(o["gamma"])["monotone_frac"])}
    # Sequential register-then-cluster (SrvfRegNet + clustering analogue)
    slab, sxt, sg = nf.register_then_cluster(x, t, C, K, seed=seed, epochs=150,
                                             warmup=45, Sode=30)
    seq = {"acc": float(nf.cluster_acc(lab, slab)), "nmi": float(nf.nmi(lab, slab)),
           "atv": float(nf.atv(sxt, lab))}
    # Ours w/o Reg (clustering without registration)
    rlab, _ = nf.kmeans_raw(x, C, K, seed)
    woreg = {"acc": float(nf.cluster_acc(lab, rlab)), "nmi": float(nf.nmi(lab, rlab)),
             "atv": float(nf.atv(x, lab))}
    res = {"scn": scn, "seed": seed, "C": C, "N": N, "T": T,
           "paper_N": pN, "paper_T": pT, "t": time.time() - t0,
           "ours": ours, "seq": seq, "woreg": woreg}
    json.dump(res, open(p, "w"), indent=1, default=float)
    print("%-12s s%d  Ours ACC=%.3f NMI=%.3f ATV=%.2f | seq ACC=%.3f | w/oReg ACC=%.3f (%.1fs)"
          % (scn, seed, ours["acc"], ours["nmi"], ours["atv"], seq["acc"],
             woreg["acc"], res["t"]))


def batch():
    for scn in SCN:
        for s in SEEDS:
            try:
                run(scn, s)
            except Exception as e:
                print("ERR", scn, s, repr(e))
    combine()
    print("C3B BATCH DONE")


def agg(rows, meth, met):
    v = [r[meth][met] for r in rows]
    return [float(np.mean(v)), float(np.std(v)), len(v)]


def combine():
    out = {"scenarios": [], "hyperparams": {"K": K, "alpha": 0.01, "seeds": SEEDS},
           "note": "UCR archive offline; matched-C synthetic analogues, paper metrics"}
    ok_order = True
    for scn in SCN:
        rows = []
        for s in SEEDS:
            tag = f"c3b_{scn.replace('(', '').replace(')', '').replace('=', '')}_s{s}"
            p = os.path.join(CACHE, tag + ".json")
            if os.path.exists(p):
                rows.append(json.load(open(p)))
        if not rows:
            continue
        C, N, T, phase, pN, pT = SCN[scn]
        entry = {"scn": scn, "C": C, "N": N, "T": T, "paper_N": pN, "paper_T": pT,
                 "n_seeds": len(rows)}
        for meth in ["ours", "seq", "woreg"]:
            for met in ["acc", "nmi", "atv"]:
                entry.setdefault(meth, {})[met] = agg(rows, meth, met)
        entry["paper_ours"] = PAPER_OURS.get(scn)
        # ordering check: Ours ACC >= seq and Ours >> w/oReg; Ours ATV <= seq
        oa = entry["ours"]["acc"][0]; sa = entry["seq"]["acc"][0]; wa = entry["woreg"]["acc"][0]
        oatv = entry["ours"]["atv"][0]; satv = entry["seq"]["atv"][0]
        order = (oa >= sa - 0.03) and (oa > wa + 0.1) and (oatv <= satv + 1e-6)
        entry["ordering_ok"] = bool(order)
        ok_order = ok_order and order
        out["scenarios"].append(entry)
    out["all_ordering_ok"] = bool(ok_order)
    json.dump(out, open(os.path.join(HERE, "results_bench.json"), "w"), indent=1, default=float)
    for e in out["scenarios"]:
        print("%-12s Ours ACC=%.3f NMI=%.3f ATV=%.2f | seq ACC=%.3f ATV=%.2f | w/oReg ACC=%.3f | order=%s"
              % (e["scn"], e["ours"]["acc"][0], e["ours"]["nmi"][0], e["ours"]["atv"][0],
                 e["seq"]["acc"][0], e["seq"]["atv"][0], e["woreg"]["acc"][0], e["ordering_ok"]))
    print("all ordering ok:", out["all_ordering_ok"])
    print("wrote results_bench.json")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "combine"
    if a == "combine":
        combine()
    elif a == "batch":
        batch()
    elif a == "run":
        run(sys.argv[2], int(sys.argv[3]))
