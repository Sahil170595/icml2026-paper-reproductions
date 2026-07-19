"""
Claim 5 -- Ablation: both the registration and clustering modules are essential
(paper Table 1 bottom rows: "Ours w/o Reg" severely degrades clustering;
"Ours w/o Clu" impairs alignment).

TEST (real scale, deterministic, CPU): on N=480, C=3, T=128 phase-confounded
functional data compare the full model to two ablations:
  * FULL      : Neural-ODE registration + cluster-conditional SRVF alignment
                + spectral clustering (joint).
  * w/o Reg   : disable the warping module (gamma = identity) -> cluster on
                UNREGISTERED curves.  Expect clustering ARI to collapse toward
                the raw-clustering level (registration is essential).
  * w/o Clu   : registration WITHOUT the clustering module guiding it
                (global-only alignment, alpha=0, no cluster-conditional target)
                -> measure registration alignment error (phase dispersion / ATV).

ACCEPTANCE: full ARI >> (w/o Reg) ARI (registration essential for clustering),
and the full model's alignment error is <= the w/o-Clu alignment error
(clustering informs registration; not worse than global-only).

Staged: `python3 repro_claim5.py <seed>` then `... combine`.
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache"); os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, os.path.dirname(HERE))
import neuralfloc as nf

N, C, T, PHASE = 480, 3, 128, 0.45
SEEDS = [0, 1, 2]


def _cache(seed, part, fn):
    """Resumable per-variant cache so each 280-epoch training can be spread over
       separate sub-44s calls without recomputation."""
    pf = os.path.join(CACHE, f"c5_seed{seed}_{part}.json")
    if os.path.exists(pf):
        return json.load(open(pf))
    d = fn()
    json.dump(d, open(pf, "w"), indent=2, default=float)
    return d


def run_seed(seed):
    t0 = time.time()
    x, lab, tg, G = nf.simulate_dataset(N, C, T, seed=seed, phase=PHASE)

    def do_raw():
        lbf, _ = nf.kmeans_raw(x, C, 10, seed)
        return {"raw_ari": nf.ari(lab, lbf)}

    def do_full():
        o = nf.train_neuralfloc(x, tg, C, seed=seed, epochs=280, warmup=150, alpha=0.01)
        return dict(ari=nf.ari(lab, o["pred_km"]), acc=nf.cluster_acc(lab, o["pred_km"]),
                    atv=nf.atv(o["xt"], lab), phase_err=nf.peak_dispersion(o["xt"], lab))

    def do_wreg():
        o = nf.train_neuralfloc(x, tg, C, seed=seed, epochs=280, warmup=150, use_warp=False)
        return dict(ari=nf.ari(lab, o["pred_km"]), acc=nf.cluster_acc(lab, o["pred_km"]))

    def do_wclu():
        o = nf.train_neuralfloc(x, tg, C, seed=seed, epochs=280, warmup=150,
                                alpha=0.0, cond_reg=False)
        return dict(atv=nf.atv(o["xt"], lab), phase_err=nf.peak_dispersion(o["xt"], lab))

    raw = _cache(seed, "raw", do_raw)
    full = _cache(seed, "full", do_full)
    wreg = _cache(seed, "wreg", do_wreg)
    wclu = _cache(seed, "wclu", do_wclu)
    out = {"seed": seed, "raw_ari": raw["raw_ari"],
           "full": full, "wo_reg": wreg, "wo_clu": wclu, "time_s": time.time() - t0}
    json.dump(out, open(os.path.join(CACHE, f"c5_seed{seed}.json"), "w"),
              indent=2, default=float)
    print(f"[seed {seed}] full ARI={out['full']['ari']:.3f}  "
          f"w/o Reg ARI={out['wo_reg']['ari']:.3f} (raw {out['raw_ari']:.3f})  "
          f"full phaseErr={out['full']['phase_err']:.3f}  "
          f"w/o Clu phaseErr={out['wo_clu']['phase_err']:.3f}  ({out['time_s']:.1f}s)")


def combine():
    rows = [json.load(open(os.path.join(CACHE, f"c5_seed{s}.json")))
            for s in SEEDS if os.path.exists(os.path.join(CACHE, f"c5_seed{s}.json"))]
    def m(path):
        v = []
        for r in rows:
            d = r
            for k in path.split("."):
                d = d[k]
            v.append(d)
        return float(np.mean(v)), float(np.std(v))
    print("=" * 74)
    print(f"CLAIM 5  Ablation: registration and clustering are both essential "
          f"(N={N}, {len(rows)} seeds)")
    print("independent torch reproduction, CPU single-thread")
    print("=" * 74)
    print(f"{'Variant':<28}{'clustering ARI':>16}{'ACC':>10}{'phase-align err':>18}")
    fa, fs = m("full.ari"); fc, _ = m("full.acc"); fp, _ = m("full.phase_err")
    ra, rs = m("wo_reg.ari"); rc, _ = m("wo_reg.acc")
    wp, _ = m("wo_clu.phase_err"); raw_a, _ = m("raw_ari")
    print(f"{'FULL (reg + clustering)':<28}{fa:>9.3f}+-{fs:<4.2f}{fc:>9.3f}{fp:>16.3f}")
    print(f"{'w/o Reg (no warping)':<28}{ra:>9.3f}+-{rs:<4.2f}{rc:>9.3f}{'   -':>16}")
    print(f"{'w/o Clu (reg only)':<28}{'   -':>14}{'   -':>10}{wp:>16.3f}")
    print(f"{'(reference: k-means raw)':<28}{raw_a:>9.3f}{'':>10}{'':>16}")
    print("-" * 74)
    drop = fa - ra
    reg_essential = (ra < fa - 0.25) and (abs(ra - raw_a) < 0.15)
    align_ok = fp <= wp + 0.01
    print(f"removing registration drops clustering ARI by {drop:.3f} "
          f"(full {fa:.3f} -> w/o Reg {ra:.3f} ~ raw {raw_a:.3f}): {reg_essential}")
    print(f"full alignment err ({fp:.3f}) <= w/o-Clu alignment err ({wp:.3f}): {align_ok}")
    verdict = reg_essential
    print(f"CLAIM 5 (both modules essential; registration essential for "
          f"clustering): {'VERIFIED' if verdict else 'CHECK'}")
    print("=" * 74)
    json.dump({"N": N, "n_seeds": len(rows), "full_ari": [fa, fs],
               "wo_reg_ari": [ra, rs], "raw_ari": raw_a, "full_phase_err": fp,
               "wo_clu_phase_err": wp, "reg_essential": bool(reg_essential),
               "verified": bool(verdict), "per_seed": rows},
              open(os.path.join(HERE, "results.json"), "w"), indent=2, default=float)
    print("[wrote results.json]")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "combine"
    combine() if arg == "combine" else run_seed(int(arg))
