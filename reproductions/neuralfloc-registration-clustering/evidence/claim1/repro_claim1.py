"""
Claim 1 -- NeuralFLoC is a fully unsupervised, end-to-end framework that JOINTLY
performs functional registration and clustering using Neural ODEs.

We verify the four operative parts of that claim on realistic phase-confounded
functional data (N=600, C=3, T=128), CPU, deterministic:
  (a) END-TO-END / JOINT: one differentiable objective L_total = L_reg + alpha
      L_clu is minimised jointly; both terms DECREASE during training.
  (b) NEURAL-ODE DIFFEOMORPHIC WARPS: the learned warps gamma_i lie in Gamma --
      strictly monotone (min increment > 0), boundary-preserving
      (gamma(0)=0, gamma(1)=1).
  (c) FULLY UNSUPERVISED: no labels enter training (used only to score);
      the model still recovers the clusters (high ARI).
  (d) DOES BOTH TASKS: it simultaneously outputs aligned curves (registration:
      phase-dispersion drops sharply) AND cluster assignments (clustering: ARI).

ACCEPTANCE: L_total decreases; all warps are valid diffeomorphisms
(monotone_frac == 1.0, boundary error ~ 0); clustering ARI high and phase
alignment error much lower than raw -- all from a single unsupervised run.
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import neuralfloc as nf

N, C, T, PHASE, SEED = 600, 3, 128, 0.45, 0


def main():
    t0 = time.time()
    x, lab, tg, G = nf.simulate_dataset(N, C, T, seed=SEED, phase=PHASE)
    raw_pe = nf.peak_dispersion(x, lab)
    lbf, _ = nf.kmeans_raw(x, C, 10, SEED); raw_ari = nf.ari(lab, lbf)
    print("=" * 72)
    print(f"CLAIM 1  Fully-unsupervised end-to-end joint NeuralFLoC "
          f"(N={N}, C={C}, T={T})")
    print("independent torch reproduction, CPU single-thread; NO labels used in training")
    print("=" * 72)
    o = nf.train_neuralfloc(x, tg, C, seed=SEED, epochs=280, warmup=150, alpha=0.01,
                            verbose=True, log_every=40)
    h = o["hist"]
    diag = nf.warp_diffeo_diag(o["gamma"])
    ari = nf.ari(lab, o["pred_km"]); acc = nf.cluster_acc(lab, o["pred_km"])
    pe = nf.peak_dispersion(o["xt"], lab)
    # loss decreases (compare early avg vs final avg to be robust to warmup switch)
    Ltot0 = float(np.mean(h["total"][:10])); LtotF = float(np.mean(h["total"][-10:]))
    Lreg0 = float(np.mean(h["reg"][:10])); LregF = float(np.mean(h["reg"][-10:]))
    Lclu0 = float(np.mean(h["clu"][:10])); LcluF = float(np.mean(h["clu"][-10:]))
    print("-" * 72)
    print("(a) JOINT objective decreases (mean of first/last 10 epochs):")
    print(f"    L_total {Ltot0:.4f} -> {LtotF:.4f} | L_reg {Lreg0:.4f} -> {LregF:.4f}"
          f" | L_clu {Lclu0:.4f} -> {LcluF:.4f}")
    print("(b) Neural-ODE warps are valid diffeomorphisms (Gamma):")
    print(f"    monotone_frac={diag['monotone_frac']:.4f} (=1 => all strictly increasing)"
          f"  min_increment={diag['min_increment']:.2e} (>0)"
          f"  max_boundary_err={diag['max_boundary_err']:.2e}")
    print("(c) Fully unsupervised: labels used ONLY to score.")
    print(f"    clustering ARI={ari:.3f}  ACC={acc:.3f}  (vs k-means-on-raw ARI={raw_ari:.3f})")
    print("(d) Simultaneously performs BOTH tasks from one model:")
    print(f"    registration: phase-alignment error {raw_pe:.3f} (raw) -> {pe:.3f} (aligned)")
    print(f"    clustering:   ARI {raw_ari:.3f} (raw) -> {ari:.3f} (joint)")
    ok = (LtotF < Ltot0) and (diag["monotone_frac"] == 1.0) and \
         (diag["max_boundary_err"] < 1e-4) and (ari > 0.7) and (pe < 0.5 * raw_pe)
    print("-" * 72)
    print(f"CLAIM 1 (fully-unsupervised end-to-end joint registration+clustering "
          f"with Neural ODEs): {'VERIFIED' if ok else 'CHECK'}")
    print("=" * 72)
    res = {"N": N, "C": C, "T": T, "seed": SEED,
           "L_total_start": Ltot0, "L_total_final": LtotF,
           "L_reg_start": Lreg0, "L_reg_final": LregF,
           "L_clu_start": Lclu0, "L_clu_final": LcluF,
           "warp_diffeo": diag, "ari": ari, "acc": acc,
           "raw_ari": raw_ari, "phase_err_raw": raw_pe, "phase_err_aligned": pe,
           "loss_curve_total": h["total"], "verified": bool(ok),
           "runtime_s": time.time() - t0}
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=2, default=float)
    print(f"[wrote results.json]  runtime={res['runtime_s']:.1f}s")


if __name__ == "__main__":
    main()
