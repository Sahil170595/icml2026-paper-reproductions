#!/usr/bin/env python3
r"""
control_quantized_identity.py -- post-hoc CONTROL (not a predeclared rule).

Question: on cells where the raw model already passes the SKCE auto-calibration
test, CKME's acceptance drops (e.g. housing/gdn 0.90 -> 0.50).  Is that drop a
genuine miscalibration introduced by CKME, or an artifact of representing the
recalibrated predictive as an n_val-atom empirical distribution (the paper's
Eq. 22 output format, atoms = validation observations)?

Control: replace each raw test predictive F_i by its QUANTIZED IDENTITY -- the
projection onto the same y_val atom set (atom a_j gets the F_i-mass of the
midpoint cell around a_j), i.e. the SAME output format as CKME but with NO
recalibration map.  If acceptance drops similarly, the drop is a resolution
artifact of the representation under the (exact, conditional-MC) SKCE test,
not evidence against CKME.

Writes control_quantized.json; prints per-cell acceptance for
raw / quantized-identity / ckme.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from uci_repro import (N_SPLITS, cdf_on_grid, load_split, fit_predict_nn,
                       make_grid, masses_from_cdf, skce_test)

CELLS_CONTROL = [("bostonHousing", "gdn"), ("concrete", "bnn"), ("energy", "mdn")]


def quantize_to_atoms(F_te, g, atoms):
    """Project grid-CDF predictives onto the atom set (midpoint cells)."""
    aa = np.sort(atoms)
    edges = np.concatenate([[-np.inf], 0.5 * (aa[1:] + aa[:-1]), [np.inf]])
    # evaluate each row's CDF at the cell edges; cell mass = CDF difference
    W = np.empty((F_te.shape[0], aa.size))
    for i in range(F_te.shape[0]):
        Fi = np.interp(edges[1:], g, F_te[i], left=0.0, right=1.0)
        W[i] = np.diff(np.concatenate([[0.0], Fi]))
    W = np.maximum(W, 0.0)
    W /= W.sum(1, keepdims=True)
    return {"kind": "empirical", "atoms": aa, "W": W}


def main():
    t0 = time.time()
    res = {}
    for ds, model in CELLS_CONTROL:
        acc = {"raw": [], "quantized_identity": []}
        skce = {"raw": [], "quantized_identity": []}
        for si in range(N_SPLITS):
            Xtr, ytr, Xva, yva, Xte, yte = load_split(ds, si)
            (rep_va, rep_te), _ = fit_predict_nn(ds, model, Xtr, ytr, Xva, yva,
                                                 [Xva, Xte])
            g = make_grid(np.concatenate([ytr, yva, yte]), [rep_va, rep_te])
            F_te = cdf_on_grid(rep_te, g)
            rep_q = quantize_to_atoms(F_te, g, yva)
            F_q = cdf_on_grid(rep_q, g)
            for name, F in [("raw", F_te), ("quantized_identity", F_q)]:
                F = np.maximum.accumulate(np.clip(F, 0, 1), axis=1)
                Wg = masses_from_cdf(F)
                rng = np.random.default_rng(10_000 + 97 * si)  # same as 'none'
                s, p = skce_test(Wg, F, yte, g, rng)
                acc[name].append(bool(p >= 0.05))
                skce[name].append(s)
            print(f"  {ds}/{model} split {si}: raw p-acc={acc['raw'][-1]} "
                  f"quant p-acc={acc['quantized_identity'][-1]}", flush=True)
        res[f"{ds}__{model}"] = {
            "accept_raw": float(np.mean(acc["raw"])),
            "accept_quantized_identity": float(np.mean(acc["quantized_identity"])),
            "skce_raw_mean": float(np.mean(skce["raw"])),
            "skce_quantized_mean": float(np.mean(skce["quantized_identity"]))}
        print(ds, model, res[f"{ds}__{model}"], flush=True)
    res["elapsed_s"] = time.time() - t0
    (HERE / "control_quantized.json").write_text(json.dumps(res, indent=2))
    print("wrote control_quantized.json")


if __name__ == "__main__":
    main()
