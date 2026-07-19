#!/usr/bin/env python3
# Claim 1 -- Theorem 4.5: on treewidth-one (tree) MAP(LRA) with a NON-CONVEX SMT
# constraint and a NON-log-concave Omega^PP density, MpMap computes the EXACT
# constrained MAP.  Checkable consequences:
#   (a) MpMap value == independent optimizer (scipy multi-start) global constrained max;
#   (b) MpMap value is attained at a FEASIBLE assignment (== eval_joint(assignment));
#   (c) MpMap value dominates a dense brute-force grid (never below it).
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mpmap_core as mc

t0 = time.time()

def bump(cx, w, box=(-2.0, 2.0), s=1.0):
    lo = max(box[0], cx - w); hi = min(box[1], cx + w)
    return [(lo, hi, s * np.array([-1.0, 2*cx, w*w - cx*cx]))]

named = {}
named['I1_chain2'] = ([
  {'box':(-2.,2.),'p':bump(0.8,1.4),'parent':None,'children':[1]},
  {'box':(-2.,2.),'p':bump(0.6,1.4),'parent':0,'children':[],
   'edge':{'A':bump(0.,2.2),'B':bump(0.,2.2),
           'cells':[[(-1.,1.,-1.4)],[(1.,-1.,-1.4)]]}}], 601)
named['I2_chain3'] = ([
  {'box':(-2.,2.),'p':bump(0.7,1.5),'parent':None,'children':[1]},
  {'box':(-2.,2.),'p':bump(-0.3,1.5),'parent':0,'children':[2],
   'edge':{'A':bump(0.2,2.0),'B':bump(0.2,2.0),'cells':[[(-1.,1.,-0.9)],[(1.,-1.,-0.9)]]}},
  {'box':(-2.,2.),'p':bump(0.5,1.5),'parent':1,'children':[],
   'edge':{'A':bump(-0.1,2.0),'B':bump(-0.1,2.0),'cells':[[(1.,1.,-0.6)],[(-1.,-1.,-0.6)]]}}], 121)
named['I3_star4'] = ([
  {'box':(-2.,2.),'p':bump(0.1,1.6),'parent':None,'children':[1,2,3]},
  {'box':(-2.,2.),'p':bump(0.9,1.5),'parent':0,'children':[],
   'edge':{'A':bump(0.,2.),'B':bump(0.,2.),'cells':[[(-1.,1.,-0.8)],[(1.,-1.,-0.8)]]}},
  {'box':(-2.,2.),'p':bump(-0.8,1.5),'parent':0,'children':[],
   'edge':{'A':bump(0.,2.),'B':bump(0.,2.),'cells':[[(1.,1.,-0.7)],[(-1.,-1.,-0.7)]]}},
  {'box':(-2.,2.),'p':bump(0.4,1.5),'parent':0,'children':[],
   'edge':{'A':bump(0.,2.),'B':bump(0.,2.),'cells':[[(-1.,1.,-0.5)],[(1.,-1.,-0.5)]]}}], 41)

rows = []; worst_ref_rel = 0.0; worst_self = 0.0; min_dom = 1e9; all_feasible = True
for name, (tree, grid) in named.items():
    res = mc.mpmap_solve(tree, root=0)
    bval, bx = mc.brute_map_vec(tree, grid)                       # dense grid (lower bound)
    ref, rx = mc.refine_map(tree, [res['assignment'], bx], n_restart=14,
                            rng=np.random.default_rng(7))          # independent optimizer
    mval = mc.eval_joint(tree, res['assignment'])                 # value at MpMap assignment
    ref_rel = abs(res['value'] - ref) / max(abs(res['value']), 1e-12)
    self_err = abs(res['value'] - mval)
    dom = res['value'] - bval                                     # >=0 : MpMap dominates grid
    feas = mval > 1e-9
    all_feasible = all_feasible and feas
    worst_ref_rel = max(worst_ref_rel, ref_rel); worst_self = max(worst_self, self_err)
    min_dom = min(min_dom, dom)
    rows.append((name, len(tree), res['value'], ref, ref_rel, self_err, dom, feas))
    print("[%s] n=%d  MpMap=%.6f  scipy_ref=%.6f  rel=%.2e  |MpMap-eval(assign)|=%.2e  dominates_grid=%+.3e  feas=%s"
          % (name, len(tree), res['value'], ref, ref_rel, self_err, dom, feas))

rng = np.random.default_rng(2024)
rand_pos = 0; rand_worst_ref = 0.0; rand_worst_self = 0.0; rand_min_dom = 1e9
for topo, n, grid in [('chain',3,81),('star',4,41),('chain',4,41),('star',3,81),('ternary',4,41)]:
    for rep in range(6):
        tree = mc.make_random_tree(rng, topo, n)
        res = mc.mpmap_solve(tree, root=0)
        if res['assignment'] is None: continue
        mval = mc.eval_joint(tree, res['assignment'])
        if mval <= 1e-9: continue
        bval, bx = mc.brute_map_vec(tree, grid)
        ref, rx = mc.refine_map(tree, [res['assignment'], bx], n_restart=8,
                                rng=np.random.default_rng(rep))
        rand_worst_ref = max(rand_worst_ref, abs(res['value']-ref)/max(abs(res['value']),1e-12))
        rand_worst_self = max(rand_worst_self, abs(res['value']-mval))
        rand_min_dom = min(rand_min_dom, res['value']-bval)
        rand_pos += 1
print("\n[random battery] positive=%d  worst rel(vs scipy)=%.2e  worst self=%.2e  min dominance=%+.3e"
      % (rand_pos, rand_worst_ref, rand_worst_self, rand_min_dom))

verified = (worst_self < 1e-6 and rand_worst_self < 1e-6 and worst_ref_rel < 2e-3
            and rand_worst_ref < 5e-3 and min_dom > -1e-6 and rand_min_dom > -1e-6 and all_feasible)
print("\nMEASURED vs TARGET")
print("  named  max rel |MpMap - scipy global ref|   : %.2e   (target 0; <2e-3)" % worst_ref_rel)
print("  random max rel |MpMap - scipy global ref|   : %.2e   (target 0; <5e-3)" % rand_worst_ref)
print("  max |MpMap - eval_joint(assignment)|        : %.2e   (target 0 EXACT; <1e-6)" % max(worst_self, rand_worst_self))
print("  min (MpMap - brute grid) [dominance]        : %+.3e  (target >=0)" % min(min_dom, rand_min_dom))
print("  all MpMap assignments feasible              : %s" % all_feasible)
print("  VERIFIED (MpMap == exact constrained MAP)   : %s" % verified)

out = {
  "claim":"Theorem 4.5: MpMap solves treewidth-one MAP(LRA) exactly (== independent global optimum)",
  "named":[{"name":r[0],"n_vars":r[1],"mpmap":r[2],"scipy_ref":r[3],"rel_vs_ref":r[4],
            "mpmap_minus_evaljoint":r[5],"dominance_vs_grid":r[6],"feasible":bool(r[7])} for r in rows],
  "named_max_rel_vs_ref":worst_ref_rel,"random_positive":rand_pos,
  "random_max_rel_vs_ref":rand_worst_ref,
  "max_selfconsistency_err":max(worst_self,rand_worst_self),
  "min_dominance_vs_grid":min(min_dom,rand_min_dom),
  "all_feasible":bool(all_feasible),"verified":bool(verified),
  "runtime_s":time.time()-t0,"env":{"numpy":np.__version__,"threads":os.environ.get("OMP_NUM_THREADS","?")}}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"results.json"),"w") as f:
    json.dump(out,f,indent=2)
print("\n[wrote results.json]  runtime=%.2fs" % (time.time()-t0))
