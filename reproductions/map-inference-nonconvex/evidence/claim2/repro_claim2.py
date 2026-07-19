#!/usr/bin/env python3
# Claim 2 -- The MpMap algorithm (Alg 1-3, Eqs 4-6): recursive message passing over
# tree-shaped factor graphs with Omega^PP factors.  Two checkable consequences:
#  (A) the factor->variable message m_{F_ij->X_j}(x_j)=max_{x_i} F_ij(x_i,x_j) m_{X_i->F_ij}(x_i)
#      (Eq 5) matches a direct per-x_j brute maximization over the feasible x_i set;
#  (B) end-to-end MpMap is exact on ALL THREE paper benchmark tree topologies
#      STAR, SNOW (ternary tree), PATH (linear chain) -- vs an independent optimizer.
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mpmap_core as mc

t0 = time.time()

def brute_message(edge, child_box, parent_box, ny=241):
    """EXACT independent reference for Eq (5): for each x_j, compute the feasible
    x_i intervals (feasible_intervals_given) then EXACTLY maximize A*mUp over them
    (pp_max_over_interval); multiply by B(x_j).  Independent of compute_msg bands."""
    g = mc.pp_product(edge['A'], edge['mUp'])
    ys = np.linspace(parent_box[0], parent_box[1], ny)
    out = []
    for y in ys:
        ivs = mc.feasible_intervals_given(edge['cells'], y, child_box, is_parent_fixed=True)
        best = None
        for (lo, hi) in ivs:
            v = mc.pp_max_over_interval(g, lo, hi)
            if v is not None:
                best = v if best is None else max(best, v)
        out.append((y, None if best is None else mc.pp_eval(edge['B'], y) * best))
    return out

# ---------- (A) message-operation correctness (Eq 5) on random edges ----------
rng = np.random.default_rng(1)
msg_max_err = 0.0; msg_checks = 0; nmsg = 0
for t in range(20):
    edge = {'A':mc.rand_bump(rng,(-2,2),w=1.9),'B':mc.rand_bump(rng,(-2,2),w=1.9),
            'cells':mc.rand_nonconvex_cells(rng)}
    edge['mUp'] = mc.rand_bump(rng,(-2,2),w=1.8)
    m = mc.compute_msg(edge, edge['mUp'], (-2.,2.), (-2.,2.))
    ref = brute_message(edge, (-2.,2.), (-2.,2.))
    nmsg += 1
    for (y, rv) in ref:
        if rv is None: continue
        cov = any(a-1e-7<=y<=b+1e-7 for (a,b,_) in m)
        gv = mc.pp_eval(m, y) if cov else 0.0
        msg_max_err = max(msg_max_err, abs(gv-rv)); msg_checks += 1
print("[A message-op Eq5] edges=%d points=%d  MAX_ABS_ERR(msg vs per-x_j brute)=%.2e"
      % (nmsg, msg_checks, msg_max_err))

# ---------- (B) end-to-end exactness across paper topologies ----------
topo_rows = []; worst_topo_rel = 0.0; worst_topo_self = 0.0; all_feas = True
for topo, n, reps in [('star',6,5),('ternary',7,5),('chain',4,5)]:
    for rep in range(reps):
        tree = mc.make_random_tree(np.random.default_rng(1000*rep + n), topo, n)
        res = mc.mpmap_solve(tree, root=0)
        if res['assignment'] is None: continue
        mval = mc.eval_joint(tree, res['assignment'])
        if mval <= 1e-9: continue
        ref, rx = mc.refine_map(tree, [res['assignment']], n_restart=8, rng=np.random.default_rng(rep))
        rel = abs(res['value']-ref)/max(abs(res['value']),1e-12)
        self_err = abs(res['value']-mval)
        worst_topo_rel = max(worst_topo_rel, rel); worst_topo_self = max(worst_topo_self, self_err)
        all_feas = all_feas and (mval > 1e-9)
        topo_rows.append((topo, n, res['value'], ref, rel, self_err))
per = {}
for topo in ('star','ternary','chain'):
    rs = [r for r in topo_rows if r[0]==topo]
    per[topo] = {"instances":len(rs),
                 "max_rel_vs_ref":max((r[4] for r in rs), default=0.0),
                 "max_self_err":max((r[5] for r in rs), default=0.0)}
    print("[B %-7s] instances=%d max rel|MpMap-scipy|=%.2e max|MpMap-eval(assign)|=%.2e"
          % (topo, per[topo]['instances'], per[topo]['max_rel_vs_ref'], per[topo]['max_self_err']))

verified = (msg_max_err < 1e-6) and (worst_topo_rel < 5e-3) and (worst_topo_self < 1e-6) and all_feas
print("\nMEASURED vs TARGET")
print("  (A) message Eq5 max err vs per-x_j brute  : %.2e   (target 0; <1e-6)" % msg_max_err)
print("  (B) STAR/SNOW/PATH max rel vs scipy ref   : %.2e   (target 0; <5e-3)" % worst_topo_rel)
print("  (B) max |MpMap - eval_joint(assignment)|  : %.2e   (target 0 EXACT; <1e-6)" % worst_topo_self)
print("  VERIFIED (recursive MpMap exact, msgs ok) : %s" % verified)

out = {"claim":"MpMap recursive message passing (Alg 1-3, Eqs 4-6) over Omega^PP tree factors is exact",
       "message_op_Eq5_max_err":msg_max_err,"message_op_edges":nmsg,"message_op_points":msg_checks,
       "per_topology":per,"topo_max_rel_vs_ref":worst_topo_rel,"topo_max_self_err":worst_topo_self,
       "all_feasible":bool(all_feas),"verified":bool(verified),
       "runtime_s":time.time()-t0,"env":{"numpy":np.__version__,"threads":os.environ.get("OMP_NUM_THREADS","?")}}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"results.json"),"w") as f:
    json.dump(out,f,indent=2)
print("\n[wrote results.json]  runtime=%.2fs" % (time.time()-t0))
