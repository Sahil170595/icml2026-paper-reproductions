#!/usr/bin/env python3
# Claim 3 -- Correctness of the piecewise-polynomial (Omega^PP) message operations.
#  (i)   Theorem A.5  (max-outPP correct): for piecewise-poly q and affine l,u,
#        max_out(q,l,u)(y) == sup_{x in [l(y),u(y)]} q(x)  (exact per-interval ref).
#  (ii)  product closure (TMC-i): pp_product(f,g)(x) == f(x)*g(x) exactly.
#  (iii) pointwise max (TMC-iii): pp_pointmax(f,g)(x) == max(f(x),g(x)) exactly.
#  (iv)  Prop A.6 piece bound: #pieces(max-out on deg-q, m-piece poly) <= 8mq+4m+4.
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mpmap_core as mc

t0 = time.time(); rng = np.random.default_rng(0)

def rand_pp(rng, npc, degmax, lo=-2.0, hi=2.0):
    bnds = np.concatenate([[lo], np.sort(rng.uniform(lo, hi, npc-1)), [hi]])
    return [(float(bnds[i]), float(bnds[i+1]),
             rng.uniform(-1.5, 1.5, int(rng.integers(0, degmax+1))+1)) for i in range(npc)]

# ---- (i) Theorem A.5: max-outPP correctness ----
mo_err = 0.0; mo_cmp = 0; ninst = 0
for t in range(300):
    npc = int(rng.integers(1, 6)); q = rand_pp(rng, npc, 3)
    la, lb = rng.uniform(-1.5,1.5), rng.uniform(-2,2)
    ua, ub = rng.uniform(-1.5,1.5), rng.uniform(-2,2)
    m = mc.max_out(q, la, lb, ua, ub, -1.6, 1.6); ninst += 1
    for y in np.linspace(-1.6, 1.6, 33):
        ref = mc.pp_max_over_interval(q, la*y+lb, ua*y+ub)
        if ref is None: continue
        cov = any(a-1e-7<=y<=b+1e-7 for (a,b,_) in m)
        got = mc.pp_eval(m, y) if cov else None
        mo_err = max(mo_err, abs(ref) if got is None else abs(got-ref)); mo_cmp += 1
print("[i  Thm A.5 max-outPP] instances=%d points=%d  MAX_ABS_ERR=%.2e" % (ninst, mo_cmp, mo_err))

# ---- (ii) product closure (TMC-i) ----
prod_err = 0.0; pn = 0
for t in range(200):
    f = rand_pp(rng, int(rng.integers(1,5)), 3); g = rand_pp(rng, int(rng.integers(1,5)), 3)
    fg = mc.pp_product(f, g); pn += 1
    for x in np.linspace(-1.9, 1.9, 40):
        ref = mc.pp_eval(f, x) * mc.pp_eval(g, x)
        prod_err = max(prod_err, abs(mc.pp_eval(fg, x) - ref))
print("[ii  product closure ] pairs=%d  MAX_ABS_ERR(prod vs f*g)=%.2e" % (pn, prod_err))

# ---- (iii) pointwise max (TMC-iii) ----
pmax_err = 0.0; mn = 0
for t in range(200):
    f = rand_pp(rng, int(rng.integers(1,5)), 3); g = rand_pp(rng, int(rng.integers(1,5)), 3)
    mx = mc.pp_pointmax(f, g); mn += 1
    for x in np.linspace(-1.9, 1.9, 60):
        # pp_pointmax uses 0 outside domains -> compare to max over present pieces
        fv = mc.pp_eval(f, x); gv = mc.pp_eval(g, x)
        covf = any(a-1e-7<=x<=b+1e-7 for (a,b,_) in f)
        covg = any(a-1e-7<=x<=b+1e-7 for (a,b,_) in g)
        if not covf and not covg: continue
        ref = max(fv if covf else -np.inf, gv if covg else -np.inf)
        pmax_err = max(pmax_err, abs(mc.pp_eval(mx, x) - ref))
print("[iii pointwise max   ] pairs=%d  MAX_ABS_ERR(pmax vs max(f,g))=%.2e" % (mn, pmax_err))

# ---- (iv) Prop A.6 piece bound: #pieces(max-out) <= 8mq+4m+4 ----
worst_ratio = 0.0; nb = 0; max_pieces = 0; bound_holds = True
for t in range(200):
    m_pieces = int(rng.integers(1,6)); qdeg = int(rng.integers(1,4))
    q = rand_pp(rng, m_pieces, qdeg)
    la, lb = rng.uniform(-1.2,1.2), rng.uniform(-1.5,1.5)
    ua, ub = rng.uniform(-1.2,1.2), rng.uniform(-1.5,1.5)
    res = mc.max_out(q, la, lb, ua, ub, -1.6, 1.6)
    bound = 8*m_pieces*qdeg + 4*m_pieces + 4
    npieces = len(res); max_pieces = max(max_pieces, npieces); nb += 1
    worst_ratio = max(worst_ratio, npieces / bound)
    if npieces > bound: bound_holds = False
print("[iv Prop A.6 bound   ] tested=%d  max observed pieces=%d  worst(pieces/bound)=%.3f  bound_8mq+4m+4_holds=%s"
      % (nb, max_pieces, worst_ratio, bound_holds))

verified = (mo_err < 1e-6) and (prod_err < 1e-9) and (pmax_err < 1e-6) and bound_holds
print("\nMEASURED vs TARGET")
print("  (i)   max-outPP correctness (Thm A.5) err : %.2e   (target 0; <1e-6)" % mo_err)
print("  (ii)  product closure err                 : %.2e   (target 0; <1e-9)" % prod_err)
print("  (iii) pointwise-max err                   : %.2e   (target 0; <1e-6)" % pmax_err)
print("  (iv)  Prop A.6 piece bound 8mq+4m+4 holds : %s  (worst ratio %.3f)" % (bound_holds, worst_ratio))
print("  VERIFIED (Omega^PP message ops correct)   : %s" % verified)

out = {"claim":"Correctness of Omega^PP piecewise-polynomial message operations (Thm A.5, TMC ops, Prop A.6)",
       "maxout_ThmA5_err":mo_err,"maxout_points":mo_cmp,"product_err":prod_err,"pointmax_err":pmax_err,
       "propA6_bound_holds":bool(bound_holds),"propA6_worst_ratio":worst_ratio,"propA6_max_pieces":max_pieces,
       "verified":bool(verified),"runtime_s":time.time()-t0,
       "env":{"numpy":np.__version__,"threads":os.environ.get("OMP_NUM_THREADS","?")}}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"results.json"),"w") as f:
    json.dump(out,f,indent=2)
print("\n[wrote results.json]  runtime=%.2fs" % (time.time()-t0))
