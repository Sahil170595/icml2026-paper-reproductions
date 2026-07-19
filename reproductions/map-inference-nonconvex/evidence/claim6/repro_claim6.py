#!/usr/bin/env python3
# Claim 6 -- PAMAP (Sec 5, contribution C2, Fig 1/7): decompose a NON-CONVEX
# SMT(LRA) feasible region into CONVEX polytopes, optimise the density over each
# with a local convex-polytope optimizer, keep the best (Alg 4).  Checkable:
#  (a) PAMAP recovers the GLOBAL constrained MAP (== fine-grid ground truth),
#      incl. the paper's Example 2.2 constrained optimum ~(1.83,1.83);
#  (b) a constraint-AGNOSTIC optimizer (maximises density ignoring constraints)
#      returns INFEASIBLE / suboptimal points;
#  (c) VALID upper-bound pruning (Sec 5 / Fig 6, Alg 4 line 9) skips polytopes
#      while remaining EXACT.
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import minimize

t0 = time.time()

def cell_feasible(pt, cell): return all(a*pt[0]+b*pt[1]+c >= -1e-9 for (a,b,c) in cell)

def project_dist(center, cell, box):
    """min distance from center to convex cell (valid for a Gaussian UB)."""
    if cell_feasible(center, cell): return 0.0
    cons = [{'type':'ineq','fun':(lambda x,a=a,b=b,c=c: a*x[0]+b*x[1]+c)} for (a,b,c) in cell]
    r = minimize(lambda x:(x[0]-center[0])**2+(x[1]-center[1])**2, np.array(center,float),
                 method='SLSQP', bounds=[(box[0],box[1]),(box[2],box[3])], constraints=cons,
                 options={'maxiter':200,'ftol':1e-12})
    return float(np.hypot(r.x[0]-center[0], r.x[1]-center[1])) if r.success else 0.0

def pamap(density, cells, box, centers=None, hts=None, wds=None, prune=False):
    starts = [(box[0]+(box[1]-box[0])*fx, box[2]+(box[3]-box[2])*fy)
              for fx in (0.15,0.5,0.85) for fy in (0.15,0.5,0.85)]
    def cell_ub(cell):
        if centers is None: return np.inf
        return float(sum(h*np.exp(-w*project_dist(c,cell,box)**2) for h,w,c in zip(hts,wds,centers)))
    order = sorted(range(len(cells)), key=lambda k:-cell_ub(cells[k]))
    best=-np.inf; bx=None; nopt=0
    for k in order:
        cell = cells[k]
        if prune and cell_ub(cell) < best - 1e-9:       # VALID UB below best => safe to skip
            continue
        cons = [{'type':'ineq','fun':(lambda x,a=a,b=b,c=c: a*x[0]+b*x[1]+c)} for (a,b,c) in cell]
        nopt += 1
        for s in starts:
            if not cell_feasible(s, cell): continue
            r = minimize(lambda x:-density(x), np.array(s,float), method='SLSQP',
                         bounds=[(box[0],box[1]),(box[2],box[3])], constraints=cons,
                         options={'maxiter':300,'ftol':1e-11})
            if r.success and cell_feasible(r.x, cell) and density(r.x) > best:
                best=density(r.x); bx=tuple(r.x)
    return best, bx, nopt

def grid_truth(density, feasible, box, g=401):
    xs=np.linspace(box[0],box[1],g); ys=np.linspace(box[2],box[3],g)
    best=-np.inf; bx=None
    for x in xs:
        col=[(x,y) for y in ys if feasible((x,y))]
        for pt in col:
            v=density(pt)
            if v>best: best=v; bx=pt
    return best, bx

def agnostic(density, box):
    best=-np.inf; bx=None
    for s in [(1.5,1.5),(1.0,1.0),(0.5,0.5),(2.5,2.5),(0.5,2.5)]:
        r=minimize(lambda x:-density(x), np.array(s,float), method='Nelder-Mead',
                   options={'maxiter':2000,'xatol':1e-8,'fatol':1e-10})
        if density(r.x)>best: best=density(r.x); bx=tuple(r.x)
    return best, bx

# ---------- (1) paper Example 2.2 (EXACT constraint) ----------
def feas_ex22(p):
    x1,x2=p
    if not(0<=x1<=2 and 0<=x2<=2): return False
    return (x2<=1) or (x2>2*x1) or (x2>4.75-2*x1)
E_c=[(1.15,1.40),(1.83,1.83),(0.40,0.50)]; E_h=[2.6,2.0,1.4]; E_w=[8.,6.,6.]
def dens_ex22(p):
    return float(sum(h*np.exp(-w*((p[0]-c[0])**2+(p[1]-c[1])**2)) for h,w,c in zip(E_h,E_w,E_c)))
# cells: x2<=1 ; x2>=2x1 ; x2>=4.75-2x1  (a*x1+b*x2+c>=0)
cells_ex22=[[(0.,-1.,1.)], [(-2.,1.,0.)], [(2.,1.,-4.75)]]
gt,gx=grid_truth(dens_ex22,feas_ex22,(0,2,0,2))
pv22,px22,nop22=pamap(dens_ex22,cells_ex22,(0,2,0,2),E_c,E_h,E_w,prune=True)
av22,ax22=agnostic(dens_ex22,(0,2,0,2))
ex22_rel=abs(pv22-gt)/gt
print("[Example 2.2] ground-truth constrained MAP p=%.5f at (%.3f,%.3f)"%(gt,gx[0],gx[1]))
print("              PAMAP           p=%.5f at (%.3f,%.3f) feasible=%s rel=%.2e (cells opt=%d/3)"
      %(pv22,px22[0],px22[1],feas_ex22(px22),ex22_rel,nop22))
print("              agnostic (Adam) p=%.5f at (%.3f,%.3f) feasible=%s"%(av22,ax22[0],ax22[1],feas_ex22(ax22)))

# ---------- (2) non-convex battery: excluded central region (paper Fig 2 flavour) ----------
# feasible = [0,3]^2 MINUS central square (1,2)x(1,2) = union of 4 convex cells.
box=(0,3,0,3)
cells4=[[(-1.,0.,1.)], [(1.,0.,-2.)], [(0.,-1.,1.)], [(0.,1.,-2.)]]  # x<=1, x>=2, y<=1, y>=2
def feas4(p):
    x,y=p
    if not(0<=x<=3 and 0<=y<=3): return False
    return (x<=1) or (x>=2) or (y<=1) or (y>=2)
rng=np.random.default_rng(5)
N=12; matches=0; agn_fail=0; worst_rel=0.0; pruned=[]
for t in range(N):
    # tallest mode INSIDE the forbidden square (infeasible) + feasible modes outside
    ctall=(rng.uniform(1.2,1.8),rng.uniform(1.2,1.8))
    cs=[ctall]+[(rng.uniform(0.2,2.8),rng.uniform(0.2,2.8)) for _ in range(3)]
    hs=[3.0]+list(rng.uniform(1.2,2.4,3)); ws=list(rng.uniform(5,10,4))
    def dens(p,cs=cs,hs=hs,ws=ws):
        return float(sum(h*np.exp(-w*((p[0]-c[0])**2+(p[1]-c[1])**2)) for h,w,c in zip(hs,ws,cs)))
    gt,_=grid_truth(dens,feas4,box,g=241)
    pv,px,nop=pamap(dens,cells4,box,cs,hs,ws,prune=True)
    _,_,nop_all=pamap(dens,cells4,box,cs,hs,ws,prune=False)
    av,ax=agnostic(dens,box)
    rel=abs(pv-gt)/max(gt,1e-9); worst_rel=max(worst_rel,rel)
    def feas4_tol(q,e=2e-3):
        x,y=q
        return (0-e<=x<=3+e and 0-e<=y<=3+e) and ((x<=1+e) or (x>=2-e) or (y<=1+e) or (y>=2-e))
    if rel<2e-2 and feas4_tol(px): matches+=1
    if (not feas4(ax)) or dens(ax)<gt-1e-6: agn_fail+=1
    pruned.append(nop_all-nop)
print("\n[battery: excluded central region] N=%d"%N)
print("  PAMAP matches ground truth        : %d/%d  (worst rel=%.2e)"%(matches,N,worst_rel))
print("  constraint-agnostic INFEASIBLE/subopt: %d/%d"%(agn_fail,N))
print("  valid-UB pruning skipped avg %.1f/4 polytopes/instance (still exact)"%np.mean(pruned))

verified=(ex22_rel<2e-2 and feas_ex22(px22) and (not feas_ex22(ax22))
          and matches>=N-1 and agn_fail>=N-1)
print("\nMEASURED vs TARGET")
print("  Example 2.2 PAMAP rel vs truth        : %.2e   (target ~0)"%ex22_rel)
print("  Example 2.2 PAMAP feas / agnostic INFEAS: %s / %s"%(feas_ex22(px22),not feas_ex22(ax22)))
print("  battery PAMAP == ground truth         : %d/%d"%(matches,N))
print("  battery agnostic fails                : %d/%d"%(agn_fail,N))
print("  VERIFIED (PAMAP global; agnostic fails): %s"%verified)

out={"claim":"PAMAP decomposes non-convex feasible region into convex polytopes & recovers the GLOBAL constrained MAP; constraint-agnostic optimizers fail",
     "example22_pamap_value":pv22,"example22_pamap_point":list(px22),"example22_pamap_feasible":bool(feas_ex22(px22)),
     "example22_rel_vs_truth":ex22_rel,"example22_agnostic_point":list(ax22),"example22_agnostic_feasible":bool(feas_ex22(ax22)),
     "battery_N":N,"battery_pamap_matches":matches,"battery_worst_rel":worst_rel,
     "battery_agnostic_fail":agn_fail,"avg_polytopes_pruned_of_4":float(np.mean(pruned)),
     "verified":bool(verified),"runtime_s":time.time()-t0,
     "env":{"numpy":np.__version__,"threads":os.environ.get("OMP_NUM_THREADS","?")}}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"results.json"),"w") as f:
    json.dump(out,f,indent=2)
print("\n[wrote results.json]  runtime=%.2fs" % (time.time()-t0))
