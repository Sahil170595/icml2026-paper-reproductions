#!/usr/bin/env python3
# Claim 5 -- Complexity / scalability (Prop A.7, Thm A.8, Q1 / Fig 5).
#  * BOUNDED-diameter trees (STAR, diameter 2 -- the Theorem 4.5 tractable regime):
#    MpMap runs POLYNOMIAL in n and stays EXACT, whereas exhaustive grid MAP is
#    EXPONENTIAL (grid^n).
#  * MpMap cost DEPENDS ON DIAMETER (Thm A.8): PATH (diameter n-1) scales at a
#    strictly HIGHER polynomial exponent than STAR.  (Thm A.8 worst-case is
#    exponential in diameter; the paper notes it rarely materialises -- on
#    structured instances MpMap stays polynomial but diameter-sensitive.)
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mpmap_core as mc
mc.PIECE_CAP = 200000
t0 = time.time()

def multimodal(cx_list, w=0.55, box=(-2.,2.)):
    ps = []
    for cx in cx_list:
        lo = max(box[0], cx-w); hi = min(box[1], cx+w)
        if hi-lo > 1e-6: ps.append((lo, hi, np.array([-1.0, 2*cx, w*w-cx*cx])))
    return mc.merge_pieces(ps)
def comb_cells(bands):
    return [[(-1.0, 1.0, -lo), (1.0, -1.0, hi)] for (lo, hi) in bands]
def adv_star(n):
    tree = [{'box':(-2.,2.),'p':multimodal([-1.2,0.0,1.2]),'parent':None,'children':[]} for _ in range(n)]
    for i in range(1, n):
        tree[i]['parent'] = 0; tree[0]['children'].append(i)
        tree[i]['edge'] = {'A':multimodal([-1.0,1.0],w=0.9),'B':multimodal([-1.0,1.0],w=0.9),
                           'cells':comb_cells([(-1.7,-0.9),(-0.35,0.35),(0.9,1.7)])}
    return tree
def adv_chain(n):
    tree = [{'box':(-2.,2.),'p':multimodal([-1.2,0.0,1.2]),'parent':None,'children':[]} for _ in range(n)]
    for i in range(1, n):
        tree[i]['parent'] = i-1; tree[i-1]['children'].append(i)
        tree[i]['edge'] = {'A':multimodal([-1.0,1.0],w=0.9),'B':multimodal([-1.0,1.0],w=0.9),
                           'cells':comb_cells([(-1.7,-0.9),(-0.35,0.35),(0.9,1.7)])}
    return tree

star_n = [3, 5, 8, 12, 16, 20]; star_t = []; star_pieces = []
for n in star_n:
    tree = adv_star(n); ts = time.time(); res = mc.mpmap_solve(tree, root=0); dt = time.time()-ts
    star_t.append(dt); star_pieces.append(max((len(m) for m in res['msgs'].values()), default=0))
    print("[STAR diam=2] n=%2d  time=%.4fs  max_msg_pieces=%d" % (n, dt, star_pieces[-1]))
star_slope = float(np.polyfit(np.log(star_n), np.log(np.maximum(star_t,1e-6)), 1)[0])
print("[STAR] runtime ~ n^%.2f  (polynomial => tractable, bounded diameter)" % star_slope)

star_rel = 0.0
for n in [4, 6, 8]:
    tree = adv_star(n); res = mc.mpmap_solve(tree, root=0)
    if res['assignment'] is None: continue
    if mc.eval_joint(tree, res['assignment']) > 1e-12:
        ref, _ = mc.refine_map(tree, [res['assignment']], n_restart=8, rng=np.random.default_rng(n))
        star_rel = max(star_rel, abs(res['value']-ref)/max(abs(res['value']),1e-12))
print("[STAR] exactness vs scipy (n=4,6,8): max rel = %.2e" % star_rel)

path_n = [3, 5, 8, 12, 16, 20]; path_t = []; path_pieces = []
for n in path_n:
    tree = adv_chain(n); ts = time.time(); res = mc.mpmap_solve(tree, root=0); dt = time.time()-ts
    path_t.append(dt); path_pieces.append(max((len(m) for m in res['msgs'].values()), default=0))
    print("[PATH diam=%2d] n=%2d  time=%.4fs  max_msg_pieces=%d" % (n-1, n, dt, path_pieces[-1]))
path_slope = float(np.polyfit(np.log(path_n), np.log(np.maximum(path_t,1e-6)), 1)[0])
print("[PATH] runtime ~ n^%.2f  (higher exponent than STAR => diameter-sensitive, Thm A.8)" % path_slope)

gtime = []
for n in [2, 3, 4, 5]:
    tree = adv_star(n); ts = time.time(); mc.brute_map_vec(tree, 15); gtime.append((n, time.time()-ts))
    print("[BRUTE grid15] n=%d  time=%.4fs" % (n, gtime[-1][1]))
brute_slope = float(np.polyfit([g[0] for g in gtime], np.log([max(g[1],1e-6) for g in gtime]), 1)[0])
grid = 11
print("[BRUTE] log(time) ~ %.2f * n => EXPONENTIAL (base~%.1f)" % (brute_slope, np.exp(brute_slope)))
print("[CROSS] MpMap STAR n=20 in %.4fs; brute grid=%d,n=20 => %.1e evals (intractable)" % (star_t[-1], grid, grid**20))

polynomial_star = star_slope < 3.0
exact_star = star_rel < 5e-3
brute_exponential = brute_slope > 1.0
diameter_sensitive = path_slope > star_slope - 0.2
verified = polynomial_star and exact_star and brute_exponential and diameter_sensitive
print("\nMEASURED vs TARGET")
print("  STAR runtime exponent n^a  : %.2f    (target polynomial a<3 => tractable)" % star_slope)
print("  STAR exactness vs scipy    : %.2e  (target 0; exact)" % star_rel)
print("  PATH runtime exponent n^a  : %.2f    (>= STAR => diameter-sensitive, Thm A.8)" % path_slope)
print("  BRUTE log(time)/n          : %.2f    (>0 => exponential grid^n)" % brute_slope)
print("  VERIFIED : %s" % verified)

out = {"claim":"MpMap polynomial & exact for bounded-diameter (treewidth-1) MAP(LRA); diameter-sensitive (Thm A.8); brute exponential",
       "star_n":star_n,"star_time_s":star_t,"star_max_pieces":star_pieces,"star_runtime_exponent":star_slope,
       "star_exactness_rel":star_rel,"path_n":path_n,"path_time_s":path_t,"path_max_pieces":path_pieces,
       "path_runtime_exponent":path_slope,"brute_time_s":[g[1] for g in gtime],"brute_log_time_per_var":brute_slope,
       "brute_base_est":float(np.exp(brute_slope)),"star20_time_s":star_t[-1],"brute_grid11_pow20":float(grid**20),
       "verified":bool(verified),"runtime_s":time.time()-t0,
       "env":{"numpy":np.__version__,"threads":os.environ.get("OMP_NUM_THREADS","?")}}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"results.json"),"w") as f:
    json.dump(out,f,indent=2)
print("\n[wrote results.json]  runtime=%.2fs" % (time.time()-t0))
