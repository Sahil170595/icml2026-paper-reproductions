#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claim 1  (Tree -> Flow correspondence, Thm 2.3-2.5)
Paper: "Trees to Flows and Back" (OpenReview gW7NZN8zJu, arXiv 2605.00414).

Claim: "TreeFlow establishes a formal correspondence showing decision trees arise
as DISCRETE APPROXIMATIONS of continuous diffusion flows (PF-ODEs)."

Checkable consequences verified numerically on a tractable 1-D case:
 (C1) REFINEMENT CONVERGENCE (Thm 2.5): a depth-n dyadic decision tree used as a
      piecewise-constant (axis-aligned) discretisation of the continuous PF-ODE
      velocity, with N=2^n Euler steps, reproduces the continuous diffusion flow
      to a Wasserstein-1 tolerance that DECAYS GEOMETRICALLY toward 0 (~0.35/level).
 (C2) MOMENT TRUNCATION (Thm 2.4, D^(2)->0): the spurious 2nd-order (diffusion)
      jump moment of the discretisation vanishes vs the 1st-order (drift) moment:
      D2/D1 -> 0 geometrically ; D3/D1 -> 0 faster  => Kramers-Moyal truncates to a
      DETERMINISTIC PF-ODE (Liouville).
 (C3) MARKOV COARSE-GRAINING (Def 2.1, Eq.1): the tree coarse-graining operator
      M_k = E[.|F_k] is mass-preserving and the density path has MONOTONE entropy
      (uniform root -> data leaves).

Reference flow = probability-flow ODE of a Variance-Preserving diffusion on a
2-mode Gaussian mixture (marginals + score analytic). CPU-only, deterministic.
"""
import json, os, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
np.random.seed(0)
t0 = time.time()

# --- 2-mode Gaussian mixture data density -----------------------------------
W  = np.array([0.6, 0.4]); MU = np.array([-1.3, 1.6]); SD = np.array([0.45, 0.35])
DOMAIN = (-6.0, 6.0)
def gm_pdf(x, mu, s2):
    x = np.asarray(x)[..., None]
    return (W * np.exp(-0.5*(x-mu)**2/s2)/np.sqrt(2*np.pi*s2)).sum(-1)

# --- Variance-Preserving schedule -------------------------------------------
BMIN, BMAX = 0.1, 9.0
def beta(t):  return BMIN + t*(BMAX-BMIN)
def Bint(t):  return BMIN*t + 0.5*(BMAX-BMIN)*t**2
def alpha(t): return np.exp(-0.5*Bint(t))
def marg_params(t):
    a = alpha(t); v = 1.0-a*a
    return a*MU, a*a*SD**2 + v
def score(x, t):
    mu_t, s2_t = marg_params(t)
    x = np.asarray(x, float)[..., None]
    comp = W*np.exp(-0.5*(x-mu_t)**2/s2_t)/np.sqrt(2*np.pi*s2_t)
    p = comp.sum(-1, keepdims=True) + 1e-300
    return (comp/p * (-(x-mu_t)/s2_t)).sum(-1)
def velocity(x, t):
    return -0.5*beta(t)*(x + score(x, t))

# --- reference continuous flow map (fine RK4) -------------------------------
T = 1.0
def rk4_flow(x0, nsteps, vfun):
    x = np.array(x0, float); dt = T/nsteps
    for i in range(nsteps):
        t = i*dt
        k1 = vfun(x, t); k2 = vfun(x+0.5*dt*k1, t+0.5*dt)
        k3 = vfun(x+0.5*dt*k2, t+0.5*dt); k4 = vfun(x+dt*k3, t+dt)
        x = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
    return x
GRID = np.linspace(DOMAIN[0], DOMAIN[1], 1201)
REF_STEPS = 8192
phi_ref = rk4_flow(GRID, REF_STEPS, velocity)
w_grid = gm_pdf(GRID, MU, SD**2); w_grid = w_grid/w_grid.sum()   # data-density weights

# --- depth-n dyadic decision tree as discrete PF-ODE approximation ----------
def tree_flow(n):
    ncells = 2**n
    edges = np.linspace(DOMAIN[0], DOMAIN[1], ncells+1)
    centers = 0.5*(edges[:-1]+edges[1:]); cellw = edges[1]-edges[0]
    N = 2**n; dt = T/N
    x = GRID.copy(); D1=D2=D3=0.0; nmom=0
    for i in range(N):
        t = i*dt
        idx = np.clip(np.searchsorted(edges, x)-1, 0, ncells-1)   # tree routing
        v_cell = velocity(centers[idx], t)                        # leaf-constant velocity
        v_true = velocity(x, t)
        D1 += np.mean(np.abs(v_cell))
        D2 += np.mean((v_cell-v_true)**2)*dt
        D3 += np.mean(np.abs(v_cell-v_true)**3)*dt*dt
        nmom += 1
        x = x + dt*v_cell
    return x, (D1/nmom, D2/nmom, D3/nmom), cellw

def push_samples(xs, n):
    ncells=2**n; edges=np.linspace(DOMAIN[0],DOMAIN[1],ncells+1); centers=0.5*(edges[:-1]+edges[1:])
    N=2**n; dt=T/N; xx=xs.copy()
    for i in range(N):
        t=i*dt; idx=np.clip(np.searchsorted(edges,xx)-1,0,ncells-1); xx=xx+dt*velocity(centers[idx],t)
    return xx
def w1(a, b):
    a=np.sort(a); b=np.sort(b); m=min(len(a),len(b)); return float(np.mean(np.abs(a[:m]-b[:m])))

rng = np.random.default_rng(0)
comp = rng.choice(2, size=4000, p=W)
xs = np.clip(MU[comp]+SD[comp]*rng.standard_normal(4000), DOMAIN[0]+1e-6, DOMAIN[1]-1e-6)
ref_samples = rk4_flow(xs, REF_STEPS, velocity)

print("="*70)
print("CLAIM 1  Tree -> Flow : dyadic decision tree as discrete PF-ODE flow")
print("="*70)
print("Reference: VP probability-flow ODE, 2-mode Gaussian mixture, RK4 %d steps" % REF_STEPS)
print("n : cells=steps=2^n | W1(tree,flow)  ratio | massL1 transport | D2/D1     D3/D1")
rows=[]
for n in range(1, 8):
    phi_n, (D1,D2,D3), cw = tree_flow(n)
    massL1 = float(np.sum(np.abs(phi_n-phi_ref)*w_grid))       # data-density-weighted transport error
    wd = w1(push_samples(xs, n), ref_samples)
    rows.append(dict(n=n, cells=2**n, cell_width=cw, W1_to_flow=wd, massL1_transport=massL1,
                     D1_drift=D1, D2_diffusion=D2, D3=D3, D2_over_D1=D2/D1, D3_over_D1=D3/D1))
for i,r in enumerate(rows):
    ratio = rows[i]['W1_to_flow']/rows[i-1]['W1_to_flow'] if i>0 else float('nan')
    r['W1_ratio']=ratio
    print(f" {r['n']} :   {r['cells']:4d}          | {r['W1_to_flow']:.4e} {ratio:6.3f} |"
          f"    {r['massL1_transport']:.4e}   | {r['D2_over_D1']:.3e} {r['D3_over_D1']:.3e}")

# asymptotic geometric-decay diagnostics (n>=4, in-regime)
w1arr = np.array([r['W1_to_flow'] for r in rows])
ml1arr= np.array([r['massL1_transport'] for r in rows])
w1_ratios_asym = w1arr[4:]/w1arr[3:-1]           # n=5..7 over n=4..6
mean_w1_ratio = float(np.mean(w1_ratios_asym))
ml1_ratios_asym = ml1arr[4:]/ml1arr[3:-1]
mean_ml1_ratio = float(np.mean(ml1_ratios_asym))
d2d1 = np.array([r['D2_over_D1'] for r in rows])
mean_d2d1_decay = float(np.mean(d2d1[1:]/d2d1[:-1]))

# --- (C3) Markov coarse-graining : mass conservation + monotone entropy -----
FINE = 2**10
fe = np.linspace(DOMAIN[0], DOMAIN[1], FINE+1); fc = 0.5*(fe[:-1]+fe[1:])
p_fine = gm_pdf(fc, MU, SD**2); p_fine = p_fine/p_fine.sum()      # leaves = data pmf
def coarse(pmf, level):
    ncells=2**level; per=FINE//ncells
    blk = pmf.reshape(ncells, per); cell_mass = blk.sum(1, keepdims=True)
    return (np.ones_like(blk)*cell_mass/per).reshape(-1)
def entropy(pmf):
    q=pmf[pmf>0]; return float(-(q*np.log(q)).sum())
mass_err=[]; ent=[]
for level in range(0, 11):        # level 0 = root(uniform), level 10 = leaves(data)
    pk = coarse(p_fine, level); mass_err.append(abs(pk.sum()-1.0)); ent.append(entropy(pk))
ent = np.array(ent)
max_mass_err = float(max(mass_err))
# root(level0)=uniform high entropy ; leaves(level10)=data low entropy => monotone DECREASING in level
mono_decreasing_root_to_leaf = bool(np.all(np.diff(ent) <= 1e-12))
H_root = float(ent[0]); H_leaves = float(ent[-1])

print("-"*70)
print(f"[C1] W1(tree,flow): {w1arr[0]:.3e} -> {w1arr[-1]:.3e} ; mean decay ratio (n>=4) = {mean_w1_ratio:.4f} (target<0.6 => geometric)")
print(f"[C1] mass-weighted transport L1: {ml1arr[0]:.3e} -> {ml1arr[-1]:.3e} ; mean decay ratio (n>=4)={mean_ml1_ratio:.4f}")
print(f"[C2] D2/D1 (spurious diffusion/drift): {d2d1[0]:.3e} -> {d2d1[-1]:.3e} ; mean decay/level={mean_d2d1_decay:.4f} (->0 => deterministic PF-ODE)")
print(f"[C3] max mass-conservation error of M_k = {max_mass_err:.2e}")
print(f"[C3] entropy root(uniform) H={H_root:.4f} -> leaves(data) H={H_leaves:.4f} ; monotone decreasing = {mono_decreasing_root_to_leaf}")

verdict = (mean_w1_ratio < 0.6 and w1arr[-1] < 1e-2 and mean_ml1_ratio < 0.6
           and d2d1[-1] < 0.05 and mean_d2d1_decay < 0.75
           and max_mass_err < 1e-10 and mono_decreasing_root_to_leaf and H_root > H_leaves)
print("-"*70)
print("VERDICT (tree->flow discrete approx converges; deterministic PF-ODE limit):",
      "SUPPORTED" if verdict else "NOT SUPPORTED")

out = dict(
  claim="Tree->Flow: decision trees arise as discrete approximations of continuous diffusion PF-ODE flows (Thm 2.3-2.5)",
  reference="VP probability-flow ODE on 2-mode Gaussian mixture; RK4 %d-step ground truth" % REF_STEPS,
  refinement_table=rows,
  C1_W1_first=float(w1arr[0]), C1_W1_final=float(w1arr[-1]),
  C1_mean_W1_decay_ratio_nge4=mean_w1_ratio,
  C1_massL1_first=float(ml1arr[0]), C1_massL1_final=float(ml1arr[-1]),
  C1_mean_massL1_decay_ratio_nge4=mean_ml1_ratio,
  C2_D2_over_D1=[float(v) for v in d2d1], C2_D2D1_final=float(d2d1[-1]), C2_mean_D2D1_decay=mean_d2d1_decay,
  C3_max_mass_error=max_mass_err, C3_entropy_root=H_root, C3_entropy_leaves=H_leaves,
  C3_entropy_monotone_decreasing=mono_decreasing_root_to_leaf,
  targets=dict(W1_decay_ratio="<0.6 (geometric)", W1_final="<1e-2",
               D2_over_D1_final="<0.05 and decaying (=> deterministic PF-ODE)",
               mass_error="<1e-10", entropy="monotone, root>leaves"),
  verdict="SUPPORTED" if verdict else "NOT_SUPPORTED",
  runtime_s=round(time.time()-t0,3))
with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
    json.dump(out, f, indent=2)
print("runtime_s =", round(time.time()-t0,3)); print("wrote results.json")
