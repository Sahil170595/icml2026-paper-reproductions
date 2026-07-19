"""
mpmap_core.py -- Independent NumPy implementation of the MpMap constrained-MAP
message-passing algorithm and its piecewise-polynomial (Omega^PP / Omega^PEP)
operations, from Kurscheidt, Masina, Sebastiani & Vergari, "The Theory and
Practice of MAP Inference over Non-Convex Constraints"
(arXiv 2602.08681 / OpenReview jIZqAemuqk).  CPU-only, deterministic.

Univariate piecewise polynomials (PP) are lists of pieces (lo, hi, coeffs),
coeffs highest-degree-first (np.polyval); value is 0 outside all listed pieces
(an infeasible / zero-density region).  Densities are non-negative.
"""
import numpy as np

TOL = 1e-9
RTOL = 1e-7
PIECE_CAP = 60000        # guard: abort a message that blows up (Thm A.8 diameter blowup)

class PieceBlowup(Exception):
    pass

# ----------------------------- polynomial helpers ---------------------------
def poly_compose_affine(coeffs, a, b):
    """coeffs of p(a*y+b) given p(x) with coeffs highest-first (Horner)."""
    res = np.array([0.0])
    for c in np.atleast_1d(coeffs).astype(float):
        res = np.polymul(res, np.array([a, b]))
        res = np.polyadd(res, np.array([c]))
    return np.atleast_1d(res).astype(float)

def roots_in(coeffs, lo, hi, tol=RTOL):
    coeffs = np.atleast_1d(coeffs).astype(float)
    nz = np.nonzero(np.abs(coeffs) > 1e-14)[0]
    if len(nz) == 0:
        return []
    coeffs = coeffs[nz[0]:]
    if len(coeffs) <= 1:
        return []
    try:
        r = np.roots(coeffs)
    except Exception:
        return []
    out = []
    for z in r:
        if abs(z.imag) < 1e-6:
            x = float(z.real)
            if lo - tol <= x <= hi + tol:
                out.append(min(max(x, lo), hi))
    return sorted(out)

def piece_at(PP, lo, hi):
    mid = 0.5 * (lo + hi)
    for (a, b, c) in PP:
        if a - 1e-9 <= mid <= b + 1e-9:
            return c
    return None

def pp_eval(PP, x):
    for (a, b, c) in PP:
        if a - 1e-9 <= x <= b + 1e-9:
            return float(np.polyval(c, x))
    return 0.0

def pp_eval_vec(PP, X):
    X = np.asarray(X, float); out = np.zeros_like(X)
    for (a, b, c) in PP:
        m = (X >= a - 1e-7) & (X <= b + 1e-7)
        if m.any():
            out[m] = np.polyval(np.atleast_1d(c).astype(float), X[m])
    return out

def merge_pieces(PP, tol=1e-7):
    P = [(a, b, np.atleast_1d(c).astype(float)) for (a, b, c) in PP if b - a > 1e-11]
    P.sort(key=lambda t: t[0])
    out = []
    for (a, b, c) in P:
        if out:
            pa, pb, pc = out[-1]
            if abs(pb - a) < 1e-9 and len(pc) == len(c) and np.allclose(pc, c, atol=tol, rtol=0):
                out[-1] = (pa, b, pc); continue
        out.append((a, b, c))
    return out

def pp_pointmax(A, B):
    pts = sorted(set([p for (lo, hi, _) in list(A) + list(B) for p in (lo, hi)]))
    out = []
    for k in range(len(pts) - 1):
        lo, hi = pts[k], pts[k + 1]
        if hi - lo < 1e-12: continue
        ca = piece_at(A, lo, hi); cb = piece_at(B, lo, hi)
        if ca is None and cb is None: continue
        if ca is None: out.append((lo, hi, cb)); continue
        if cb is None: out.append((lo, hi, ca)); continue
        rts = roots_in(np.polysub(ca, cb), lo, hi)
        cut = [lo] + rts + [hi]
        for j in range(len(cut) - 1):
            aa, bb = cut[j], cut[j + 1]
            if bb - aa < 1e-12: continue
            mid = 0.5 * (aa + bb)
            out.append((aa, bb, ca if np.polyval(ca, mid) >= np.polyval(cb, mid) else cb))
    return merge_pieces(out)

def pp_product(A, B):
    pts = sorted(set([p for (lo, hi, _) in list(A) + list(B) for p in (lo, hi)]))
    out = []
    for k in range(len(pts) - 1):
        lo, hi = pts[k], pts[k + 1]
        if hi - lo < 1e-12: continue
        ca = piece_at(A, lo, hi); cb = piece_at(B, lo, hi)
        if ca is None or cb is None: continue
        out.append((lo, hi, np.polymul(ca, cb)))
    return merge_pieces(out)

def pp_breakpoints(PP):
    bps = set()
    for (a, b, _) in PP:
        bps.add(a); bps.add(b)
    return sorted(bps)

def pp_stationary(PP):
    xs = []
    for (a, b, c) in PP:
        xs += roots_in(np.polyder(np.atleast_1d(c).astype(float)), a, b)
    return xs

def gval_sup(PP, x):
    v = -np.inf
    for (a, b, c) in PP:
        if a - 1e-7 <= x <= b + 1e-7:
            v = max(v, float(np.polyval(c, x)))
    return v if v > -np.inf else 0.0

# ----------------------------- max-out (Thm A.5) ----------------------------
def max_out(g, la, lb, ua, ub, y_lo, y_hi):
    """Symbolic max-outPP: PP in y equal to sup_{x in [l(y),u(y)] cap dom(g)} g(x),
    with l(y)=la*y+lb, u(y)=ua*y+ub. g univariate PP."""
    if not g:
        return []
    gdlo = min(p[0] for p in g); gdhi = max(p[1] for p in g)
    xcand = set(pp_breakpoints(g)) | set(pp_stationary(g)) | {gdlo, gdhi}
    yb = {y_lo, y_hi}
    for xc in xcand:
        if abs(la) > 1e-12: yb.add((xc - lb) / la)
        if abs(ua) > 1e-12: yb.add((xc - ub) / ua)
    if abs(la - ua) > 1e-12: yb.add((ub - lb) / (la - ua))
    yb = sorted(set(min(max(y, y_lo), y_hi) for y in yb if y_lo - 1e-9 <= y <= y_hi + 1e-9))
    out = []
    for k in range(len(yb) - 1):
        yc0, yc1 = yb[k], yb[k + 1]
        if yc1 - yc0 < 1e-11: continue
        ym = 0.5 * (yc0 + yc1)
        L = la * ym + lb; U = ua * ym + ub
        if L >= gdlo - 1e-12: Le_a, Le_b, Lval = la, lb, L
        else:                 Le_a, Le_b, Lval = 0.0, gdlo, gdlo
        if U <= gdhi + 1e-12: Ue_a, Ue_b, Uval = ua, ub, U
        else:                 Ue_a, Ue_b, Uval = 0.0, gdhi, gdhi
        if Lval > Uval + 1e-9: continue
        cands = []
        pieceL = piece_at(g, Lval - 1e-9, Lval + 1e-9)
        if pieceL is not None:
            cands.append([(yc0, yc1, poly_compose_affine(pieceL, Le_a, Le_b))])
        pieceU = piece_at(g, Uval - 1e-9, Uval + 1e-9)
        if pieceU is not None:
            cands.append([(yc0, yc1, poly_compose_affine(pieceU, Ue_a, Ue_b))])
        for xc in xcand:
            if Lval - 1e-9 <= xc <= Uval + 1e-9:
                cands.append([(yc0, yc1, np.array([gval_sup(g, xc)]))])
        if not cands: continue
        m_cell = cands[0]
        for cc in cands[1:]:
            m_cell = pp_pointmax(m_cell, cc)
        out += m_cell
        if len(out) > PIECE_CAP:
            raise PieceBlowup("max_out out pieces %d" % len(out))
    return merge_pieces(out)

def pp_max_over_interval(g, L, U):
    """Independent exact reference: exact max of PP g over fixed numeric [L,U]."""
    gdlo = min(p[0] for p in g); gdhi = max(p[1] for p in g)
    L = max(L, gdlo); U = min(U, gdhi)
    if L > U + 1e-12: return None
    best = -np.inf
    for (a, b, c) in g:
        lo = max(a, L); hi = min(b, U)
        if lo > hi + 1e-12: continue
        for x in [lo, hi] + roots_in(np.polyder(np.atleast_1d(c).astype(float)), lo, hi):
            best = max(best, float(np.polyval(c, x)))
    return best if best > -np.inf else None

def pp_argmax_full(PP):
    best = -np.inf; bx = None
    for (a, b, c) in PP:
        for x in [a, b] + roots_in(np.polyder(np.atleast_1d(c).astype(float)), a, b):
            v = float(np.polyval(c, x))
            if v > best: best = v; bx = x
    return best, bx

# ============================ MpMap tree solver =============================
def feasible_intervals_given(cells, xfix, box_self, is_parent_fixed=True):
    lo0, hi0 = box_self; out = []
    for cell in cells:
        lo, hi = lo0, hi0; ok = True
        for (ac, ap, c) in cell:
            a_self = ac if is_parent_fixed else ap
            a_other = ap if is_parent_fixed else ac
            rhs = a_other * xfix + c
            if abs(a_self) < 1e-12:
                if rhs < -1e-9: ok = False; break
            elif a_self > 0: lo = max(lo, -rhs / a_self)
            else:            hi = min(hi, -rhs / a_self)
        if ok and lo <= hi + 1e-12:
            out.append((lo, hi))
    return out

def compute_msg(edge, mUp_child, child_box, parent_box):
    """Factor->parent message (Eq 5): m(x_par)=max_{x_ch feasible} F(x_ch,x_par)*mUp(x_ch)."""
    A = edge['A']; B = edge['B']; cells = edge['cells']
    g = pp_product(A, mUp_child)
    if not g: return []
    contribs = []
    for cell in cells:
        lowers = []; uppers = []; ygates = []
        for (ac, ap, c) in cell:
            if abs(ac) < 1e-12: ygates.append((ap, c))
            elif ac > 0: lowers.append((-ap / ac, -c / ac))
            else:        uppers.append((-ap / ac, -c / ac))
        lowers.append((0.0, child_box[0])); uppers.append((0.0, child_box[1]))
        yranges = [tuple(parent_box)]
        for (ap, c) in ygates:
            nr = []
            for (a0, b0) in yranges:
                if abs(ap) < 1e-12:
                    if c >= -1e-9: nr.append((a0, b0))
                elif ap > 0:
                    lo = max(a0, -c / ap)
                    if lo < b0 + 1e-12: nr.append((lo, b0))
                else:
                    hi = min(b0, -c / ap)
                    if a0 < hi + 1e-12: nr.append((a0, hi))
            yranges = nr
        if not yranges: continue
        ycrit = set(); allb = lowers + uppers
        for i1 in range(len(allb)):
            for i2 in range(i1 + 1, len(allb)):
                a1, b1 = allb[i1]; a2, b2 = allb[i2]
                if abs(a1 - a2) > 1e-12: ycrit.add((b2 - b1) / (a1 - a2))
        for (a0, b0) in yranges:
            pts = sorted(set([a0, b0] + [y for y in ycrit if a0 - 1e-9 <= y <= b0 + 1e-9]))
            for k in range(len(pts) - 1):
                s0, s1 = pts[k], pts[k + 1]
                if s1 - s0 < 1e-11: continue
                ym = 0.5 * (s0 + s1)
                la_, lb_ = max(lowers, key=lambda ab: ab[0] * ym + ab[1])
                ua_, ub_ = min(uppers, key=lambda ab: ab[0] * ym + ab[1])
                if (la_ * ym + lb_) > (ua_ * ym + ub_) + 1e-12: continue
                mo = max_out(g, la_, lb_, ua_, ub_, s0, s1)
                if mo:
                    contribs.append(mo)
                    if len(mo) > PIECE_CAP: raise PieceBlowup("mo %d" % len(mo))
    if not contribs: return []
    m = contribs[0]
    for cc in contribs[1:]:
        m = pp_pointmax(m, cc)
        if len(m) > PIECE_CAP: raise PieceBlowup("pointmax %d" % len(m))
    return pp_product(B, m)

def mpmap_solve(tree, root=0):
    """MpMap upward pass (Eqs 4-6) + argmax backtracking on a tree of Omega^PP factors."""
    n = len(tree); order = []
    def post(u):
        for c in tree[u]['children']: post(c)
        order.append(u)
    post(root)
    up_to_parent = {}; mUp = {}
    for u in order:
        m = list(tree[u]['p'])
        for c in tree[u]['children']:
            m = pp_product(m, up_to_parent[c])
        mUp[u] = m
        if tree[u]['parent'] is not None:
            up_to_parent[u] = compute_msg(tree[u]['edge'], m, tree[u]['box'],
                                          tree[tree[u]['parent']]['box'])
    R = mUp[root]
    value, xr = pp_argmax_full(R)
    if xr is None or not np.isfinite(value):
        return {'value': 0.0, 'assignment': None, 'msgs': up_to_parent, 'feasible': False}
    assign = {root: xr}
    def down(u):
        for c in tree[u]['children']:
            edge = tree[c]['edge']; pv = assign[u]
            ivs = feasible_intervals_given(edge['cells'], pv, tree[c]['box'], True)
            h = pp_product(edge['A'], mUp[c])
            best = -np.inf; bx = None
            for (lo, hi) in ivs:
                for (a, b, cf) in h:
                    L = max(a, lo); Hh = min(b, hi)
                    if L > Hh + 1e-12: continue
                    for x in [L, Hh] + roots_in(np.polyder(np.atleast_1d(cf).astype(float)), L, Hh):
                        v = float(np.polyval(cf, x))
                        if v > best: best = v; bx = x
            assign[c] = bx if bx is not None else 0.5 * (tree[c]['box'][0] + tree[c]['box'][1])
            down(c)
    down(root)
    return {'value': value, 'assignment': [assign[i] for i in range(n)],
            'msgs': up_to_parent, 'feasible': True}

# ---------------------- brute-force grid MAP reference ----------------------
def eval_joint(tree, x):
    val = 1.0
    for i, nd in enumerate(tree):
        val *= pp_eval(nd['p'], x[i])
        if val == 0.0: return 0.0
    for i, nd in enumerate(tree):
        if nd['parent'] is not None:
            j = nd['parent']; edge = nd['edge']
            feas = any(all(ac * x[i] + ap * x[j] + c >= -1e-9 for (ac, ap, c) in cell)
                       for cell in edge['cells'])
            if not feas: return 0.0
            val *= pp_eval(edge['A'], x[i]) * pp_eval(edge['B'], x[j])
    return val

def brute_map_vec(tree, grid):
    """vectorized exhaustive grid MAP (small n)."""
    n = len(tree)
    axes = [np.linspace(tree[i]['box'][0], tree[i]['box'][1], grid) for i in range(n)]
    mesh = np.meshgrid(*axes, indexing='ij')
    val = np.ones_like(mesh[0])
    for i in range(n):
        val = val * pp_eval_vec(tree[i]['p'], mesh[i])
    for i in range(n):
        if tree[i]['parent'] is not None:
            j = tree[i]['parent']; edge = tree[i]['edge']
            Xi = mesh[i]; Xj = mesh[j]
            val = val * pp_eval_vec(edge['A'], Xi) * pp_eval_vec(edge['B'], Xj)
            feas = np.zeros_like(val, dtype=bool)
            for cell in edge['cells']:
                ok = np.ones_like(val, dtype=bool)
                for (ac, ap, c) in cell:
                    ok &= (ac * Xi + ap * Xj + c >= -1e-9)
                feas |= ok
            val = np.where(feas, val, 0.0)
    idx = np.unravel_index(np.argmax(val), val.shape)
    return float(val[idx]), [float(axes[k][idx[k]]) for k in range(n)]

# ------------------------- random Omega^PP tree builder ---------------------
def rand_bump(rng, box, w=None):
    lo0, hi0 = box
    cx = rng.uniform(lo0 + 0.9, hi0 - 0.9)
    if w is None: w = rng.uniform(1.5, 2.2)
    scale = rng.uniform(0.6, 1.4)
    lo = max(lo0, cx - w); hi = min(hi0, cx + w)
    return [(lo, hi, scale * np.array([-1.0, 2 * cx, w * w - cx * cx]))]

def rand_nonconvex_cells(rng):
    s = rng.uniform(0.3, 1.2); off = rng.uniform(-0.6, 0.6)
    return [[(-1.0, 1.0, -s + off)], [(1.0, -1.0, -s - off)]]

def make_random_tree(rng, topology, n, box=(-2.0, 2.0)):
    tree = [{'box': box, 'p': rand_bump(rng, box), 'parent': None, 'children': []}
            for _ in range(n)]
    if topology == 'chain':   edges = [(i, i - 1) for i in range(1, n)]
    elif topology == 'star':  edges = [(i, 0) for i in range(1, n)]
    elif topology == 'ternary': edges = [(i, (i - 1) // 3) for i in range(1, n)]
    else: raise ValueError(topology)
    for (ch, pa) in edges:
        tree[ch]['parent'] = pa; tree[pa]['children'].append(ch)
        tree[ch]['edge'] = {'A': rand_bump(rng, box, w=1.8),
                            'B': rand_bump(rng, box, w=1.8),
                            'cells': rand_nonconvex_cells(rng)}
    return tree

def refine_map(tree, seeds, n_restart=10, rng=None):
    """Independent reference optimum: scipy multi-start local maximization of the
    joint eval_joint over the boxes (0 outside feasible).  Not used by MpMap."""
    from scipy.optimize import minimize
    n = len(tree)
    bounds = [tuple(tree[i]['box']) for i in range(n)]
    if rng is None:
        rng = np.random.default_rng(0)
    starts = [list(s) for s in seeds if s is not None]
    for _ in range(n_restart):
        starts.append([rng.uniform(*bounds[i]) for i in range(n)])
    best = -np.inf; bx = None
    for s in starts:
        for meth in ('Nelder-Mead', 'Powell'):
            try:
                r = minimize(lambda x: -eval_joint(tree, x), np.array(s, float),
                             method=meth,
                             bounds=(bounds if meth == 'Powell' else None),
                             options=({'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-12}
                                      if meth == 'Nelder-Mead' else {'maxiter': 4000}))
                v = eval_joint(tree, r.x)
                if v > best:
                    best = v; bx = list(r.x)
            except Exception:
                pass
    return best, bx
