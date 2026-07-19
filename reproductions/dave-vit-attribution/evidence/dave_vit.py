"""Compact real ViT + DAVE gradient decomposition (Eq. 3), pure PyTorch, CPU, float64.

The DAVE decomposition (paper Eq. 3) states that each ViT layer F(X)=L(X)(X)+B has
    D_X F = L(X)  +  ((D_X L(X)(.)) X)
             ^effective        ^operator-variation
This is exactly the multivariate chain rule for F(X)=g(X,X): total derivative =
partial wrt the acted-upon argument C (the effective transformation L(X), a linear
operator) plus partial wrt the operator-source argument A (operator variation).
We implement each ViT sublayer in dual-input form g(A,C) matching the paper's layer
taxonomy (Eq. 2): attention/LayerNorm are type-(I) W_t(A) C W_d, GELU is type-(II)
gate(A) (.) C. The identity then holds to machine precision by construction.
"""
import torch, math
torch.set_default_dtype(torch.float64)

def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

# ---- dual-input sublayers: g(A, C). g(X,X) reproduces the true layer. ----
def lin(C, W, b):                       # linear: operator-free (opvar = 0)
    return C @ W + b

def layernorm(A, C, gamma, beta, eps=1e-5):
    # operator = normalization statistics from A; acts linearly on C (type I)
    mu = A.mean(-1, keepdim=True)
    var = A.var(-1, unbiased=False, keepdim=True)
    inv = 1.0 / torch.sqrt(var + eps)
    muC = C.mean(-1, keepdim=True)
    return (C - muC) * inv * gamma + beta   # linear in C for fixed A

def mhsa(A, C, Wq, Wk, Wv, Wo, nh):
    # operator = attention matrix P(A) (type I W_t); values from C, static W_d=Wv,Wo
    t, d = A.shape
    dh = d // nh
    Q = (A @ Wq).view(t, nh, dh).transpose(0, 1)   # nh,t,dh  (from A)
    K = (A @ Wk).view(t, nh, dh).transpose(0, 1)
    P = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(dh), dim=-1)  # nh,t,t
    V = (C @ Wv).view(t, nh, dh).transpose(0, 1)   # from C
    O = (P @ V).transpose(0, 1).contiguous().view(t, d)
    return O @ Wo

def mlp(A, C, W1, b1, W2, b2):
    # type II: GELU gate from A multiplies fc1(C); fc1,fc2 linear
    hA = A @ W1 + b1
    hC = C @ W1 + b1
    gate = gelu(hA) / torch.where(hA.abs() < 1e-12, torch.ones_like(hA), hA)
    act = gate * hC                    # = gelu(hA) when A==C ; linear in C
    return act @ W2 + b2

class TinyViT:
    """Small real ViT: patch-embed conv, cls token, pos-embed, blocks, linear head->1 logit."""
    def __init__(self, img=32, patch=8, dim=32, depth=2, nh=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.img, self.patch, self.dim, self.nh, self.depth = img, patch, dim, nh, depth
        self.np = (img // patch) ** 2
        pin = 3 * patch * patch
        r = lambda *s, sc=0.02: (torch.randn(*s, generator=g) * sc)
        self.Wpe = r(pin, dim, sc=0.08); self.bpe = r(dim)
        self.cls = r(1, dim); self.pos = r(self.np + 1, dim)
        self.blocks = []
        for _ in range(depth):
            self.blocks.append(dict(
                g1=torch.ones(dim), b1=torch.zeros(dim),
                Wq=r(dim, dim), Wk=r(dim, dim), Wv=r(dim, dim), Wo=r(dim, dim),
                g2=torch.ones(dim), b2=torch.zeros(dim),
                W1=r(dim, 2*dim), c1=r(2*dim), W2=r(2*dim, dim), c2=r(dim)))
        self.gN = torch.ones(dim); self.bN = torch.zeros(dim)
        self.Wh = r(dim, 1); self.bh = r(1)

    def patch_embed(self, X):
        # X: 3,img,img -> tokens (np, pin) -> linear -> (np, dim)
        p = self.patch; n = self.img // p
        patches = []
        for i in range(n):
            for j in range(n):
                patches.append(X[:, i*p:(i+1)*p, j*p:(j+1)*p].reshape(-1))
        P = torch.stack(patches, 0)          # np, pin
        return P @ self.Wpe + self.bpe

    def forward(self, X):
        tok = self.patch_embed(X)                       # np,dim
        h = torch.cat([self.cls, tok], 0) + self.pos    # t,dim
        for blk in self.blocks:
            a = layernorm(h, h, blk['g1'], blk['b1'])
            h = h + mhsa(a, a, blk['Wq'], blk['Wk'], blk['Wv'], blk['Wo'], self.nh)
            a2 = layernorm(h, h, blk['g2'], blk['b2'])
            h = h + mlp(a2, a2, blk['W1'], blk['c1'], blk['W2'], blk['c2'])
        hN = layernorm(h, h, self.gN, self.bN)
        return (hN[0:1] @ self.Wh + self.bh).squeeze()   # scalar logit (cls token)


def jac(fn, X):
    """Full Jacobian of matrix-valued fn at X (X requires no grad here)."""
    X = X.clone().detach().requires_grad_(True)
    Y = fn(X)
    ysh, xsh = Y.shape, X.shape
    Yf = Y.reshape(-1)
    J = torch.zeros(Yf.numel(), X.numel())
    if not Yf.requires_grad:          # output independent of X => Jacobian is 0
        return J, ysh, xsh
    for i in range(Yf.numel()):
        g, = torch.autograd.grad(Yf[i], X, retain_graph=True, allow_unused=True)
        if g is not None:
            J[i] = g.reshape(-1)
    return J, ysh, xsh


def check_layer(name, g_fn, X):
    """Verify D_X[g(X,X)] = D_C g(A,C)|eff + D_A g(A,C)|opvar to machine precision."""
    Xd = X.detach()
    Jfull, _, _ = jac(lambda Z: g_fn(Z, Z), Xd)                 # true layer derivative
    Jeff,  _, _ = jac(lambda C: g_fn(Xd.clone(), C), Xd)        # effective L(X)
    Jop,   _, _ = jac(lambda A: g_fn(A, Xd.clone()), Xd)        # operator variation
    recon = Jeff + Jop
    err = (Jfull - recon).abs().max().item()
    rel = err / (Jfull.abs().max().item() + 1e-300)
    eff_n = Jeff.norm().item(); op_n = Jop.norm().item(); full_n = Jfull.norm().item()
    return dict(name=name, max_abs_err=err, rel_err=rel,
                eff_norm=eff_n, opvar_norm=op_n, full_norm=full_n,
                opvar_frac=op_n / (eff_n + op_n + 1e-300))

def forward_mode(m, X, detach_op=False):
    """Full ViT forward. detach_op=True freezes every operator (attention matrix,
    LayerNorm stats, GELU gate) at the current running state -> gradient flows only
    through the effective (input-conditioned linear) path W_L(X) (paper eq.4,
    'practical realisation'). detach_op=False = true autograd gradient."""
    def op(h):
        return h.detach() if detach_op else h
    tok = m.patch_embed(X)
    h = torch.cat([m.cls, tok], 0) + m.pos
    for blk in m.blocks:
        a = layernorm(op(h), h, blk['g1'], blk['b1'])
        h = h + mhsa(op(a), a, blk['Wq'], blk['Wk'], blk['Wv'], blk['Wo'], m.nh)
        a2 = layernorm(op(h), h, blk['g2'], blk['b2'])
        h = h + mlp(op(a2), a2, blk['W1'], blk['c1'], blk['W2'], blk['c2'])
    hN = layernorm(op(h), h, m.gN, m.bN)
    return (hN[0:1] @ m.Wh + m.bh).reshape(())

def input_grad(m, X, **kw):
    X = X.clone().detach().requires_grad_(True)
    y = forward_mode(m, X, **kw)
    g, = torch.autograd.grad(y, X)
    return g, float(y.detach())

def forward_head(m, X, Wh, bh, detach_op=False):
    """Same forward as forward_mode but with an arbitrary K-class head Wh (dim x K); returns (K,).
    detach_op=True freezes all operators -> gradient flows through the effective transformation only."""
    def op(h):
        return h.detach() if detach_op else h
    tok = m.patch_embed(X)
    h = torch.cat([m.cls, tok], 0) + m.pos
    for blk in m.blocks:
        a = layernorm(op(h), h, blk['g1'], blk['b1'])
        h = h + mhsa(op(a), a, blk['Wq'], blk['Wk'], blk['Wv'], blk['Wo'], m.nh)
        a2 = layernorm(op(h), h, blk['g2'], blk['b2'])
        h = h + mlp(op(a2), a2, blk['W1'], blk['c1'], blk['W2'], blk['c2'])
    hN = layernorm(op(h), h, m.gN, m.bN)
    return (hN[0:1] @ Wh + bh).reshape(-1)          # K logits from the cls token

def attribution_map(m, X, Wh, bh, target, method="dave"):
    """Per-pixel attribution (H,W) for a target class. method: 'dave' (effective transform),
    'ixg' (inputxgradient, full), or 'grad' (raw gradient)."""
    Xr = X.clone().detach().requires_grad_(True)
    detach_op = (method == "dave")
    y = forward_head(m, Xr, Wh, bh, detach_op=detach_op)[target]
    g, = torch.autograd.grad(y, Xr)
    if method == "grad":
        a = g
    else:
        a = g * Xr.detach()                          # input x (effective or full) gradient
    return a.sum(0).detach()                          # sum over channels -> (H,W)
