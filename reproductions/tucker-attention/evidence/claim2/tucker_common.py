"""
Shared CPU-exact (float64, numpy) mechanics for the Tucker Attention reproduction.
Paper: "Tucker Attention: A generalization of approximate attention mechanisms"
       (arXiv 2603.30033 / OpenReview ErcPPRZaiq).

Implements, from first principles and with no fabrication:
  * n-mode tensor products (Kolda & Bader, 2009 convention),
  * the tensor reformulation of MHA  (Eqs. 2-6),
  * a generic *factored* Tucker Attention forward pass (Def. 2.1 + Sec. 3 impl.),
  * reference MHA / GQA / MLA layers,
  * constructions of the Tucker core+factors that recover each of MHA/GQA/MLA
    EXACTLY (Theorems B.2/B.3/B.4), and rank measurement of the attention tensors.

Everything is deterministic (numpy default_rng with fixed seeds) and single thread.
"""
import numpy as np


# ----------------------------------------------------------------------------- 
# n-mode products.  Tensor T has shape (I0, I1, I2).  Mode-n product with matrix
# U of shape (J, I_n) replaces mode n by J:  (T x_n U)[..,j,..] = sum_in T[..,in,..] U[j,in].
# ----------------------------------------------------------------------------- 
def mode1(T, U):  # U: (J, I0) -> (J, I1, I2)
    return np.einsum('abc,ja->jbc', T, U)

def mode2(T, U):  # U: (J, I1) -> (I0, J, I2)
    return np.einsum('abc,jb->ajc', T, U)

def mode3(T, U):  # U: (J, I2) -> (I0, I1, J)
    return np.einsum('abc,jc->abj', T, U)

def tucker_reconstruct(C, U1, U2, U3):
    """W = C x1 U1 x2 U2 x3 U3."""
    return mode3(mode2(mode1(C, U1), U2), U3)


def softmax_rows(M):
    M = M - M.max(axis=-1, keepdims=True)
    E = np.exp(M)
    return E / E.sum(axis=-1, keepdims=True)


def matn_rank(T, mode, tol=1e-9):
    """Numerical rank of the mode-`mode` matricization (0-indexed) of tensor T."""
    T = np.moveaxis(T, mode, 0)
    M = T.reshape(T.shape[0], -1)
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > tol * s[0]).sum()) if s.size and s[0] > 0 else 0


# ----------------------------------------------------------------------------- 
# Tensor reformulation forward (Eqs. 4-6): given the pre/post-softmax tensors
# W (nH,d,d) and Wt (nH,d,d) [= stack of Wtilde_i^T], compute attention output.
# ----------------------------------------------------------------------------- 
def attention_from_tensors(W, Wt, X, dH):
    logits = np.einsum('nm,imk,lk->inl', X, W, X) / np.sqrt(dH)   # X W_i X^T / sqrt(dH)
    H1 = softmax_rows(logits)                                     # (nH, N, N)
    H2 = mode3(Wt, X)                                             # (nH, d, N)
    out = np.einsum('ijl,ikl->jk', H1, H2)                        # sum_{i,l}
    return out, H1


# ----------------------------------------------------------------------------- 
# Faithful *factored* Tucker Attention forward (Sec. 3 "Implementation"): never
# builds the full W; projects keys/values then contracts the small core.
# ----------------------------------------------------------------------------- 
def tucker_attention_forward(C, U1, U2, U3, Ct, U1t, U2t, U3t, X, dH):
    P2 = X @ U2                                   # (N, r2)  query projection
    P3 = X @ U3                                   # (N, r3)  key   projection
    T  = mode1(C, U1)                             # (nH, r2, r3)
    logits = np.einsum('nb,ibc,mc->inm', P2, T, P3) / np.sqrt(dH)   # (nH,N,N)
    H1 = softmax_rows(logits)
    V  = X @ U3t                                  # (N, r3t) value projection
    S  = np.einsum('iac,mc->iam', mode1(Ct, U1t), V)   # (nH, r2t, N)
    H2 = np.einsum('iam,ja->ijm', S, U2t)              # (nH, d, N)
    out = np.einsum('ijl,ikl->jk', H1, H2)
    return out


# ----------------------------------------------------------------------------- 
# Reference layers.  All use the paper's refined head definition where the output
# projection W^O is row-blocked into per-head W_i^O in R^{dH x d}.
# ----------------------------------------------------------------------------- 
def mha_reference(X, WQ, WK, WV, WO, nH, dH):
    """Standard MHA = sum_i softmax(X WiQ (X WiK)^T/sqrt dH) X WiV WiO."""
    d = X.shape[1]
    out = np.zeros((X.shape[0], d))
    for i in range(nH):
        Qi = X @ WQ[:, i*dH:(i+1)*dH]
        Ki = X @ WK[:, i*dH:(i+1)*dH]
        Vi = X @ WV[:, i*dH:(i+1)*dH]
        Oi = WO[i*dH:(i+1)*dH, :]
        A  = softmax_rows(Qi @ Ki.T / np.sqrt(dH))
        out += A @ Vi @ Oi
    return out


def build_mha_tensors(WQ, WK, WV, WO, nH, dH):
    d = WQ.shape[0]
    W  = np.zeros((nH, d, d)); Wt = np.zeros((nH, d, d))
    for i in range(nH):
        WiQ = WQ[:, i*dH:(i+1)*dH]; WiK = WK[:, i*dH:(i+1)*dH]
        WiV = WV[:, i*dH:(i+1)*dH]; WiO = WO[i*dH:(i+1)*dH, :]
        W[i]  = WiQ @ WiK.T                 # W_i
        Wt[i] = (WiV @ WiO).T               # Wtilde_i^T
    return W, Wt
