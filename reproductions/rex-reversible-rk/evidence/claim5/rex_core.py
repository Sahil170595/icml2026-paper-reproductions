"""
rex_core.py  --  Independent, faithful implementation of the Rex family of
Reversible Exponential (Stochastic) Runge-Kutta solvers.

Paper: Blasingame & Liu, "Rex: (A Family of) Reversible (Exponential Stochastic
Runge-Kutta) Solvers for Diffusion Models", OpenReview 7pQIzVNctu / arXiv 2502.08834.

Rex = McCallum-Foster algebraically-reversible construction (Eqs. 6-7 of the paper)
applied to an exponential (Lawson) Runge-Kutta base method for a semilinear ODE/SDE
    dx/dt = a(t) x + N(t, x)          (+ g(t) dW  for the SDE case)
where the linear (stiff) part a(t)x is integrated EXACTLY via the integrating
factor (the "exponential") and the nonlinear remainder N by an explicit RK scheme.

McCallum-Foster reversible step (paper Def., Eqs 6-7), coupling zeta in (0,1]:
  forward :  x_{n+1} = zeta x_n + (1-zeta) xhat_n + Phi_h(t_n, xhat_n)
             xhat_{n+1} = xhat_n - Phi_{-h}(t_{n+1}, x_{n+1})
  backward:  xhat_n = xhat_{n+1} + Phi_{-h}(t_{n+1}, x_{n+1})
             x_n = zeta^{-1} x_{n+1} + (1-zeta^{-1}) xhat_n - zeta^{-1} Phi_h(t_n, xhat_n)
The backward step is the EXACT algebraic inverse of the forward step for ANY base
increment Phi (this is what makes Rex reversible without storing the trajectory).
Working in the Lawson variable y = exp(-Lambda(t)) x makes Phi an additive RK
increment, so the exponential solver is reversible in x as well (y<->x is an exact
diagonal rescaling).  All operations are pure numpy; deterministic.
"""
import numpy as np


# ----------------------------------------------------------------------------
# Explicit Runge-Kutta increments for  dy/dt = G(t, y)  over a step of size h.
# Phi_h(t, y) := (RK-p one-step map)(y) - y   (the additive increment).
# order p in {1,2,3}  ->  classical Euler / Heun(RK2) / Kutta(RK3).
# ----------------------------------------------------------------------------
def rk_increment(G, t, y, h, order):
    if order == 1:                              # explicit Euler (order 1)
        return h * G(t, y)
    if order == 2:                              # Heun / explicit trapezoid (order 2)
        k1 = G(t, y)
        k2 = G(t + h, y + h * k1)
        return h * 0.5 * (k1 + k2)
    if order == 3:                              # Kutta's third-order method (order 3)
        k1 = G(t, y)
        k2 = G(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = G(t + h, y - h * k1 + 2.0 * h * k2)
        return h * (k1 + 4.0 * k2 + k3) / 6.0
    raise ValueError("order must be 1, 2, or 3")


# ----------------------------------------------------------------------------
# Rex (reversible exponential RK) -- single forward / backward step in the
# Lawson variable y.  G is the transformed (exponential) vector field.
# ----------------------------------------------------------------------------
def rex_forward(G, tn, h, order, zeta, y, yhat):
    Phi = rk_increment(G, tn, yhat, h, order)
    y1 = zeta * y + (1.0 - zeta) * yhat + Phi
    Phi_m = rk_increment(G, tn + h, y1, -h, order)
    yhat1 = yhat - Phi_m
    return y1, yhat1


def rex_backward(G, tn, h, order, zeta, y1, yhat1):
    Phi_m = rk_increment(G, tn + h, y1, -h, order)
    yhat = yhat1 + Phi_m
    Phi = rk_increment(G, tn, yhat, h, order)
    y = (1.0 / zeta) * y1 + (1.0 - 1.0 / zeta) * yhat - (1.0 / zeta) * Phi
    return y, yhat


# Non-reversible baseline of the SAME order: plain exponential (Lawson) RK,
# single state, y_{n+1} = y_n + Phi_h(t_n, y_n).  (First-order case == DDIM/
# exponential-Euler.)  Its "inverse" (a backward exp-RK step) is NOT exact.
def exprk_forward(G, tn, h, order, y):
    return y + rk_increment(G, tn, y, h, order)


def exprk_backward(G, tn, h, order, y1):
    # naive reverse integration: one backward RK step from (t_{n+1}, y1)
    return y1 + rk_increment(G, tn + h, y1, -h, order)


# ----------------------------------------------------------------------------
# Lawson transform helpers for a CONSTANT linear coefficient a (scalar or diag).
#   Lambda(t) = a * (t - t0);   x = exp(Lambda) y;   y = exp(-Lambda) x
#   transformed field  G(t, y) = exp(-Lambda(t)) * N(t, exp(Lambda(t)) y)
# ----------------------------------------------------------------------------
def make_lawson_field(a, N, t0):
    def G(t, y):
        e = np.exp(a * (t - t0))
        return np.exp(-a * (t - t0)) * N(t, e * y)   # = e^{-Lam} N(t, e^{Lam} y)
    return G


def y_to_x(a, t0, t, y):
    return np.exp(a * (t - t0)) * y


def x_to_y(a, t0, t, x):
    return np.exp(-a * (t - t0)) * x


# ----------------------------------------------------------------------------
# Fixed-step Rex integrator over [t0, t1] with Nsteps.  Returns final (y, yhat)
# and (optionally) the recorded backward reconstruction error.
# ----------------------------------------------------------------------------
def integrate_rex(G, t0, t1, Nsteps, order, zeta, y0):
    h = (t1 - t0) / Nsteps
    y = y0.copy()
    yhat = y0.copy()                 # xhat_0 = x_0  (paper initialization)
    ys = [y.copy()]
    for n in range(Nsteps):
        tn = t0 + n * h
        y, yhat = rex_forward(G, tn, h, order, zeta, y, yhat)
        ys.append(y.copy())
    return y, yhat, h, ys


def loglog_slope(hs, errs):
    hs = np.asarray(hs, float); errs = np.asarray(errs, float)
    m = errs > 0
    s, _ = np.polyfit(np.log10(hs[m]), np.log10(errs[m]), 1)
    return float(s)
