"""Shared tabular four-room + SR + eigenoptions + HSR core (numpy, CPU)."""
import numpy as np

def four_room(size=11):
    """Classic four-room grid. Returns walls grid, open-state list, coords, and
    neighbor transition (deterministic 4-action). ~104 states for size=11."""
    G = np.zeros((size, size), dtype=int)  # 0 open, 1 wall
    G[0,:]=1; G[-1,:]=1; G[:,0]=1; G[:,-1]=1
    mid = size//2
    G[mid,:]=1; G[:,mid]=1
    # doorways (bottlenecks): one gap in each wall segment
    q=size//4
    doors=[(mid, q),(mid, size-1-q),(q, mid),(size-1-q, mid)]
    for (r,c) in doors: G[r,c]=0
    coords=[(r,c) for r in range(size) for c in range(size) if G[r,c]==0]
    idx={rc:i for i,rc in enumerate(coords)}
    N=len(coords)
    A=4
    moves=[(-1,0),(1,0),(0,-1),(0,1)]  # U D L R
    # P[a] deterministic transition matrix (stay if blocked)
    P=np.zeros((A,N,N))
    for i,(r,c) in enumerate(coords):
        for a,(dr,dc) in enumerate(moves):
            nr,nc=r+dr,c+dc
            if 0<=nr<size and 0<=nc<size and G[nr,nc]==0:
                j=idx[(nr,nc)]
            else:
                j=i
            P[a,i,j]=1.0
    doors_idx=[idx[d] for d in doors]
    return dict(G=G,coords=coords,idx=idx,N=N,A=A,P=P,moves=moves,doors=doors_idx,size=size)

def rw_transition(env):
    """Uniform random-walk transition matrix (avg over 4 actions)."""
    return env['P'].mean(axis=0)

def sr_matrix(Pmat, gamma):
    N=Pmat.shape[0]
    return np.linalg.solve(np.eye(N)-gamma*Pmat, np.eye(N))

def value_iteration(env, reward_vec, gamma, tol=1e-8, max_it=2000):
    """Value iteration -> greedy deterministic policy (action per state).
    reward_vec: reward as function of NEXT state (length N)."""
    P=env['P']; N=env['N']; A=env['A']
    V=np.zeros(N)
    for _ in range(max_it):
        Q=np.stack([P[a]@(reward_vec+gamma*V) for a in range(A)],axis=1)  # N x A
        Vn=Q.max(axis=1)
        if np.max(np.abs(Vn-V))<tol: V=Vn; break
        V=Vn
    Q=np.stack([P[a]@(reward_vec+gamma*V) for a in range(A)],axis=1)
    pol=Q.argmax(axis=1)
    return pol,V

def policy_transition(env, pol):
    """Deterministic policy -> transition matrix."""
    P=env['P']; N=env['N']
    Ppi=np.zeros((N,N))
    for i in range(N): Ppi[i]=P[pol[i],i]
    return Ppi

def eigenoptions(env, gamma, K):
    """Discover K eigenoptions from SVD of RW-SR. Returns list of options,
    each dict with internal transition P_int (N x N), termination beta (N,)."""
    Prw=rw_transition(env)
    Mrw=sr_matrix(Prw,gamma)
    U,S,Vt=np.linalg.svd(Mrw)
    opts=[]
    N=env['N']
    for k in range(1,K+1):
        v=Vt[k]  # k-th right singular vector (skip 0th = stationary/global)
        # pseudo-reward r(s')=v(s') (potential-based reward v(s')-v(s) handled by VI on next-state v)
        pol,Vval=value_iteration(env, v, gamma)
        # termination: terminate at the maximizer region of v (option goal) -> top quantile
        thr=np.quantile(v, 0.9)
        beta=(v>=thr).astype(float)
        beta=np.clip(beta,0.02,1.0)  # avoid never-terminating
        Pint=policy_transition(env,pol)
        opts.append(dict(P=Pint,beta=beta,pol=pol,v=v))
    return opts, Mrw

def termination_kernel(Pint, beta, gamma):
    """F = discounted termination kernel: F = (I - gamma P diag(1-beta))^{-1} gamma P diag(beta).
    Row sums = E[gamma^tau], tau>=1. Exact."""
    N=Pint.shape[0]
    cont=Pint*(1.0-beta)[None,:]  # P diag(1-beta)
    term=Pint*beta[None,:]        # P diag(beta)
    F=np.linalg.solve(np.eye(N)-gamma*cont, gamma*term)
    return F

def option_sr(Pint, beta, gamma):
    """Intra-option SR contribution: SR accumulated WHILE option runs (before termination).
    B-part M^abar: expected discounted occupancy until termination.
    M = (I - gamma P diag(1-beta))^{-1}."""
    N=Pint.shape[0]
    cont=Pint*(1.0-beta)[None,:]
    M=np.linalg.solve(np.eye(N)-gamma*cont, np.eye(N))
    return M

def build_augmented(env, opts, gamma):
    """Build per-(pseudo)action B_a (intra-option SR) and F_a (continuation kernel).
    Actions: 4 primitives (tau=1,beta=1) + len(opts) options."""
    P=env['P']; N=env['N']; A=env['A']
    Blist=[]; Flist=[]
    # primitives: beta=1 everywhere, tau=1
    for a in range(A):
        Pa=P[a]
        beta=np.ones(N)
        Blist.append(option_sr(Pa,beta,gamma))   # = (I - gamma*0)^{-1} = I  (occupancy just current state)
        Flist.append(termination_kernel(Pa,beta,gamma))  # = gamma * Pa
    for o in opts:
        Blist.append(option_sr(o['P'],o['beta'],gamma))
        Flist.append(termination_kernel(o['P'],o['beta'],gamma))
    return Blist, Flist  # each length (A+K), matrices N x N

def hsr_operator(mu, Blist, Flist):
    """Given high-level policy mu (N x nact), build B^mu and G^mu.
    T(M) = B + G M ; fixed point M* = (I-G)^{-1} B."""
    N=Blist[0].shape[0]; nact=len(Blist)
    B=np.zeros((N,N)); G=np.zeros((N,N))
    for a in range(nact):
        w=mu[:,a][:,None]  # N x 1 weighting per start state
        B+=w*Blist[a]
        G+=w*Flist[a]
    return B,G

if __name__=="__main__":
    np.random.seed(0)
    gamma=0.95
    env=four_room(11)
    print("N states",env['N'],"doors",env['doors'])
    opts,Mrw=eigenoptions(env,gamma,K=8)
    Blist,Flist=build_augmented(env,opts,gamma)
    print("num augmented actions",len(Blist))
    # uniform random high-level policy
    nact=len(Blist); N=env['N']
    mu=np.ones((N,nact))/nact
    B,Gm=hsr_operator(mu,Blist,Flist)
    rowsum=Gm.sum(axis=1)
    print("G row-sum: min %.4f max %.4f (gamma=%.2f)"%(rowsum.min(),rowsum.max(),gamma))
    print("max row-sum <= gamma?",rowsum.max()<=gamma+1e-9)
    # primitive-only policy -> row sum should be gamma exactly
    mu2=np.zeros((N,nact)); mu2[:,:4]=0.25
    B2,G2=hsr_operator(mu2,Blist,Flist)
    print("primitive-only G row-sum max %.6f (should be gamma=%.2f)"%(G2.sum(axis=1).max(),gamma))
