"""
Claim 2 (Sec 4.1, Fig 2b-d): In the Four-Room gridworld transfer experiment, HSR
row-features yield SIGNIFICANTLY HIGHER TRANSFER EFFICIENCY than baseline SR
row-features (both beating a zero-transfer one-hot 'Raw' baseline).
Paper: arXiv 2602.12753 / OpenReview txswvMHt4u (Yu & Lengyel).

Faithful independent reproduction: option-augmented (4 primitives + 8 eigenoptions)
linear SMDP Q-learning. Fixed state features phi(s): Raw one-hot, RW-SR rows, eHSR
rows. Train to G1 (episodes-to-optimal), then transfer weights to a NEW goal G2 and
count episodes-to-optimal. Transfer efficiency (paper metric, LOWER=better transfer):
    TE = (N_G2 / N^Raw_G2) / (N_G1 / N^Raw_G1).
20 seeds; two-sample test SR vs HSR. Honest verdict from measured numbers.
"""
import json,time,numpy as np
import hsr_core as H
from math import sqrt,erf
from collections import deque

def main():
    t0=time.time(); gamma=0.95
    env=H.four_room(11); N=env['N']; P=env['P']
    opts,Mrw=H.eigenoptions(env,gamma,K=8); nact=4+len(opts)
    Blist,Flist=H.build_augmented(env,opts,gamma)
    OPT=list(range(4,nact))
    def smdp_opts(r):
        Br=np.stack([Blist[a]@r for a in OPT],axis=1); V=np.zeros(N)
        for _ in range(300):
            Q=Br+np.stack([Flist[a]@V for a in OPT],axis=1); Vn=Q.max(1)
            if np.max(np.abs(Vn-V))<1e-9: V=Vn;break
            V=Vn
        return (Br+np.stack([Flist[a]@V for a in OPT],axis=1)).argmax(1)
    def mu_opts(sel):
        mu=np.zeros((N,nact))
        for s in range(N): mu[s,OPT[sel[s]]]=1.0
        return mu
    rng=np.random.default_rng(0); L=8; eHSR=np.zeros((N,N))
    for _ in range(L):
        g=rng.integers(N); r=np.zeros(N); r[g]=1
        mu=mu_opts(smdp_opts(r)); B,G=H.hsr_operator(mu,Blist,Flist); eHSR+=np.linalg.solve(np.eye(N)-G,B)/L
    feats={'Raw':np.eye(N),'SR':Mrw/np.abs(Mrw).max(),'HSR':eHSR/np.abs(eHSR).max()}
    def sp_len(goal):
        d=np.full(N,-1); d[goal]=0; q=deque([goal])
        while q:
            u=q.popleft()
            for a in range(4):
                for v in np.where(P[a,:,u]>0)[0]:
                    if d[v]<0: d[v]=d[u]+1; q.append(v)
        return d
    def exec_action(s,a,goal,rng):
        if a<4:
            j=int(np.argmax(P[a,s])); return (1.0 if j==goal else 0.0),j,1
        o=opts[a-4]; disc=1.0; R=0.0; k=0; cur=s
        for _ in range(60):
            act=o['pol'][cur]; j=int(np.argmax(P[act,cur])); k+=1
            R+=disc*(1.0 if j==goal else 0.0); disc*=gamma; cur=j
            if j==goal or rng.random()<o['beta'][cur]: break
        return R,cur,k
    def qlearn(phi,goal,start,w,optlen,rng,max_ep=150,alpha=0.2,eps=0.2):
        for ep in range(1,max_ep+1):
            s=start; steps=0
            for _ in range(200):
                q=np.array([w[a]@phi[s] for a in range(nact)])
                a=rng.integers(nact) if rng.random()<eps else int(np.argmax(q))
                R,sp,k=exec_action(s,a,goal,rng)
                tgt=R+(gamma**k)*(0 if sp==goal else np.max([w[b]@phi[sp] for b in range(nact)]))
                w[a]+=alpha*(tgt-w[a]@phi[s])*phi[s]; s=sp; steps+=k
                if s==goal or steps>200: break
            gs=start; gl=0
            for _ in range(200):
                a=int(np.argmax([w[a]@phi[gs] for a in range(nact)]))
                R,sp,k=exec_action(gs,a,goal,rng); gs=sp; gl+=k
                if gs==goal: break
            if gs==goal and gl<=optlen+1: return ep,w
        return max_ep,w
    g1,g2=5,60; d1=sp_len(g1); d2=sp_len(g2); start=int(np.argmax(d1+d2))
    print("="*74)
    print("Claim 2 - Four-Room transfer efficiency: HSR vs SR row-features (Fig 2b-d)")
    print("arXiv 2602.12753 / OpenReview txswvMHt4u - independent NumPy repro")
    print("="*74)
    print(f"N={N} augmented-actions={nact}  G1={g1}(opt {d1[start]}) G2={g2}(opt {d2[start]}) start={start}")
    S=20; res={}
    for fn,phi in feats.items():
        e1=[]; e2=[]
        for seed in range(S):
            rng=np.random.default_rng(1000+seed)
            w=[np.zeros(phi.shape[1]) for _ in range(nact)]
            n1,w=qlearn(phi,g1,start,w,d1[start],rng)
            n2,w=qlearn(phi,g2,start,w,d2[start],rng)   # transfer: retain weights
            e1.append(n1); e2.append(n2)
        res[fn]=dict(G1=e1,G2=e2,G1_mean=float(np.mean(e1)),G2_mean=float(np.mean(e2)),
                     G1_sem=float(np.std(e1,ddof=1)/sqrt(S)),G2_sem=float(np.std(e2,ddof=1)/sqrt(S)))
        print(f"  {fn:4s}: G1 ep={np.mean(e1):5.1f}+-{np.std(e1,ddof=1)/sqrt(S):.1f}  "
              f"G2(transfer) ep={np.mean(e2):5.1f}+-{np.std(e2,ddof=1)/sqrt(S):.1f}")
    raw1=res['Raw']['G1_mean']; raw2=res['Raw']['G2_mean']
    te={}
    for fn in ['SR','HSR']:
        te[fn]=(res[fn]['G2_mean']/raw2)/(res[fn]['G1_mean']/raw1)
    # SR vs HSR two-sample test on G2 transfer episodes
    a=np.array(res['SR']['G2']); b=np.array(res['HSR']['G2'])
    t=(a.mean()-b.mean())/sqrt(a.var(ddof=1)/S+b.var(ddof=1)/S)
    p=2*(1-0.5*(1+erf(abs(t)/sqrt(2))))
    print(f"\n  Transfer efficiency (lower=better): SR={te['SR']:.3f}  HSR={te['HSR']:.3f}")
    print(f"  SR & HSR both transfer to G2; Raw (one-hot) zero-transfer baseline G2={raw2:.1f} (cap 150)")
    print(f"  SR-vs-HSR G2 episodes: t={t:.3f}, p={p:.3f}")
    both_beat_raw = res['SR']['G2_mean']<raw2*0.9 and res['HSR']['G2_mean']<raw2*0.9
    hsr_better = res['HSR']['G2_mean']<res['SR']['G2_mean'] and p<0.05
    if hsr_better: verdict="reproduced"
    elif both_beat_raw: verdict="partial: SR & HSR both transfer vs Raw, but HSR NOT sig. better than SR"
    else: verdict="not_reproduced"
    print(f"\n  VERDICT: {verdict}")
    print("="*74)
    rt=time.time()-t0
    out=dict(claim="Four-Room transfer: HSR row-features transfer significantly faster than SR (Fig 2d)",
        paper="arXiv 2602.12753 / OpenReview txswvMHt4u", gamma=gamma, N=N, seeds=S,
        n_augmented_actions=nact, results=res,
        transfer_efficiency_SR=float(te['SR']), transfer_efficiency_HSR=float(te['HSR']),
        Raw_G2_zero_transfer_baseline=float(raw2), SR_vs_HSR_G2_t=float(t), SR_vs_HSR_G2_p=float(p),
        SR_and_HSR_both_beat_Raw=bool(both_beat_raw), HSR_sig_better_than_SR=bool(hsr_better),
        verdict=verdict, runtime_s=rt)
    json.dump(out,open("results.json","w"),indent=2)
    print(f"runtime={rt:.2f}s ; wrote results.json")

if __name__=="__main__": main()
