"""
Claim 4 (Sec 4.2, Fig 3-4): NMF applied to the (expected) HSR yields a SPARSE,
low-rank basis that (i) matches SR-SVD reconstruction, (ii) is elevated at
BOTTLENECK (doorway) states, whereas NMF on the SR suffers 'feature collapse'.
Paper: arXiv 2602.12753 / OpenReview txswvMHt4u (Yu & Lengyel).
Independent build: eSR & eHSR averaged over 12 pretraining option-policies; SVD &
multiplicative-update NMF; measure sparsity (Gini), bottleneck ratio, recon MSE.
"""
import json,time,numpy as np
import hsr_core as H
def nmf(X,k,it=500,seed=1):
    rr=np.random.default_rng(seed); X=np.maximum(X,0.0)
    W=rr.random((X.shape[0],k))+1e-3; Hh=rr.random((k,X.shape[1]))+1e-3
    for _ in range(it):
        Hh*=(W.T@X)/(W.T@W@Hh+1e-9); W*=(X@Hh.T)/(W@(Hh@Hh.T)+1e-9)
    return W,Hh
def gini(v):
    v=np.sort(np.abs(v))+1e-12; n=len(v); idx=np.arange(1,n+1)
    return float(np.sum((2*idx-n-1)*v)/(n*np.sum(v)))
def main():
    t0=time.time(); gamma=0.95
    env=H.four_room(11); N=env['N']; doors=env['doors']
    opts,Mrw=H.eigenoptions(env,gamma,K=8); nact=4+len(opts)
    Blist,Flist=H.build_augmented(env,opts,gamma); OPT=list(range(4,nact))
    def smdp_opts(r):
        Br=np.stack([Blist[a]@r for a in OPT],axis=1); V=np.zeros(N)
        for _ in range(400):
            Q=Br+np.stack([Flist[a]@V for a in OPT],axis=1); Vn=Q.max(1)
            if np.max(np.abs(Vn-V))<1e-9: V=Vn;break
            V=Vn
        return (Br+np.stack([Flist[a]@V for a in OPT],axis=1)).argmax(1)
    def mu_opts(sel):
        mu=np.zeros((N,nact))
        for s in range(N): mu[s,OPT[sel[s]]]=1.0
        return mu
    rng=np.random.default_rng(0); L=12; eSR=np.zeros((N,N)); eHSR=np.zeros((N,N))
    for _ in range(L):
        g=rng.integers(N); r=np.zeros(N); r[g]=1
        pol,_=H.value_iteration(env,r,gamma); eSR+=H.sr_matrix(H.policy_transition(env,pol),gamma)/L
        mu=mu_opts(smdp_opts(r)); B,G=H.hsr_operator(mu,Blist,Flist); eHSR+=np.linalg.solve(np.eye(N)-G,B)/L
    dmask=np.zeros(N,bool); dmask[doors]=True
    print("="*74)
    print("Claim 4 - HSR-NMF sparse basis: sparsity, bottleneck, reconstruction (Fig 4)")
    print("arXiv 2602.12753 / OpenReview txswvMHt4u - independent NumPy repro")
    print("="*74)
    K=8; out={}
    for name,M in [("eSR",eSR),("eHSR",eHSR)]:
        W,_=nmf(M,K); g=float(np.mean([gini(W[:,j]) for j in range(K)]))
        act=np.abs(W).sum(1); bratio=float(act[dmask].mean()/act[~dmask].mean())
        out[name]=dict(nmf_gini=g,bottleneck_ratio=bratio)
        print(f"  {name}: NMF mean-Gini(sparsity)={g:.3f}   bottleneck-activation-ratio={bratio:.3f}")
    def rsvd(M,k):
        U,S,Vt=np.linalg.svd(M); return float(np.mean((M-(U[:,:k]*S[:k])@Vt[:k])**2))
    def rnmf(M,k):
        W,Hh=nmf(M,k); return float(np.mean((np.maximum(M,0)-W@Hh)**2))
    print("\n  Reconstruction MSE vs rank:")
    print("  rank   SR-SVD    SR-NMF    HSR-SVD   HSR-NMF")
    recon={}
    for k in [4,8,16]:
        a,b,c,d=rsvd(eSR,k),rnmf(eSR,k),rsvd(eHSR,k),rnmf(eHSR,k)
        recon[str(k)]=dict(SR_SVD=a,SR_NMF=b,HSR_SVD=c,HSR_NMF=d)
        print(f"  {k:4d}   {a:.5f}  {b:.5f}  {c:.5f}  {d:.5f}")
    hsr_sparser = out['eHSR']['nmf_gini'] > out['eSR']['nmf_gini']+0.03
    hsr_bottleneck = out['eHSR']['bottleneck_ratio'] > 1.15
    hsrnmf_matches_srsvd = recon['8']['HSR_NMF'] <= recon['8']['SR_SVD']*1.5
    sr_nmf_collapse = recon['8']['SR_NMF'] > recon['8']['SR_SVD']*3.0
    print(f"\n  paper-predicted signatures (measured):")
    print(f"    HSR-NMF sparser than SR-NMF          : {hsr_sparser}  (Gini {out['eHSR']['nmf_gini']:.3f} vs {out['eSR']['nmf_gini']:.3f})")
    print(f"    HSR-NMF elevated at bottlenecks(>1.15): {hsr_bottleneck}  (ratio {out['eHSR']['bottleneck_ratio']:.3f})")
    print(f"    HSR-NMF ~ SR-SVD reconstruction      : {hsrnmf_matches_srsvd}")
    print(f"    SR-NMF 'feature collapse' (>>SR-SVD) : {sr_nmf_collapse}")
    verdict = hsr_bottleneck and hsr_sparser
    vtxt = "reproduced" if verdict else "not_reproduced - HSR-NMF basis ~ SR basis"
    print(f"\n  VERDICT: {vtxt}")
    print("="*74)
    rt=time.time()-t0
    js=dict(claim="NMF of HSR yields sparse low-rank basis elevated at bottlenecks, matching SR-SVD reconstruction; NMF on SR collapses (Fig 4)",
        paper="arXiv 2602.12753 / OpenReview txswvMHt4u", gamma=gamma, N=N, doors=doors, K=K,
        basis_metrics=out, reconstruction_mse=recon,
        HSR_NMF_sparser=bool(hsr_sparser), HSR_NMF_bottleneck_elevated=bool(hsr_bottleneck),
        HSR_NMF_matches_SR_SVD=bool(hsrnmf_matches_srsvd), SR_NMF_feature_collapse=bool(sr_nmf_collapse),
        overall_reproduced=bool(verdict), verdict=("reproduced" if verdict else "not_reproduced"), runtime_s=rt)
    json.dump(js,open("results.json","w"),indent=2)
    print(f"runtime={rt:.2f}s ; wrote results.json")
if __name__=="__main__": main()
