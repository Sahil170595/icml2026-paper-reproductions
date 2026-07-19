import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import sgshift_core as sg
from sklearn.linear_model import LogisticRegression
OUT=os.path.join(os.path.dirname(__file__),'..','_cache','diag.txt')
rng=np.random.default_rng(1); p=25; n=20000
Araw=rng.standard_normal((p,p)); Sig=Araw@Araw.T/p+np.eye(p); Dd=np.sqrt(np.diag(Sig)); Sig/=np.outer(Dd,Dd)
Lc=np.linalg.cholesky(Sig); Z=rng.standard_normal((n,p))@Lc.T
beta=rng.standard_normal(p)*0.5; groups=[[j] for j in range(p)]
out=[]
def run(off,y,tag,t01=None):
    mW,selK=sg.sgshift_K(Z,Z,y,off,groups,'clf',sg.KnockoffSampler(Z),np.random.default_rng(3),B=8,qs=(0.1,0.2,0.3),pi=0.5)
    if t01 is None: t01=np.zeros(p,int)
    nb=mW[t01==0]
    out.append(f"{tag}: nullW {nb.mean():+.2f} #>0 {int((nb>0).sum())}/{len(nb)} | "+" ".join(
        f"q{q}:nsel{int(selK[q].sum())},fdp{sg.fdp(t01,selK[q]):.2f},pow{sg.power(t01,selK[q]):.2f}" for q in (0.1,0.2,0.3)))
y1=(rng.random(n)<1/(1+np.exp(-(Z@beta)))).astype(float)
run(Z@beta,y1,"T1 exact-offset NO-shift")
base=LogisticRegression(max_iter=300).fit(Z,y1); run(base.decision_function(Z),y1,"T2 est-offset NO-shift")
A=np.sort(rng.choice(p,6,replace=False)); delta=np.zeros(p); delta[A]=rng.choice([-1.,1.],6)*np.random.default_rng(5).uniform(1,2,6)*0.15
y3=(rng.random(n)<1/(1+np.exp(-(Z@beta+Z@delta)))).astype(float); t01=np.zeros(p,int); t01[A]=1
run(Z@beta,y3,"T3 exact-offset WITH-shift(1rep)",t01)
# T4: AVERAGE FDP over many replicates with estimated offset (true FDR test)
fdps={q:[] for q in (0.1,0.2,0.3)}; pows={q:[] for q in (0.1,0.2,0.3)}
for s in range(12):
    rr=np.random.default_rng(100+s)
    Zr=rr.standard_normal((n,p))@Lc.T
    bb=rr.standard_normal(p)*0.5
    ys=(rr.random(n)<1/(1+np.exp(-(Zr@bb)))).astype(float)
    Ar=np.sort(rr.choice(p,6,replace=False)); dl=np.zeros(p); dl[Ar]=rr.choice([-1.,1.],6)*rr.uniform(1,2,6)*0.18
    yt=(rr.random(n)<1/(1+np.exp(-(Zr@bb+Zr@dl)))).astype(float)
    bs=LogisticRegression(max_iter=300).fit(Zr,ys); offr=bs.decision_function(Zr)
    t=np.zeros(p,int); t[Ar]=1
    mW,selK=sg.sgshift_K(Zr,Zr,yt,offr,groups,'clf',sg.KnockoffSampler(Zr),rr,B=6,qs=(0.1,0.2,0.3),pi=0.5)
    for q in (0.1,0.2,0.3): fdps[q].append(sg.fdp(t,selK[q])); pows[q].append(sg.power(t,selK[q]))
out.append("T4 AVG over 12 reps (est-offset,shift): "+" ".join(
    f"q{q}: meanFDP {np.mean(fdps[q]):.3f} meanPow {np.mean(pows[q]):.3f}" for q in (0.1,0.2,0.3)))
open(OUT,'w').write("\n".join(out)+"\n")
print("DONE")
