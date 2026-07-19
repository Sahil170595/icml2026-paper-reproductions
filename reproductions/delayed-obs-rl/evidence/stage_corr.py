import os,sys,json
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
def build_base_mdp(rng,S,A,B,gap):
    P0=np.zeros((S,A,S)); r=np.zeros((S,A))
    for s in range(S):
        sup=rng.choice(S,size=min(B,S),replace=False); base=rng.dirichlet(np.ones(len(sup))); br=rng.uniform(0.2,0.8)
        for a in range(A):
            pert=rng.dirichlet(np.ones(len(sup))); mix=(1-gap)*base+gap*pert; P0[s,a,sup]=mix/mix.sum()
            r[s,a]=np.clip(br+gap*rng.uniform(-1,1),0,1)
    return P0,r
def build_aug(S,A,d):
    AD=A**d; M=S*AD; m=np.arange(M); x_of=m//AD; buf=m%AD; Adm1=A**(d-1); b1=buf//Adm1; rest=buf%Adm1
    NEXT=[(np.arange(S)[None,:]*AD+(rest*A+a)[:,None]).astype(np.int64) for a in range(A)]
    return dict(AD=AD,M=M,x_of=x_of,b1_of=b1,rest_of=rest,NEXT=NEXT,Adm1=Adm1)
def vi_opt(aug,P0,r,H,bonus=None):
    x=aug["x_of"];b1=aug["b1_of"];NEXT=aug["NEXT"];A=len(NEXT); Pmat=P0[x,b1,:]; Rv=r[x,b1]+(bonus[x,b1] if bonus is not None else 0.0)
    V=np.zeros(aug["M"]); pol=np.zeros((H,aug["M"]),dtype=np.int64)
    for h in range(H-1,-1,-1):
        Qs=np.empty((A,aug["M"]))
        for a in range(A): Qs[a]=Rv+(V[NEXT[a]]*Pmat).sum(1)
        pol[h]=Qs.argmax(0); V=Qs.max(0)
        if bonus is not None: np.clip(V,0,H,out=V)
    return pol,V
def peval(aug,P0,r,pol,H):
    x=aug["x_of"];b1=aug["b1_of"];rest=aug["rest_of"];AD=aug["AD"];A=len(aug["NEXT"]); Pmat=P0[x,b1,:];Rv=r[x,b1];S=P0.shape[0];xr=np.arange(S);V=np.zeros(aug["M"])
    for h in range(H-1,-1,-1):
        a=pol[h];idx=xr[None,:]*AD+(rest*A+a)[:,None];V=Rv+(V[idx]*Pmat).sum(1)
    return V
def run_pooled(rng,S,A,d,H,K,B,c,gap,rec=0,control=False):
    P0,r=build_base_mdp(rng,S,A,B,gap);aug=build_aug(S,A,d);m0=0; AD=aug["AD"];rest=aug["rest_of"];Adm1=aug["Adm1"]
    Vstar=vi_opt(aug,P0,r,H)[1][m0]; N=np.zeros((S,A));Ntr=np.zeros((S,A,S));Rs=np.zeros((S,A));Kf=float(S*A*K);creg=0.0;ks=[];cum=[]
    for k in range(1,K+1):
        Ne=np.maximum(N,1.0);P0h=Ntr/Ne[:,:,None];P0h[N==0]=1.0/S; rh=np.where(N>0,Rs/Ne,0.5);bon=c*np.sqrt(np.log(Kf+1)/Ne)
        pol=(rng.integers(0,A,size=(H,aug["M"])) if control else vi_opt(aug,P0h,rh,H,bon)[0])
        creg+=(Vstar-peval(aug,P0,r,pol,H)[m0]); m=m0
        for h in range(H):
            a=int(pol[h][m]);x=m//AD;b1=(m%AD)//Adm1;xp=rng.choice(S,p=P0[x,b1]); N[x,b1]+=1;Ntr[x,b1,xp]+=1;Rs[x,b1]+=r[x,b1];m=xp*AD+(rest[m]*A+a)
        if rec and k%rec==0: ks.append(k);cum.append(creg)
    return (np.array(ks),np.array(cum)) if rec else creg
def fit(x,y):
    x=np.log(np.asarray(x,float)); y=np.log(np.asarray(y,float)); b,a=np.polyfit(x,y,1)
    return float(b),float(1-np.sum((y-(a+b*x))**2)/np.sum((y-np.mean(y))**2))
mode=sys.argv[1]
if mode=="rate":
    Sc,Ac,dc,Hc,Bc,cc,gapc,Kc=6,3,2,4,3,0.5,0.12,6000; rec=50; cum=[];ctl=[];ks=None
    for sd in range(3):
        ks,cu=run_pooled(np.random.default_rng(7000+sd),Sc,Ac,dc,Hc,Kc,Bc,cc,gapc,rec=rec); cum.append(cu)
        _,cl=run_pooled(np.random.default_rng(9000+sd),Sc,Ac,dc,Hc,Kc,Bc,cc,gapc,rec=rec,control=True); ctl.append(cl)
    cm=np.mean(cum,0); cl=np.mean(ctl,0); msk=ks>400
    sK=fit(ks[msk],cm[msk])[0]; sKc=fit(ks[msk],cl[msk])[0]
    json.dump(dict(S=Sc,A=Ac,delay=dc,H=Hc,K=Kc,seeds=3,slope_ucbvi=sK,slope_control=sKc,
        R_final_ucbvi=float(cm[-1]),R_final_control=float(cl[-1]),
        checkpoints=[int(v) for v in ks[::4]],cum_ucbvi=[float(v) for v in cm[::4]],cum_control=[float(v) for v in cl[::4]]),
        open("_cache/corr_rate.json","w"),indent=1)
    open("_cache/corr_rate.DONE","w").write("ok"); print("rate slope_ucbvi=%.3f control=%.3f Rf=%.1f/%.1f"%(sK,sKc,cm[-1],cl[-1]))
elif mode=="hardeasy":
    Sc,Ac,dc,Hc,Bc,cc=6,3,2,4,3,0.5; Kl=[1500,4500,13500]
    hard=[float(np.mean([run_pooled(np.random.default_rng(300+sd),Sc,Ac,dc,Hc,K,Bc,cc,2.2/np.sqrt(K)) for sd in range(3)])) for K in Kl]
    easy=[float(np.mean([run_pooled(np.random.default_rng(600+sd),Sc,Ac,dc,Hc,K,Bc,cc,0.45) for sd in range(3)])) for K in Kl]
    json.dump(dict(S=Sc,A=Ac,delay=dc,H=Hc,K_list=Kl,hard_R=hard,easy_R=easy,
        slope_hard=fit(Kl,hard)[0],slope_easy=fit(Kl,easy)[0]),open("_cache/corr_he.json","w"),indent=1)
    open("_cache/corr_he.DONE","w").write("ok"); print("hard slope=%.3f easy slope=%.3f"%(fit(Kl,hard)[0],fit(Kl,easy)[0]))
