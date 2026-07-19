import os,json
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
def kl(p,q): return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
def gap_floor(N,H,D,K,grid=6000):
    g=np.linspace(1e-4,0.49,grid); n=K/N
    val=H*g*K*0.25*np.exp(-n*kl(0.5+g,0.5)/D); i=int(np.argmax(val)); return float(g[i]),float(val[i])
def run_doe_mdp(N,H,D,K,delta,sign_seed,run_seed,c=1.0):
    zeta=np.random.default_rng(sign_seed).choice([-1.0,1.0],size=N)
    rng=np.random.default_rng(run_seed); per=int(np.ceil(K/N)); m=np.zeros(N); n=np.zeros(N); wrong=0.0; gi=np.arange(N); lt=np.log(max(K,2))
    for t in range(per):
        conf=np.where(n>0,np.sqrt(c*D*lt/np.maximum(n,1.0)),1e9)
        commit=(m+conf)>0.0; wrong+=float(np.sum(commit!=(zeta>0)))
        q=0.5+delta*zeta; bit=(rng.random(N)<q).astype(float)
        nuis=rng.binomial(D-1,0.5,size=N) if D>1 else np.zeros(N)
        deb=bit+nuis-(D-1)/2.0-0.5; n[gi]+=1.0; m[gi]+=(deb-m[gi])/n[gi]
    return H*delta*wrong
def ms(N,H,D,K,d,seeds,base,c=1.0): return float(np.mean([run_doe_mdp(N,H,D,K,d,base+7*s,base+1000+s,c) for s in range(seeds)]))
def fit(x,y):
    x=np.log(np.asarray(x,float)); y=np.log(np.asarray(y,float)); b,a=np.polyfit(x,y,1)
    yp=a+b*x; return float(b),float(1-np.sum((y-yp)**2)/np.sum((y-np.mean(y))**2))
SEEDS=16; S0,A0,H0,D0,K0=12,4,4,4,25000
grids={"K":[6250,12500,25000,50000,100000],"S":[5,8,12,18,25,30],"A":[3,4,5,6,8,10],
       "D_max":[1,2,4,8,16,32,64],"H":[2,3,4,5,6,8]}
fbase={"K":1000,"S":2000,"A":3000,"D_max":4000,"H":5000}
res={"baseline":dict(S=S0,A=A0,H=H0,D_max=D0,K=K0,SA_kernels=S0*A0,seeds=SEEDS),"sweeps":{}}
for fac,vals in grids.items():
    uy=[];fy=[];gg=[]
    for v in vals:
        S,A,H,D,K=S0,A0,H0,D0,K0
        if fac=="K":K=v
        elif fac=="S":S=v
        elif fac=="A":A=v
        elif fac=="D_max":D=v
        elif fac=="H":H=v
        N=S*A; g,f=gap_floor(N,H,D,K); u=ms(N,H,D,K,g,SEEDS,fbase[fac]+v)
        uy.append(u);fy.append(f);gg.append(g)
    su,ru=fit(vals,uy); sf,rf=fit(vals,fy); tgt=1.0 if fac=="H" else 0.5
    res["sweeps"][fac]=dict(values=vals,ucb=uy,floor=fy,gap_star=gg,ucb_slope=su,ucb_r2=ru,
        floor_slope=sf,floor_r2=rf,ratio=[float(u/f) for u,f in zip(uy,fy)],target=tgt)
json.dump(res,open("_cache/gates.json","w"),indent=1)
open("_cache/gates.DONE","w").write("ok")
print("DONE gates")
