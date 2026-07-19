import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import run_dataset as R

LAM=0.2
def run(dataset, model, seeds, conds):
    cfg=R.CFG[dataset]; X,y=R.load_dataset(dataset)
    acc={c:{m:{k:[] for k in ["condece","ace","crps"]} for m in ["kuleshov","song","ckme"]} for c in conds}
    rawcond=[]
    for si in range(seeds):
        rng=np.random.default_rng(7000+101*si); n=len(y); pool=rng.permutation(n)[:cfg["nsub"]]
        ntr,nca,nte=cfg["ntr"],cfg["ncal"],cfg["ntest"]
        itr=pool[:ntr]; ica=pool[ntr:ntr+nca]; ite=pool[ntr+nca:ntr+nca+nte]
        xm,xs=X[itr].mean(0),X[itr].std(0)+1e-8; ym,ys=y[itr].mean(),y[itr].std()+1e-8
        Xtr=(X[itr]-xm)/xs; Xca=(X[ica]-xm)/xs; Xte=(X[ite]-xm)/xs
        ytr=(y[itr]-ym)/ys; yca=(y[ica]-ym)/ys; yte=(y[ite]-ym)/ys
        grid=R.make_grid(np.concatenate([ytr,yca,yte]))
        if model=="gbm":
            (muc,sdc),(mut,sdt)=R.fit_predict_gbm(Xtr,ytr,[Xca,Xte],cfg["gbt"],si)
            Fc,Zc=R.gbm_cdf_pit(muc,sdc,yca,grid); Ft,Zt=R.gbm_cdf_pit(mut,sdt,yte,grid)
            lsc,lst=np.log(sdc),np.log(sdt); locc,loct=muc,mut
        else:
            Pc,Pt=R.fit_predict_rf(Xtr,ytr,[Xca,Xte],cfg["rft"],si)
            Fc,Zc=R.rf_cdf_pit(Pc,yca,grid); Ft,Zt=R.rf_cdf_pit(Pt,yte,grid)
            lsc,lst=np.log(Pc.std(1)+1e-6),np.log(Pt.std(1)+1e-6); locc,loct=Pc.mean(1),Pt.mean(1)
        Zc=np.clip(Zc,1e-6,1-1e-6); Zt=np.clip(Zt,1e-6,1-1e-6); s_bin=loct
        base=float(R.crps_from_cdf(Ft,yte,grid).mean())+1e-12
        Rk=R.recal_kuleshov(Zc); Rs,_=R.recal_song(Zc)
        rawcond.append(R.cond_ece(Zt,s_bin))
        U,S,Vt=np.linalg.svd(Xtr-Xtr.mean(0),full_matrices=False)
        for c in conds:
            if c=="fullX": sca,ste=Xca,Xte
            elif c=="mu": sca,ste=locc[:,None],loct[:,None]
            elif c=="mu_ls": sca,ste=np.c_[locc,lsc],np.c_[loct,lst]
            elif c=="mu_pc2": sca,ste=np.c_[locc,Xca@Vt[:2].T],np.c_[loct,Xte@Vt[:2].T]
            cm,cs=sca.mean(0),sca.std(0)+1e-8; sca=(sca-cm)/cs; ste=(ste-cm)/cs
            sub=sca[rng.permutation(sca.shape[0])[:min(400,sca.shape[0])]]
            dd=sub[:,None,:]-sub[None,:,:]; med=np.median(np.sqrt(np.einsum("ijk,ijk->ij",dd,dd))+1e-12)
            h=float(max(med/np.sqrt(2.0)*(sca.shape[0]**(-1.0/(4+sca.shape[1]))),1e-3))
            ck=R.ckme_maps(sca,Zc,ste,h,lam=LAM)
            for m,Umap,Fmap in [("kuleshov",Rk(Zt),Rk(Ft)),("song",Rs(Zt),Rs(Ft)),
                                ("ckme",np.clip(ck(Zt[:,None])[:,0],1e-6,1-1e-6),ck(Ft))]:
                acc[c][m]["condece"].append(R.cond_ece(Umap,s_bin))
                acc[c][m]["ace"].append(R.ace_edk(Umap,s_bin))
                acc[c][m]["crps"].append(float(R.crps_from_cdf(Fmap,yte,grid).mean())/base)
    print(f"=== {dataset}/{model} seeds={seeds}  raw condece={np.mean(rawcond):.4f} ===")
    for c in conds:
        print(f"  {c:7s} | " + " | ".join(
            f"{mt}: K={np.mean(acc[c]['kuleshov'][mt]):.4f} S={np.mean(acc[c]['song'][mt]):.4f} CKME={np.mean(acc[c]['ckme'][mt]):.4f}"
            for mt in ["condece","ace","crps"]))

if __name__=="__main__":
    ds,md=sys.argv[1],sys.argv[2]; seeds=int(sys.argv[3]); conds=sys.argv[4].split(",")
    LAM=float(sys.argv[5]) if len(sys.argv)>5 else 0.2
    print(f"[lam={LAM}]")
    run(ds,md,seeds,conds)
