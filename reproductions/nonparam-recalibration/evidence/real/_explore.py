import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import run_dataset as R
from scipy.special import ndtr

def one(dataset, model, cond, seeds, ncal_over=None):
    cfg = dict(R.CFG[dataset]);
    if ncal_over: cfg.update(ncal_over)
    X,y = R.load_dataset(dataset)
    methods=["raw","kuleshov","song","ckme"]; mets=["condece","ace","crps","ece"]
    acc={m:{mt:[] for mt in mets} for m in methods}
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
            lsc=np.log(sdc); lst=np.log(sdt); locc,loct=muc,mut
        else:
            Pc,Pt=R.fit_predict_rf(Xtr,ytr,[Xca,Xte],cfg["rft"],si)
            Fc,Zc=R.rf_cdf_pit(Pc,yca,grid); Ft,Zt=R.rf_cdf_pit(Pt,yte,grid)
            lsc=np.log(Pc.std(1)+1e-6); lst=np.log(Pt.std(1)+1e-6); locc,loct=Pc.mean(1),Pt.mean(1)
        Zc=np.clip(Zc,1e-6,1-1e-6); Zt=np.clip(Zt,1e-6,1-1e-6)
        # conditioning summary
        if cond=="fullX": sca,ste=Xca,Xte
        elif cond=="mu": sca,ste=locc[:,None],loct[:,None]
        elif cond=="mu_ls": sca,ste=np.c_[locc,lsc],np.c_[loct,lst]
        elif cond=="pca3":
            U,S,Vt=np.linalg.svd(Xtr-Xtr.mean(0),full_matrices=False)
            sca=(Xca)@Vt[:3].T; ste=(Xte)@Vt[:3].T
        s_bin=loct
        # standardize summary by cal stats for bandwidth stability
        cm,cs=sca.mean(0),sca.std(0)+1e-8; sca=(sca-cm)/cs; ste=(ste-cm)/cs
        sub=sca[rng.permutation(sca.shape[0])[:min(400,sca.shape[0])]]
        dd=sub[:,None,:]-sub[None,:,:]; med=np.median(np.sqrt(np.einsum("ijk,ijk->ij",dd,dd))+1e-12)
        h=float(max(med/np.sqrt(2.0)*(sca.shape[0]**(-1.0/(4+sca.shape[1]))),1e-3))
        Rk=R.recal_kuleshov(Zc); Rs,_=R.recal_song(Zc); ck=R.ckme_maps(sca,Zc,ste,h)
        U={"raw":Zt,"kuleshov":Rk(Zt),"song":Rs(Zt),"ckme":np.clip(ck(Zt[:,None])[:,0],1e-6,1-1e-6)}
        Fr={"raw":Ft,"kuleshov":Rk(Ft),"song":Rs(Ft),"ckme":ck(Ft)}
        base=float(R.crps_from_cdf(Ft,yte,grid).mean())+1e-12
        for m in methods:
            u=np.clip(U[m],1e-9,1-1e-9)
            acc[m]["ece"].append(R.pit_ece(u)); acc[m]["condece"].append(R.cond_ece(u,s_bin))
            acc[m]["ace"].append(R.ace_edk(u,s_bin)); acc[m]["crps"].append(float(R.crps_from_cdf(Fr[m],yte,grid).mean())/base)
    print(f"  cond={cond:6s} | " + " | ".join(
        f"{mt}: K={np.mean(acc['kuleshov'][mt]):.4f} S={np.mean(acc['song'][mt]):.4f} CKME={np.mean(acc['ckme'][mt]):.4f}"
        for mt in ["condece","ace","crps"]))

if __name__=="__main__":
    ds,md=sys.argv[1],sys.argv[2]; seeds=int(sys.argv[3]) if len(sys.argv)>3 else 4
    print(f"=== {ds}/{md} (raw shown once) seeds={seeds} ===")
    for cond in ["fullX","mu","mu_ls","pca3"]:
        one(ds,md,cond,seeds)
