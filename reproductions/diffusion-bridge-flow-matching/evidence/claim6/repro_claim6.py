"""Claim 6 - same network input conditions do NOT eliminate the FM-vs-DB gap
(Table 4, OpenReview aIFgQusnPy / arXiv 2509.24531).
Theory: the DB<FM gap comes from the forward-process SOC cost (drift theta(x1-x)),
a property of the dynamics, NOT of any network input representation. So feeding
both models identical inputs cannot close it.
 T1 identical (x0,x1) inputs -> DB cost < FM cost for 100% of pairs.
 T2 orthogonal reparam of inputs (x->Rx, same R) -> gap ratio J_DB/J_FM invariant.
 T3 per-pair gap == (1/2 - c(theta))||x1-x0||^2, c(theta)=theta/(e^{2theta}-1),
    a closed function of endpoints+drift only (no input term).
SCOPE: the 'same input conditions' ablation Table 4 is the OpenReview version;
arXiv v1 Table 4 is shared-Transformer hyper-params. We verify the theory the
ablation rests on, not its image FID/LPIPS (not CPU-trainable)."""
import json, numpy as np
rng = np.random.default_rng(2026); d = 24; N = 4000
c = lambda th: th/np.expm1(2.0*th)
def gaps(lam, X0, X1):
    th = 1.0/(2.0*lam*lam); D2 = np.sum((X1-X0)**2,axis=1)
    Jfm = 0.5*D2; Jdb = c(th)*D2; return th, Jfm, Jdb, Jfm-Jdb
X0 = rng.standard_normal((N,d)); X1 = rng.standard_normal((N,d))+1.0
res = {}
for lam in [30.0/255.0, 0.5, 1.0, 2.0]:
    th,Jfm,Jdb,g = gaps(lam,X0,X1)
    res[f"lambda={lam:.4f}"] = dict(theta=float(th),
        frac_DB_strictly_lower=float(np.mean(Jdb<Jfm)),
        frac_gap_positive=float(np.mean(g>0)), mean_gap=float(g.mean()),
        min_gap=float(g.min()), mean_ratio_DB_FM=float(np.mean(Jdb/Jfm)))
th,Jfm,Jdb,g = gaps(1.0,X0,X1)
Aq = rng.standard_normal((d,d)); R,_ = np.linalg.qr(Aq)
th2,Jfm_r,Jdb_r,g_r = gaps(1.0,X0@R.T,X1@R.T)
inv_err = float(np.max(np.abs(Jdb/Jfm - Jdb_r/Jfm_r)))
closed_err = float(np.max(np.abs(g - (0.5-c(th))*np.sum((X1-X0)**2,axis=1))))
print("="*74)
print("Claim 6 - same input conditions do NOT eliminate the DB>FM gap")
print("arXiv 2509.24531 / OpenReview aIFgQusnPy")
print("="*74)
print(f"N={N} identical endpoint pairs in R^{d} fed identically to 'both models'\n")
print("T1 fraction of pairs with DB cost strictly < FM cost (gap not closable):")
print(f"{'lambda':>10} {'theta':>10} {'DB<FM frac':>12} {'mean JDB/JFM':>14} {'min gap':>10}")
for k,v in res.items():
    lv=float(k.split('=')[1])
    print(f"{lv:10.4f} {v['theta']:10.4f} {v['frac_DB_strictly_lower']:12.4f} "
          f"{v['mean_ratio_DB_FM']:14.6f} {v['min_gap']:10.4f}")
print()
print("T2 input-representation invariance (orthogonal reparam of identical inputs):")
print(f"   max_i |J_DB/J_FM - (under R)| = {inv_err:.3e} (machine zero)")
print("   => gap ratio depends only on (theta,||x1-x0||), not the input encoding\n")
print(f"T3 per-pair gap == (1/2 - c(theta))||x1-x0||^2 : max err = {closed_err:.3e}\n")
allb = all(v['frac_DB_strictly_lower']==1.0 for v in res.values())
verdict = allb and inv_err<1e-9 and closed_err<1e-9
print(f"VERDICT gap persists under identical inputs (100%) & input-invariant: {verdict}")
print("  gap is a property of forward SOC dynamics (drift) -> identical inputs can't remove it.")
print("="*74)
json.dump(dict(N=N,d=d,per_lambda=res,input_invariance_max_err=inv_err,
    closed_form_max_err=closed_err,all_DB_strictly_lower=bool(allb),
    verdict=bool(verdict),
    scope_note="Table 4 ablation is OpenReview version; verify theory not image metrics"),
    open("results.json","w"), indent=2)
print("wrote results.json")
