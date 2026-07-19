"""
Claim 3 - Under a shared Transformer architecture, Diffusion Bridge (DB)
outperforms Flow Matching (FM) across image restoration and translation tasks
(Table 1 of arXiv 2509.24531; OpenReview aIFgQusnPy).

CPU-REPRODUCIBILITY: The headline numbers are FID/LPIPS/PSNR/SSIM from training
latent DiT Transformers on CelebA-HQ 256x256 (Table 3: 16-66 GPU-hours per run on
an H20). This is NOT reproducible on a CPU in seconds, and we do NOT fabricate
image metrics. What we DO execute here (real numbers): a faithful re-tabulation of
the paper's reported Table 1 with derived win-counts and mean gaps, to test the
DIRECTION of the claim as stated, plus the theory mechanism (Claim 2) that the
paper offers as the cause of DB's perceptual edge.

RULE: The claim "DB outperforms FM across tasks" is SUPPORTED iff, on the
perceptual metrics the paper emphasises (FID, LPIPS), DB wins in a clear majority
of the task cells; we also report where FM wins (pixel metric SSIM) honestly.
"""
import json, numpy as np

# Paper Table 1 (arXiv 2509.24531). higher-better: PSNR, SSIM; lower-better: LPIPS, FID.
# task -> {FM:[PSNR,SSIM,LPIPS,FID], DB:[...]}
T1 = {
 "Inpaint-Box64": {"FM":[28.03,0.840,0.039,5.13],  "DB":[27.90,0.813,0.038,5.11]},
 "Inpaint-Box128":{"FM":[23.54,0.760,0.106,17.84], "DB":[23.57,0.741,0.078,7.71]},
 "4x-SuperRes":   {"FM":[27.11,0.789,0.088,11.61], "DB":[27.47,0.762,0.077,8.50]},
 "Deblur-15x15":  {"FM":[27.24,0.793,0.088,10.49], "DB":[27.26,0.757,0.087,8.77]},
 "Deblur-61x61":  {"FM":[24.67,0.683,0.228,38.18], "DB":[24.13,0.661,0.172,19.03]},
 "Denoising":     {"FM":[27.02,0.760,0.093,16.04], "DB":[27.30,0.757,0.086,10.16]},
}
metrics = ["PSNR","SSIM","LPIPS","FID"]
higher_better = {"PSNR":True,"SSIM":True,"LPIPS":False,"FID":False}

def db_wins(task, mi):
    fm = T1[task]["FM"][mi]; db = T1[task]["DB"][mi]
    if higher_better[metrics[mi]]:
        return db > fm, db - fm            # positive gap => DB better
    else:
        return db < fm, fm - db            # positive gap => DB better (FM-DB)

wins = {m:0 for m in metrics}
gaps = {m:[] for m in metrics}
percell = []
for task in T1:
    row = {"task":task}
    for mi,m in enumerate(metrics):
        w,g = db_wins(task,mi)
        wins[m]+= int(w); gaps[m].append(g)
        row[m] = {"FM":T1[task]["FM"][mi],"DB":T1[task]["DB"][mi],"DB_better":bool(w),"gap_DBadv":g}
    percell.append(row)

ntask = len(T1)
mean_gap = {m: float(np.mean(gaps[m])) for m in metrics}
# relative FID/LPIPS improvement of DB over FM
rel_fid = float(np.mean([(T1[t]["FM"][3]-T1[t]["DB"][3])/T1[t]["FM"][3] for t in T1]))
rel_lpips = float(np.mean([(T1[t]["FM"][2]-T1[t]["DB"][2])/T1[t]["FM"][2] for t in T1]))

print("="*74)
print("Claim 3  -  DB vs FM across restoration tasks (Table 1 re-tabulation)")
print("arXiv 2509.24531 / OpenReview aIFgQusnPy")
print("NOTE: image metrics come from GPU training on CelebA-HQ (16-66 GPU-h/run,")
print("Table 3); NOT CPU-reproducible. Below: derived stats from the paper's Table 1.")
print("="*74)
print(f"{'task':16s} {'PSNR(F/D)':>15} {'SSIM(F/D)':>15} {'LPIPS(F/D)':>15} {'FID(F/D)':>15}")
for task in T1:
    f=T1[task]["FM"]; d=T1[task]["DB"]
    print(f"{task:16s} {f[0]:6.2f}/{d[0]:<6.2f}   {f[1]:5.3f}/{d[1]:<5.3f}   "
          f"{f[2]:5.3f}/{d[2]:<5.3f}   {f[3]:6.2f}/{d[3]:<6.2f}")
print()
print(f"DB-wins / {ntask} tasks:  FID {wins['FID']}/{ntask}   LPIPS {wins['LPIPS']}/{ntask}   "
      f"SSIM {wins['SSIM']}/{ntask}   PSNR {wins['PSNR']}/{ntask}")
print(f"mean DB-advantage gap:  FID {mean_gap['FID']:+.3f}  LPIPS {mean_gap['LPIPS']:+.4f}  "
      f"SSIM {mean_gap['SSIM']:+.4f}  PSNR {mean_gap['PSNR']:+.3f}")
print(f"mean relative improvement of DB:  FID {100*rel_fid:.1f}%   LPIPS {100*rel_lpips:.1f}%")
print()
perceptual_support = (wins['FID'] >= 5 and wins['LPIPS'] >= 5)
fm_pixel = (wins['SSIM'] <= 1)   # FM wins SSIM in >=5/6
print("FINDINGS (as reported by the paper, re-derived):")
print(f"  * DB wins perceptual metrics (FID {wins['FID']}/6, LPIPS {wins['LPIPS']}/6) -> claim SUPPORTED on FID/LPIPS")
print(f"  * FM wins pixel metric SSIM ({ntask-wins['SSIM']}/6) -> nuance the paper itself states")
print(f"  * mechanism: DB's lower SOC control cost (Claim 2/Thm 4.2) is the paper's")
print(f"    proposed cause of the smoother/more-natural trajectories -> better FID/LPIPS")
print()
print(f"VERDICT (direction of claim on perceptual metrics): SUPPORTED = {perceptual_support}")
print("SCOPE: re-tabulation of paper numbers + theory mechanism; independent CelebA-HQ")
print("       training is out of CPU scope (documented, not fabricated).")
print("="*74)

with open("results.json","w") as f:
    json.dump(dict(source="paper Table 1 (arXiv 2509.24531)",
                   cpu_reproducible=False,
                   reason="latent DiT training on CelebA-HQ, 16-66 GPU-hours/run (Table 3)",
                   ntask=ntask, db_wins=wins, mean_gap_DBadv=mean_gap,
                   mean_rel_improvement_FID=rel_fid, mean_rel_improvement_LPIPS=rel_lpips,
                   per_task=percell,
                   perceptual_claim_supported=bool(perceptual_support),
                   fm_wins_pixel_ssim=bool(fm_pixel)), f, indent=2)
print("wrote results.json")
