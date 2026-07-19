"""
Claim 4 - Diffusion Bridge stays stronger than Flow Matching as the inpainting
mask (distributional discrepancy / task difficulty) increases (Table 2, Fig 3a of
arXiv 2509.24531 / OpenReview aIFgQusnPy).

Two real, executed pieces of evidence:
 (a) FACTUAL: re-tabulate paper Table 2 (FID, LPIPS vs mask side 50..128) and test
     the CLAIM's content -> the FM-minus-DB perceptual gap increases MONOTONICALLY
     with mask size, and correlates with mask area (a discrepancy proxy).
 (b) TOY MECHANISM (exact, CPU): the paper attributes DB's robustness to the drift
     term lowering the SOC cost (Thm 4.2). With g_t=1, theta=1/(2 lambda^2):
        J_FM = (1/2) D^2 ,   J_DB = theta/(e^{2theta}-1) D^2 = c(theta) D^2,
     D = ||x1-x0|| ~ distributional discrepancy.  The ABSOLUTE cost gap
        Delta J(D) = (1/2 - c(theta)) D^2
     grows monotonically with discrepancy D -> DB's advantage widens with
     difficulty, matching the widening Table-2 gap.
"""
import json, numpy as np

# ---- (a) paper Table 2 (full data, 27000 imgs): mask side -> (FM, DB) ----
side = np.array([50, 64, 72, 80, 96, 128], float)
FID  = {"FM":np.array([4.93,5.13,5.43,5.86,8.18,17.84]),
        "DB":np.array([4.93,5.11,5.25,5.34,6.25, 7.71])}
LPIPS= {"FM":np.array([0.035,0.039,0.042,0.047,0.060,0.106]),
        "DB":np.array([0.035,0.038,0.041,0.044,0.052,0.078])}
fid_gap   = FID["FM"] - FID["DB"]      # DB advantage
lpips_gap = LPIPS["FM"] - LPIPS["DB"]
def is_monotone_nondec(a): return bool(np.all(np.diff(a) >= -1e-12))
area = side**2                          # mask area ~ discrepancy proxy
def pearson(a,b):
    a=a-a.mean(); b=b-b.mean(); return float(a@b/np.sqrt((a@a)*(b@b)))
corr_fid = pearson(area, fid_gap)
corr_lpips = pearson(area, lpips_gap)

# ---- (b) toy SOC cost gap vs discrepancy D ----
lam2 = (30.0/255.0)**2                  # paper steady variance level
theta = 1.0/(2.0*lam2)
c_theta = theta/np.expm1(2.0*theta)     # J_DB / D^2
D = np.linspace(0.5, 8.0, 16)           # increasing discrepancy (proxy for mask size)
J_FM = 0.5*D**2
J_DB = c_theta*D**2
gapJ = J_FM - J_DB
mono_toy = is_monotone_nondec(gapJ)
# also a moderate-drift case (lambda=1) so the gap is a visible fraction, not ~=J_FM
c1 = 0.5/np.expm1(1.0)                   # theta=0.5 -> c=theta/(e^{2theta}-1)=0.5/(e-1)
gapJ_lam1 = (0.5 - c1)*D**2

print("="*74)
print("Claim 4  -  DB robustness grows with mask size / distributional discrepancy")
print("arXiv 2509.24531 / OpenReview aIFgQusnPy")
print("="*74)
print("(a) PAPER Table 2 (FID/LPIPS vs mask side); DB advantage = FM - DB")
print(f"{'side':>6} {'area':>7} {'FID_FM':>7} {'FID_DB':>7} {'FIDgap':>7} {'LPIPSgap':>9}")
for i in range(len(side)):
    print(f"{side[i]:6.0f} {area[i]:7.0f} {FID['FM'][i]:7.2f} {FID['DB'][i]:7.2f} "
          f"{fid_gap[i]:7.2f} {lpips_gap[i]:9.3f}")
print(f"  FID gap monotone nondecreasing in mask size : {is_monotone_nondec(fid_gap)}")
print(f"  LPIPS gap monotone nondecreasing            : {is_monotone_nondec(lpips_gap)}")
print(f"  Pearson corr(mask area, FID gap)  = {corr_fid:.4f}")
print(f"  Pearson corr(mask area, LPIPS gap)= {corr_lpips:.4f}")
print(f"  FID gap: Box50 {fid_gap[0]:.2f} -> Box128 {fid_gap[-1]:.2f}  ({fid_gap[-1]/max(fid_gap[1],1e-9):.0f}x from Box64)")
print()
print("(b) TOY SOC cost gap  DeltaJ(D) = (1/2 - c(theta)) D^2  grows with discrepancy D")
print(f"    paper lambda^2={lam2:.4e} -> theta={theta:.3f}, c(theta)=J_DB/D^2={c_theta:.3e}")
print(f"{'D':>6} {'J_FM':>10} {'J_DB(paper th)':>15} {'gap':>10} {'gap(lam=1)':>12}")
for i in [0,4,8,12,15]:
    print(f"{D[i]:6.2f} {J_FM[i]:10.4f} {J_DB[i]:15.3e} {gapJ[i]:10.4f} {gapJ_lam1[i]:12.4f}")
print(f"  toy gap monotone increasing in D (paper theta): {mono_toy}")
print(f"  toy gap monotone increasing in D (lambda=1)    : {is_monotone_nondec(gapJ_lam1)}")
print()
verdict = (is_monotone_nondec(fid_gap) and is_monotone_nondec(lpips_gap)
           and corr_fid > 0.9 and mono_toy)
print(f"VERDICT (DB advantage widens with difficulty): {verdict}")
print("  paper Table-2 gap widens monotonically AND toy SOC gap grows with discrepancy,")
print("  consistent with the drift-term robustness the paper predicts.")
print("SCOPE: (a) is the paper's own reported numbers (image FID/LPIPS not CPU-trainable);")
print("       (b) is an exact toy of the SOC-cost mechanism, not a CelebA reproduction.")
print("="*74)

with open("results.json","w") as f:
    json.dump(dict(side=side.tolist(), area=area.tolist(),
                   fid_fm=FID["FM"].tolist(), fid_db=FID["DB"].tolist(),
                   fid_gap=fid_gap.tolist(), lpips_gap=lpips_gap.tolist(),
                   fid_gap_monotone=is_monotone_nondec(fid_gap),
                   lpips_gap_monotone=is_monotone_nondec(lpips_gap),
                   corr_area_fidgap=corr_fid, corr_area_lpipsgap=corr_lpips,
                   toy_lam2_paper=lam2, toy_theta=theta, toy_c_theta=c_theta,
                   toy_D=D.tolist(), toy_gapJ_paper=gapJ.tolist(),
                   toy_gapJ_lam1=gapJ_lam1.tolist(), toy_gap_monotone=mono_toy,
                   verdict=bool(verdict)), f, indent=2)
print("wrote results.json")
