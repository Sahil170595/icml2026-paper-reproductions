import json, numpy as np
AL=[1.3,1.5,2.0]
adv={a:json.load(open('_cache/adv_%s.json'%a)) for a in AL}
sto={a:json.load(open('_cache/stoch_%s.json'%a)) for a in AL}
H=adv[1.3]['H']; S=adv[1.3]['S']; A=adv[1.3]['A']
print("=== SCALE: H=%d  states/multi-layer S=%d  A=%d  |S|_total=%d (1 start +%d+%d)  layers=%d ==="%(H,S,A,1+2*S,S,S,H))
print("\n=== ADVERSARIAL  Reg_T ~ T^{1/alpha}  (OM known-P, UOB unknown-P) ===")
print("%5s %8s %9s %9s %8s %8s"%("alpha","1/alpha","OM_slope","UOB_slope","mOM","mUOB"))
for a in AL:
    d=adv[a]; print("%5s %8.3f %9.3f %9.3f %8s %8s"%(a,d['target'],d['slope_om'],d['slope_uob'],d['match_om'],d['match_uob']))
mono_om=all(adv[AL[i]]['slope_om']>=adv[AL[i+1]]['slope_om']-0.03 for i in range(2))
mono_uob=all(adv[AL[i]]['slope_uob']>=adv[AL[i+1]]['slope_uob']-0.03 for i in range(2))
print("monotone OM=%s UOB=%s ; horizons=%s C0=%s"%(mono_om,mono_uob,adv[1.3]['horizons'],adv[1.3]['C0']))
print("\n=== STOCHASTIC  OM: O(log T)  UOB: O(log^2 T)  (fixed gap=%.2f, Tmax=%d) ==="%(sto[1.3]['gap'],sto[1.3]['tmax']))
print("%5s | %8s %8s %7s %7s %7s | %8s %8s %7s %7s %7s %6s"%(
  "alpha","OMwin","OMfull","R2log","R2log2","R2sqrt","UOBwin","UOBfull","R2log","R2log2","R2sqrt","falls"))
for a in AL:
    d=sto[a]; ro=d['r2_om']; ru=d['r2_uob']
    print("%5s | %8.3f %8.3f %7.3f %7.3f %7.3f | %8.3f %8.3f %7.3f %7.3f %7.3f %6s"%(
      a,d['slope_om_win'],d['slope_om_full'],ro['r2_log'],ro['r2_log2'],ro['r2_sqrt'],
      d['slope_uob_win'],d['slope_uob_full'],ru['r2_log'],ru['r2_log2'],ru['r2_sqrt'],d['uob_falls']))
print("\n=== CONTROL (bounded-loss, no skipping) dispersion vs skipping-UOB ===")
print("%5s %10s %10s %9s %10s %10s"%("alpha","std_UOB","std_CTL","ratio","uobP90fall","ctlP90fall"))
for a in AL:
    c=sto[a]['ctl']; print("%5s %10.1f %10.1f %9.2f %10s %10s"%(a,c['std_uob'],c['std_ctl'],c['std_ratio'],c['uob_p90_falls'],c['ctl_p90_falls']))
print("\n=== Reg/sqrt(T) UOB (peaks then falls = polylog) ===")
for a in AL:
    print(" a=%s ck=%s"%(a,sto[a]['ckpts']))
    print("      %s"%[round(v,2) for v in sto[a]['regsqrt_uob']])
print("\n=== noise moments E|noise|^a (<=1) ===", {a:round(sto[a]['noise_moment'],3) for a in AL})
out=dict(scale=dict(H=H,S=S,A=A,total_states=1+2*S,layers=H,actions=A,is_bandit=False),
    adversarial={str(a):adv[a] for a in AL}, stochastic={str(a):sto[a] for a in AL},
    adv_mono_om=mono_om, adv_mono_uob=mono_uob)
json.dump(out,open('results.json','w'),indent=1)
print("\n[wrote results.json  bytes=%d]"%len(open('results.json').read()))
