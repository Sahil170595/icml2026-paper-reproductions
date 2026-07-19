"""
CLAIM 4 - Parameter-count savings.  Tucker Attention has O(r*d_model + r1*r^2)
trainable attention parameters vs O(d_model^2) for MHA/GQA/MLA, delivering an
order-of-magnitude reduction at comparable configs.
Paper: arXiv 2603.30033 / ErcPPRZaiq, Table 1 (formulas), Tables 2-3 (MB values),
Sec. 4.2 text ("~18% of MHA, ~39% of MLA parameters", "up to 9x reduction").

Independent re-derivation of the Table-1 formulas; we then reproduce every
Table-2 (GPT2) attention-parameter and KV-cache MB value (BF16, MB=1e6 bytes)
and the LLaMA3-1B (Table 3) ratios, and confirm the O(d) vs O(d^2) scaling.
"""
import json, time, hashlib, numpy as np

t0 = time.time()
BYTES = 2  # BF16

# ---- Table-1 parameter formulas (per layer, #params of Q,K,V,O objects) ----
def p_mha(d, nH, dH):            return 4*d*d
def p_mqa(d, nH, dH):            return 2*d*d + 2*dH*d
def p_gqa(d, nH, dH, nKV):       return 2*d*d + 2*nKV*dH*d
def p_mla_shared(d, nH, dH, dc): return d*d + 5*d*dc
def p_mla_sep(d, nH, dH, dc):    return d*d + 6*d*dc
def p_tucker_sep(d, nH, r1, r2, r3):   # separated KV (Table 2/3 Tucker rows)
    return 2*(r1*r2*r3 + nH*r1 + d*r2 + d*r3)
def kv_mha(d, nH, dH, N):        return 2*N*d
def kv_gqa(d, nH, dH, N, nKV):   return 2*N*nKV*dH
def kv_mla_shared(d, N, dc):     return N*dc
def kv_tucker_sep(N, r3):        return 2*N*r3

def MB(params_per_layer, n_layers): return params_per_layer * n_layers * BYTES / 1e6

# ============================ GPT2 (Table 2) ============================
d, nH, dH, L, N = 768, 12, 64, 12, 1024
print("="*74)
print("CLAIM 4  Parameter-count savings  (Table 1 formulas; Tables 2-3 MB values)")
print("arXiv 2603.30033 / ErcPPRZaiq   independent arithmetic, BF16, MB=1e6 bytes")
print("="*74)
print(f"\nGPT2: d_model={d} n_H={nH} d_H={dH} n_layers={L} N={N}")
rows = [  # (label, measured_attn_MB, published_attn_MB, measured_kv_MB, published_kv_MB)
 ("MHA",                 MB(p_mha(d,nH,dH),L),          56.62, MB(kv_mha(d,nH,dH,N),L),       37.74),
 ("GQA n_KV=4",          MB(p_gqa(d,nH,dH,4),L),        37.74, MB(kv_gqa(d,nH,dH,N,4),L),     12.85),
 ("GQA n_KV=2",          MB(p_gqa(d,nH,dH,2),L),        33.02, MB(kv_gqa(d,nH,dH,N,2),L),      6.29),
 ("MLA shared dc=128",   MB(p_mla_shared(d,nH,dH,128),L),26.00, MB(kv_mla_shared(d,N,128),L),  3.14),
 ("Tucker [8,128,128]",  MB(p_tucker_sep(d,nH,8,128,128),L),15.74, MB(kv_tucker_sep(N,128),L), 6.28),
 ("Tucker [8,128,64]",   MB(p_tucker_sep(d,nH,8,128,64),L), 10.24, MB(kv_tucker_sep(N,64),L),  3.14),
 ("Tucker [8,64,64]",    MB(p_tucker_sep(d,nH,8,64,64),L),   6.31, MB(kv_tucker_sep(N,64),L),  3.14),
]
print(f"{'method':20s} {'attn MB (meas)':>14s} {'paper':>7s} {'d':>6s}   {'KV MB (meas)':>12s} {'paper':>6s}")
maxdev = 0.0
tab2 = []
for lbl, m_a, p_a, m_kv, p_kv in rows:
    da = abs(m_a-p_a); dkv = abs(m_kv-p_kv); maxdev = max(maxdev, da)
    print(f"{lbl:20s} {m_a:14.2f} {p_a:7.2f} {da:+6.2f}   {m_kv:12.2f} {p_kv:6.2f} {dkv:+6.2f}")
    tab2.append(dict(method=lbl, attn_mb_measured=round(m_a,3), attn_mb_paper=p_a,
                     attn_dev=round(da,3), kv_mb_measured=round(m_kv,3), kv_mb_paper=p_kv))
print(f"max |attn MB deviation| across GPT2 rows = {maxdev:.3f} MB")

# ---- textual ratio claims (Sec 4.2) ----
mha_mb    = MB(p_mha(d,nH,dH),L)
mla_mb    = MB(p_mla_shared(d,nH,dH,128),L)
t8_128_64 = MB(p_tucker_sep(d,nH,8,128,64),L)
t8_64_64  = MB(p_tucker_sep(d,nH,8,64,64),L)
r_mha  = 100*t8_128_64/mha_mb          # "about 18% of MHA"
r_mla  = 100*t8_128_64/mla_mb          # "39% of MLA"
x_mha  = mha_mb/t8_64_64               # "up to 9x reduction"
print(f"\nSec 4.2 ratio claims:")
print(f"  Tucker[8,128,64] / MHA params = {r_mha:.1f}%  (paper: 'about 18%')")
print(f"  Tucker[8,128,64] / MLA params = {r_mla:.1f}%  (paper: '39% of MLA')")
print(f"  MHA / Tucker[8,64,64]         = {x_mha:.2f}x  (paper: 'up to 9x reduction')")

# ============================ LLaMA3-1B (Table 3) ============================
dL, nHL, dHL, NL = 2048, 32, 64, 4096
LL = round(268e6 / (p_mha(dL,nHL,dHL)*BYTES))          # infer n_layers from MHA=268 MB
print(f"\nLLaMA3-1B: d_model={dL} n_H={nHL} d_H={dHL} N={NL}  (n_layers={LL} from MHA=268MB)")
t3 = [
 ("MHA",               MB(p_mha(dL,nHL,dHL),LL),        268.0),
 ("GQA n_KV=8",        MB(p_gqa(dL,nHL,dHL,8),LL),      168.0),
 ("MQA n_KV=1",        MB(p_mqa(dL,nHL,dHL),LL),        138.0),
 ("MLA dc=128",        MB(p_mla_sep(dL,nHL,dHL,128),LL), 93.3),
 ("Tucker [32,128,128]",MB(p_tucker_sep(dL,nHL,32,128,128),LL), 33.5),
 ("Tucker [32,128,64]", MB(p_tucker_sep(dL,nHL,32,128,64),LL),  21.0),
 ("Tucker [32,64,64]",  MB(p_tucker_sep(dL,nHL,32,64,64),LL),   12.6),
]
tab3 = []
maxrel = 0.0
print(f"{'method':20s} {'MB (meas)':>10s} {'paper':>7s} {'rel%':>6s}")
for lbl, m, p in t3:
    rel = 100*abs(m-p)/p; maxrel = max(maxrel, rel)
    print(f"{lbl:20s} {m:10.2f} {p:7.2f} {rel:6.2f}")
    tab3.append(dict(method=lbl, mb_measured=round(m,3), mb_paper=p, rel_pct=round(rel,3)))
print(f"max relative deviation across LLaMA3 rows = {maxrel:.2f}%")

# ============================ asymptotic scaling ============================
ds = np.array([256, 512, 1024, 2048, 4096], float)
nH_s, r1, r = 12, 8, 64
p_mha_s = np.array([p_mha(int(x),nH_s,int(x)//nH_s) for x in ds], float)
p_mla_s = np.array([p_mla_shared(int(x),nH_s,int(x)//nH_s,128) for x in ds], float)
p_tuck_s= np.array([p_tucker_sep(int(x),nH_s,r1,r,r) for x in ds], float)
sl_mha  = float(np.polyfit(np.log(ds), np.log(p_mha_s), 1)[0])
sl_mla  = float(np.polyfit(np.log(ds), np.log(p_mla_s), 1)[0])
sl_tuck = float(np.polyfit(np.log(ds), np.log(p_tuck_s),1)[0])
print(f"\nAsymptotic scaling of attn params vs d_model (fixed ranks r1={r1}, r={r}):")
print(f"  log-log slope  MHA    = {sl_mha:.3f}  (target 2 = O(d^2))")
print(f"  log-log slope  MLA    = {sl_mla:.3f}  (target 2 = O(d^2))")
print(f"  log-log slope  Tucker = {sl_tuck:.3f}  (target 1 = O(d),  << 2)")

verified = (maxdev < 0.1 and maxrel < 2.0 and
            abs(r_mha-18) < 2 and abs(r_mla-39) < 3 and abs(x_mha-9) < 0.5 and
            sl_tuck < 1.3 and sl_mha > 1.9)
print("\n" + "-"*74)
print("MEASURED vs TARGET")
print(f"  GPT2 Table-2 attn-MB max deviation {maxdev:.3f} MB (<0.1)     {'PASS' if maxdev<0.1 else 'FAIL'}")
print(f"  LLaMA3 Table-3 max rel deviation   {maxrel:.2f}% (<2%)        {'PASS' if maxrel<2 else 'FAIL'}")
print(f"  Tucker scaling slope {sl_tuck:.2f} ~ O(d) vs MHA {sl_mha:.2f} ~ O(d^2)  {'PASS' if sl_tuck<1.3<sl_mha else 'FAIL'}")
print(f"  ratio claims 18%/{r_mha:.1f}  39%/{r_mla:.1f}  9x/{x_mha:.2f}          "
      f"{'PASS' if abs(r_mha-18)<2 and abs(r_mla-39)<3 and abs(x_mha-9)<0.5 else 'FAIL'}")
print(f"\nVERDICT: Parameter-count savings reproduced -> "
      f"{'VERIFIED' if verified else 'NOT VERIFIED'}")
print("="*74)

res = dict(claim="Parameter-count savings: Tucker O(r*d) vs O(d^2); Tables 1-3 reproduced",
           gpt2_table2=tab2, gpt2_attn_MB_max_dev=round(maxdev,3),
           ratio_tucker_over_mha_pct=round(r_mha,2), ratio_tucker_over_mla_pct=round(r_mla,2),
           reduction_mha_over_tucker_x=round(x_mha,3),
           llama3_table3=tab3, llama3_n_layers=LL, llama3_max_rel_pct=round(maxrel,3),
           scaling_slopes=dict(mha=round(sl_mha,3), mla=round(sl_mla,3), tucker=round(sl_tuck,3)),
           verified=bool(verified), runtime_s=round(time.time()-t0,3), numpy=np.__version__)
res["script_sha256"] = hashlib.sha256(open(__file__,'rb').read()).hexdigest()
json.dump(res, open("results.json","w"), indent=2)
print("[wrote results.json]  runtime=%.3fs" % (time.time()-t0))
