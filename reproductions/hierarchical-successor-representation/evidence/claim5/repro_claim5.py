"""
Claim 5 (Sec 4.3, Fig 5): HSR's temporally-extended structure enables more
efficient / SCALABLE exploration than standard SR: HSR-augmented agents cover a
larger fraction of the state space within a fixed budget, and the gap GROWS with
maze size (SR coverage degrades in larger mazes; Fig 5b,c).
Paper: arXiv 2602.12753 / OpenReview txswvMHt4u (Yu & Lengyel).

Independent build: count/novelty-driven (SPIE-style r ~ 1/visit, a proxy for the
SR row-norm intrinsic reward) softmax exploration in four-room mazes of growing
size. 'SR' agent = primitive actions only (single-step, diffusive). 'HSR' agent =
primitives + 8 eigenoptions (temporally-extended jumps escaping local barriers).
Fixed step budget = 2*N; coverage = fraction of states visited. Honest verdict.
"""
import json,time,numpy as np
import hsr_core as H
def explore(env, opts, budget, use_options, seed, temp=0.5):
    rng=np.random.default_rng(seed); N=env['N']; P=env['P']
    visit=np.ones(N); s=int(rng.integers(N)); covered={s}; steps=0
    while steps<budget:
        cands=[('p',a,int(np.argmax(P[a,s]))) for a in range(4)]
        if use_options:
            for k,o in enumerate(opts): cands.append(('o',k,int(np.argmax(o['F'][s]))))
        nov=np.array([1.0/visit[c[2]] for c in cands])
        w=np.exp((nov-nov.max())/temp); w/=w.sum()
        c=cands[int(rng.choice(len(cands),p=w))]
        if c[0]=='p':
            s=int(np.argmax(P[c[1],s])); visit[s]+=1; covered.add(s); steps+=1
        else:
            o=opts[c[1]]
            for _ in range(40):
                a=o['pol'][s]; s=int(np.argmax(P[a,s])); visit[s]+=1; covered.add(s); steps+=1
                if steps>=budget or rng.random()<o['beta'][s]: break
    return len(covered)/N
def main():
    t0=time.time(); gamma=0.99; SEEDS=8
    print("="*74)
    print("Claim 5 - Scalable exploration: HSR(options) vs SR(primitive) coverage (Fig 5)")
    print("arXiv 2602.12753 / OpenReview txswvMHt4u - independent NumPy repro")
    print("="*74)
    print(f"budget=2*N, softmax novelty (temp 0.5), {SEEDS} seeds; coverage=frac states visited")
    print("  size    N   SR-cov(prim)   HSR-cov(opt)     gap")
    rows=[]
    for size in [9,11,13,15,17]:
        env=H.four_room(size); N=env['N']
        opts,_=H.eigenoptions(env,gamma,K=8)
        for o in opts: o['F']=H.termination_kernel(o['P'],o['beta'],gamma)
        budget=2*N
        sr=[explore(env,opts,budget,False,s) for s in range(SEEDS)]
        hs=[explore(env,opts,budget,True,s) for s in range(SEEDS)]
        srm,hsm=float(np.mean(sr)),float(np.mean(hs))
        rows.append(dict(size=size,N=N,SR_cov=srm,HSR_cov=hsm,gap=hsm-srm))
        print(f"  {size:4d} {N:4d}    {srm:.3f}         {hsm:.3f}       {hsm-srm:+.3f}")
    gaps=[r['gap'] for r in rows]; sizes=[r['N'] for r in rows]
    mean_gap=float(np.mean(gaps)); slope=float(np.polyfit(sizes,gaps,1)[0])
    hsr_helps=mean_gap>0.02; gap_grows=slope>0
    if hsr_helps and gap_grows: verdict="reproduced"
    elif hsr_helps: verdict="partial: HSR improves coverage but gap does NOT grow with maze size (opposite Fig 5c)"
    else: verdict="not_reproduced: no consistent coverage advantage"
    print(f"\n  mean coverage gap (HSR-SR) over sizes = {mean_gap:+.3f}")
    print(f"  gap-vs-size slope = {slope:+.2e}  (paper Fig 5c: gap GROWS, slope>0)")
    print(f"\n  VERDICT: {verdict}")
    print("="*74)
    rt=time.time()-t0
    js=dict(claim="HSR (temporally-extended) enables scalable exploration: higher coverage than SR, gap grows with maze size (Fig 5)",
        paper="arXiv 2602.12753 / OpenReview txswvMHt4u", gamma=gamma, seeds=SEEDS, budget_rule="2*N",
        per_size=rows, mean_gap=mean_gap, gap_vs_size_slope=slope,
        HSR_improves_coverage=bool(hsr_helps), gap_grows_with_size=bool(gap_grows),
        overall_reproduced=bool(hsr_helps and gap_grows), verdict=verdict, runtime_s=rt)
    json.dump(js,open("results.json","w"),indent=2)
    print(f"runtime={rt:.2f}s ; wrote results.json")
if __name__=="__main__": main()
