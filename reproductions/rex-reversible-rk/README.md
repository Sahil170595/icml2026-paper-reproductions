# Rex: A Family of Reversible Exponential (Stochastic) Runge-Kutta Solvers

✅✅🟡🟡🟡  **7 pts** — 2/5 full-credit  (verified, verified, toy, toy, toy)

[arXiv 2502.08834](https://arxiv.org/abs/2502.08834) · [OpenReview](https://openreview.net/forum?id=7pQIzVNctu) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-rex-reversible-rk-repro)

## Scoreboard — measured vs. paper target

| # | Paper claim | Target + acceptance rule | Measured (this repro) | Verdict |
|---|---|---|---|---|
| 1 | Rex = algebraically **reversible exponential (S)RK** for diffusion ODE **and** SDE (Sec. 3) | round-trip reconstruction ≤ **1e-9** (machine precision), h-independent, ≥1e6× below same-order non-reversible control | ODE p=1/2/3: **1.2e-14 / 6.9e-15 / 8.7e-15**; SDE: **3.0e-12**; controls 5.1e-2…1.3e-5 (ratio ≥1.5e9) | **reproduced** |
| 2 | ODE Rex inherits **arbitrary order** + **non-zero linear stability region** from McCallum-Foster (Thm A.1) | fitted order slope = base p (±0.25); MF stability-region area > 0 for ζ<1, ≈0 at ζ=1; exponential stable on stiff | slopes **0.999 / 1.986 / 3.010**; area ζ=1 **0.00** → ζ=0.5 **1.5–2.8**; stiff Rex **8e-15** vs non-exp **blow-up** | **reproduced** |
| 3 | **Near-machine-precision reconstruction** under exact inversion (Fig. 7) | Rex recon ≤ 1e-9, NFE-independent, ≥1e6× below DDIM inversion | Rex **3–6e-14** (flat, NFE 10→100); DDIM **1.8e-1→1.9e-2**; ratio **~1e12** | **reproduced (mechanism)** |
| 4 | **Improves/competitive** vs prior reversible solvers on generation/editing (Figs. 7-9) | proxy: sampling error ≤ DDIM & EDICT at matched NFE; exact inversion | Rex-best/DDIM = **2.1× / 8.4× / 84×** (NFE 48/96/192); recon Rex **4.7e-15** vs DDIM 3.4e-2 vs EDICT 22.1 | **supported (proxy)** |
| 5 | **Accurate likelihood-based Boltzmann sampling** with flow models (Table 1) | proxy: flow log-lik inherits order; reversible density-consistency; correct Boltzmann IS | log-lik err **0.29/0.021/4.3e-5** (rate 1.97); round-trip **3.8e-14**; Boltzmann ESS/N **1.00** (Rex) vs **0.001** (crude) | **supported (proxy)** |

**Claims 1-3** reproduce the paper's core numerical properties (reversibility, order, stability, exact-inversion reconstruction) *exactly* with real CPU runs. **Claims 4-5** are large-scale empirical results (image FID/CLIP; trained tri-alanine Boltzmann generator) that are **not CPU-reproducible**; we reproduce the underlying *solver mechanism* on analytic proxies with real numbers and label them honestly — the FID/CLIP/Table-1 quantities themselves are out of scope.

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`19` files).
