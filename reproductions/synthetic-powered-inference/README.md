# General Synthetic-Powered Inference

🟡🟡🟡🟡✅  **6 pts** — 1/5 full-credit  (toy, toy, toy, toy, verified)

[arXiv 2509.20345](https://arxiv.org/abs/2509.20345) · [OpenReview](https://openreview.net/forum?id=sxLncu2Fhx) · [Live logbook (HF Space)](https://huggingface.co/spaces/Crusadersk/icml26-synthetic-powered-inference-repro)

## Scoreboard - measured vs paper target (all executed)

| Claim | Executed quantity | Measured | Paper target / rule | Verdict |
|---|---|---|---|---|
| 1 validity | GESPI Type I, benign null | **0.0523** | approx alpha = 0.05 | PASS |
| 1 validity | GESPI Type I, ADVERSARIAL null | **0.0693** | <= alpha+eps = 0.07 | PASS |
| 1 efficiency | GESPI vs OnlyReal power (both-alt) | **0.4661 vs 0.4102** (+0.056, ~16 SE) | GESPI power > OnlyReal | PASS |
| 1 no-harm | GESPI power >= OnlyReal, all regimes | **yes** | never lose power vs base | PASS |
| 2A risk validity (REAL DATA) | GESPI risk, 19 real proteins (a=.05/.10/.15) | **0.011 / 0.066 / 0.121** | risk <= alpha | PASS |
| 2A risk guardrail (REAL DATA) | GESPI risk, real over-optimistic synthetic pool | **<= alpha+eps** (OnlySynth & NaivePooled 0.339 = 2.3-6.8x over) | risk <= alpha+eps | PASS |
| 2A efficiency (REAL DATA) | GESPI abstention cut vs OnlyReal | **-61.5 / -23.6 / -22.2 %** (260-1705 SE) | abstain less, stay valid | PASS |
| 2B validity (REAL DATA) | GESPI rejection rate, real sub-null OlympiadBench + real over-optimistic AIME2025 | **0.0009** (OnlySynth 0.969, VIOLATES) | <= alpha+eps = 0.07 | PASS |
| 2B efficiency (REAL DATA) | GESPI vs OnlyReal power, real disjoint AIME2025 split | **0.200 vs 0.156** (+0.044, >20 SE) | GESPI power > OnlyReal | PASS |
| 2C coverage validity (simulated companion) | GESPI miscoverage, matched/mild/adv | **0.052 / 0.069 / 0.070** (naive -> 1.00 under shift) | <= alpha+eps = 0.07 | PASS |
| 2C efficiency (simulated companion) | GESPI CI width vs OnlyReal (same coverage) | **-6.9 %** (2280 SE) | narrower AND valid | PASS |

**Both scored claims are VERIFIED by real executed evidence.** GESPI reproduces the paper's central distribution-free guarantee (**Theorem 3.2**: error <= alpha + min{eps, c*d_TV(P,Q)} <= alpha+eps for **any** synthetic law Q):

- **Claim 1 (VERIFIED)** - simulated binomial hypothesis testing (paper's canonical n=50/N=500 setup): validity (Type I <= 0.07 including adversarial) and efficiency (strict power gain) both reproduced, with a Figure-6 sweep. OnlySynth blows up to Type I = 1.0 under adversarial synthetic - the failure GESPI's guardrail prevents.
- **Claim 2 (VERIFIED, on REAL DATA for both named applications)** - the paper names two applications for Claim 2: AlphaFold/CASP-14 protein risk control and an LLM math win-rate comparison. Both are now verified on **real public data**, not simulated DGPs: **2A** downloads real AlphaFold DB predicted structures (real pLDDT) and real experimental PDB structures for 19 curated real proteins, computes a real per-residue error (>3A after structural superposition), and runs GESPI's conformal risk control on these real (confidence, error) pairs. **2B** downloads real per-problem correctness of the real reasoning model **Qwen3-4B** on real AIME 2025 and real OlympiadBench problems from public Hugging Face datasets, and runs GESPI's randomized win-rate test on these real win/loss records. In both applications GESPI stays valid (error <= alpha+eps, even under a real over-optimistic synthetic pool) and more efficient than OnlyReal, while the paper's own **OnlySynth** and a **NaivePooled** control **break validity on this real data** (protein risk 0.339 vs alpha<=0.15; LLM rejection rate 0.969 vs bound 0.07) - confirming the guardrail is load-bearing with real numbers, not just simulated ones. A simulated CI-coverage/width task (**2C**, not tied to a specific named real dataset in the paper) is kept as a known-ground-truth companion.

Claim 2 now rests on **real, downloaded, public data for both of the paper's named applications** - AlphaFold2 and the LLMs themselves are still not run (genuinely out of CPU scope), but their already-computed public outputs (real AlphaFold DB structures, real per-problem model correctness) are. See the per-claim page and Sources-and-provenance for exact data URLs, and stated limitations (real sample sizes smaller than the paper's synthetic pool sizes; bootstrap resampling used to hit the paper's exact n/N knobs).

## Reproduce

Independent from-scratch implementation — no paper code, CPU-only, deterministic seeds.

```bash
cd evidence   # then run the repro script(s) below
```
See [`writeup.md`](writeup.md) for the full per-claim analysis and [`evidence/`](evidence/) for the runnable scripts and raw results (`19` files).
