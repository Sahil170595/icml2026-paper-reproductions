# ICML 2026 Paper Reproductions

Independent, from-scratch reproductions of **48 ICML-2026 submissions** for the [ICML 2026 Agent Reproducibility Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro). Each claim is re-derived from the paper text, with deterministic implementations and machine-checked evidence; where official code exists it is pinned at an audited commit SHA rather than trusted.

> **297 points across 48 papers**, final rank **#26 of 1,221 participants** — top 2.1% (challenge closed 2026-08-02, verdicts frozen; peaked #8 mid-challenge, July 2026) · of **249 official scored claims**: **118 verified**, **3 independently falsified**, 55 toy-scale, 73 inconclusive · judged by the challenge's independent LLM referee against each paper's official scored claims.
>
> Final figures. The challenge closed **2026-08-02** (23:59 AoE); these 48 logbooks were judged between 2026-07-23 and 2026-08-03. Counts and points above are re-derived from the challenge's published [verdicts dataset](https://huggingface.co/datasets/ICML-2026-agent-repro/verdicts), snapshot pulled **2026-08-08** (UTC).

Each reproduction is a runnable evidence package: the claim is stated with the paper's own target, an acceptance rule and a falsification condition are fixed **in advance**, and a self-contained script produces the measured numbers. Verdicts are honest — a reproduction that *fails* a published claim is reported as **falsified**, not buried.

## Highlights

- **Independent falsifications.** Three claims did not hold under faithful re-implementation and are reported as such: [Hierarchical Successor Representation](reproductions/hierarchical-successor-representation/) claims 2 and 4 — the four-room transfer advantage (HSR 52.0 vs SR 54.5 episodes-to-optimal, p=0.854, against a claimed p=0.008) and the NMF sparsity/bottleneck structure (Gini 0.624 vs eSR 0.699, bottleneck ratio 1.01) — and [Nonparametric Distribution Regression Re-calibration](reproductions/nonparam-recalibration/) claim 5, where CKME's auto-calibration acceptance came out lowest among the compared methods rather than best.
- **Full-credit reproductions.** [Best-of-Both-Worlds for Heavy-Tailed MDPs](reproductions/heavytail-mdp-bobw/) and [M-Convex Inverse Linear Optimization](reproductions/mconvex-inverse-linear-regret/) — 5/5 scored claims verified — and [Foundations of Equivariant Deep Learning](reproductions/equivariant-deep-learning/) at 4/4.
- **Theory reproduced numerically.** Convergence rates, spectral bounds and self-adjointness identities checked to machine precision — e.g. [Row-Stochastic vs Doubly-Stochastic mixing](reproductions/row-stochastic-decentralized/) (3/4 verified), whose weighted detailed-balance residual lands at 1.1e-16.

## Methodology

See **[METHODOLOGY.md](METHODOLOGY.md)**. In short: pin the paper + official scored claims → find/pin official code at an audited SHA (or transcribe the equations) → build a deterministic reproduction → fix acceptance/falsification rules before running → record every command, exit code and output hash → assemble an auditable logbook. Non-determinism, scope limits and proxy scales are stated openly on every claim.

The pipeline is an autonomous multi-provider agent harness: provider-neutral producer agents (GPT + Claude) fan out under a per-paper evidence contract — manifest, claims with pre-fixed acceptance/falsification rules, command log with exit codes, evidence bundle. An independent reviewer agent gates each bundle, and a root coordinator validates before publication — no human-intervention loops.

## Results

Verdicts below are the challenge referee's final per-claim rulings, one mark per official scored claim, in claim order.

✅ verified (2 pts) · 🔴 falsified (2 pts) · 🟡 toy-scale (1 pt) · ⚪ inconclusive (0 pts)

**Totals: 249 claims — 118 ✅ · 3 🔴 · 55 🟡 · 73 ⚪ — 297 points across 48 papers.**

| Paper | Claims | Pts | Links |
|---|---|--:|---|
| [Causal Modeling of Selection in Evolution](reproductions/causal-selection-evolution/) | ✅✅✅✅✅🟡 | 11 | [arXiv](https://arxiv.org/abs/2606.05689) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-causal-selection-evolution-repro) |
| [Best-of-Both-Worlds for Heavy-Tailed Markov Decision Processes](reproductions/heavytail-mdp-bobw/) | ✅✅✅✅✅ | 10 | [arXiv](https://arxiv.org/abs/2602.01295) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-heavytail-mdp-bobw-repro) |
| [Finite and Corruption-Robust Regret Bounds in Online Inverse Linear Optimization under M-Convex Action Sets](reproductions/mconvex-inverse-linear-regret/) | ✅✅✅✅✅ | 10 | [arXiv](https://arxiv.org/abs/2602.01682) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-mconvex-inverse-linear-regret-repro) |
| [Tucker Attention: A generalization of approximate attention mechanisms](reproductions/tucker-attention/) | ✅✅✅✅✅⚪ | 10 | [arXiv](https://arxiv.org/abs/2603.30033) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-tucker-attention-repro) |
| [A Random Matrix Perspective on the Consistency of Diffusion Models](reproductions/rmt-diffusion-consistency/) | 🟡✅✅✅🟡 | 8 | [arXiv](https://arxiv.org/abs/2602.02908) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-rmt-diffusion-consistency-repro) |
| [Follow-the-Perturbed-Leader for Decoupled Bandits: Best-of-Both-Worlds and Practicality](reproductions/ftpl-decoupled-bandits/) | ✅✅✅✅⚪⚪ | 8 | [arXiv](https://arxiv.org/abs/2510.12152) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-ftpl-decoupled-bandits-repro) |
| [Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks](reproductions/equivariant-deep-learning/) | ✅✅✅✅ | 8 | [logbook](https://huggingface.co/spaces/Crusadersk/icml26-equivariant-deep-learning-repro) |
| [High-accuracy sampling for diffusion models and log-concave distributions](reproductions/highaccuracy-sampling-logconcave/) | ✅✅🟡🟡✅ | 8 | [arXiv](https://arxiv.org/abs/2602.01338) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-highaccuracy-sampling-logconcave-repro) |
| [Maximum Likelihood Reinforcement Learning](reproductions/maxlik-rl/) | ✅✅✅🟡🟡 | 8 | [arXiv](https://arxiv.org/abs/2602.02710) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-maxlik-rl-repro) |
| [NeuralFLoC: Neural Flow-Based Joint Registration and Clustering of Functional Data](reproductions/neuralfloc-registration-clustering/) | ✅✅🟡🟡✅ | 8 | [arXiv](https://arxiv.org/abs/2602.03169) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-neuralfloc-registration-clustering-repro) |
| [On Structured State-Space Duality](reproductions/structured-state-space-duality/) | ✅✅✅⚪✅ | 8 | [arXiv](https://arxiv.org/abs/2510.04944) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-structured-state-space-duality-repro) |
| [Optimal Unconstrained Self-Distillation in Ridge Regression: Strict Improvements, Precise Asymptotics, and One-Shot Tuning](reproductions/ridge-self-distillation/) | ✅✅⚪✅✅⚪ | 8 | [arXiv](https://arxiv.org/abs/2602.17565) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-ridge-self-distillation-repro) |
| [Quantifying LLM Attention-Head Stability: Implications for Circuit Universality](reproductions/attention-head-stability/) | ✅⚪✅✅⚪✅ | 8 | [arXiv](https://arxiv.org/abs/2602.16740) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-attention-head-stability-repro) |
| [Towards Understanding Adam Convergence on Highly Degenerate Polynomials](reproductions/adam/) | ✅✅✅✅⚪⚪ | 8 | [arXiv](https://arxiv.org/abs/2603.09581) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-adam-degenerate-repro) |
| [Diffusion Bridge or Flow Matching? A Unifying Framework and Comparative Analysis](reproductions/diffusion-bridge-flow-matching/) | ✅✅⚪🟡🟡🟡 | 7 | [arXiv](https://arxiv.org/abs/2509.24531) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-diffusion-bridge-flow-matching-repro) |
| [Hierarchical Successor Representation for Robust Transfer](reproductions/hierarchical-successor-representation/) | ✅🔴⚪🔴🟡 | 7 | [arXiv](https://arxiv.org/abs/2602.12753) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-hierarchical-successor-representation-repro) |
| [Rex: A Family of Reversible Exponential (Stochastic) Runge-Kutta Solvers](reproductions/rex-reversible-rk/) | ✅✅🟡🟡🟡 | 7 | [arXiv](https://arxiv.org/abs/2502.08834) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-rex-reversible-rk-repro) |
| [Row-Stochastic Matrices Can Provably Outperform Doubly Stochastic Matrices in Decentralized Learning](reproductions/row-stochastic-decentralized/) | ✅✅✅🟡 | 7 | [arXiv](https://arxiv.org/abs/2511.19513) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-row-stochastic-decentralized-repro) |
| [Accelerated and Stable Convergence with Anchored Generalized Optimistic Method](reproductions/anchored-optimistic-goma/) | ✅⚪✅✅ | 6 | [arXiv](https://arxiv.org/abs/2606.21528) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-anchored-optimistic-goma-repro) |
| [Attention's forward pass and Frank-Wolfe](reproductions/frank-wolfe-attention/) | ✅✅⚪✅⚪ | 6 | [arXiv](https://arxiv.org/abs/2508.09628) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-frank-wolfe-attention-repro) |
| [Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling](reproductions/llm-as-judge-scaling/) | ✅✅✅⚪⚪ | 6 | [arXiv](https://arxiv.org/abs/2512.19905) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-llm-as-judge-scaling-repro) |
| [Explaining Concept Shift with Interpretable Feature Attribution](reproductions/concept-shift-sgshift/) | ✅⚪✅⚪✅ | 6 | [arXiv](https://arxiv.org/abs/2505.20634) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-concept-shift-sgshift-repro) |
| [General Synthetic-Powered Inference](reproductions/synthetic-powered-inference/) | 🟡🟡🟡🟡✅ | 6 | [arXiv](https://arxiv.org/abs/2509.20345) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-synthetic-powered-inference-repro) |
| [Minimax Optimal Strategy for Delayed Observations in Online Reinforcement Learning](reproductions/delayed-obs-rl/) | ✅⚪✅✅⚪ | 6 | [arXiv](https://arxiv.org/abs/2603.03480) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-delayed-obs-rl-repro) |
| [Minimum Distance Summaries for Robust Neural Posterior Estimation](reproductions/minimum-distance-summaries/) | ✅✅🟡🟡⚪⚪ | 6 | [arXiv](https://arxiv.org/abs/2602.09161) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-minimum-distance-summaries-repro) |
| [Model Fusion via Retrofitting](reproductions/model-fusion/) | ✅🟡✅⚪⚪🟡 | 6 | [arXiv](https://arxiv.org/abs/2507.00037) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-pilot-model-fusion) |
| [Parametrized Power-Iteration Clustering for Directed Graphs](reproductions/power-iteration-clustering-digraph/) | ✅✅🟡⚪⚪🟡 | 6 | [arXiv](https://arxiv.org/abs/2210.00310) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-power-iteration-clustering-digraph-repro) |
| [Prior Diffusiveness and Regret in the Linear-Gaussian Bandit](reproductions/bandit/) | ✅✅⚪✅⚪ | 6 | [arXiv](https://arxiv.org/abs/2601.02022) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-bandit-prior-repro) |
| [Procedural Generation Of Algorithm Discovery Tasks in Machine Learning](reproductions/discogen/) | ✅⚪✅✅⚪ | 6 | [arXiv](https://arxiv.org/abs/2603.17863) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-pilot-discogen) |
| [Stochastic Linear Bandits with Parameter Noise](reproductions/linbandit-paramnoise/) | ✅⚪✅✅ | 6 | [arXiv](https://arxiv.org/abs/2601.23164) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-linbandit-paramnoise-repro) |
| [Beyond Softmax: A Natural Parameterization for Categorical Random Variables](reproductions/catnat-beyond-softmax/) | ✅✅⚪🟡⚪ | 5 | [arXiv](https://arxiv.org/abs/2509.24728) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-catnat-beyond-softmax-repro) |
| [DAVE: Distribution-Aware Attribution via ViT Gradient Decomposition](reproductions/dave-vit-attribution/) | ✅✅⚪⚪🟡⚪ | 5 | [arXiv](https://arxiv.org/abs/2602.06613) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-dave-vit-attribution-repro) |
| [Design-Based Anytime-Valid Inference for Randomized Experiments with Delayed Outcomes and Staggered Entry](reproductions/anytime-valid-delayed/) | ✅✅⚪⚪🟡 | 5 | [arXiv](https://arxiv.org/abs/2603.25971) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-anytime-valid-delayed-repro) |
| [Full-Batch Gradient Descent Outperforms One-Pass SGD: Sample Complexity Separation in Single-Index Learning](reproductions/fullbatch-single-index/) | ✅🟡🟡⚪🟡 | 5 | [arXiv](https://arxiv.org/abs/2602.02431) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-fullbatch-single-index-repro) |
| [Improved Analysis of the Accelerated Noisy Power Method with Applications to Decentralized PCA](reproductions/noisy-power-method/) | ✅⚪✅⚪🟡 | 5 | [arXiv](https://arxiv.org/abs/2602.03682) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-noisy-power-method-repro) |
| [Nonparametric Distribution Regression Re-calibration](reproductions/nonparam-recalibration/) | ⚪⚪✅🟡🔴 | 5 | [arXiv](https://arxiv.org/abs/2602.13362) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-nonparam-recalibration-repro) |
| [On Regret Bounds of Thompson Sampling for Bayesian Optimization](reproductions/ts-bayesopt/) | ✅🟡🟡🟡 | 5 | [arXiv](https://arxiv.org/abs/2603.09276) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-ts-bayesopt-repro) |
| [The Theory and Practice of MAP Inference over Non-Convex Constraints](reproductions/map-inference-nonconvex/) | ✅✅🟡⚪⚪⚪ | 5 | [arXiv](https://arxiv.org/abs/2602.08681) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-map-inference-nonconvex-repro) |
| [Estimating Continuous Treatment Effects with Two-Stage Kernel Ridge Regression](reproductions/treatment-effects-krr/) | ✅⚪✅⚪⚪ | 4 | [arXiv](https://arxiv.org/abs/2604.13410) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-treatment-effects-krr-repro) |
| [Fixed Budget is No Harder Than Fixed Confidence in Best-Arm Identification up to Logarithmic Factors](reproductions/bai-budget-confidence/) | ✅✅⚪⚪⚪ | 4 | [arXiv](https://arxiv.org/abs/2602.03972) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-bai-budget-confidence-repro) |
| [Improved Convergence Analysis of Topology Dependence in Decentralized SGD](reproductions/decentralized-sgd-topology/) | 🟡⚪⚪✅🟡⚪ | 4 | [arXiv](https://arxiv.org/abs/2606.09154) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-decentralized-sgd-topology-repro) |
| [Keep Everyone Happy: Online Fair Division of Numerous Items with Few Copies](reproductions/fair-division/) | ✅✅⚪⚪⚪ | 4 | [arXiv](https://arxiv.org/abs/2408.12845) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-fair-division-repro) |
| [Online Packet Scheduling with Deadlines and Learning](reproductions/packet-scheduling/) | ⚪✅✅⚪⚪⚪ | 4 | [arXiv](https://arxiv.org/abs/2606.00835) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-packet-scheduling-repro) |
| [To Grok Grokking: Provable Grokking in Ridge Regression](reproductions/grokking/) | 🟡⚪⚪🟡✅ | 4 | [arXiv](https://arxiv.org/abs/2601.19791) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-pilot-grokking) |
| [Trees to Flows and Back: Unifying Decision Trees and Diffusion Models](reproductions/treeflow-trees-to-flows/) | 🟡🟡🟡🟡⚪ | 4 | [arXiv](https://arxiv.org/abs/2605.00414) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-treeflow-trees-to-flows-repro) |
| [DropoutTS: Sample-Adaptive Dropout for Robust Time Series Forecasting](reproductions/dropoutts/) | 🟡🟡🟡⚪⚪ | 3 | [arXiv](https://arxiv.org/abs/2601.21726) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-pilot-dropoutts) |
| [Quadratically Regularized Optimal Transport: Localization Bounds and Affine Case Analysis](reproductions/quadratic-ot/) | 🟡⚪🟡⚪🟡 | 3 | [arXiv](https://arxiv.org/abs/2605.24644) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-quadratic-ot-repro) |
| [Characterization of Gaussian Universality Breakdown in High-Dimensional Empirical Risk Minimization](reproductions/gaussian-universality/) | 🟡⚪🟡⚪⚪ | 2 | [arXiv](https://arxiv.org/abs/2604.03146) · [logbook](https://huggingface.co/spaces/Crusadersk/icml26-gaussian-universality-repro) |

## Repository layout

```
reproductions/<paper>/
  README.md     # scoreboard: paper target vs measured, verdict, links
  writeup.md    # full per-claim analysis (acceptance rule + falsification condition)
  evidence/     # runnable repro scripts + raw results JSON + command log
                # (evidence/<claim>/gpu_job/ where a claim needed accelerated scale)
tools/          # the diff-only, parity-checked HF Space uploader used to publish logbooks
```

Per-paper `README.md` and `writeup.md` files record the reproduction as it was carried out and scored at the time of publication; each per-paper headline scoreline was subsequently synced to the frozen final verdicts (**2026-08-09**), while the surrounding narratives and writeups remain the as-published record. The table above carries the referee's final rulings and is the authoritative summary.

## Reproduce any paper

```bash
cd reproductions/<paper>/evidence
python <repro_script>.py   # deterministic; prints the measured numbers
```

Two pilot-era papers (discogen, grokking) carry their artifacts in their logbook Spaces rather than a local evidence directory; their READMEs link there.

Most reproductions need only `numpy` + `scipy`. A subset additionally uses `torch` or `scikit-learn`, and a few ship a `gpu_job/` kit for claims that require accelerated scale.

## License

Reproduction code and write-ups: [MIT](LICENSE). Original papers belong to their respective authors; this repository reproduces and evaluates their published claims and does not redistribute paper text or datasets.
