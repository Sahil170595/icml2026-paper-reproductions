# ICML 2026 Paper Reproductions

Independent, from-scratch reproductions of **48 ICML-2026 submissions** for the [ICML 2026 Agent Reproducibility Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro). Every claim is re-derived from the paper text — **no paper code reused** — with CPU-only, deterministic implementations and machine-checked evidence.

> **297 points across 48 papers**, rank **#26 of 372** on the live leaderboard — top 7% (peaked #8, July 2026) · of **249 claims**: **118 verified**, **3 independently falsified**, 55 toy-scale, 73 inconclusive · judged by an independent LLM referee against each paper's official scored claims.

Each reproduction is a runnable evidence package: the claim is stated with the paper's own target, an acceptance rule and a falsification condition are fixed **in advance**, and a self-contained `numpy`/`scipy` script produces the measured numbers. Verdicts are honest — a reproduction that *fails* a published claim is reported as **falsified**, not buried.

## Highlights

- **Independent falsifications.** Several published claims did not hold under faithful re-implementation and are reported as such — e.g. [Hierarchical Successor Representation](reproductions/hierarchical-successor-representation/) (3/3 falsified) and [DropoutTS](reproductions/dropoutts/) (the headline +46% robustness claim does not reproduce at scale, corroborated by an independent run).
- **Full 4-claim reproductions.** [Minimum Distance Summaries](reproductions/minimum-distance-summaries/) and [Attention-Head Stability](reproductions/attention-head-stability/) — 4/4 claims verified on the papers' own protocols.
- **Theory reproduced numerically.** Convergence rates, spectral bounds and self-adjointness identities checked to machine precision (e.g. [Row-Stochastic vs Doubly-Stochastic mixing](reproductions/row-stochastic-decentralized/), residuals at 1e-16).

## Methodology

See **[METHODOLOGY.md](METHODOLOGY.md)**. In short: pin the paper + official scored claims → find/pin official code (or transcribe the equations) → build a deterministic CPU reproduction → fix acceptance/falsification rules before running → record every command, exit code and output hash → assemble an auditable logbook. Non-determinism, scope limits and proxy scales are stated openly on every claim.

The pipeline is an autonomous multi-provider agent harness: provider-neutral producer agents (GPT + Claude) fan out under a per-paper evidence contract — manifest, claims with pre-fixed acceptance/falsification rules, command log with exit codes, evidence bundle. An independent reviewer agent gates each bundle, and a root coordinator validates before publication — no human-intervention loops.

## Results

✅ verified · 🔴 falsified · 🟡 toy-scale · ⚪ inconclusive · 🕒 deploy pending

| Paper | Claims | Pts | Links |
|---|---|--:|---|
| [Minimum Distance Summaries for Robust Neural Posterior Est](reproductions/minimum-distance-summaries/) | ✅✅✅✅ | 8 | [arXiv](https://arxiv.org/abs/2602.09161) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-minimum-distance-summaries-repro) |
| [Quantifying LLM Attention-Head Stability: Implications for](reproductions/attention-head-stability/) | ✅✅✅✅ | 8 | [arXiv](https://arxiv.org/abs/2602.16740) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-attention-head-stability-repro) |
| [Attention's forward pass and Frank-Wolfe](reproductions/frank-wolfe-attention/) | ✅✅✅ | 6 | [arXiv](https://arxiv.org/abs/2508.09628) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-frank-wolfe-attention-repro) |
| [Demystifying LLM-as-a-Judge: Inference-Time Scaling](reproductions/llm-as-judge-scaling/) | ✅✅✅ | 6 | [arXiv](https://arxiv.org/abs/2512.19905) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-llm-as-judge-scaling-repro) |
| [Finite and Corruption-Robust Regret Bounds in Online Inver](reproductions/mconvex-inverse-linear-regret/) | ✅✅✅ | 6 | [arXiv](https://arxiv.org/abs/2602.01682) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-mconvex-inverse-linear-regret-repro) |
| [Hierarchical Successor Representation for Robust Transfer](reproductions/hierarchical-successor-representation/) | 🔴🔴🔴 | 6 | [arXiv](https://arxiv.org/abs/2602.12753) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-hierarchical-successor-representation-repro) |
| [High-accuracy sampling for diffusion models and log-concav](reproductions/highaccuracy-sampling-logconcave/) | ✅✅✅ | 6 | [arXiv](https://arxiv.org/abs/2602.01338) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-highaccuracy-sampling-logconcave-repro) |
| [Improved Analysis of the Accelerated Noisy Power Method wi](reproductions/noisy-power-method/) | ✅✅✅ | 6 | [arXiv](https://arxiv.org/abs/2602.03682) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-noisy-power-method-repro) |
| [On Regret Bounds of Thompson Sampling for Bayesian Optimiz](reproductions/ts-bayesopt/) | ✅✅✅ | 6 | [arXiv](https://arxiv.org/abs/2603.09276) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-ts-bayesopt-repro) |
| [On Structured State-Space Duality](reproductions/structured-state-space-duality/) | ✅✅✅ | 6 | [arXiv](https://arxiv.org/abs/2510.04944) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-structured-state-space-duality-repro) |
| [Online Packet Scheduling with Deadlines and Learning](reproductions/packet-scheduling/) | ✅✅✅ | 6 | [arXiv](https://arxiv.org/abs/2606.00835) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-packet-scheduling-repro) |
| [Optimal Unconstrained Self-Distillation in Ridge Regressio](reproductions/ridge-self-distillation/) | ✅✅✅ | 6 | [arXiv](https://arxiv.org/abs/2602.17565) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-ridge-self-distillation-repro) |
| [Row-Stochastic Matrices Can Provably Outperform Doubly Sto](reproductions/row-stochastic-decentralized/) | ✅✅✅ | 6 | [arXiv](https://arxiv.org/abs/2511.19513) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-row-stochastic-decentralized-repro) |
| [To Grok Grokking: Provable Grokking in Ridge Regression](reproductions/grokking/) | ✅✅✅ | 6 | [arXiv](https://arxiv.org/abs/2601.19791) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-grokking-repro) |
| [A Random Matrix Perspective on the Consistency of Diffusio](reproductions/rmt-diffusion-consistency/) | ✅✅🟡 | 5 | [arXiv](https://arxiv.org/abs/2602.02908) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-rmt-diffusion-consistency-repro) |
| [Accelerated and Stable Convergence with Anchored Generaliz](reproductions/anchored-optimistic-goma/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2606.21528) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-anchored-optimistic-goma-repro) |
| [Best-of-Both-Worlds for Heavy-Tailed MDPs on full episodic](reproductions/heavytail-mdp-bobw/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2602.01295) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-heavytail-mdp-bobw-repro) |
| [Causal Modeling of Selection in Evolution](reproductions/causal-selection-evolution/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2606.05689) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-causal-selection-evolution-repro) |
| [Characterization of Gaussian Universality Breakdown in Hig](reproductions/gaussian-universality/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2604.03146) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-gaussian-universality-repro) |
| [DAVE: Distribution-Aware Attribution via ViT Gradient Deco](reproductions/dave-vit-attribution/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2602.06613) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-dave-vit-attribution-repro) |
| [Design-Based Anytime-Valid Inference for Randomized Experi](reproductions/anytime-valid-delayed/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2603.25971) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-anytime-valid-delayed-repro) |
| [DropoutTS: Sample-Adaptive Dropout for Robust Time Series ](reproductions/dropoutts/) | 🔴✅ | 4 | [arXiv](https://arxiv.org/abs/2601.21726) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-dropoutts-repro) |
| [Estimating Continuous Treatment Effects with Two-Stage Ker](reproductions/treatment-effects-krr/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2604.13410) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-treatment-effects-krr-repro) |
| [Explaining Concept Shift with Interpretable Feature Attrib](reproductions/concept-shift-sgshift/) | ✅✅⚪ | 4 | [arXiv](https://arxiv.org/abs/2505.20634) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-concept-shift-sgshift-repro) |
| [Fixed Budget is No Harder Than Fixed Confidence in Best-Ar](reproductions/bai-budget-confidence/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2602.03972) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-bai-budget-confidence-repro) |
| [Follow-the-Perturbed-Leader for Decoupled Bandits: Best-of](reproductions/ftpl-decoupled-bandits/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2510.12152) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-ftpl-decoupled-bandits-repro) |
| [Foundations of Equivariant Deep Learning](reproductions/equivariant-deep-learning/) | ✅✅ | 4 | [Space](https://huggingface.co/spaces/Crusadersk/icml26-equivariant-deep-learning-repro) |
| [Full-Batch Gradient Descent Outperforms One-Pass SGD: Samp](reproductions/fullbatch-single-index/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2602.02431) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-fullbatch-single-index-repro) |
| [General Synthetic-Powered Inference](reproductions/synthetic-powered-inference/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2509.20345) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-synthetic-powered-inference-repro) |
| [Improved Convergence Analysis of Topology Dependence in De](reproductions/decentralized-sgd-topology/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2606.09154) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-decentralized-sgd-topology-repro) |
| [Keep Everyone Happy: Online Fair Division of Numerous Item](reproductions/fair-division/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2408.12845) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-fair-division-repro) |
| [Minimax Optimal Strategy for Delayed Observations in Onlin](reproductions/delayed-obs-rl/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2603.03480) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-delayed-obs-rl-repro) |
| [NeuralFLoC: Neural Flow-Based Joint Registration and Clust](reproductions/neuralfloc-registration-clustering/) | 🟡🟡✅ | 4 | [arXiv](https://arxiv.org/abs/2602.03169) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-neuralfloc-registration-clustering-repro) |
| [Parametrized Power-Iteration Clustering for Directed Graph](reproductions/power-iteration-clustering-digraph/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2210.00310) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-power-iteration-clustering-digraph-repro) |
| [Prior Diffusiveness and Regret in the Linear-Gaussian Band](reproductions/bandit/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2601.02022) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-bandit-repro) |
| [Procedural Generation Of Algorithm Discovery Tasks in Mach](reproductions/discogen/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2603.17863) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-discogen-repro) |
| [Quadratically Regularized Optimal Transport: Localization ](reproductions/quadratic-ot/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2605.24644) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-quadratic-ot-repro) |
| [Stochastic Linear Bandits with Parameter Noise](reproductions/linbandit-paramnoise/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2601.23164) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-linbandit-paramnoise-repro) |
| [The Theory and Practice of MAP Inference over Non-Convex C](reproductions/map-inference-nonconvex/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2602.08681) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-map-inference-nonconvex-repro) |
| [Towards Understanding Adam Convergence on Highly Degenerat](reproductions/adam/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2603.09581) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-adam-repro) |
| [Trees to Flows and Back: Unifying Decision Trees and Diffu](reproductions/treeflow-trees-to-flows/) | ✅✅ | 4 | [arXiv](https://arxiv.org/abs/2605.00414) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-treeflow-trees-to-flows-repro) |
| [Tucker Attention: A generalization of approximate attentio](reproductions/tucker-attention/) | ⚪✅✅ | 4 | [arXiv](https://arxiv.org/abs/2603.30033) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-tucker-attention-repro) |
| [Beyond Softmax: A Natural Parameterization for Categorical](reproductions/catnat-beyond-softmax/) | ✅🟡 | 3 | [arXiv](https://arxiv.org/abs/2509.24728) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-catnat-beyond-softmax-repro) |
| [Diffusion Bridge or Flow Matching? A Unifying Framework](reproductions/diffusion-bridge-flow-matching/) | ✅🟡 | 3 | [arXiv](https://arxiv.org/abs/2509.24531) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-diffusion-bridge-flow-matching-repro) |
| [Maximum Likelihood Reinforcement Learning](reproductions/maxlik-rl/) | 🟡✅ | 3 | [arXiv](https://arxiv.org/abs/2602.02710) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-maxlik-rl-repro) |
| [Repro — Model Fusion via Neuron Interpolation](reproductions/model-fusion/) | 🟡🟡🟡 | 3 | [arXiv](https://arxiv.org/abs/2507.00037) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-model-fusion-repro) |
| [Rex: A Family of Reversible Exponential (Stochastic) Runge](reproductions/rex-reversible-rk/) | ✅🟡 | 3 | [arXiv](https://arxiv.org/abs/2502.08834) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-rex-reversible-rk-repro) |
| [Nonparametric Distribution Regression Re-calibration](reproductions/nonparam-recalibration/) | ⚪✅ | 2 | [arXiv](https://arxiv.org/abs/2602.13362) · [Space](https://huggingface.co/spaces/Crusadersk/icml26-nonparam-recalibration-repro) |

## Repository layout

```
reproductions/<paper>/
  README.md     # scoreboard: paper target vs measured, verdict, links
  writeup.md    # full per-claim analysis (acceptance rule + falsification condition)
  evidence/     # runnable repro scripts + raw results JSON + command log
tools/          # the diff-only, parity-checked HF Space uploader used to publish logbooks
```

## Reproduce any paper

```bash
cd reproductions/<paper>/evidence
python <repro_script>.py   # numpy + scipy only; deterministic; prints the measured numbers
```

## License

Reproduction code and write-ups: [MIT](LICENSE). Original papers belong to their respective authors; this repository reproduces and evaluates their published claims and does not redistribute paper text or datasets.
