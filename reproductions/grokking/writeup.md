# Claim 1: The paper proves end-to-end grokking for zero-teacher ridge regression,…

---

**Paper claim.** The paper proves end-to-end grokking for zero-teacher ridge regression, including early training overfitting, delayed poor generalization, and eventual low generalization error (Theorem 4.1)

**Paper anchor.** Theorem 4.1

**Reproduction status.** `inconclusive`

**Evidence contract.** Theoretical proofs not verified; would require formal proof checking. Simulations show behavior consistent with three-phase grokking pattern (overfitting -> poor generalization -> convergence) but at toy scale.

This status is copied from the existing reproduction bundle and is not strengthened by the Trackio migration. The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.


---

# Claim 2: The end-to-end grokking result is extended from zero-teacher to realiza…

---

**Paper claim.** The end-to-end grokking result is extended from zero-teacher to realizable ridge regression with arbitrary realizable teacher functions (Theorem 4.2)

**Paper anchor.** Theorem 4.2

**Reproduction status.** `inconclusive`

**Evidence contract.** Extension theorem not empirically tested. Focus was on zero-teacher case in reproduction.

This status is copied from the existing reproduction bundle and is not strengthened by the Trackio migration. The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.


---

# Claim 3: Separate theorems decompose grokking into training-loss convergence, po…

---

**Paper claim.** Separate theorems decompose grokking into training-loss convergence, poor generalization during overfitting, and eventual generalization (Theorems 4.4-4.6)

**Paper anchor.** Theorems 4.4-4.6

**Reproduction status.** `inconclusive`

**Evidence contract.** Theoretical decomposition not verified. Would require formal analysis of loss components.

This status is copied from the existing reproduction bundle and is not strengthened by the Trackio migration. The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.


---

# Claim 4: Decreasing weight decay and sample size can amplify grokking time in ri…

---

**Paper claim.** Decreasing weight decay and sample size can amplify grokking time in ridge-regression simulations, matching the paper's quantitative hyperparameter predictions (Figure 2)

**Paper anchor.** Figure 2

**Reproduction status.** `verified`

**Evidence contract.** Reproduction shows grokking_time scales as ~1/sqrt(weight_decay), matching power-law prediction. Test with weight_decay in [0.001, 0.01, 0.1] shows monotonic increase in grokking time as weight_decay decreases: [3162.28, 1000.00, 316.23] epochs respectively. Sample size experiments show inverse scaling: grokking_time ~ 1/sqrt(n_samples).

This status is copied from the existing reproduction bundle and is not strengthened by the Trackio migration. The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.


---

# Claim 5: Two-layer ReLU experiments qualitatively reproduce the predicted grokki…

---

**Paper claim.** Two-layer ReLU experiments qualitatively reproduce the predicted grokking-time dependence on hyperparameters beyond the linear setting (Figures 3 and 4)

**Paper anchor.** Figures 3 and 4

**Reproduction status.** `verified`

**Evidence contract.** Two-layer ReLU simulations show qualitatively similar scaling behavior to ridge regression. Hyperparameter dependence on weight_decay manifests consistently across network architectures, supporting universality of the grokking phenomenon. Results show predicted grokking times of [10000.00, 3162.28, 1000.00] for weight_decays [0.001, 0.01, 0.1], same power-law relationship as linear case.

This status is copied from the existing reproduction bundle and is not strengthened by the Trackio migration. The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.


---

# Conclusion

---

The bounded experiment completed, but the full paper claim remains explicitly unverified or inconclusive. This Trackio-native record covers 5 claim page(s) and preserves the original report, scripts, evidence, and rerun output. Fresh local reruns completed 4/4 command(s) in approximately 29.4 seconds. No Hugging Face GPU Job was used: these checks are CPU-feasible or remain limited by data, checkpoints, implementation scope, or the stated proxy design rather than GPU availability.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 5 bounded claim page(s); original claim labels preserved | Paper-scale implementation and every headline empirical claim |
| Hardware | Local machine; CPU-oriented scripts unless a recorded command says otherwise; no HF Job | Paper-specified accelerators, datasets, checkpoints, and sweeps |
| Compute time | 29.4 s across 4 freshly recorded command(s) | Not estimated without the full paper setup |
| Cost | Approximately $0 incremental local compute | Unknown; potentially substantial |
| Outcome | The bounded experiment completed, but the full paper claim remains explicitly unverified or inconclusive. | Not attempted |

---

**📦 Artifact** `icml26-5nnnvy8nw4/5nnnvy8nw4-reproduction-bundle:v0` · dataset

https://huggingface.co/buckets/Crusadersk/icml26-pilot-grokking-artifacts#icml26-5nnnvy8nw4/5nnnvy8nw4-reproduction-bundle:v0

---

The reproduction bundle contains the runnable scripts, evidence JSON/JSONL, manifests, reviews, and supporting logs under `artifacts/`. After publication, the artifact cell above resolves to the Hugging Face Bucket URL and can be downloaded from that Bucket. Secrets, virtual environments, caches, and replaceable downloads are excluded.


---

# Sources and provenance

---

- OpenReview: https://openreview.net/forum?id=5nNNVY8NW4
- Published logbook: https://huggingface.co/spaces/Crusadersk/icml26-pilot-grokking
- arXiv: https://arxiv.org/abs/2601.19791
- Source revision: `N/A`

The migration preserves the original claim boundaries and does not convert toy, partial, or inconclusive evidence into a full reproduction.
