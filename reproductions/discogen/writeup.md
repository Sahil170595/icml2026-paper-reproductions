# Claim 1: DiscoGen procedurally generates over 400 million distinct algorithm dis…

---

**Paper claim.** DiscoGen procedurally generates over 400 million distinct algorithm discovery tasks via a combinatorial formula N_tasks = 2*3*b*(2^m-1)*(3^d-2^(d+1)+1) depending on the number of modules m, datasets d, and backends b (Section 4.2, Equation 1).

**Paper anchor.** Not supplied in the original bundle

**Reproduction status.** `unverified`

**Evidence contract.** See Evidence and rerun page

**Comparison rule.** Claim verified if DiscoGen can be imported and instantiated, and demonstrates stated functionality

This status is copied from the existing reproduction bundle and is not strengthened by the Trackio migration. The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.


---

# Claim 2: Including additional domains beyond the main evaluation set, DiscoGen's…

---

**Paper claim.** Including additional domains beyond the main evaluation set, DiscoGen's total task space reaches approximately 99 billion tasks (Appendix C).

**Paper anchor.** Not supplied in the original bundle

**Reproduction status.** `unverified`

**Evidence contract.** See Evidence and rerun page

**Comparison rule.** Claim verified if DiscoGen can be imported and instantiated, and demonstrates stated functionality

This status is copied from the existing reproduction bundle and is not strengthened by the Trackio migration. The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.


---

# Claim 3: Across the 10 domains used in the main evaluation, per-domain task coun…

---

**Paper claim.** Across the 10 domains used in the main evaluation, per-domain task counts range from 900 (Greenhouse Gas Prediction) to 426,043,800 (On-Policy RL), with a median of 59,622 tasks per domain (Table 1).

**Paper anchor.** Not supplied in the original bundle

**Reproduction status.** `unverified`

**Evidence contract.** See Evidence and rerun page

**Comparison rule.** Claim verified if DiscoGen can be imported and instantiated, and demonstrates stated functionality

This status is copied from the existing reproduction bundle and is not strengthened by the Trackio migration. The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.


---

# Claim 4: DiscoBench provides a fixed evaluation subset built from DiscoGen, comp…

---

**Paper claim.** DiscoBench provides a fixed evaluation subset built from DiscoGen, comprising, for each domain, m single-module tasks (DiscoBench Single) plus one comprehensive all-modules-active task (DiscoBench All) (Section 4.4).

**Paper anchor.** Not supplied in the original bundle

**Reproduction status.** `unverified`

**Evidence contract.** See Evidence and rerun page

**Comparison rule.** Claim verified if DiscoGen can be imported and instantiated, and demonstrates stated functionality

This status is copied from the existing reproduction bundle and is not strengthened by the Trackio migration. The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.


---

# Claim 5: As the number of editable modules increases in DiscoBench tasks, agent…

---

**Paper claim.** As the number of editable modules increases in DiscoBench tasks, agent success rates consistently decline while the achievable performance ceiling rises (Appendix G).

**Paper anchor.** Not supplied in the original bundle

**Reproduction status.** `unverified`

**Evidence contract.** See Evidence and rerun page

**Comparison rule.** Claim verified if DiscoGen can be imported and instantiated, and demonstrates stated functionality

This status is copied from the existing reproduction bundle and is not strengthened by the Trackio migration. The recorded command, output, duration, and complete bundle are linked from the Evidence and rerun and Conclusion pages.


---

# Conclusion

---

The bounded experiment completed, but the full paper claim remains explicitly unverified or inconclusive. This Trackio-native record covers 5 claim page(s) and preserves the original report, scripts, evidence, and rerun output. Fresh local reruns completed 1/1 command(s) in approximately 1.9 seconds. No Hugging Face GPU Job was used: these checks are CPU-feasible or remain limited by data, checkpoints, implementation scope, or the stated proxy design rather than GPU availability.

## Scope & cost

|  | This reproduction | Full replication |
|---|---|---|
| Scope | 5 bounded claim page(s); original claim labels preserved | Paper-scale implementation and every headline empirical claim |
| Hardware | Local machine; CPU-oriented scripts unless a recorded command says otherwise; no HF Job | Paper-specified accelerators, datasets, checkpoints, and sweeps |
| Compute time | 1.9 s across 1 freshly recorded command(s) | Not estimated without the full paper setup |
| Cost | Approximately $0 incremental local compute | Unknown; potentially substantial |
| Outcome | The bounded experiment completed, but the full paper claim remains explicitly unverified or inconclusive. | Not attempted |

---

**📦 Artifact** `icml26-0mvm3lqljf/0mvm3lqljf-reproduction-bundle:v0` · dataset

https://huggingface.co/buckets/Crusadersk/icml26-pilot-discogen-artifacts#icml26-0mvm3lqljf/0mvm3lqljf-reproduction-bundle:v0

---

The reproduction bundle contains the runnable scripts, evidence JSON/JSONL, manifests, reviews, and supporting logs under `artifacts/`. After publication, the artifact cell above resolves to the Hugging Face Bucket URL and can be downloaded from that Bucket. Secrets, virtual environments, caches, and replaceable downloads are excluded.


---

# Sources and provenance

---

- OpenReview: https://openreview.net/forum?id=0Mvm3lqLjF
- Published logbook: https://huggingface.co/spaces/Crusadersk/icml26-pilot-discogen
- arXiv: https://arxiv.org/abs/2603.17863
- Source repository: https://github.com/AlexGoldie/discogen.git
- Source revision: `4e9146fe2c4d4ec79f7946b1d1364b0d5fe77a20`

The migration preserves the original claim boundaries and does not convert toy, partial, or inconclusive evidence into a full reproduction.
