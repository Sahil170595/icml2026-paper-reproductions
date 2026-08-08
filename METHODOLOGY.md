# Methodology

How the 48 reproductions in this repository were produced, scored and published. Nothing here is a claim about a specific result — per-paper numbers live in `reproductions/<paper>/` and the final referee verdicts are summarised in the [README results table](README.md#results).

## Principles

- **Re-derivation, not re-execution.** Each scored claim is re-derived from the paper text. Where official code, a dataset or a checkpoint exists it is pinned at the exact commit SHA that was audited and treated as an artifact to check against, not as an oracle to trust — the default branch can move and break reproducibility.
- **Deterministic by default.** Reproductions are built to run deterministically and to be re-runnable by a third party from this repository alone. Where a claim genuinely requires accelerated scale, the job kit is shipped alongside the CPU evidence rather than hidden.
- **Rules fixed in advance.** An acceptance rule and a falsification condition are written down *before* the reproduction runs, so the verdict is decided by the pre-registered rule and not chosen after seeing the number.
- **Honest verdicts.** A reproduction that fails a published claim is reported as **falsified**, not buried. Non-determinism, scope limits and proxy scales are stated openly on every claim.

## Per-paper pipeline

1. **Pin the paper and its official scored claims.** The claim list is the paper's own scored claim set, not a set chosen by the reproducer.
2. **Find and pin official code**, at an audited commit SHA — or, where no code exists, transcribe the equations directly from the paper text.
3. **Build a deterministic reproduction** of each claim, stating the paper's own target for that claim.
4. **Fix the acceptance rule and the falsification condition before running.** Each claim page records both, so the reader can check the verdict against the rule.
5. **Run and record.** Every command, its exit code and the output hash are captured verbatim.
6. **Assemble an auditable logbook** — the published Hugging Face Space for that paper — carrying the claim statements, the pre-fixed rules, the command log and the evidence bundle.

## Agent harness

The pipeline is an autonomous multi-provider agent harness with no human-intervention loops:

- **Producer agents** (provider-neutral; GPT and Claude) fan out per paper under a shared evidence contract: a manifest, the claim list with pre-fixed acceptance and falsification rules, a command log with exit codes, and an evidence bundle.
- **An independent reviewer agent** gates each bundle before it can be published.
- **A root coordinator** validates the gated bundle as the last step before publication.

The evidence contract is what makes the bundles comparable across papers and across providers: every paper produces the same artifacts in the same shape, whichever producer agent built it.

## Evidence structure

Each paper directory carries the same three surfaces:

```
reproductions/<paper>/
  README.md     # scoreboard: paper target vs measured, verdict, links
  writeup.md    # full per-claim analysis (acceptance rule + falsification condition)
  evidence/     # runnable repro scripts + raw results JSON + command log
                # evidence/<claim>/gpu_job/ where a claim needed accelerated scale
```

`tools/` holds the diff-only, parity-checked Hugging Face Space uploader used to publish the logbooks, so that what is published matches what is in this repository.

Any paper can be re-run directly from the evidence directory:

```bash
cd reproductions/<paper>/evidence
python <repro_script>.py   # deterministic; prints the measured numbers
```

Most reproductions need only `numpy` + `scipy`. A subset additionally uses `torch` or `scikit-learn`, and a few ship a `gpu_job/` kit for claims that require accelerated scale.

## Judging and scoring

Verdicts are **not** self-assigned. Every published logbook is judged by the challenge's own independent LLM referee against that paper's official scored claims, and the rulings are published in the challenge's [verdicts dataset](https://huggingface.co/datasets/ICML-2026-agent-repro/verdicts). The referee issues one verdict per official scored claim:

| Verdict | Meaning | Points |
|---|---|--:|
| ✅ verified | the claim reproduces under the pre-fixed acceptance rule | 2 |
| 🔴 falsified | the reproduction meets the pre-fixed falsification condition | 2 |
| 🟡 toy-scale | the mechanism was tested, but at reduced scale, on a proxy task, or on a subset | 1 |
| ⚪ inconclusive | the claim was not addressed, or the evidence does not settle it | 0 |

Falsifying a claim scores the same as verifying it: the challenge rewards resolving a claim either way, and scores nothing for leaving it open. A paper's point total is the sum over its official scored claims, and an agent's leaderboard total is the sum over its logbooks.

The totals reported in the README are re-derived from the published verdicts dataset rather than counted by hand, so the headline summary and the per-paper table are guaranteed to agree.
