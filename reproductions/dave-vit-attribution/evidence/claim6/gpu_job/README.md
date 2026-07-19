# GPU kit - Claim 6: pixel-deletion faithfulness (paper Figure 6)

Runs the ImageNet-1k deletion-faithfulness benchmark on GPU. `benchmark.py` loads real pretrained
ViT-B/16 and DeiT-III-B/16 (timm), computes DAVE (effective transformation + Reynolds + low-pass,
50 samples) and the six baselines, then the deletion curve (remove pixels least->most important,
track target-class softmax probability) and its AUC. `run_job.sh` = `hf jobs run` launcher.

Acceptance: DAVE yields the flattest curve (highest deletion AUC) on both models. (The CPU pilot
already implements the deletion metric and shows DAVE/I x G beat random ordering - repro_claim6.py.)
