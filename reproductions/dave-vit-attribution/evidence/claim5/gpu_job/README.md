# GPU kit - Claim 5: GridPG/EnergyPG localization on B-cos ViTs (paper Table 2)

Runs the ImageNet-1k B-cos localization benchmark on GPU. `benchmark.py` loads the authors'
pretrained B-cos-ViT / B-cos-ViT-C, computes the DAVE effective transformation (B-cos gating
|cos|^(B-1) treated as the dynamic-linear operator, detached; Reynolds + low-pass, 50 samples),
the inherent B-cos explanation, and gradient baselines. `run_job.sh` = `hf jobs run` launcher.

Expected DAVE: B-cos-ViT 84.00/78.55, B-cos-ViT-C 88.43/79.63. Acceptance: DAVE > inherent B-cos
on GridPG and EnergyPG for both models. (The CPU pilot already verifies the DAVE decomposition
extends to real B-cos dynamic-linear layers to machine precision - see repro_claim5.py.)
