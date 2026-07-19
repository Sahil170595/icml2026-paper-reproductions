# GPU kit - Claim 4: GridPG/EnergyPG localization on conventional ViTs (paper Table 1)

Runs the full ImageNet-1k localization benchmark on GPU (not runnable on the CPU pilot host).

- `benchmark.py` - loads real pretrained ViT-B/16, DeiT-B/16, DeiT-III-B/16, DINO-B/16 (timm),
  computes the DAVE effective transformation (detached-operator forward/backward + Reynolds
  translation/rotation averaging +/-20 deg + Gaussian low-pass, 50 samples, Eq. 8-9) and the six
  baselines, then GridPG (3x3 montage) and EnergyPG (GT bbox) per Table 1.
- `run_job.sh` - `hf jobs run` launcher (A10G-large, ImageNet val + bbox mount).

Expected DAVE numbers: GridPG 60.19 / 63.52 / 65.76 / 51.33 %; EnergyPG best on DeiT-B (82.23),
DeiT-III-B (82.43), DINO-B (83.38). Acceptance: DAVE GridPG >= best baseline on all 4 models.
