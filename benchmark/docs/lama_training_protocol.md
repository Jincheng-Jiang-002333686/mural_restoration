# LaMa Training Protocol (mural, mixed masks, hole range [0.2, 0.3])

Date: 2026-06-09 (training completed 2026-06-10)

## FINAL RESULTS (test_ref, 2,649 images, fixed seeded mixed [0.2,0.3] masks,
## full-image metrics computed with MAT's exact metric code)

| Checkpoint | PSNR | SSIM | L1 | FID | P-IDS | U-IDS |
|---|---|---|---|---|---|---|
| epoch 50 (best val FID) | 25.2992 | 0.8980 | 0.0193 | 17.7745 | 0.0000 | 0.0008 |
| epoch 54 (last) | 25.1852 | 0.8951 | 0.0200 | 20.6057 | 0.0000 | 0.0000 |

- Experiment dir: `experiments/jincheng_2026-06-09_20-37-39_train_mural-lama-fourier_full`
- Outputs: `outputs/lama_mixed02_03_best` (epoch 50) and `outputs/lama_mixed02_03` (last)
- CSV: `results/test_metrics.csv`; wandb: training run `0x01yvi9`,
  eval runs `lama-fourier-test-eval` / `lama-fourier-bestfid-test-eval`
- 55 epochs = ~582 kimg, ~17.5 h on one RTX 3090, batch 10, fp32
- Caution when comparing to the HMAT/MADF/MAT paper-table rows: those were
  measured under different harnesses (110-image random-mask in-run eval for
  HMAT, different data/masks for MADF) — see `docs/hmat_code_review.md`.
  These LaMa numbers are the first rows on the shared fixed-mask protocol.

## Environment

- Conda env: `lama` (Python 3.8, torch 1.8.1+cu111 for the RTX 3090,
  pytorch-lightning 1.2.9, kornia 0.5.0, hydra-core 1.1.0, albumentations 0.5.2,
  opencv-python-headless 4.5.5.64, numpy 1.20.3, pillow 9.5.0, wandb).
- Separate from the `mural` env used by MAT — no dependency conflicts.
- `TORCH_HOME=/home/jincheng/Mural/mural_project/lama_mat_comparison/torch_home`
  holds the ade20k ResNet50dilated perceptual-loss encoder and the FID
  inception weights (pre-downloaded).

## Fairness settings (matched to MAT run 00010-train_ref-places256-kimg3000-batch2-noaug)

| Setting | Value | Matches MAT |
|---|---|---|
| Resolution | 256x256, cv2.resize (INTER_LINEAR) from 512 | yes (MAT dataset_256.py) |
| Train images | `train_ref` (10,584) | yes |
| Val/test images | `test_ref` (2,649) | yes (MAT validated on test_ref) |
| Mask generator | `MAT/datasets/mask_generator_256.py` (frozen copy at `external/lama/saicinpainting/training/data/mat_mask_generator_256.py`) | yes |
| Mask settings | `mask_type='mixed'`, `hole_range=[0.2, 0.3]` | yes |
| Training masks | sampled online per item, like MAT | yes |
| Test masks | fixed, one per test image, seed = crc32(filename) XOR 42; stored in `masks/test_mixed_hole02_03/` and `data/mural_lama/val/` | fixed for reproducibility |
| Augmentation | none (`no_augs`) | yes (MAT --aug=noaug) |
| Budget | 55 epochs x 10,584 = ~582 kimg | MAT stopped at ~572 kimg |

Note: the user-referenced `backup/MAT/datasets/mask_generator_256.py` is an older
brush-only version without `mask_type`; only the main
`MAT/datasets/mask_generator_256.py` implements the `mixed` (crack+peel) masks
used in the MAT run, so that is the version frozen here.

Mask conventions: MAT 1=preserved/0=damaged; LaMa 1=damaged. Training masks are
inverted inside `MATMuralMaskGenerator`; saved test mask PNGs use 255=damaged.

## Model / loss config

`configs/training/mural-lama-fourier.yaml` = official `lama-fourier`
(ffc_resnet_075 generator, pix2pixhd discriminator, L1 known=10 + adv 10 +
feature matching 100 + resnet_pl 30), batch size 10, Adam lr G=1e-3/D=1e-4,
single GPU, precision 32.

## Code changes to external/lama (commit 786f5936)

- `saicinpainting/training/data/mat_mask_generator_256.py`: frozen MAT generator.
- `saicinpainting/training/data/masks.py`: `MATMuralMaskGenerator` + registry kind `mat_mural`.
- `saicinpainting/training/data/datasets.py`: `img_suffix` option for `InpaintingTrainDataset` (PNG support).
- `saicinpainting/evaluation/losses/base_loss.py`: `PSNRScore`, `L1Score` (formulas equivalent to MAT's `calculate_psnr`/`calculate_l1`).
- `saicinpainting/evaluation/__init__.py`: `psnr`/`l1` flags in `make_evaluator`.
- `saicinpainting/training/trainers/base.py`: wandb logging — `Metrics/psnr|ssim|l1|fid` + `val/*` each validation epoch, `Loss/*` every 50 steps.
- `bin/train.py`: `wandb.init` from new `wandb:` config section (project `mural_inpainting`).
- `bin/predict.py`: honor `device` from prediction config (was hardcoded CPU).

## Commands

Train (from `external/lama`):

```bash
export TORCH_HOME=/home/jincheng/Mural/mural_project/lama_mat_comparison/torch_home
export PYTHONPATH=$(pwd)
conda run -n lama --no-capture-output python bin/train.py -cn mural-lama-fourier
```

Predict on the fixed test masks:

```bash
conda run -n lama --no-capture-output python bin/predict.py \
  model.path=<experiment_dir> model.checkpoint=<best.ckpt name> \
  indir=/home/jincheng/Mural/mural_project/lama_mat_comparison/data/mural_lama/val \
  outdir=/home/jincheng/Mural/mural_project/lama_mat_comparison/outputs/lama_mixed02_03
# then strip the _mask001 suffix from output names
```

Final metrics (MAT implementations, `mural` env):

```bash
conda run -n mural python ../scripts/eval_outputs.py \
  <renamed_pred_dir> ../data/mural_lama/gt_256 lama-fourier
```

## Validation-time metric notes

- During training, SSIM/PSNR/L1/FID are computed on the full val set (2,649
  test_ref images with the fixed masks) every epoch and logged to wandb
  (`Metrics/*`). LaMa's own implementations are used online (SSIM window 11 on
  [0,1] RGB; FID torchvision-style inception). Checkpoint selection:
  `val_ssim_fid100_f1_total_mean` (max), top-5 + last saved.
- Final reported numbers come from MAT's exact metric code on saved PNGs
  (`scripts/eval_outputs.py`), so they are directly comparable to MAT's table
  (MAT training-time wandb values: PSNR 23.44, SSIM 0.8926, L1 0.0231,
  FID-2649 139.70).
