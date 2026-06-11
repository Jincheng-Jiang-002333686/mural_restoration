# LaMa Source Inspection Notes

Official source location:

- `../external/lama`
- commit: `786f5936b27fb3dacd2b1ad799e4de968ea697e7`

## Relevant Entry Points

- Training: `external/lama/bin/train.py`
- Prediction: `external/lama/bin/predict.py`
- Evaluation: `external/lama/bin/evaluate_predicts.py`
- Main training config: `external/lama/configs/training/lama-fourier.yaml`
- Default 256 data config:
  `external/lama/configs/training/data/abl-04-256-mh-dist.yaml`
- Dataset code:
  `external/lama/saicinpainting/training/data/datasets.py`
- Mask generator code:
  `external/lama/saicinpainting/training/data/masks.py`
- Evaluation dataset code:
  `external/lama/saicinpainting/evaluation/data.py`

## LaMa Mask Convention

LaMa uses:

- `1` for missing / inpainted region.
- `0` for known / preserved region.

This is confirmed by the prediction path, where LaMa builds masked input as:

```text
masked_img = image * (1 - mask)
```

MAT's `mask_generator_256.py` returns the opposite:

- `1` for preserved / known region.
- `0` for damaged / missing region.

Therefore, when using MAT masks inside LaMa:

```text
lama_mask = 1 - mat_mask
```

When saving masks as PNG files for LaMa prediction/evaluation, use:

```text
lama_mask_png = lama_mask * 255
```

## LaMa Training Mask Path

The default `lama-fourier.yaml` uses:

```text
data: abl-04-256-mh-dist
```

That data config generates training masks online via:

```text
mask_gen_kwargs
mask_generator_kind: mixed
```

The code path is:

```text
make_default_train_dataloader(...)
  -> get_mask_generator(...)
  -> MixedMaskGenerator(...)
```

For our comparison, this should be adapted so online training masks come from
MAT's mask generator instead of LaMa's default irregular/box generator.

## LaMa Validation/Test Mask Path

Validation, prediction, and evaluation use fixed image/mask files.

Expected format:

```text
image1.png
image1_mask001.png
image2.png
image2_mask001.png
```

The default evaluation dataset finds mask files with:

```text
*mask*.png
```

Then it maps each mask to its image by removing `_mask...` and appending
`img_suffix`.

For our PNG data, `img_suffix` must be `.png`.

## Dataset Findings

Current project data:

- `train_ref`: 10,584 PNG images
- `test_ref`: 2,649 PNG images

Sampled image sizes:

- `train_ref`: mostly `512 x 512`, with a few `256 x 256`
- `test_ref`: mostly `512 x 512`, with a few `256 x 256`

Important issue:

LaMa's default training dataset currently only searches for `*.jpg` files:

```text
glob(os.path.join(indir, '**', '*.jpg'), recursive=True)
```

Since our data is PNG-only, we need to adapt the dataset loader or prepare a
compatible dataset copy.

## Recommended Adaptation Plan

1. Keep LaMa source mostly unchanged, but add a small custom config/code path for
   this mural comparison.

2. Add MAT-style mask generator support to LaMa's
   `saicinpainting/training/data/masks.py`.

   Proposed generator behavior:

   ```text
   mat_mask = MAT RandomMask(...)
   lama_mask = 1 - mat_mask
   return lama_mask
   ```

3. Adapt LaMa training dataset loading to accept PNG images.

   Preferred change:

   - add an `img_suffix` option to `InpaintingTrainDataset`
   - default remains `.jpg` so original LaMa behavior is preserved
   - our config sets `img_suffix: .png`

4. Prepare fixed validation/evaluation folders using LaMa naming:

   ```text
   image_name.png
   image_name_mask001.png
   ```

5. Use the same fixed masks for all LaMa test inference and metrics.

## Decisions Needed Before Editing Code

1. Resolution:
   - If MAT metrics were produced at 256, resize/crop data to 256 and use
     `mask_generator_256.py`.
   - If MAT metrics were produced at 512, use `mask_generator_512.py` or generate
     512 masks with equivalent settings.

2. Validation split:
   - Use an existing MAT validation split if one exists.
   - Otherwise hold out a small subset from `train_ref` for LaMa validation.

3. Training schedule:
   - Match MAT's training budget if possible.
   - Otherwise record LaMa epoch/iteration count clearly and compare as a trained
     baseline with transparent settings.
