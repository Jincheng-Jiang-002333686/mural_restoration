#!/bin/bash
# HMAT v2 on the Nine-Colored Deer dataset (3,500 train / 500 test, 256px),
# mixed masks hole range [0.2, 0.3]. From scratch, v2 recipe, ~600 kimg (~33h).
# Run when the GPU is free. Then predict on the 500 fixed masks + metrics.
set -euo pipefail

PROJ=/home/jincheng/Mural/mural_project
COMP=$PROJ/lama_mat_comparison
cd $PROJ/MAT

echo "[1/3] training HMAT v2 on Nine-Colored Deer..."
conda run -n mural --no-capture-output python train.py \
  --outdir=outputs \
  --data=$PROJ/train_mural_png \
  --data_val=$PROJ/test_mural_png \
  --dataloader=datasets.dataset_256.ImageFolderMaskDataset \
  --metrics=fid500_full,psnr500_full \
  --gpus=1 --batch=16 --batch-gpu=4 --kimg=600 --snap=10 \
  --cfg=places256 --aug=noaug --workers=2 \
  --style_mix=0.0 --pr=0.0 --glr=0.001 --dlr=0.0001 --gamma=2.0 \
  --wandb-project=mural_inpainting > $COMP/logs/deer_hmat_train.log 2>&1

EXP=$(ls -td $PROJ/MAT/outputs/*train_mural_png* | head -1)
echo "[2/3] predicting with $EXP (last snapshot) on fixed deer masks..."
SNAP=$(ls -t $EXP/network-snapshot-*.pkl | head -1)
conda run -n mural --no-capture-output python $COMP/scripts/hmat_predict_fixed.py \
  $SNAP $COMP/outputs/hmat_v2_deer \
  $COMP/data/deer/gt_256 $COMP/masks/deer_test_mixed_hole02_03 \
  > $COMP/logs/deer_hmat_predict.log 2>&1

echo "[3/3] metrics..."
conda run -n mural --no-capture-output python $COMP/scripts/eval_outputs.py \
  $COMP/outputs/hmat_v2_deer $COMP/data/deer/gt_256 hmat-v2-deer \
  $COMP/masks/deer_test_mixed_hole02_03 > $COMP/logs/deer_hmat_eval.log 2>&1
grep "psnr:" $COMP/logs/deer_hmat_eval.log
echo "DEER_HMAT_DONE"
