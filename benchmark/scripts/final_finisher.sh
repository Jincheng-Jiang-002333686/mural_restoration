#!/bin/bash
# GPU is free (deer run stopped early at k200, converged). Run the three
# remaining stages back-to-back, no gating:
#   1. Deer HMAT v2 benchmark from best snapshot (k200) -> hmat-v2-deer row
#   2. EdgeConnect test + eval (models trained, mask bug fixed) -> edgeconnect row
#   3. HMAT continuation +1000 kimg from run 00012 k1000 -> benchmark -> hmat-v3 rows
set -uo pipefail

PROJ=/home/jincheng/Mural/mural_project
COMP=$PROJ/lama_mat_comparison
EC=$PROJ/edge-connect
LOG=$COMP/logs/final_finisher.log
DEER=$PROJ/MAT/outputs/00013-train_mural_png-places256-glr0.001-dlr0.0001-gamma2-pr0-kimg600-batch16-bgpu4-sm0.0-noaug
SRC=$PROJ/MAT/outputs/00012-train_ref-places256-glr0.001-dlr0.0001-gamma2-pr0-kimg1000-batch16-bgpu4-sm0.0-noaug-resumecustom/network-snapshot-001000.pkl

echo "[$(date '+%F %T')] START final_finisher" | tee -a $LOG

# ---- 1. Deer benchmark (best snapshot = k200) ----
{
  DSNAP=$(ls -t $DEER/network-snapshot-*.pkl | head -1)
  echo "[$(date '+%F %T')] STAGE 1 deer benchmark: $(basename $DSNAP)" | tee -a $LOG
  conda run -n mural --no-capture-output python $COMP/scripts/hmat_predict_fixed.py \
      "$DSNAP" $COMP/outputs/hmat_v2_deer \
      $COMP/data/deer/gt_256 $COMP/masks/deer_test_mixed_hole02_03 >> $LOG 2>&1
  conda run -n mural --no-capture-output python $COMP/scripts/eval_outputs.py \
      $COMP/outputs/hmat_v2_deer $COMP/data/deer/gt_256 hmat-v2-deer \
      $COMP/masks/deer_test_mixed_hole02_03 >> $LOG 2>&1
  echo "[$(date '+%F %T')] STAGE 1 done" | tee -a $LOG
} || echo "[$(date '+%F %T')] STAGE 1 (deer) FAILED" | tee -a $LOG

# ---- 2. EdgeConnect test + eval (DHMural) ----
{
  echo "[$(date '+%F %T')] STAGE 2 EdgeConnect test+eval" | tee -a $LOG
  cd $EC
  conda run -n mural --no-capture-output python test.py \
      --path checkpoints/mural --model 3 \
      --input $EC/datasets/mural_test.flist \
      --mask $EC/datasets/mural_test_masks.flist \
      --output $COMP/outputs/edgeconnect_mixed02_03 >> $LOG 2>&1
  echo "[$(date '+%F %T')] EC outputs: $(ls $COMP/outputs/edgeconnect_mixed02_03 2>/dev/null | wc -l)" | tee -a $LOG
  conda run -n mural --no-capture-output python $COMP/scripts/eval_outputs.py \
      $COMP/outputs/edgeconnect_mixed02_03 $COMP/data/mural_lama/gt_256 edgeconnect >> $LOG 2>&1
  echo "[$(date '+%F %T')] STAGE 2 done" | tee -a $LOG
} || echo "[$(date '+%F %T')] STAGE 2 (EdgeConnect) FAILED" | tee -a $LOG

# ---- 3. HMAT continuation +1000 kimg (DHMural) ----
{
  echo "[$(date '+%F %T')] STAGE 3 HMAT continuation +1000 kimg" | tee -a $LOG
  cd $PROJ/MAT
  conda run -n mural --no-capture-output python train.py \
    --outdir=outputs \
    --data=$PROJ/train_ref --data_val=$PROJ/test_ref \
    --dataloader=datasets.dataset_256.ImageFolderMaskDataset \
    --metrics=fid2649_full,psnr2649_full \
    --gpus=1 --batch=16 --batch-gpu=4 --kimg=1000 --snap=10 \
    --cfg=places256 --aug=noaug --workers=2 --style_mix=0.0 --pr=0.0 \
    --glr=0.001 --dlr=0.0001 --gamma=2.0 \
    --resume=$SRC --wandb-project=mural_inpainting >> $LOG 2>&1
  CONT=$(ls -td $PROJ/MAT/outputs/0001[4-9]*resumecustom 2>/dev/null | head -1)
  echo "[$(date '+%F %T')] continuation dir: $CONT" | tee -a $LOG
  mapfile -t SNAPS < <(conda run -n mural python $COMP/scripts/select_best_snapshot.py "$CONT" | grep network-snapshot)
  for entry in "${SNAPS[@]}"; do
      tag=$(echo "$entry" | cut -d' ' -f1); snap=$(echo "$entry" | cut -d' ' -f2)
      conda run -n mural --no-capture-output python $COMP/scripts/hmat_predict_fixed.py \
          "$snap" $COMP/outputs/hmat_v3_$tag >> $LOG 2>&1
      conda run -n mural --no-capture-output python $COMP/scripts/eval_outputs.py \
          $COMP/outputs/hmat_v3_$tag $COMP/data/mural_lama/gt_256 "hmat-v3-k2000-$tag" >> $LOG 2>&1
  done
  echo "[$(date '+%F %T')] STAGE 3 done" | tee -a $LOG
} || echo "[$(date '+%F %T')] STAGE 3 (continuation) FAILED" | tee -a $LOG

echo "[$(date '+%F %T')] FINAL_FINISHER_DONE" | tee -a $LOG
