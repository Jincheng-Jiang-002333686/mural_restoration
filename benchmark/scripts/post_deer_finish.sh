#!/bin/bash
# Self-gating: after the baseline queue finishes (deer run = last stage, QUEUE_DONE)
# and the GPU is free, do two things in order on the freed 3090:
#   1. EdgeConnect TEST + EVAL only (models already trained in checkpoints/mural;
#      the earlier full run trained fine, only the test-mask loader bug failed,
#      now fixed) -> edgeconnect row
#   2. HMAT continuation +1000 kimg from run 00012 k1000 (same v2 recipe) ->
#      benchmark best snapshot -> hmat-v3-k2000 rows
set -uo pipefail

PROJ=/home/jincheng/Mural/mural_project
COMP=$PROJ/lama_mat_comparison
EC=$PROJ/edge-connect
QLOG=$COMP/logs/post_00012_queue.log
LOG=$COMP/logs/post_deer_finish.log
SRC=$PROJ/MAT/outputs/00012-train_ref-places256-glr0.001-dlr0.0001-gamma2-pr0-kimg1000-batch16-bgpu4-sm0.0-noaug-resumecustom/network-snapshot-001000.pkl

echo "[$(date '+%F %T')] waiting for QUEUE_DONE (deer finished) + free GPU..." | tee -a $LOG
until grep -q "QUEUE_DONE" "$QLOG" 2>/dev/null && ! pgrep -f "train.py --outdir" >/dev/null; do
    sleep 120
done
sleep 30

# ---- 1. EdgeConnect test + eval (no retraining) ----
{
  echo "[$(date '+%F %T')] EC test (model 3) on fixed masks..." | tee -a $LOG
  cd $EC
  conda run -n mural --no-capture-output python test.py \
      --path checkpoints/mural --model 3 \
      --input $EC/datasets/mural_test.flist \
      --mask $EC/datasets/mural_test_masks.flist \
      --output $COMP/outputs/edgeconnect_mixed02_03 >> $LOG 2>&1
  echo "[$(date '+%F %T')] EC outputs: $(ls $COMP/outputs/edgeconnect_mixed02_03 2>/dev/null | wc -l)" | tee -a $LOG
  conda run -n mural --no-capture-output python $COMP/scripts/eval_outputs.py \
      $COMP/outputs/edgeconnect_mixed02_03 $COMP/data/mural_lama/gt_256 edgeconnect >> $LOG 2>&1
  grep -m1 "psnr:" <(tail -40 $LOG) || true
} || echo "[$(date '+%F %T')] EC TEST/EVAL FAILED" | tee -a $LOG

# ---- 2. HMAT continuation +1000 kimg ----
{
  echo "[$(date '+%F %T')] HMAT continuation +1000 kimg from k1000..." | tee -a $LOG
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

  CONT=$(ls -td $PROJ/MAT/outputs/0001[3-9]*resumecustom 2>/dev/null | head -1)
  echo "[$(date '+%F %T')] continuation done: $CONT" | tee -a $LOG
  mapfile -t SNAPS < <(conda run -n mural python $COMP/scripts/select_best_snapshot.py "$CONT" | grep network-snapshot)
  for entry in "${SNAPS[@]}"; do
      tag=$(echo "$entry" | cut -d' ' -f1); snap=$(echo "$entry" | cut -d' ' -f2)
      conda run -n mural --no-capture-output python $COMP/scripts/hmat_predict_fixed.py \
          "$snap" $COMP/outputs/hmat_v3_$tag >> $LOG 2>&1
      conda run -n mural --no-capture-output python $COMP/scripts/eval_outputs.py \
          $COMP/outputs/hmat_v3_$tag $COMP/data/mural_lama/gt_256 "hmat-v3-k2000-$tag" >> $LOG 2>&1
  done
} || echo "[$(date '+%F %T')] HMAT CONTINUATION FAILED" | tee -a $LOG

echo "[$(date '+%F %T')] POST_DEER_DONE" | tee -a $LOG
