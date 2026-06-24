#!/bin/bash
# After the component ablation finishes (ABLATION_DONE) + GPU free:
#   1. hole-region metrics on nomgstyle (completes Table 3 hole columns)
#   2. train TF-free architecture (mat_notf) with the OLD loss (adv + VGG only,
#      no hole-L1 / FM / HRF) @600 kimg  ->  loss-ablation row
#   3. full + hole benchmark of the old-loss model
# Loss comparison = abl-notf-k600 (new loss, already done) vs notf-oldloss.
set -uo pipefail

PROJ=/home/jincheng/Mural/mural_project
COMP=$PROJ/lama_mat_comparison
ALOG=$COMP/logs/ablation.log
LOG=$COMP/logs/post_ablation_loss.log
GT=$COMP/data/mural_lama/gt_256
MASKS=$COMP/masks/test_mixed_hole02_03

echo "[$(date '+%F %T')] waiting for ABLATION_DONE + free GPU..." | tee -a $LOG
until grep -q "ABLATION_DONE" "$ALOG" 2>/dev/null && ! pgrep -f "train.py --outdir" >/dev/null; do
    sleep 120
done
sleep 30

# 1. hole metrics for the remaining trained ablations (CPU)
{
  for v in nomgstyle; do
    [ -d $COMP/outputs/abl_$v ] && conda run -n mural --no-capture-output python $COMP/scripts/hole_metrics.py \
        $COMP/outputs/abl_$v $GT $MASKS abl-$v-k600 >> $LOG 2>&1
  done
  echo "[$(date '+%F %T')] hole metrics (nomgstyle) done" | tee -a $LOG
} || echo "[$(date '+%F %T')] hole metrics step FAILED" | tee -a $LOG

# 2. train TF-free + OLD loss @600
{
  echo "[$(date '+%F %T')] TRAIN notf + OLD loss (adv+VGG only) @600" | tee -a $LOG
  cd $PROJ/MAT
  conda run -n mural --no-capture-output python train.py \
    --outdir=outputs \
    --data=$PROJ/train_ref --data_val=$PROJ/test_ref \
    --dataloader=datasets.dataset_256.ImageFolderMaskDataset \
    --generator=networks.mat_notf.Generator \
    --metrics=fid2649_full,psnr2649_full \
    --gpus=1 --batch=16 --batch-gpu=4 --kimg=600 --snap=10 \
    --cfg=places256 --aug=noaug --workers=2 --style_mix=0.0 \
    --pr=1.0 --l1w=0 --fmw=0 --hrfw=0 --glr=0.001 --dlr=0.0001 --gamma=2.0 \
    --wandb-project=mural_inpainting >> $LOG 2>&1

  EXP=$(ls -td $PROJ/MAT/outputs/*-mat_notf-*pr1-l1w0-fmw0-hrfw0* 2>/dev/null | head -1)
  SNAP=$(conda run -n mural python $COMP/scripts/select_best_snapshot.py "$EXP" | grep network-snapshot | head -1 | cut -d' ' -f2)
  echo "[$(date '+%F %T')] BENCH notf-oldloss $(basename $SNAP)" | tee -a $LOG
  conda run -n mural --no-capture-output python $COMP/scripts/hmat_predict_fixed.py \
      "$SNAP" $COMP/outputs/loss_notf_oldloss >> $LOG 2>&1
  conda run -n mural --no-capture-output python $COMP/scripts/eval_outputs.py \
      $COMP/outputs/loss_notf_oldloss $GT loss-notf-oldloss-k600 >> $LOG 2>&1
  conda run -n mural --no-capture-output python $COMP/scripts/hole_metrics.py \
      $COMP/outputs/loss_notf_oldloss $GT $MASKS loss-notf-oldloss-k600 >> $LOG 2>&1
} || echo "[$(date '+%F %T')] OLD-loss run FAILED" | tee -a $LOG

echo "[$(date '+%F %T')] POST_ABLATION_LOSS_DONE" | tee -a $LOG
