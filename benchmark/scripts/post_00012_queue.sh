#!/bin/bash
# Self-gating GPU queue: waits for run 00012 to finish, then runs the three
# ready stages back-to-back so the 3090 is never idle. Each stage is error-
# isolated (a failure logs and the queue continues). All rows land in
# lama_mat_comparison/results/test_metrics.csv + wandb project mural_inpainting.
set -uo pipefail

PROJ=/home/jincheng/Mural/mural_project
COMP=$PROJ/lama_mat_comparison
RUN=$PROJ/MAT/outputs/00012-train_ref-places256-glr0.001-dlr0.0001-gamma2-pr0-kimg1000-batch16-bgpu4-sm0.0-noaug-resumecustom
Q=$COMP/logs/post_00012_queue.log

echo "[$(date '+%F %T')] waiting for run 00012 to finish (need snapshot 001000 + trainer exit)..." | tee -a $Q
until [ -f "$RUN/network-snapshot-001000.pkl" ] && ! pgrep -f "train.py --outdir" >/dev/null; do
    sleep 60
done
# settle: let the final metric pass + wandb flush complete
sleep 30
echo "[$(date '+%F %T')] 00012 done. GPU free. starting queue." | tee -a $Q

# ---- Stage 1: benchmark best HMAT v2 snapshot(s) on DHMural fixed masks ----
{
  echo "[$(date '+%F %T')] STAGE 1: HMAT v2 final benchmark" | tee -a $Q
  mapfile -t SNAPS < <(conda run -n mural python $COMP/scripts/select_best_snapshot.py $RUN | grep network-snapshot)
  for entry in "${SNAPS[@]}"; do
      tag=$(echo "$entry" | cut -d' ' -f1); snap=$(echo "$entry" | cut -d' ' -f2)
      echo "[$(date '+%F %T')]   benchmarking $tag = $(basename $snap)" | tee -a $Q
      conda run -n mural --no-capture-output python $COMP/scripts/hmat_predict_fixed.py \
          "$snap" $COMP/outputs/hmat_v2_$tag >> $Q 2>&1
      conda run -n mural --no-capture-output python $COMP/scripts/eval_outputs.py \
          $COMP/outputs/hmat_v2_$tag $COMP/data/mural_lama/gt_256 "hmat-v2-$tag" >> $Q 2>&1
      grep "psnr:" $COMP/logs/*.log 2>/dev/null | tail -1
  done
} || echo "[$(date '+%F %T')] STAGE 1 FAILED" | tee -a $Q

# ---- Stage 2: EdgeConnect (DHMural 4th baseline), GPU-smoke-gated ----
{
  echo "[$(date '+%F %T')] STAGE 2: EdgeConnect GPU smoke (20 iters each model)" | tee -a $Q
  EC=$PROJ/edge-connect
  rm -rf $EC/checkpoints/gpu_smoke; mkdir -p $EC/checkpoints/gpu_smoke
  ls $COMP/data/ec_train_masks/*.png | head -200 > $EC/datasets/gpu_smoke_masks.flist
  sed -e 's#datasets/mural_train_masks.flist#datasets/gpu_smoke_masks.flist#' \
      -e 's/^MAX_ITERS: 75000/MAX_ITERS: 20/' -e 's/^SAVE_INTERVAL: 5000/SAVE_INTERVAL: 20/' \
      -e 's/^SAMPLE_INTERVAL: 5000/SAMPLE_INTERVAL: 0/' -e 's/^BATCH_SIZE: 8/BATCH_SIZE: 4/' \
      $EC/checkpoints/mural/config.yml > $EC/checkpoints/gpu_smoke/config.yml
  cd $EC
  if conda run -n mural --no-capture-output python train.py --path checkpoints/gpu_smoke --model 1 >> $Q 2>&1 \
     && conda run -n mural --no-capture-output python train.py --path checkpoints/gpu_smoke --model 2 >> $Q 2>&1; then
      echo "[$(date '+%F %T')] EC smoke OK -> full run" | tee -a $Q
      bash $COMP/scripts/run_edgeconnect.sh >> $Q 2>&1
  else
      echo "[$(date '+%F %T')] STAGE 2 EC GPU SMOKE FAILED - skipping full EdgeConnect" | tee -a $Q
  fi
} || echo "[$(date '+%F %T')] STAGE 2 FAILED" | tee -a $Q

# ---- Stage 3: HMAT v2 on Nine-Colored Deer ----
{
  echo "[$(date '+%F %T')] STAGE 3: deer HMAT v2" | tee -a $Q
  bash $COMP/scripts/run_deer_hmat.sh >> $Q 2>&1
} || echo "[$(date '+%F %T')] STAGE 3 FAILED" | tee -a $Q

echo "[$(date '+%F %T')] QUEUE_DONE" | tee -a $Q
