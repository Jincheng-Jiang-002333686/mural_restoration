#!/bin/bash
# Self-gating HMAT continuation: waits for the baseline queue (EdgeConnect + deer)
# to finish, then resumes HMAT from run 00012's k1000 snapshot for +1000 kimg
# (same v2 recipe), then benchmarks the best snapshot on the DHMural fixed masks.
# Resume restarts the kimg counter at 0, so --kimg=1000 == effective k1000->k2000.
set -uo pipefail

PROJ=/home/jincheng/Mural/mural_project
COMP=$PROJ/lama_mat_comparison
QLOG=$COMP/logs/post_00012_queue.log
LOG=$COMP/logs/hmat_continue.log
SRC=$PROJ/MAT/outputs/00012-train_ref-places256-glr0.001-dlr0.0001-gamma2-pr0-kimg1000-batch16-bgpu4-sm0.0-noaug-resumecustom/network-snapshot-001000.pkl

echo "[$(date '+%F %T')] waiting for baseline queue (QUEUE_DONE) + free GPU..." | tee -a $LOG
until grep -q "QUEUE_DONE" "$QLOG" 2>/dev/null && ! pgrep -f "train.py --outdir" >/dev/null; do
    sleep 120
done
sleep 30
echo "[$(date '+%F %T')] queue done. resuming HMAT +1000 kimg from k1000." | tee -a $LOG

cd $PROJ/MAT
conda run -n mural --no-capture-output python train.py \
  --outdir=outputs \
  --data=$PROJ/train_ref --data_val=$PROJ/test_ref \
  --dataloader=datasets.dataset_256.ImageFolderMaskDataset \
  --metrics=fid2649_full,psnr2649_full \
  --gpus=1 --batch=16 --batch-gpu=4 --kimg=1000 --snap=10 \
  --cfg=places256 --aug=noaug --workers=2 --style_mix=0.0 --pr=0.0 \
  --glr=0.001 --dlr=0.0001 --gamma=2.0 \
  --resume=$SRC \
  --wandb-project=mural_inpainting >> $LOG 2>&1

CONT=$(ls -td $PROJ/MAT/outputs/0001[3-9]*resumecustom 2>/dev/null | head -1)
echo "[$(date '+%F %T')] continuation done: $CONT. benchmarking best snapshot." | tee -a $LOG
mapfile -t SNAPS < <(conda run -n mural python $COMP/scripts/select_best_snapshot.py "$CONT" | grep network-snapshot)
for entry in "${SNAPS[@]}"; do
    tag=$(echo "$entry" | cut -d' ' -f1); snap=$(echo "$entry" | cut -d' ' -f2)
    conda run -n mural --no-capture-output python $COMP/scripts/hmat_predict_fixed.py \
        "$snap" $COMP/outputs/hmat_v3_$tag >> $LOG 2>&1
    conda run -n mural --no-capture-output python $COMP/scripts/eval_outputs.py \
        $COMP/outputs/hmat_v3_$tag $COMP/data/mural_lama/gt_256 "hmat-v3-k2000-$tag" >> $LOG 2>&1
done
echo "[$(date '+%F %T')] HMAT_CONTINUE_DONE" | tee -a $LOG
