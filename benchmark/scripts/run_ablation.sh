#!/bin/bash
# HMAT component ablation on DHMural (20-30% masks). Trains the full model and
# three ablations FROM SCRATCH under the identical v2 recipe to an equal budget,
# then benchmarks each on the fixed test masks -> ablation rows in the CSV.
# Fair Table 3: only the ablated component differs between runs.
set -uo pipefail

PROJ=/home/jincheng/Mural/mural_project
COMP=$PROJ/lama_mat_comparison
LOG=$COMP/logs/ablation.log
KIMG=${1:-600}   # per-run budget (override: run_ablation.sh 400)

echo "[$(date '+%F %T')] ABLATION START, budget=${KIMG} kimg/run" | tee -a $LOG
cd $PROJ/MAT

# variant tag -> generator class
declare -A GEN=(
  [full]="networks.mat.Generator"
  [nomadf]="networks.mat_nomadf.Generator"
  [notf]="networks.mat_notf.Generator"
  [nomgstyle]="networks.mat_nomgstyle.Generator"
)

for tag in full nomadf notf nomgstyle; do
  {
    echo "[$(date '+%F %T')] TRAIN ablation:$tag (${GEN[$tag]})" | tee -a $LOG
    conda run -n mural --no-capture-output python train.py \
      --outdir=outputs \
      --data=$PROJ/train_ref --data_val=$PROJ/test_ref \
      --dataloader=datasets.dataset_256.ImageFolderMaskDataset \
      --generator=${GEN[$tag]} \
      --metrics=fid2649_full,psnr2649_full \
      --gpus=1 --batch=16 --batch-gpu=4 --kimg=$KIMG --snap=10 \
      --cfg=places256 --aug=noaug --workers=2 --style_mix=0.0 --pr=0.0 \
      --glr=0.001 --dlr=0.0001 --gamma=2.0 \
      --wandb-project=mural_inpainting >> $LOG 2>&1

    GTAG=$(echo "${GEN[$tag]}" | cut -d. -f2)   # mat | mat_nomadf | mat_notf | mat_nomgstyle
    EXP=$(ls -td $PROJ/MAT/outputs/*-train_ref-places256-${GTAG}-glr* 2>/dev/null | head -1)
    SNAP=$(conda run -n mural python $COMP/scripts/select_best_snapshot.py "$EXP" | grep network-snapshot | head -1 | cut -d' ' -f2)
    echo "[$(date '+%F %T')] BENCH ablation:$tag $(basename $SNAP)" | tee -a $LOG
    conda run -n mural --no-capture-output python $COMP/scripts/hmat_predict_fixed.py \
        "$SNAP" $COMP/outputs/abl_$tag >> $LOG 2>&1
    conda run -n mural --no-capture-output python $COMP/scripts/eval_outputs.py \
        $COMP/outputs/abl_$tag $COMP/data/mural_lama/gt_256 "abl-$tag-k$KIMG" >> $LOG 2>&1
  } || echo "[$(date '+%F %T')] ablation:$tag FAILED" | tee -a $LOG
done

echo "[$(date '+%F %T')] ABLATION_DONE" | tee -a $LOG
