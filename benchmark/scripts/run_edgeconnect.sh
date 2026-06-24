#!/bin/bash
# EdgeConnect baseline: train edge model -> inpaint model -> test -> metrics.
# Run AFTER the HMAT retrain releases the GPU. ~600 kimg budget per stage
# (75k iters x batch 8), matching HMAT/LaMa.
set -euo pipefail

EC=/home/jincheng/Mural/mural_project/edge-connect
COMP=/home/jincheng/Mural/mural_project/lama_mat_comparison
cd $EC

N_MASKS=$(ls $COMP/data/ec_train_masks/*.png | wc -l)
[ "$N_MASKS" -ge 20000 ] || { echo "mask bank incomplete ($N_MASKS/20000)"; exit 1; }
ls $COMP/data/ec_train_masks/*.png | sort > datasets/mural_train_masks.flist
wc -l datasets/mural_train_masks.flist

echo "[1/4] training edge model (model 1)..."
conda run -n mural --no-capture-output python train.py \
    --path checkpoints/mural --model 1 > $COMP/logs/ec_train_edge.log 2>&1

echo "[2/4] training inpaint model (model 2)..."
conda run -n mural --no-capture-output python train.py \
    --path checkpoints/mural --model 2 > $COMP/logs/ec_train_inpaint.log 2>&1

echo "[3/4] testing (model 3: edge-inpaint) on fixed masks..."
conda run -n mural --no-capture-output python test.py \
    --path checkpoints/mural --model 3 \
    --input $EC/datasets/mural_test.flist \
    --mask $EC/datasets/mural_test_masks.flist \
    --output $COMP/outputs/edgeconnect_mixed02_03 > $COMP/logs/ec_test.log 2>&1
ls $COMP/outputs/edgeconnect_mixed02_03 | wc -l

echo "[4/4] metrics (MAT implementations)..."
conda run -n mural --no-capture-output python $COMP/scripts/eval_outputs.py \
    $COMP/outputs/edgeconnect_mixed02_03 $COMP/data/mural_lama/gt_256 edgeconnect \
    > $COMP/logs/ec_eval.log 2>&1
grep "psnr:" $COMP/logs/ec_eval.log
echo "EC_PIPELINE_DONE"
