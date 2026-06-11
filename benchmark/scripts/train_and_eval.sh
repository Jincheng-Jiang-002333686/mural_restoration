#!/bin/bash
# Full LaMa pipeline: train -> predict on fixed test masks -> MAT-style metrics.
# Logs: lama_mat_comparison/logs/. Metrics go to wandb project mural_inpainting
# and results/test_metrics.csv.
set -euo pipefail

COMP=/home/jincheng/Mural/mural_project/lama_mat_comparison
export TORCH_HOME=$COMP/torch_home
cd $COMP/external/lama
export PYTHONPATH=$PWD

echo "[1/4] training..."
conda run -n lama --no-capture-output python bin/train.py -cn mural-lama-fourier \
    run_title=full > $COMP/logs/full_train.log 2>&1

EXP=$(ls -td $COMP/experiments/*mural-lama-fourier_full* | head -1)
echo "[2/4] predicting with $EXP (last.ckpt)..."
conda run -n lama --no-capture-output python bin/predict.py \
    model.path=$EXP model.checkpoint=last.ckpt \
    indir=$COMP/data/mural_lama/val \
    outdir=$COMP/outputs/lama_mixed02_03_raw > $COMP/logs/predict.log 2>&1

echo "[3/4] renaming outputs to GT names..."
mkdir -p $COMP/outputs/lama_mixed02_03
for f in $COMP/outputs/lama_mixed02_03_raw/*_mask001.png; do
    b=$(basename "$f")
    cp "$f" "$COMP/outputs/lama_mixed02_03/${b%_mask001.png}.png"
done
ls $COMP/outputs/lama_mixed02_03 | wc -l

echo "[4/4] final metrics (MAT implementations)..."
conda run -n mural --no-capture-output python $COMP/scripts/eval_outputs.py \
    $COMP/outputs/lama_mixed02_03 $COMP/data/mural_lama/gt_256 lama-fourier \
    > $COMP/logs/final_eval.log 2>&1
tail -5 $COMP/logs/final_eval.log
echo "PIPELINE_DONE"
