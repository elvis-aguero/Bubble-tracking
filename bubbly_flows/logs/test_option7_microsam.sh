#!/bin/bash
#SBATCH -J test_eval_microsam
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -o /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/test_option7_microsam_%j.out
#SBATCH -e /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/test_option7_microsam_%j.err

set -e
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate bubbly-train-env

REPO=/oscar/data/dharri15/eaguerov/Github/Bubble-tracking
SCRIPTS=$REPO/bubbly_flows/scripts
CKPT=~/scratch/bubble-models/trained/microsam_seed_v04_run2/checkpoints/microsam_seed_v04_run2/best.pt
TEST_IMGS=$REPO/bubbly_flows/microsam/datasets/seed_v04_test/images
TEST_LBLS=$REPO/bubbly_flows/microsam/datasets/seed_v04_test/labels
PRED_DIR=$REPO/bubbly_flows/tests/output/eval_preds/microsam_seed_v04_run2

mkdir -p $PRED_DIR

echo "=== Running MicroSAM inference on test set ==="
for img in $TEST_IMGS/*.png; do
    stem=$(basename $img .png)
    out=$PRED_DIR/${stem}.png
    echo "Inferring $stem ..."
    python3 $SCRIPTS/inference.py \
        --model_path $CKPT \
        --image $img \
        --output $out \
        --model_type vit_b
done

echo "=== Running evaluation ==="
python3 $SCRIPTS/evaluate.py \
    --preds $PRED_DIR \
    --gts $TEST_LBLS \
    --iou_threshold 0.5 \
    --output $PRED_DIR/results.csv

echo "=== Done. Results at $PRED_DIR/results.csv ==="
