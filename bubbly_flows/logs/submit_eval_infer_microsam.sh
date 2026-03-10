#!/bin/bash
#SBATCH --job-name=eval_infer_microsam
#SBATCH --time=0:30:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.out
#SBATCH -e /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.err

module purge
module load miniforge3
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate bubbly-train-env

for img in /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/microsam/datasets/seed_v04_test/images/*.png; do
    stem=$(basename "$img" .png)
    python3 /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/scripts/inference.py \
        --model_path /users/eaguerov/scratch/bubble-models/trained/microsam_seed_v04_run2/checkpoints/microsam_seed_v04_run2/best.pt \
        --image "$img" \
        --output /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/tests/output/eval_preds/microsam/${stem}.png \
        --model_type vit_b
done
