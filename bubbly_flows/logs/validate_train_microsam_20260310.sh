#!/bin/bash
#SBATCH --job-name=wfval_msam
#SBATCH --time=01:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.out
#SBATCH -e /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.err
module purge
module load miniforge3
module load cuda/11.8
cd /oscar/data/dharri15/eaguerov/Github/Bubble-tracking
export MICROSAM_CACHEDIR=$HOME/scratch/bubble-models/pipeline
eval "$(conda shell.bash hook)"
conda activate bubbly-train-env
python3 bubbly_flows/scripts/train.py \
  --dataset /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/pipeline/datasets/seed_v04_train \
  --name wf_validate_microsam_20260310 \
  --config /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/configs/microsam.json \
  --save_root $HOME/scratch/bubble-models/trained
