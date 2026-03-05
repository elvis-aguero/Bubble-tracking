#!/bin/bash
#SBATCH --job-name=dryrun_stardist
#SBATCH --time=1:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/dryrun_stardist_%j.out
#SBATCH -e /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/dryrun_stardist_%j.err

module purge
module load miniforge3
module load cuda/11.8

eval "$(conda shell.bash hook)"
conda activate bubbly-train-env

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Model:  stardist"
echo "---"

export MICROSAM_CACHEDIR=/users/eaguerov/scratch/bubble-models/microsam

/users/eaguerov/.conda/envs/bubbly-train-env/bin/python /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/scripts/train_stardist.py \
    --dataset /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/microsam/datasets/v00_train \
    --name dryrun_stardist \
    --epochs 2 \
    --save_root /users/eaguerov/scratch/bubble-models/trained
