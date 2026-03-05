#!/bin/bash
#SBATCH --job-name=dryrun_yolov9
#SBATCH --time=1:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/dryrun_yolov9_%j.out
#SBATCH -e /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/dryrun_yolov9_%j.err

module purge
module load miniforge3
module load cuda/11.8

eval "$(conda shell.bash hook)"
conda activate bubbly-train-env

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Model:  yolov9"
echo "---"

export MICROSAM_CACHEDIR=/users/eaguerov/scratch/bubble-models/microsam

/users/eaguerov/.conda/envs/bubbly-train-env/bin/python /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/scripts/train_yolov9.py \
    --dataset /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/microsam/datasets/v00_train \
    --name dryrun_yolov9 \
    --epochs 2 \
    --save_root /users/eaguerov/scratch/bubble-models/trained
