#!/bin/bash
#SBATCH --job-name=microsam_seed_v04_run1
#SBATCH --time=4:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.out
#SBATCH -e /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.err

# Load Modules (Oscar Standard)
# We use minimal modules and rely on the conda env for python packages
module purge
module load miniforge3
module load cuda/11.8

# Activate Conda Env
# We use the universal hook to initialize conda, avoiding ambiguous CONDA_PREFIX paths
eval "$(conda shell.bash hook)"
conda activate bubbly-train-env

# Echo Info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Dataset: seed_v04_test"

# Run Training
export MICROSAM_CACHEDIR=/users/eaguerov/scratch/bubble-models/microsam
python3 /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/scripts/train.py \
    --dataset /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/microsam/datasets/seed_v04_test \
    --name microsam_seed_v04_run1 \
    --epochs 100 \
    --save_root /users/eaguerov/scratch/bubble-models/trained
