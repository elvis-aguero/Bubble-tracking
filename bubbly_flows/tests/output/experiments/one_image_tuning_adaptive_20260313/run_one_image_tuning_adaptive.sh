#!/usr/bin/env bash
#SBATCH -J oneimg_adapt
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o bubbly_flows/tests/output/experiments/one_image_tuning_adaptive_20260313/slurm_%j.out
#SBATCH -e bubbly_flows/tests/output/experiments/one_image_tuning_adaptive_20260313/slurm_%j.err

set -euo pipefail
cd /oscar/data/dharri15/eaguerov/Github/Bubble-tracking
PYTHON_BIN="/users/eaguerov/.conda/envs/sam3/bin/python"
echo "Using orchestrator: ${PYTHON_BIN}"
"${PYTHON_BIN}" bubbly_flows/tests/run_one_image_tuning.py \
  --strategy adaptive \
  --image-stem ZeroG_FlightDay_Test_C1S0014_img006001 \
  --output-root bubbly_flows/tests/output/experiments/one_image_tuning_adaptive_20260313/results
