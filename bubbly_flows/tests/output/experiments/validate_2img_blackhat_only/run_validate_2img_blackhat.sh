#!/usr/bin/env bash
#SBATCH --job-name=bh2img
#SBATCH --output=/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/tests/output/experiments/validate_2img_blackhat_only/slurm_%j.out
#SBATCH --error=/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/tests/output/experiments/validate_2img_blackhat_only/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -eo pipefail
source ~/.bashrc
conda activate bubbly-train-env
cd /oscar/data/dharri15/eaguerov/Github/Bubble-tracking
python - <<'PY'
import json
from pathlib import Path
from bubbly_flows.tests.src.experiments.gold_eval import prepare_gold_evaluation_set
from bubbly_flows.tests.src.experiments.provenance import ExperimentSpec
from bubbly_flows.tests.src.experiments.runner import execute_real_experiment_batch

root = Path('/oscar/data/dharri15/eaguerov/Github/Bubble-tracking')
gold_json_dir = root / 'bubbly_flows' / 'annotations' / 'gold' / 'gold_seed_v04' / 'labels_json'
output_eval_root = root / 'bubbly_flows' / 'tests' / 'output' / 'eval_sets' / 'gold_seed_v04'
image_search_dirs = [
    root / 'bubbly_flows' / 'data' / 'frames' / 'images_16bit_png',
    root / 'bubbly_flows' / 'data' / 'frames' / 'images_clahe',
    root / 'bubbly_flows' / 'data' / 'frames' / 'images_raw',
    root / 'bubbly_flows' / 'annotations' / 'pool',
]
manifest = prepare_gold_evaluation_set(gold_json_dir, output_eval_root, image_search_dirs)
manifest['items'] = manifest['items'][:2]
manifest['count'] = len(manifest['items'])

batch_root = root / 'bubbly_flows' / 'tests' / 'output' / 'experiments' / 'validate_2img_blackhat_only'
(batch_root / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')

specs = [
    ExperimentSpec('validate_blackhat_only', 'blackhat_only', 'baseline', {
        'blackhat.radius': 5,
        'blackhat.percentile': 99.0,
        'blackhat.area_min': 30,
        'blackhat.area_max': 120,
    }),
]
summary = execute_real_experiment_batch(specs, manifest, batch_root)
(batch_root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
print(batch_root)
PY
