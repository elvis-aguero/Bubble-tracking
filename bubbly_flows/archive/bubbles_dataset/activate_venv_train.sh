module load python/3.11
module load cudnn/9
module load cuda/12

# Determine this script directory and repository root (two levels up from this file)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Name for the training venv expected at repo root; keep original name
VENV_NAME="ultralytics-bubble"
VENV_PATH="$REPO_ROOT/$VENV_NAME"

if [ -f "$VENV_PATH/bin/activate" ]; then
	source "$VENV_PATH/bin/activate"
	echo "Environment activated."
	python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
else
	echo "Warning: training venv not found at $VENV_PATH; falling back to system python"
fi
