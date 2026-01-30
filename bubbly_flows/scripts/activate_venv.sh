module load python/3.11
module load cudnn/8.9
module load cuda/11

# Determine this script directory and repository root (parent of bubbly_flows)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default venv name (matches xanylabel.sh)
VENV_NAME="x-labeling-env"
VENV_PATH="$REPO_ROOT/$VENV_NAME"

if [ -f "$VENV_PATH/bin/activate" ]; then
	source "$VENV_PATH/bin/activate"
else
	echo "Warning: venv not found at $VENV_PATH; falling back to system python"
fi
