#!/bin/bash

# ================= CONFIGURATION =================
# Name of the virtual environment directory
VENV_NAME="x-labeling-env"
# Where to store the environment
# Default: place the virtual environment at the repository root (one level
# above this script). This avoids using a fragile user-specific path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# repo root is the folder above bubbly_flows/
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$BASE_DIR/$VENV_NAME"

# Force ONNX Runtime to use only 1 thread (prevents the affinity crash)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Tell ONNX Runtime to be passive, not aggressive
export OMP_WAIT_POLICY=PASSIVE

# OSCAR MODULES (Adjust these if you used specific versions)
# We load these to ensure the system has the right drivers before Python starts
echo " [1/5] Loading Oscar Modules..."
module purge
module load python/3.11
module load cudnn/8.9
module load cuda/11

# ================= SETUP LOGIC =================
# Check if the Base Directory exists, create if not
if [ ! -d "$BASE_DIR" ]; then
    echo " [2/5] Creating workspace at $BASE_DIR..."
    mkdir -p "$BASE_DIR"
fi

# Check if Venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "==================================================="
    echo " FIRST RUN DETECTED: Setting up environment..."
    echo " This may take 2-5 minutes. Please wait."
    echo "==================================================="
    
    # Create Venv
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Packages (The specific GPU configuration)
    echo " Installing X-AnyLabeling and GPU Runtime..."
    pip install "numpy<2.0"
    pip install "x-anylabeling-cvhub[gpu]"

    
    # Force GPU version of onnxruntime (uninstall cpu version if present)
    pip uninstall -y onnxruntime
    pip install onnxruntime-gpu==1.18.0
    
    echo " Installation Complete!"
else
    echo " [2/5] Environment found. Activating..."
    source "$VENV_PATH/bin/activate"
fi

# ================= GPU CHECK =================
echo " [3/5] Checking GPU Status..."
python3 -c "import onnxruntime as ort; print('Available Providers:', ort.get_available_providers()); assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'ERROR: GPU NOT DETECTED'"

if [ $? -eq 0 ]; then
    echo " [SUCCESS] GPU is active and ready."
else
    echo " [WARNING] GPU was NOT detected. The tool will run slowly on CPU."
    read -p "Press Enter to continue anyway..."
fi

# ================= LAUNCH =================
echo " [4/5] Launching X-AnyLabeling..."
echo " Please keep this black terminal window OPEN while using the tool."
echo "==================================================="

ORT_CAPI_PATH=$(python3 -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))")

# Force Linux to look in that folder for the missing library
export LD_LIBRARY_PATH=$ORT_CAPI_PATH:$LD_LIBRARY_PATH

echo "Debug: Added ONNX Runtime path: $ORT_CAPI_PATH"

# Launch the tool
xanylabeling

echo " [5/5] Session Closed."
