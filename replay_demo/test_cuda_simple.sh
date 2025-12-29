#!/bin/bash
# Quick CUDA availability test for Singularity container

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/seir_cuda.sif" ]; then
    echo "Error: seir_cuda.sif not found in $SCRIPT_DIR"
    exit 1
fi

echo "=== Testing CUDA availability in container ==="
echo ""

# Use the workaround script if it exists, otherwise try --nv
if [ -f "$SCRIPT_DIR/run_gpu_workaround.sh" ]; then
    echo "Using manual NVIDIA library bindings..."
    echo ""
    "$SCRIPT_DIR/run_gpu_workaround.sh" --device gpu --n-sims 10
else
    echo "Using --nv flag..."
    echo ""
    singularity exec --nv "$SCRIPT_DIR/seir_cuda.sif" python3 -c "
from numba import cuda
import sys

print('Checking CUDA availability...')
if cuda.is_available():
    print('✓ CUDA is available!')
    print(f'  Device: {cuda.get_current_device().name}')
    print(f'  Compute Capability: {cuda.get_current_device().compute_capability}')
    sys.exit(0)
else:
    print('✗ CUDA is NOT available')
    print('  This usually means:')
    print('  1. --nv flag is not working (ldconfig not configured)')
    print('  2. NVIDIA drivers are not loaded')
    print('  3. No GPU is present')
    sys.exit(1)
"
fi
