#!/bin/bash
# Check what's causing the overlay conflicts

echo "=== Checking container's existing NVIDIA files ==="
echo ""

echo "What NVIDIA files exist in the container (without --nv)?"
singularity exec seir_cuda.sif bash -c "ls -la /usr/lib/x86_64-linux-gnu/libcuda* 2>/dev/null || echo 'No libcuda files in container'"
echo ""

singularity exec seir_cuda.sif bash -c "ls -la /usr/lib/x86_64-linux-gnu/libnvidia* 2>/dev/null | head -10 || echo 'No libnvidia files in container'"
echo ""

echo "=== With --nv flag (should add host NVIDIA libs) ==="
singularity exec --nv seir_cuda.sif bash -c "ls -la /usr/lib/x86_64-linux-gnu/libcuda* 2>&1"
echo ""

echo "=== Checking if CUDA runtime exists in container ==="
singularity exec seir_cuda.sif bash -c "ls -la /usr/local/cuda/lib64/libcudart* 2>/dev/null || echo 'No CUDA runtime in /usr/local/cuda/lib64'"
echo ""

echo "=== Testing CUDA with verbose Numba output ==="
singularity exec --nv seir_cuda.sif bash -c "NUMBA_ENABLE_CUDASIM=0 python3 -c '
import os
os.environ[\"NUMBA_CUDA_LOG_LEVEL\"] = \"DEBUG\"

print(\"Attempting to import and use CUDA...\")
try:
    from numba import cuda
    print(f\"CUDA module imported\")
    print(f\"CUDA available: {cuda.is_available()}\")

    if not cuda.is_available():
        print(\"\\nAttempting to detect CUDA...\")
        try:
            detected = cuda.detect()
            print(f\"Detected: {detected}\")
        except Exception as e:
            print(f\"Detection error: {e}\")
            import traceback
            traceback.print_exc()
except Exception as e:
    print(f\"Import/check failed: {e}\")
    import traceback
    traceback.print_exc()
'" 2>&1 | head -50
