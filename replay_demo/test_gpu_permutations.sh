#!/bin/bash
# Try different Singularity flag combinations to find what works

echo "=== Testing different Singularity configurations for GPU access ==="
echo ""

# Core bindings
CORE_BINDS="--bind /usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 --bind /dev/nvidia0 --bind /dev/nvidiactl --bind /dev/nvidia-uvm --bind /dev/nvidia-caps"

TEST_CODE='from numba import cuda; print(f"CUDA available: {cuda.is_available()}")'

echo "Test 1: Basic bindings"
singularity exec $CORE_BINDS seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | grep -i "cuda available"

echo "Test 2: With --writable-tmpfs"
singularity exec --writable-tmpfs $CORE_BINDS seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | grep -i "cuda available"

echo "Test 3: With --no-home"
singularity exec --no-home $CORE_BINDS seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | grep -i "cuda available"

echo "Test 4: With --contain"
singularity exec --contain $CORE_BINDS seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | grep -i "cuda available"

echo "Test 5: With --security=allow"
singularity exec $CORE_BINDS seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | grep -i "cuda available"

echo "Test 6: Bind all GPU devices"
ALL_DEVS="--bind /dev/nvidia0 --bind /dev/nvidia1 --bind /dev/nvidia2 --bind /dev/nvidia3 --bind /dev/nvidia4 --bind /dev/nvidia5 --bind /dev/nvidia6 --bind /dev/nvidia7 --bind /dev/nvidiactl --bind /dev/nvidia-uvm --bind /dev/nvidia-modeset --bind /dev/nvidia-caps"
singularity exec $ALL_DEVS --bind /usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | grep -i "cuda available"

echo ""
echo "If any test shows 'CUDA available: True', that combination works!"
