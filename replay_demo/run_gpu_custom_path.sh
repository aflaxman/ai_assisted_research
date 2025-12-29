#!/bin/bash
# Try to work around Singularity overlay limits by using explicit library list
# Instead of binding individual files, try binding to custom mount points

set -e

echo "=== Workaround: Custom NVIDIA library paths ==="
echo ""

# Key NVIDIA libraries needed by CUDA
CUDA_LIBS=(
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    "/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1"
    "/usr/lib/x86_64-linux-gnu/libnvidia-fatbinaryloader.so.535.216.03"
)

# Build bind args - bind to /nvlib instead of /usr/lib to avoid overlay
BIND_ARGS=""
for lib in "${CUDA_LIBS[@]}"; do
    if [ -f "$lib" ]; then
        basename=$(basename "$lib")
        # Bind to custom /nvlib path to avoid overlay conflicts
        BIND_ARGS="$BIND_ARGS --bind $lib:/nvlib/$basename"
        echo "Binding: $lib -> /nvlib/$basename"
    fi
done

# Bind devices
for i in {0..7}; do
    if [ -e "/dev/nvidia$i" ]; then
        BIND_ARGS="$BIND_ARGS --bind /dev/nvidia$i"
    fi
done
BIND_ARGS="$BIND_ARGS --bind /dev/nvidiactl --bind /dev/nvidia-uvm"

echo ""
echo "Running with custom library paths..."

# Run with custom LD_LIBRARY_PATH pointing to our custom mount
singularity exec \
    $BIND_ARGS \
    --env LD_LIBRARY_PATH="/nvlib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH" \
    seir_cuda_minimal.sif \
    python3 /app/seir_cuda_simulation.py "$@"
