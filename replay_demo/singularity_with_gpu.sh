#!/bin/bash
# Flexible GPU workaround for Singularity when --nv doesn't work
# Usage: ./singularity_with_gpu.sh [singularity exec args] <container> <command>
# Example: ./singularity_with_gpu.sh seir_cuda.sif python3 -c "from numba import cuda; print(cuda.is_available())"

# NVIDIA library names from nvliblist.conf
NVIDIA_LIBS=(
    "libcuda.so" "libnvidia-ml.so" "libnvidia-fatbinaryloader.so"
    "libnvidia-ptxjitcompiler.so" "libnvcuvid.so" "libnvidia-encode.so"
    "libnvidia-opticalflow.so" "libnvidia-compiler.so"
)

# Search paths
SEARCH_PATHS=(
    "/usr/lib/x86_64-linux-gnu"
    "/usr/lib64"
    "/lib/x86_64-linux-gnu"
    "/lib64"
)

# Build bind arguments
BIND_ARGS=""
for lib in "${NVIDIA_LIBS[@]}"; do
    for search_path in "${SEARCH_PATHS[@]}"; do
        for lib_file in "$search_path/$lib"*; do
            if [ -f "$lib_file" ]; then
                BIND_ARGS="$BIND_ARGS --bind $lib_file"
                break 2
            fi
        done
    done
done

# Add device bindings
for device in /dev/nvidia* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    if [ -e "$device" ]; then
        BIND_ARGS="$BIND_ARGS --bind $device"
    fi
done

# Run singularity with bindings
exec singularity exec $BIND_ARGS "$@"
