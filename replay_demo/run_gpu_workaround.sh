#!/bin/bash
# Workaround for Singularity --nv when ldconfig path is not configured
# This script manually binds NVIDIA libraries listed in nvliblist.conf

set -e

# NVIDIA library names from nvliblist.conf (libraries only, ending in .so)
NVIDIA_LIBS=(
    "libcuda.so"
    "libEGL_installertest.so"
    "libEGL_nvidia.so"
    "libEGL.so"
    "libGLdispatch.so"
    "libGLESv1_CM_nvidia.so"
    "libGLESv1_CM.so"
    "libGLESv2_nvidia.so"
    "libGLESv2.so"
    "libGL.so"
    "libGLX_installertest.so"
    "libGLX_nvidia.so"
    "libglx.so"
    "libGLX.so"
    "libnvcuvid.so"
    "libnvidia-cbl.so"
    "libnvidia-cfg.so"
    "libnvidia-compiler.so"
    "libnvidia-eglcore.so"
    "libnvidia-egl-wayland.so"
    "libnvidia-encode.so"
    "libnvidia-fatbinaryloader.so"
    "libnvidia-fbc.so"
    "libnvidia-glcore.so"
    "libnvidia-glsi.so"
    "libnvidia-glvkspirv.so"
    "libnvidia-gpucomp.so"
    "libnvidia-gtk2.so"
    "libnvidia-gtk3.so"
    "libnvidia-ifr.so"
    "libnvidia-ml.so"
    "libnvidia-nvvm.so"
    "libnvidia-opencl.so"
    "libnvidia-opticalflow.so"
    "libnvidia-ptxjitcompiler.so"
    "libnvidia-rtcore.so"
    "libnvidia-tls.so"
    "libnvidia-wfb.so"
    "libnvoptix.so.1"
    "libOpenCL.so"
    "libOpenGL.so"
    "libvdpau_nvidia.so"
)

# Search paths for NVIDIA libraries
SEARCH_PATHS=(
    "/usr/lib/x86_64-linux-gnu"
    "/usr/lib64"
    "/usr/lib"
    "/lib/x86_64-linux-gnu"
    "/lib64"
    "/lib"
)

# Build bind arguments for libraries
BIND_ARGS=""
BOUND_LIBS=()
echo "Searching for NVIDIA libraries..."
for lib in "${NVIDIA_LIBS[@]}"; do
    for search_path in "${SEARCH_PATHS[@]}"; do
        # Look for the library and all versioned variants (e.g., libcuda.so.1)
        # Bind ALL versions, not just the first match
        found_any=0
        for lib_file in "$search_path/$lib"*; do
            # Skip if it's a symbolic link - we want the real file
            if [ -f "$lib_file" ] && [ ! -L "$lib_file" ]; then
                BIND_ARGS="$BIND_ARGS --bind $lib_file"
                BOUND_LIBS+=("$lib_file")
                echo "  Found (real): $lib_file"
                found_any=1
            fi
        done

        # If we found real files, also check for symlinks pointing to them
        if [ $found_any -eq 1 ]; then
            for lib_file in "$search_path/$lib"*; do
                if [ -L "$lib_file" ]; then
                    BIND_ARGS="$BIND_ARGS --bind $lib_file"
                    BOUND_LIBS+=("$lib_file")
                    echo "  Found (link): $lib_file"
                fi
            done
            break  # Move to next library
        fi
    done
done

# Add NVIDIA device bindings
echo "Adding NVIDIA device bindings..."
BOUND_DEVICES=()

# Process devices, avoiding duplicates
for device in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/nvidia-nvswitchctl; do
    if [ -e "$device" ] && [[ ! " ${BOUND_DEVICES[@]} " =~ " ${device} " ]]; then
        BIND_ARGS="$BIND_ARGS --bind $device"
        BOUND_DEVICES+=("$device")
        echo "  Binding: $device"
    fi
done

# Bind nvidia-caps directory if it exists
if [ -d /dev/nvidia-caps ] && [[ ! " ${BOUND_DEVICES[@]} " =~ " /dev/nvidia-caps " ]]; then
    BIND_ARGS="$BIND_ARGS --bind /dev/nvidia-caps"
    BOUND_DEVICES+=("/dev/nvidia-caps")
    echo "  Binding: /dev/nvidia-caps"
fi

echo ""
echo "Summary: Bound ${#BOUND_LIBS[@]} libraries and ${#BOUND_DEVICES[@]} devices"
echo ""

# Check if user wants diagnostics
if [[ "$1" == "--debug" ]]; then
    echo "=== DEBUG MODE ==="
    echo "Running CUDA diagnostics in container..."
    singularity exec $BIND_ARGS seir_cuda.sif python3 << 'PYEOF'
import os
import sys

print("\n=== Library Check ===")
for lib in ['libcuda.so.1', 'libnvidia-ml.so.1']:
    for path in ['/usr/lib/x86_64-linux-gnu', '/usr/local/cuda/lib64']:
        full_path = os.path.join(path, lib)
        exists = os.path.exists(full_path)
        print(f"{full_path}: {'EXISTS' if exists else 'MISSING'}")

print("\n=== Device Check ===")
for i in range(8):
    dev = f'/dev/nvidia{i}'
    print(f"{dev}: {'EXISTS' if os.path.exists(dev) else 'MISSING'}")

print("\n=== CUDA Check ===")
try:
    from numba import cuda
    print(f"CUDA available: {cuda.is_available()}")
    if not cuda.is_available():
        try:
            cuda.detect()
        except Exception as e:
            print(f"Error: {e}")
except Exception as e:
    print(f"Failed to import CUDA: {e}")
PYEOF
else
    # Run the simulation
    singularity exec $BIND_ARGS seir_cuda.sif python3 /app/seir_cuda_simulation.py "$@"
fi
