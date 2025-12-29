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
echo "Searching for NVIDIA libraries..."
for lib in "${NVIDIA_LIBS[@]}"; do
    for search_path in "${SEARCH_PATHS[@]}"; do
        # Look for the library and all versioned variants (e.g., libcuda.so.1)
        for lib_file in "$search_path/$lib"*; do
            if [ -f "$lib_file" ]; then
                BIND_ARGS="$BIND_ARGS --bind $lib_file"
                echo "  Found: $lib_file"
                break 2  # Move to next library once found
            fi
        done
    done
done

# Add NVIDIA device bindings
echo "Adding NVIDIA device bindings..."
for device in /dev/nvidia* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
    if [ -e "$device" ]; then
        BIND_ARGS="$BIND_ARGS --bind $device"
        echo "  Binding: $device"
    fi
done

# Also bind nvidia-caps directory if it exists
if [ -d /dev/nvidia-caps ]; then
    BIND_ARGS="$BIND_ARGS --bind /dev/nvidia-caps"
    echo "  Binding: /dev/nvidia-caps"
fi

echo ""
echo "Running container with manual NVIDIA bindings..."
echo "Command: singularity exec $BIND_ARGS seir_cuda.sif python3 /app/seir_cuda_simulation.py $@"
echo ""

# Run the container
singularity exec $BIND_ARGS seir_cuda.sif python3 /app/seir_cuda_simulation.py "$@"
