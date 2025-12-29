#!/bin/bash
# Workaround for Singularity --nv not working due to missing ldconfig path
# Manually bind all NVIDIA libraries and devices

# Find all NVIDIA libraries on the host
NVIDIA_LIBS=$(find /usr/lib/x86_64-linux-gnu -name "libnvidia*.so*" -o -name "libcuda*.so*" 2>/dev/null | tr '\n' ',')

# Add device bindings
DEVICE_BINDS="/dev/nvidia0,/dev/nvidia1,/dev/nvidia2,/dev/nvidia3,/dev/nvidia4,/dev/nvidia5,/dev/nvidia6,/dev/nvidia7,/dev/nvidiactl,/dev/nvidia-uvm,/dev/nvidia-uvm-tools,/dev/nvidia-modeset"

singularity exec \
  --bind "$NVIDIA_LIBS$DEVICE_BINDS" \
  seir_cuda.sif \
  python3 /app/seir_cuda_simulation.py "$@"
