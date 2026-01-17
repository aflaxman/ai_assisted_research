#!/bin/bash
# Simplest possible GPU workaround - bind entire directories

set -e

echo "=== Simple GPU binding approach ==="
echo "Binding entire /usr/lib/x86_64-linux-gnu to container..."
echo ""

# Bind entire library directory and all devices
singularity exec \
  --bind /usr/lib/x86_64-linux-gnu \
  --bind /dev/nvidia0 \
  --bind /dev/nvidia1 \
  --bind /dev/nvidia2 \
  --bind /dev/nvidia3 \
  --bind /dev/nvidia4 \
  --bind /dev/nvidia5 \
  --bind /dev/nvidia6 \
  --bind /dev/nvidia7 \
  --bind /dev/nvidiactl \
  --bind /dev/nvidia-uvm \
  --bind /dev/nvidia-caps \
  --env LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH" \
  seir_cuda.sif \
  python3 /app/seir_cuda_simulation.py "$@"
