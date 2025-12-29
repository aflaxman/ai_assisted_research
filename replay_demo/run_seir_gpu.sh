#!/bin/bash
singularity exec \
  --bind /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  --bind /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
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
  seir_cuda.sif \
  python3 /app/seir_cuda_simulation.py "$@"
