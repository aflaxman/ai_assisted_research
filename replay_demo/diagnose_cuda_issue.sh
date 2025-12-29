#!/bin/bash
# Comprehensive CUDA diagnostic for Singularity issue

echo "======================================================================"
echo "CUDA/Singularity Diagnostic Report"
echo "======================================================================"
echo ""

echo "=== Host System Information ==="
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo ""

echo "=== NVIDIA Driver Information ==="
if [ -f /proc/driver/nvidia/version ]; then
    cat /proc/driver/nvidia/version
else
    echo "ERROR: /proc/driver/nvidia/version not found"
fi
echo ""

echo "=== NVIDIA Devices ==="
ls -la /dev/nvidia* 2>/dev/null | head -20
echo ""

echo "=== NVIDIA Libraries on Host ==="
ls -la /usr/lib/x86_64-linux-gnu/libcuda.so* 2>/dev/null
ls -la /usr/lib/x86_64-linux-gnu/libnvidia-ml.so* 2>/dev/null
echo ""

echo "=== Singularity Configuration ==="
SING_CONF="/opt/singularity/etc/singularity/singularity.conf"
if [ -f "$SING_CONF" ]; then
    echo "ldconfig setting:"
    grep -i "ldconfig" "$SING_CONF" | grep -v "^#" || echo "  (not set or commented out)"
    echo ""
fi

echo "=== Test 1: Host CUDA Access (without container) ==="
python3 -c "
try:
    from numba import cuda
    print(f'Host CUDA available: {cuda.is_available()}')
    if cuda.is_available():
        print(f'  Device: {cuda.get_current_device().name}')
except Exception as e:
    print(f'Host CUDA check failed: {e}')
" 2>&1
echo ""

echo "=== Test 2: Container with --nv flag ==="
singularity exec --nv seir_cuda.sif python3 -c "
from numba import cuda
print(f'Container --nv CUDA available: {cuda.is_available()}')
if not cuda.is_available():
    try:
        cuda.detect()
    except Exception as e:
        print(f'  Error: {e}')
" 2>&1
echo ""

echo "=== Test 3: Manual Library Binding ==="
singularity exec \
  --bind /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  --bind /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  --bind /dev/nvidia0 \
  --bind /dev/nvidiactl \
  --bind /dev/nvidia-uvm \
  seir_cuda.sif python3 -c "
import os
print('libcuda.so.1 exists:', os.path.exists('/usr/lib/x86_64-linux-gnu/libcuda.so.1'))
print('/dev/nvidia0 exists:', os.path.exists('/dev/nvidia0'))
print('/dev/nvidiactl exists:', os.path.exists('/dev/nvidiactl'))

from numba import cuda
print(f'Manual binding CUDA available: {cuda.is_available()}')
if not cuda.is_available():
    try:
        cuda.detect()
    except Exception as e:
        print(f'  Error: {e}')
" 2>&1
echo ""

echo "======================================================================"
echo "Summary:"
echo "  If host CUDA works but container doesn't:"
echo "    -> Singularity binding issue (ldconfig fix needed)"
echo "  If host CUDA fails too:"
echo "    -> Check numba installation or NVIDIA driver"
echo "  If --nv works:"
echo "    -> Admin needs to configure ldconfig in singularity.conf"
echo "======================================================================"
