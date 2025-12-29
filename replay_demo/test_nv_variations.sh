#!/bin/bash
# Test different --nv variations to find what works

TEST_CODE='from numba import cuda; print(f"CUDA: {cuda.is_available()}")'

echo "======================================================================"
echo "Testing Singularity --nv with different flag combinations"
echo "======================================================================"
echo ""

echo "Test 1: Plain --nv"
singularity exec --nv seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | tail -3
echo ""

echo "Test 2: --nv with --cleanenv"
singularity exec --nv --cleanenv seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | tail -3
echo ""

echo "Test 3: --nv with --contain"
singularity exec --nv --contain seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | tail -3
echo ""

echo "Test 4: --nv with --no-home"
singularity exec --nv --no-home seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | tail -3
echo ""

echo "Test 5: --nv with --writable-tmpfs"
singularity exec --nv --writable-tmpfs seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | tail -3
echo ""

echo "Test 6: --nv with --no-home --writable-tmpfs"
singularity exec --nv --no-home --writable-tmpfs seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | tail -3
echo ""

echo "Test 7: --nv with minimal environment (--contain --cleanenv)"
singularity exec --nv --contain --cleanenv seir_cuda.sif python3 -c "$TEST_CODE" 2>&1 | tail -3
echo ""

echo "======================================================================"
echo "If any test shows 'CUDA: True', use those flags!"
echo "======================================================================"
