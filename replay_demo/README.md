# SEIR Disease Simulation with CUDA Acceleration

A GPU-accelerated implementation of SEIR (Susceptible-Exposed-Infectious-Recovered) disease modeling on temporal contact networks, inspired by the [Replay framework](https://github.com/HarrisonGreenlee/Replay).

## Quick Start (IHME GPU Machine)

```bash
# SSH to the GPU machine
ssh gpu-machine

# 1. Build the container (one-time setup)
cd /path/to/replay_demo
singularity build seir_cuda.sif seir_cuda.def

# 2. Run benchmark (CPU vs GPU comparison)
singularity exec --nv seir_cuda.sif python3 /app/seir_cuda_simulation.py --n-sims 10000

# 3. Run tests
singularity exec --nv seir_cuda.sif python3 /app/test_seir_cuda.py
```

The `--nv` flag enables GPU access inside the container.

## Local Development (WSL/Laptop)

Use `uv` for fast, isolated Python environment setup:

```bash
cd replay_demo

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install numpy numba matplotlib

# Run the pure Python demo
python seir_contact_simulation.py

# Run tests (CPU-only without CUDA)
python test_seir_cuda.py

# Run benchmark (CPU mode)
python seir_cuda_simulation.py --device cpu --n-sims 500

# Generate visualization
python visualize_results.py
```

To deactivate when done: `deactivate`

## Files

| File | Description |
|------|-------------|
| `seir_cuda_simulation.py` | Main CUDA-accelerated simulation |
| `seir_contact_simulation.py` | Pure Python reference implementation |
| `visualize_results.py` | Plot generation |
| `test_seir_cuda.py` | Test suite with benchmarks |
| `seir_cuda.def` | Singularity container definition |

## Usage Examples

```bash
# Basic benchmark (CPU vs GPU)
python seir_cuda_simulation.py --n-sims 1000

# Large-scale GPU run
python seir_cuda_simulation.py --n-sims 50000 --device gpu

# CPU-only (no GPU required)
python seir_cuda_simulation.py --n-sims 500 --device cpu

# Custom network size
python seir_cuda_simulation.py --n-nodes 200 --n-days 60 --n-sims 5000

# Run all tests
python test_seir_cuda.py -v
```

## Expected Speedup

Typical performance on a V100 GPU:

| Simulations | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| 1,000 | 5-10s | 0.5-1s | 5-15x |
| 10,000 | 50-100s | 2-5s | 15-30x |
| 50,000 | 250-500s | 10-20s | 20-40x |

*Speedup depends on network size, contact density, and GPU model.*

## How It Works

1. **Temporal Contact Networks**: Contacts between individuals are timestamped events
2. **SEIR Model**: Tracks disease states (Susceptible → Exposed → Infectious → Recovered)
3. **Monte Carlo**: Runs thousands of stochastic simulations
4. **GPU Parallelization**: Each simulation runs on a separate CUDA thread

## Troubleshooting

### "CUDA not available" or `CUDA_ERROR_NO_DEVICE`

**Symptoms:**
```
WARNING: Could not find any nv libraries on this host!
CUDA available: False
```

**Cause:** Singularity's `--nv` flag requires `ldconfig path` to be set in `singularity.conf`

**Fix (requires admin):**
Add this line to `/opt/singularity/etc/singularity/singularity.conf`:
```
ldconfig path = /usr/sbin/ldconfig
```

**Workaround (without admin):**
Contact your system administrator to add the ldconfig path, or request they check:
```bash
grep -i ldconfig /opt/singularity/etc/singularity/singularity.conf
cat /opt/singularity/etc/singularity/nvliblist.conf
```

**Verification:**
```bash
# Should show GPU name, not "CUDA available: False"
singularity exec --nv seir_cuda.sif python3 -c "from numba import cuda; print(cuda.is_available())"
```

### Container build fails
- Build on a node with internet access
- Or pull base image first: `singularity pull docker://nvidia/cuda:12.2.0-runtime-ubuntu22.04`
- For remote build: `singularity build --remote seir_cuda.sif seir_cuda.def`

### Out of memory
- Reduce `--n-nodes` (each node uses local GPU memory)
- Maximum ~1000 nodes with current implementation

## References

- [Replay Framework (original)](https://github.com/HarrisonGreenlee/Replay)
- [BMC Medical Informatics Paper](https://doi.org/10.1186/s12911-025-03310-2)
- [Numba CUDA Documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)
