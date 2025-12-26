# SEIR Disease Simulation with CUDA Acceleration

A GPU-accelerated implementation of SEIR (Susceptible-Exposed-Infectious-Recovered) disease modeling on temporal contact networks, inspired by the [Replay framework](https://github.com/HarrisonGreenlee/Replay).

## Quick Start (IHME Cluster)

### Option 1: Using Singularity Container (Recommended)

```bash
# 1. Build the container (one-time setup)
cd /path/to/replay_demo
singularity build seir_cuda.sif seir_cuda.def

# 2. Run benchmark on GPU node
srun --partition=gpu --gres=gpu:1 --time=00:30:00 \
    singularity exec --nv seir_cuda.sif python3 /app/seir_cuda_simulation.py --n-sims 10000

# 3. Run tests
singularity exec --nv seir_cuda.sif python3 /app/test_seir_cuda.py
```

### Option 2: Direct Python (if CUDA already configured)

```bash
# Load modules (adjust for your cluster)
module load cuda/12.2
module load python/3.11

# Install dependencies
pip install --user numpy numba matplotlib

# Run
python seir_cuda_simulation.py --n-sims 1000
```

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

## SLURM Job Script

Create `run_seir_gpu.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=seir_cuda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=seir_%j.out

# Using Singularity
singularity exec --nv seir_cuda.sif python3 /app/seir_cuda_simulation.py \
    --n-sims 50000 \
    --n-nodes 100 \
    --output results_${SLURM_JOB_ID}.json

# Or direct Python (if modules available)
# module load cuda/12.2 python/3.11
# python seir_cuda_simulation.py --n-sims 50000
```

Submit with: `sbatch run_seir_gpu.sh`

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

### "CUDA not available"
- Ensure you're on a GPU node: `srun --partition=gpu --gres=gpu:1 ...`
- Check CUDA is loaded: `nvidia-smi`
- For Singularity: use `--nv` flag

### Container build fails
- Build on a node with internet access
- Or pull base image first: `singularity pull docker://nvidia/cuda:12.2.0-runtime-ubuntu22.04`

### Out of memory
- Reduce `--n-nodes` (each node uses local GPU memory)
- Maximum ~1000 nodes with current implementation

## References

- [Replay Framework (original)](https://github.com/HarrisonGreenlee/Replay)
- [BMC Medical Informatics Paper](https://doi.org/10.1186/s12911-025-03310-2)
- [Numba CUDA Documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)
