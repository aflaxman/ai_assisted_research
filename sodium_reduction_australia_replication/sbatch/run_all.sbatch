#!/bin/bash
# Run the full scenario x seed set on a cluster node.
# Submit from the project directory:
#   sbatch sbatch/run_all.sbatch <population> <tag> <artifact> [seeds...]
#SBATCH --job-name=sodium_sim
#SBATCH --partition=all.q
#SBATCH --account=proj_simscience
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=results/slurm_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
POP=${1:-200000}; TAG=${2:-}; ARTIFACT=${3:-artifacts/australia.hdf}; shift 3 || true
JOBS=16 TAG="$TAG" ARTIFACT="$ARTIFACT" ./run_all.sh "$POP" "$@"
