#!/usr/bin/env bash
# Run every scenario x seed in parallel (4 at a time) and then summarize.
# Usage: ./run_all.sh [population_size] [seeds...]
set -euo pipefail
cd "$(dirname "$0")"
POP=${1:-200000}; shift || true
SEEDS=${@:-0 1 2}
SCENARIOS=$(.venv/bin/python -c "from sodium_sim.scenarios import SCENARIOS; print(' '.join(SCENARIOS))")
mkdir -p results/raw/logs
for seed in $SEEDS; do for s in $SCENARIOS; do echo "$s $seed"; done; done |
  xargs -P 4 -n 2 sh -c '.venv/bin/python -m sodium_sim.run --scenarios "$0" --seeds "$1" --population '"$POP"' > results/raw/logs/"$0"_"$1".log 2>&1 && echo "done $0 $1" || echo "FAILED $0 $1"'
.venv/bin/python -m sodium_sim.analyze
.venv/bin/python -m sodium_sim.plot
