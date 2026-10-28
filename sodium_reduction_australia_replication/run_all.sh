#!/usr/bin/env bash
# Run every scenario x seed in parallel and then summarize.
# Usage: ./run_all.sh [population_size] [seeds...]
# Environment: JOBS (parallel runs, default 4), ARTIFACT (default artifacts/australia.hdf),
#              TAG (results go to results/raw_<TAG>/ and comparison_<TAG>.csv),
#              SPEC (model spec YAML, default src/sodium_sim/model_spec.yaml),
#              STEP (time step in days, default 91.3125 = quarterly).
set -euo pipefail
cd "$(dirname "$0")"
POP=${1:-200000}; shift || true
SEEDS=${@:-0 1 2}
JOBS=${JOBS:-4}
ARTIFACT=${ARTIFACT:-artifacts/australia.hdf}
TAG=${TAG:-}
SPEC=${SPEC:-src/sodium_sim/model_spec.yaml}
STEP=${STEP:-91.3125}
SCENARIOS=$(.venv/bin/python -c "from sodium_sim.scenarios import SCENARIOS; print(' '.join(SCENARIOS))")
LOGS=results/raw${TAG:+_$TAG}/logs
mkdir -p "$LOGS"
for seed in $SEEDS; do for s in $SCENARIOS; do echo "$s $seed"; done; done |
  xargs -P "$JOBS" -n 2 sh -c '.venv/bin/python -m sodium_sim.run --scenarios "$0" --seeds "$1" --population '"$POP"' --artifact '"$ARTIFACT"' --tag "'"$TAG"'" --spec '"$SPEC"' --step-days '"$STEP"' > '"$LOGS"'/"$0"_"$1".log 2>&1 && echo "done $0 $1" || echo "FAILED $0 $1"'
.venv/bin/python -m sodium_sim.analyze "$TAG"
.venv/bin/python -m sodium_sim.plot "$TAG"
