#!/usr/bin/env bash
# Run all 26 fits (13 countries x {sir, sirx}) sequentially.
# Each fit: nl-sbplx multi-start scout + 4-chain adaptive MH posterior
# on the deterministic-ODE likelihood. ~3 min (SIR) / ~8 min (SIRX)
# per country on 4 cores.
set -uo pipefail
cd "$(dirname "$0")"

for toml in fits/*.toml; do
    label=$(basename "$toml" .toml)
    echo "=== $label $(date +%H:%M:%S) ==="
    camdl fit run "$toml" --label "$label" --seed 1 2>&1 | tail -3
done
echo "ALL FITS DONE $(date +%H:%M:%S)"
