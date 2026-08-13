#!/usr/bin/env bash
# End-to-end camdl pipeline for the policy-informed behavioral-change SIR.
#
# Requires `camdl` on PATH (see ../policy_behavioral_camdl/README.md for the
# no-sudo install). Regenerates every artifact the figure and README depend on.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=policy_behavioral.camdl
SEED=20

echo "==> 0. generate policy covariate + truth params"
python3 generate_inputs.py

echo "==> 1. check (dimensions + types)"
camdl check "$MODEL"

echo "==> 2. simulate the synthetic 'observed' daily case series (truth params)"
camdl simulate "$MODEL" --params params/truth.toml --backend chain_binomial \
    --dt 1.0 --seed "$SEED" --obs-only data/cases.tsv

echo "==> 3. one full-behavior trajectory (state path, for the alarm panel)"
camdl simulate "$MODEL" --params params/truth.toml --backend chain_binomial \
    --dt 1.0 --seed "$SEED" -o results/traj_full.tsv

echo "==> 4. counterfactual ensembles (40 replicates each): flattening the curve"
camdl simulate "$MODEL" --params params/truth.toml --backend chain_binomial \
    --dt 1.0 --seed 100 --replicates 40 --obs-only results/cf_full_obs.tsv
camdl simulate "$MODEL" --params params/truth.toml --param policy_weight=0.0 \
    --backend chain_binomial --dt 1.0 --seed 100 --replicates 40 \
    --obs-only results/cf_endog_obs.tsv
camdl simulate "$MODEL" --params params/truth.toml --param policy_weight=0.0 \
    --param endog_delta=0.0 --backend chain_binomial --dt 1.0 --seed 100 \
    --replicates 40 --obs-only results/cf_nobehavior_obs.tsv

echo "==> 5. fit (IF2 scout -> PGAS posterior)"
camdl fit run fit_policy.toml --label policy --seed 3

FITDIR=$(ls -dt results/fits/*policy*/ | head -1)
echo "    fit dir: $FITDIR"

echo "==> 6. fit summary (R-hat, gate verdict, MLE table)"
camdl fit summary "$FITDIR" | tee results/fit_summary.txt

echo "==> 7. export posterior draws"
DRAWS=$(ls -t "$FITDIR"*/draws.tsv "$FITDIR"draws.tsv 2>/dev/null | head -1 || true)
if [ -n "${DRAWS:-}" ]; then cp "$DRAWS" results/policy_posterior_draws.tsv; fi
echo "    draws: ${DRAWS:-<not found; check fit dir layout>}"

echo "==> 8. posterior-predictive daily incidence (300 draws)"
camdl simulate "$MODEL" --draws posterior --fit "$FITDIR" -n 300 --seed 21 \
    --obs-only results/policy_postpred_obs.tsv

echo "==> 9. plot"
uv run python plot_camdl.py

echo "==> done."
