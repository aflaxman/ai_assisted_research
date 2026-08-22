#!/usr/bin/env bash
# Build the two environments used by simple_regression_before_after.ipynb
set -euo pipefail
cd "$(dirname "$0")"

PR_COMMIT=3fc7196aa107d1bf6a0d954d7953d2fbcfd760b0

# Old: current PyPI release
uv venv .venv-old --python 3.12
uv pip install --python .venv-old/bin/python "bambi==0.20.0" matplotlib-inline

# New: the rewrite branch (PR #1002), with a one-line syntax patch it needs to import
if [ ! -d .bambi-pr1002 ]; then
    git clone https://github.com/bambinos/bambi .bambi-pr1002
    git -C .bambi-pr1002 fetch origin "$PR_COMMIT"
    git -C .bambi-pr1002 checkout "$PR_COMMIT"
fi
sed -i 's/except RuntimeError, ValueError:/except (RuntimeError, ValueError):/' \
    .bambi-pr1002/bambi/backend/pymc/model.py
uv venv .venv-new --python 3.12
uv pip install --python .venv-new/bin/python ./.bambi-pr1002 jupyter nbconvert ipykernel
