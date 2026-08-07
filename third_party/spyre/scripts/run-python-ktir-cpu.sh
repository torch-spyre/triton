#!/bin/bash
set -euo pipefail

# ----------------------------------------
# run.sh
# Run a Python script against the ktir-path build.
# Usage: run.sh <script.py>
#
# Prerequisites:
#   - PROJECT_ROOT must be set in the environment.
#   - torch-spyre, triton, and ktir-cpu must be installed in the venv at
#     ${PROJECT_ROOT}/.venv (i.e. build-ktir-path.sh must have been run).
# ----------------------------------------

source ${PROJECT_ROOT}/.venv/bin/activate

export UNROLL_LOOPS=0

export TORCH_SPYRE_TRITON=1
export TORCH_SPYRE_KTIR_CPU=1
export FLEX_DEVICE=MOCK
export FLEX_COMPUTE=NULL

rm -rf ~/.triton/cache

PYTHON=${1}

python3 ${PYTHON}
