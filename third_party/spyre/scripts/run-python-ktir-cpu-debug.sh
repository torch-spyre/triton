#!/bin/bash
set -euo pipefail

# ----------------------------------------
# run-debug.sh
# Run a Python script against the ktir-path build with debug logging enabled.
# Usage: run-debug.sh <script.py>
#
# Prerequisites:
#   - PROJECT_ROOT must be set in the environment.
#   - torch-spyre, triton, and ktir-cpu must be installed in the venv at
#     ${PROJECT_ROOT}/.venv (i.e. build-ktir-path.sh must have been run).
# ----------------------------------------

source ${PROJECT_ROOT}/.venv/bin/activate

export TORCH_LOGS=output_code
export TORCH_COMPILE_DEBUG=1
export SPYRE_BACKEND_TARGET=SuperDSC
export TORCH_SPYRE_DEBUG=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=`pwd`/triton-dump
export DXP_DEBUG=1
export SPYRE_INDUCTOR_LOG=1
export SPYRE_INDUCTOR_LOG_LEVEL=DEBUG

export UNROLL_LOOPS=0

export TORCH_SPYRE_TRITON=1
export TORCH_SPYRE_KTIR_CPU=1
export FLEX_DEVICE=MOCK
export FLEX_COMPUTE=NULL

rm -rf triton-dump
rm -rf torch_compile_debug
rm -rf /tmp/torchinductor_*
rm -rf ~/.triton/cache

PYTHON=${1}

python3 ${PYTHON}
