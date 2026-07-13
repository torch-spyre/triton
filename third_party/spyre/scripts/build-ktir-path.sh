#!/bin/bash
set -euo pipefail

# ----------------------------------------
# build-ktir-path.sh
# CI build script for:
#   1. torch-spyre  (git@github.com:tnakaike/torch-spyre.git  branch: dev/triton)
#   2. LLVM for triton (commit pinned in triton/cmake/llvm-hash-spyre.txt)
#   3. triton       (git@github.com:torch-spyre/triton.git    branch: main)
#   4. ktir-cpu     (git@github.com:torch-spyre/ktir-cpu.git  branch: main)
#
# Assumptions / prerequisites:
#   - PROJECT_ROOT must be set in the environment.
#   - A Python venv exists at ${PROJECT_ROOT}/.venv and uv is on PATH.
#   - SENLIB_INSTALL_DIR and DEEPTOOLS_INSTALL_DIR must be set in the
#     environment.
#   - SPYRE_COMMS_INSTALL_DIR must be set in the environment.
#   - SSH access to github.com is required to clone the private repositories.
#   - cmake, ninja, git, and a C++20-capable compiler (c++) must be on PATH.
# ----------------------------------------

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "ERROR: PROJECT_ROOT must be set" >&2
  exit 1
fi

# Activate Python environment
source "${PROJECT_ROOT}/.venv/bin/activate"
export UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.venv"

# Build output directory (mirrors dev-env.sh)
export PROJECT_BUILD="${PROJECT_ROOT}/build"
mkdir -p "${PROJECT_BUILD}"

uv pip install nanobind==2.9.2 ninja pybind11==3.0.1 build cmake~=3.26 regex wheel

# ----------------------------------------
# Helper: clone repo if absent, otherwise fetch + hard-reset to origin/<branch>
# Usage: clone_or_update <dir> <url> <branch>
# ----------------------------------------
clone_or_update() {
  local dir="$1"
  local url="$2"
  local branch="$3"

  if [[ ! -d "${dir}/.git" ]]; then
    echo "==> Cloning ${url} (branch: ${branch}) into ${dir}"
    git clone --branch "${branch}" "${url}" "${dir}"
  else
    echo "==> Updating $(basename "${dir}") to origin/${branch}"
    git -C "${dir}" fetch --prune origin
    git -C "${dir}" checkout "${branch}"
    git -C "${dir}" reset --hard "origin/${branch}"
  fi
}

# ----------------------------------------
# 1. Checkout repositories
# ----------------------------------------
clone_or_update "${PROJECT_ROOT}/torch-spyre" \
  "git@github.com:tnakaike/torch-spyre.git" "dev/triton"

clone_or_update "${PROJECT_ROOT}/triton" \
  "git@github.com:torch-spyre/triton.git" "main"

clone_or_update "${PROJECT_ROOT}/ktir-cpu" \
  "git@github.com:torch-spyre/ktir-cpu.git" "main"

# ----------------------------------------
# 2. Build torch-spyre
# ----------------------------------------
echo "========================================"
echo "========== Torch Spyre"
echo "========================================"

cd "${PROJECT_ROOT}/torch-spyre"
export CXX="$(command -v c++)"
uv sync --active --inexact --no-install-project --extra build
uv sync --all-extras --active --inexact --no-build-isolation --reinstall-package torch-spyre -v
unset CXX

# ----------------------------------------
# 3. Build LLVM for Triton
# ----------------------------------------
echo "========================================"
echo "========== LLVM for Triton"
echo "========================================"

export TRITON_LLVM_BUILD_DIR="${PROJECT_BUILD}/llvm-triton"
export TRITON_LLVM_SRC_DIR="${PROJECT_ROOT}/llvm-project-triton"

# Read the pinned LLVM commit from triton's own hash file
LLVM_COMMIT="$(cat "${PROJECT_ROOT}/triton/cmake/llvm-hash-spyre.txt")"
echo "==> LLVM commit: ${LLVM_COMMIT}"

# Clone llvm-project-triton if absent; otherwise fetch
if [[ ! -d "${TRITON_LLVM_SRC_DIR}/.git" ]]; then
  echo "==> Cloning llvm-project for triton"
  git clone "https://github.com/llvm/llvm-project.git" "${TRITON_LLVM_SRC_DIR}"
fi
git -C "${TRITON_LLVM_SRC_DIR}" fetch --prune origin
git -C "${TRITON_LLVM_SRC_DIR}" checkout "${LLVM_COMMIT}"

mkdir -p "${TRITON_LLVM_BUILD_DIR}"
cd "${TRITON_LLVM_BUILD_DIR}"

cmake -G Ninja "${TRITON_LLVM_SRC_DIR}/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$(which python3)" \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"

ninja

# ----------------------------------------
# 4. Build Triton
# ----------------------------------------
echo "========================================"
echo "========== Triton"
echo "========================================"

cd "${PROJECT_ROOT}/triton"
LLVM_INCLUDE_DIRS="${TRITON_LLVM_BUILD_DIR}/include" \
LLVM_LIBRARY_DIR="${TRITON_LLVM_BUILD_DIR}/lib" \
LLVM_SYSPATH="${TRITON_LLVM_BUILD_DIR}" \
LLVM_BUILD_DIR="${TRITON_LLVM_BUILD_DIR}" \
TRITON_BUILD_TUTORIALS=OFF \
TRITON_BUILD_PROTON=OFF \
uv pip install -e ".[spyre-test]" --no-build-isolation

# ----------------------------------------
# 5. Build ktir-cpu
# ----------------------------------------
echo "========================================"
echo "========== ktir-cpu"
echo "========================================"

FRONTEND_DIR="${PROJECT_ROOT}/triton/third_party/spyre/ktir-mlir-frontend"
MLIR_DIR="${PROJECT_ROOT}/build/llvm-triton/lib/cmake/mlir"

CMAKE_ARGS="-DMLIR_DIR=${MLIR_DIR}" uv pip install "${FRONTEND_DIR}"

cd "${PROJECT_ROOT}/ktir-cpu"
uv pip install -e ".[dev]"

echo "========================================"
echo "========== Build complete"
echo "========================================"
