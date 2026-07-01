# Torch-Spyre Integration with Triton

This document describes how to drive the Triton Spyre backend from PyTorch via
`torch.compile`, using [torch-spyre](https://github.com/tnakaike/torch-spyre).
When `TORCH_SPYRE_TRITON=1` is set, `torch.compile` routes fused elementwise /
reduction ops through Triton, which the Spyre backend lowers TTIR → KTIR for the
device.

## Build torch-spyre

Build the [`dev/triton` fork](https://github.com/tnakaike/torch-spyre/tree/dev/triton).

## Build Triton

Follow the [install procedure](https://github.com/torch-spyre/triton#install)
in the Triton README.

## Run a Python program on torch-spyre and Triton

Enable the Triton path and run your program:

```bash
export TORCH_SPYRE_TRITON=1
python3 <a python program>
```

To exercise the Spyre + Triton backend, `torch.compile` a PyTorch function and
create the input tensors on `device="spyre"`. `torch.compile` then traces the
function, and Inductor generates Triton kernels that the Spyre backend lowers to
KTIR for the device.

```python
import torch

DEVICE = torch.device("spyre")

def fn(a, b):
    return a + b

compiled = torch.compile(fn)

a = torch.rand(512, 1024, dtype=torch.float16).to(DEVICE)
b = torch.rand(512, 1024, dtype=torch.float16).to(DEVICE)

result = compiled(a, b).cpu()   # move back to CPU to inspect
```

## A typical run script

The script below sets the debug environment variables, cleans stale artifacts from previous runs, runs the program, and collects the dumped artifacts under `./results/<OUT_DIR>/`.

```bash
#!/bin/bash

export TORCH_LOGS=output_code          # dump Inductor-generated output_code.py
export TORCH_COMPILE_DEBUG=1           # dump the torch_compile_debug/ tree (FX graphs, Inductor IR)
export TORCH_SPYRE_DEBUG=1             # torch-spyre device debug logging
export SPYRE_INDUCTOR_LOG=1            # enable torch-spyre Inductor logging
export SPYRE_INDUCTOR_LOG_LEVEL=DEBUG  # verbosity of the Inductor logging
export TRITON_KERNEL_DUMP=1            # dump per-kernel Triton IR (.ttir/.ktir)
export TRITON_DUMP_DIR=`pwd`/triton-dump  # destination for the Triton IR dumps
export UNROLL_LOOPS=0                  # keep loops rolled in the generated kernel

export TORCH_SPYRE_TRITON=1            # route torch.compile through the Triton path

# Clean stale artifacts from previous runs
rm -rf triton-dump
rm -rf torch_compile_debug
rm -rf /tmp/torchinductor_*
rm -rf ~/.triton/cache

PYTHON=${1}
OUT_DIR=${2}

if [ -z "${OUT_DIR}" ]; then
  echo "Error: second argument (output directory name) is required" >&2
  exit 1
fi

# Collect the run's artifacts under ./results/<OUT_DIR>/
rm -rf ./results/${OUT_DIR}
mkdir -p ./results/${OUT_DIR}

python3 ${PYTHON} > ./results/${OUT_DIR}/${OUT_DIR}.log 2>&1
cp torch_compile_debug/run_*/torchinductor/model__0_inference_0.0/* ./results/${OUT_DIR}/
cp triton-dump/*/* ./results/${OUT_DIR}
```

Invoke it as `bash run.sh <program.py> <out_dir>`, e.g.
`bash run.sh my-examples/add.py add`.

### What the cleanup removes

The script deletes artifacts from prior runs so each result directory is fresh
and cache hits don't mask codegen changes:

| Path | Why it is removed |
|------|-------------------|
| `triton-dump` | Previous Triton IR dumps (`TRITON_DUMP_DIR`). |
| `torch_compile_debug` | Previous `torch.compile` debug tree. |
| `/tmp/torchinductor_*` | Inductor's on-disk compile cache. |
| `~/.triton/cache` | Triton's kernel cache — cleared so kernels are recompiled and re-dumped every run. |

### What is dumped, and by which environment variable

After a run, `./results/<OUT_DIR>/` contains the collected artifacts:

| Artifact | Source env var | Contents |
|----------|----------------|----------|
| `<OUT_DIR>.log` | (stdout/stderr redirect) | Full program output — CPU vs. Spyre results, max deltas, and any debug logging from `TORCH_SPYRE_DEBUG`, and `SPYRE_INDUCTOR_LOG*`. |
| `output_code.py` | `TORCH_LOGS=output_code` | The Inductor-generated Python wrapper: `OpSpec`s, tensor layouts, and the Triton kernel launches. |
| `fx_graph_readable.py`, `fx_graph_runnable.py`, `fx_graph_transformed.py` | `TORCH_COMPILE_DEBUG=1` | The captured FX graph (readable and standalone-runnable forms) and the post-transform graph. |
| `ir_pre_fusion.txt`, `ir_post_fusion.txt` | `TORCH_COMPILE_DEBUG=1` | Inductor's scheduler IR before and after fusion. |
| `inductor_provenance_tracking_node_mappings.json` | `TORCH_COMPILE_DEBUG=1` | Provenance mapping from generated code back to FX/IR nodes. |
| `opspec_kernel_*.json` | `SPYRE_INDUCTOR_LOG*` | The torch-spyre `OpSpec` — device dtype, tiled device sizes, and device coordinate expressions for each arg. |
| `triton_*.ttir` | `TRITON_KERNEL_DUMP=1` → `TRITON_DUMP_DIR` | The Triton TTIR for each generated kernel. |
| `triton_*.ktir` | `TRITON_KERNEL_DUMP=1` → `TRITON_DUMP_DIR` | The KTIR (KTDP dialect) the Spyre backend lowered the kernel to. |

The `.ttir`/`.ktir` pair is the Triton → Spyre lowering under test: `.ttir` is
the input to the Spyre backend and `.ktir` is its output.
