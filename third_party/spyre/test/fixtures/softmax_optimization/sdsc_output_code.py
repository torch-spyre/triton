# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from sympy import sympify
from torch_spyre._inductor.op_spec import TensorArg, TensorWorkDivision, OpSpec, UnimplementedOp, LoopSpec, spyre_constant_tensor, IndirectAccess, DebugHandle, SourceLoc, ProvenanceTransform
from torch_spyre.execution.async_compile import SpyreAsyncCompile
from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout, spyre_empty_with_layout, set_spyre_tensor_layout
import subprocess

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch_spyre._C import reinterpret_tensor as reinterpret_tensor
from torch_spyre._C import reinterpret_tensor_with_layout
del async_compile
async_compile = SpyreAsyncCompile()


# Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   softmax => amax, div, exp, sub, sum_1
# Graph fragment:
#   %arg0_1 : Tensor "f16[1, 64, 11, 2049][1442496, 22539, 2049, 1]spyre:0" = PlaceHolder[target=arg0_1]
#   %clone : Tensor "f16[1, 64, 11, 2049][1442496, 22539, 2049, 1]spyre:0" = PlaceHolder[target=clone]
#   %amax : Tensor "f16[1, 64, 11, 1][704, 11, 1, 704]spyre:0" = PlaceHolder[target=amax]
#   %sub : Tensor "f16[1, 64, 11, 2049][1442496, 22539, 2049, 1]spyre:0" = PlaceHolder[target=sub]
#   %exp : Tensor "f16[1, 64, 11, 2049][1442496, 22539, 2049, 1]spyre:0" = PlaceHolder[target=exp]
#   %sum_1 : Tensor "f16[1, 64, 11, 1][704, 11, 1, 704]spyre:0" = PlaceHolder[target=sum_1]
#   %clone : [num_users=2] = call_function[target=torch.ops.aten.clone](args = (%arg0_1,), kwargs = {})
#   %amax : Tensor "f16[1, 64, 11, 1][704, 11, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone, [-1], True), kwargs = {})
#   %sub : Tensor "f16[1, 64, 11, 2049][1442496, 22539, 2049, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %amax), kwargs = {})
#   %exp : Tensor "f16[1, 64, 11, 2049][1442496, 22539, 2049, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f16[1, 64, 11, 1][704, 11, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : Tensor "f16[1, 64, 11, 2049][1442496, 22539, 2049, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   return %clone,%amax,%sub,%exp,%sum_1,%div
sdsc_fused__softmax_0 = async_compile.sdsc('sdsc_fused__softmax_0',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('d0'): (sympify('64'), 32), sympify('d1'): (sympify('11'), 1), sympify('d2'): (sympify('2049'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2015214560273318976, source=None, aten_op=None, ir_chain=('clone', 'buf5'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[64, 11, 33, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[64, 11, 33, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'lx': 2816},
                ),
            ]
        ),
        OpSpec(
            op='max',
            is_reduction=True,
            iteration_space={sympify('d0'): (sympify('64'), 32), sympify('d1'): (sympify('11'), 1), sympify('d2'): (sympify('2049'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4638110020406371008, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._softmax.default', ir_chain=('amax', 'buf0'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[64, 11, 33, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'lx': 2816},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 11, 64, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='sub',
            is_reduction=False,
            iteration_space={sympify('d0'): (sympify('64'), 32), sympify('d1'): (sympify('11'), 1), sympify('d2'): (sympify('2049'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=9061527765045993979, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._softmax.default', ir_chain=('sub', 'buf1'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[64, 11, 33, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'lx': 2816},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 11, 64, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[64, 11, 33, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'lx': 2816},
                ),
            ]
        ),
        OpSpec(
            op='exp',
            is_reduction=False,
            iteration_space={sympify('d0'): (sympify('64'), 32), sympify('d1'): (sympify('11'), 1), sympify('d2'): (sympify('2049'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2394785117585273997, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._softmax.default', ir_chain=('exp', 'buf2'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[64, 11, 33, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'lx': 2816},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[64, 11, 33, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'lx': 2816},
                ),
            ]
        ),
        OpSpec(
            op='sum',
            is_reduction=True,
            iteration_space={sympify('d0'): (sympify('64'), 32), sympify('d1'): (sympify('11'), 1), sympify('d2'): (sympify('2049'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6819705596206386777, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._softmax.default', ir_chain=('sum_1', 'buf3'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[64, 11, 33, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'lx': 2816},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 11, 64, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='realdiv',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('64'), 32), sympify('c1'): (sympify('11'), 1), sympify('c2'): (sympify('2049'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7384475178910738790, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._softmax.default', ir_chain=('div', 'buf4'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[64, 11, 33, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 2816},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 11, 64, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c0'), sympify('0')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[64, 11, 33, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
            ]
        ),
    ]
)


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (1, 64, 11, 2049), (1442496, 22539, 2049, 1), 'input')
        buf4 = spyre_empty_with_layout((1, 64, 11, 2049), (1442496, 22539, 2049, 1), torch.float16, SpyreTensorLayout(device_size=[64, 11, 33, 1, 64], stride_map =[22539, 2049, 64, -1, 1], device_dtype=DataFormats.SEN169_FP16))
        # [Provenance debug handles] sdsc_fused__softmax_0:1
        sdsc_fused__softmax_0.run(arg0_1, buf4)
        del arg0_1
        return (buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns
