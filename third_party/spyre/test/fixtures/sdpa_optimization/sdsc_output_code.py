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


# Topologically Sorted Source Nodes: [scaled_dot_product_attention, ], Original ATen: [aten._scaled_dot_product_fused_attention_overrideable]
# Source node to ATen node mapping:
#    => batched_matmul_default, batched_matmul_default_1
#   scaled_dot_product_attention => add, add_1, amax, amax_1, amax_2, div, exp, exp_1, full_default, full_default_1, full_default_2, maximum, mul, mul_1, mul_2, mul_3, permute, sub, sub_1, sum_1, unsqueeze, unsqueeze_1, unsqueeze_2
# Graph fragment:
#   %buf0 : Tensor "f16[][]spyre:0" = PlaceHolder[target=buf0]
#   %buf2 : Tensor "f16[][]spyre:0" = PlaceHolder[target=buf2]
#   %full_default_1 : Tensor "f16[1, 32, 855, 64][1751040, 54720, 64, 1]spyre:0" = PlaceHolder[target=full_default_1]
#   %arg0_1 : Tensor "f16[1, 32, 855, 128][3502080, 109440, 128, 1]spyre:0" = PlaceHolder[target=arg0_1]
#   %buf24 : Tensor "f16[][]spyre:0" = PlaceHolder[target=buf24]
#   %arg1_1 : Tensor "f16[1, 32, 2048, 128][8388608, 262144, 128, 1]spyre:0" = PlaceHolder[target=arg1_1]
#   %mul : Tensor "f16[1, 32, 2048, 128][8388608, 262144, 128, 1]spyre:0" = PlaceHolder[target=mul]
#   %expand : Tensor "f16[1, 32, 855, 128][3502080, 109440, 128, 1]spyre:0" = PlaceHolder[target=expand]
#   %restickify_default : Tensor "f16[1, 32, 2048, 128][8388608, 262144, 128, 1]spyre:0" = PlaceHolder[target=restickify_default]
#   %batched_matmul_default_1 : Tensor "f16[1, 32, 855, 2048][56033280, 1751040, 2048, 1]spyre:0" = PlaceHolder[target=batched_matmul_default_1]
#   %amax : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0" = PlaceHolder[target=amax]
#   %amax_2 : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0" = PlaceHolder[target=amax_2]
#   %maximum : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0" = PlaceHolder[target=maximum]
#   %sub_1 : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0" = PlaceHolder[target=sub_1]
#   %full_default : Tensor "f16[1, 32, 855, 128][3502080, 109440, 128, 1]spyre:0" = PlaceHolder[target=full_default]
#   %exp_1 : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0" = PlaceHolder[target=exp_1]
#   %sub : Tensor "f16[1, 32, 855, 2048][56033280, 1751040, 2048, 1]spyre:0" = PlaceHolder[target=sub]
#   %expand_2 : Tensor "f16[1, 32, 855, 2048][56033280, 1751040, 2048, 1]spyre:0" = PlaceHolder[target=expand_2]
#   %expand_3 : Tensor "f16[1, 32, 2048, 128][8388608, 262144, 128, 1]spyre:0" = PlaceHolder[target=expand_3]
#   %mul_3 : Tensor "f16[1, 32, 855, 128][3502080, 109440, 128, 1]spyre:0" = PlaceHolder[target=mul_3]
#   %batched_matmul_default : Tensor "f16[1, 32, 855, 128][3502080, 109440, 128, 1]spyre:0" = PlaceHolder[target=batched_matmul_default]
#   %full_default_2 : Tensor "f16[1, 32, 855, 64][1751040, 54720, 64, 1]spyre:0" = PlaceHolder[target=full_default_2]
#   %amax_1 : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0" = PlaceHolder[target=amax_1]
#   %mul_2 : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0" = PlaceHolder[target=mul_2]
#   %sum_1 : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0" = PlaceHolder[target=sum_1]
#   %copy_f_1 : Tensor "f16[1, 32, 855, 128][3502080, 109440, 128, 1]spyre:0" = PlaceHolder[target=copy_f_1]
#   %copy_f : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0" = PlaceHolder[target=copy_f]
#   %restickify_default : [num_users=0] = call_function[target=torch.ops.spyre.restickify.default](args = (%mul,), kwargs = {})
#   %full_default : Tensor "f16[1, 32, 855, 128][3502080, 109440, 128, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 855, 128], 0), kwargs = {dtype: torch.float16, layout: torch.strided, device: spyre:0, pin_memory: False})
#   %full_default_1 : Tensor "f16[1, 32, 855, 64][1751040, 54720, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 855, 64], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: spyre:0, pin_memory: False})
#   %amax : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%full_default_1, [-1]), kwargs = {})
#   %mul_1 : Tensor "f16[1, 32, 855, 128][3502080, 109440, 128, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 0.29730177875068026), kwargs = {})
#   %mul : Tensor "f16[1, 32, 2048, 128][8388608, 262144, 128, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 0.29730177875068026), kwargs = {})
#   %permute : Tensor "f16[1, 32, 128, 2048][8388608, 262144, 1, 128]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 1, 3, 2]), kwargs = {})
#   %batched_matmul_default_1 : Tensor "f16[1, 32, 855, 2048][56033280, 1751040, 2048, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.batched_matmul.default](args = (%expand, %expand_1), kwargs = {})
#   %amax_2 : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%batched_matmul_default_1, [-1]), kwargs = {})
#   %maximum : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.maximum.default](args = (%amax, %amax_2), kwargs = {})
#   %sub_1 : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%amax, %maximum), kwargs = {})
#   %exp_1 : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %unsqueeze_1 : Tensor "f16[1, 32, 855, 1][27360, 855, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp_1, -1), kwargs = {})
#   %mul_3 : Tensor "f16[1, 32, 855, 128][3502080, 109440, 128, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %unsqueeze_1), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 32, 855, 1][27360, 855, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%maximum, -1), kwargs = {})
#   %sub : Tensor "f16[1, 32, 855, 2048][56033280, 1751040, 2048, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%batched_matmul_default_1, %unsqueeze), kwargs = {})
#   %exp : Tensor "f16[1, 32, 855, 2048][56033280, 1751040, 2048, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %batched_matmul_default : Tensor "f16[1, 32, 855, 128][3502080, 109440, 128, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.batched_matmul.default](args = (%expand_2, %expand_3), kwargs = {})
#   %add_1 : Tensor "f16[1, 32, 855, 128][3502080, 109440, 128, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %batched_matmul_default), kwargs = {})
#   %full_default_2 : Tensor "f16[1, 32, 855, 64][1751040, 54720, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 855, 64], 0), kwargs = {dtype: torch.float16, layout: torch.strided, device: spyre:0, pin_memory: False})
#   %amax_1 : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%full_default_2, [-1]), kwargs = {})
#   %mul_2 : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%amax_1, %exp_1), kwargs = {})
#   %sum_1 : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1]), kwargs = {})
#   %add : Tensor "f16[1, 32, 855][27360, 855, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %sum_1), kwargs = {})
#   %unsqueeze_2 : Tensor "f16[1, 32, 855, 1][27360, 855, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%copy_f, -1), kwargs = {})
#   %div : Tensor "f16[1, 32, 855, 128][3502080, 109440, 128, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%copy_f_1, %unsqueeze_2), kwargs = {})
#   return %full_default,%full_default_1,%amax,%expand,%mul,%restickify_default,%batched_matmul_default_1,%amax_2,%maximum,%sub_1,%exp_1,%mul_3,%sub,%expand_2,%batched_matmul_default,%copy_f_1,%full_default_2,%amax_1,%mul_2,%sum_1,%copy_f,%copy_f_3
sdsc_fused__scaled_dot_product_fused_attention_overrideable_0 = async_compile.sdsc('sdsc_fused__scaled_dot_product_fused_attention_overrideable_0',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('32'), 1), sympify('c1'): (sympify('855'), 19), sympify('c2'): (sympify('128'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5678567947624747936, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('full_default', 'buf1'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 2, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('32'), 1), sympify('c1'): (sympify('855'), 19), sympify('c2'): (sympify('64'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7874648198735119972, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('full_default_1', 'buf3'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 32, 855, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('c1'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 7004160},
                ),
            ]
        ),
        OpSpec(
            op='max',
            is_reduction=True,
            iteration_space={sympify('d0'): (sympify('32'), 1), sympify('d1'): (sympify('855'), 19), sympify('d2'): (sympify('64'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=565846816004769479, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('amax', 'buf4'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 32, 855, 64],
                    device_coordinates=[sympify('floor(d2/64)'), sympify('d0'), sympify('d1'), sympify('Mod(d2, 64)')],
                    allocation={'hbm_pool': 7004160},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='mul',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('32'), 1), sympify('c1'): (sympify('855'), 19), sympify('c2'): (sympify('128'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1837772491231033328, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_1', 'buf5'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 2, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 2, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 184320},
                ),
            ]
        ),
        OpSpec(
            op='mul',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('32'), 1), sympify('c1'): (sympify('2048'), 32), sympify('c2'): (sympify('128'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4130712918363663155, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul', 'buf6'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 2048, 2, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 2048, 2, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 10506240},
                ),
            ]
        ),
        OpSpec(
            op='ReStickifyOpHBM',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('32'), 32), sympify('c1'): (sympify('2048'), 1), sympify('c2'): (sympify('128'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=143104103268148660, source=None, aten_op=None, ir_chain=('restickify_default', 'buf26'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 2048, 2, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 10506240},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 128, 32, 64],
                    device_coordinates=[sympify('c0'), sympify('c2'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 27283456},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('32'), 1), sympify('c1'): (sympify('855'), 19), sympify('c2'): (sympify('2048'), 1), sympify('c3'): (sympify('128'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4586806881513237791, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op=None, ir_chain=('batched_matmul_default_1', 'permute', 'buf7'), fused_from=(DebugHandle(id=8704335446284134785, source=None, aten_op=None, ir_chain=('batched_matmul_default_1',), fused_from=(), transform_history=()), DebugHandle(id=1861707858323349723, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('permute',), fused_from=(), transform_history=())), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='insert_restickify', reason='redirect consumer to restickified input'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 2, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c3/64)'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 184320},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 128, 32, 64],
                    device_coordinates=[sympify('c0'), sympify('c3'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 27283456},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 32, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 44060672},
                ),
            ]
        ),
        OpSpec(
            op='max',
            is_reduction=True,
            iteration_space={sympify('d0'): (sympify('32'), 1), sympify('d1'): (sympify('855'), 19), sympify('d2'): (sympify('2048'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7582770498290071487, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('amax_2', 'buf8'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 32, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'hbm_pool': 44060672},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 184320},
                ),
            ]
        ),
        OpSpec(
            op='maximum',
            is_reduction=False,
            iteration_space={sympify('d0'): (sympify('32'), 1), sympify('d1'): (sympify('855'), 19)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2851990788120648259, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('maximum', 'buf9'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 184320},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 184320},
                ),
            ]
        ),
        OpSpec(
            op='sub',
            is_reduction=False,
            iteration_space={sympify('d0'): (sympify('32'), 1), sympify('d1'): (sympify('855'), 19)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1595934540802580286, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub_1', 'buf10'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 184320},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='exp',
            is_reduction=False,
            iteration_space={sympify('d0'): (sympify('32'), 1), sympify('d1'): (sympify('855'), 19)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8593127197509843497, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('exp_1', 'buf11'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='mul',
            is_reduction=False,
            iteration_space={sympify('d0'): (sympify('32'), 1), sympify('d1'): (sympify('855'), 19), sympify('d2'): (sympify('128'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1888791232144737217, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_3', 'unsqueeze_1', 'buf12'), fused_from=(DebugHandle(id=1144427306445682697, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_3',), fused_from=(), transform_history=()), DebugHandle(id=5305091296193797497, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze_1',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 2, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 2, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'lx': 368640},
                ),
            ]
        ),
        OpSpec(
            op='sub',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('32'), 1), sympify('c1'): (sympify('855'), 19), sympify('c2'): (sympify('2048'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3574041098515655281, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub', 'unsqueeze', 'buf13'), fused_from=(DebugHandle(id=9214819459581412473, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub',), fused_from=(), transform_history=()), DebugHandle(id=4339337291282323371, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 32, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 44060672},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c0'), sympify('0')],
                    allocation={'lx': 184320},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 32, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 156127232},
                ),
            ]
        ),
        OpSpec(
            op='exp',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('32'), 1), sympify('c1'): (sympify('855'), 19), sympify('c2'): (sympify('2048'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=301346763309606081, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('exp', 'buf14'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 32, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 156127232},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 32, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 44060672},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('d0'): (sympify('32'), 1), sympify('d1'): (sympify('855'), 19), sympify('d2'): (sympify('128'), 1), sympify('d3'): (sympify('2048'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8027893405369574072, source=None, aten_op=None, ir_chain=('batched_matmul_default', 'buf15'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 32, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d3/64)'), sympify('Mod(d3, 64)')],
                    allocation={'hbm_pool': 44060672},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 2048, 2, 64],
                    device_coordinates=[sympify('d0'), sympify('d3'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 2, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'lx': 737280},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('32'), 1), sympify('c1'): (sympify('855'), 19), sympify('c2'): (sympify('128'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5441582508305468545, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_1', 'buf16'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 2, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 368640},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 2, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 737280},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 2, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 10506240},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('32'), 1), sympify('c1'): (sympify('855'), 19), sympify('c2'): (sympify('64'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=841754555736628814, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('full_default_2', 'buf18'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 32, 855, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('c1'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 7004160},
                ),
            ]
        ),
        OpSpec(
            op='max',
            is_reduction=True,
            iteration_space={sympify('d0'): (sympify('32'), 1), sympify('d1'): (sympify('855'), 19), sympify('d2'): (sympify('64'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8328508986905995693, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('amax_1', 'buf19'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 32, 855, 64],
                    device_coordinates=[sympify('floor(d2/64)'), sympify('d0'), sympify('d1'), sympify('Mod(d2, 64)')],
                    allocation={'hbm_pool': 7004160},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 184320},
                ),
            ]
        ),
        OpSpec(
            op='mul',
            is_reduction=False,
            iteration_space={sympify('d0'): (sympify('32'), 1), sympify('d1'): (sympify('855'), 19)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=705486835501610757, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_2', 'buf20'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 184320},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 184320},
                ),
            ]
        ),
        OpSpec(
            op='sum',
            is_reduction=True,
            iteration_space={sympify('d0'): (sympify('32'), 1), sympify('d1'): (sympify('855'), 19), sympify('d2'): (sympify('2048'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3209298145367023122, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sum_1', 'buf21'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 32, 64],
                    device_coordinates=[sympify('d0'), sympify('d1'), sympify('floor(d2/64)'), sympify('Mod(d2, 64)')],
                    allocation={'hbm_pool': 44060672},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('d1'), sympify('d0'), sympify('0')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('32'), 1), sympify('c1'): (sympify('855'), 19)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8524803944925484016, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add', 'buf22'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c0'), sympify('0')],
                    allocation={'lx': 184320},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c0'), sympify('0')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c0'), sympify('0')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='realdiv',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('32'), 1), sympify('c1'): (sympify('855'), 19), sympify('c2'): (sympify('128'), 1)},
            op_info={},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1892775538375616293, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('div', 'unsqueeze_2', 'buf23'), fused_from=(DebugHandle(id=1271510443118617506, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('div',), fused_from=(), transform_history=()), DebugHandle(id=6364399705516583887, source=SourceLoc(file='torch-spyre/tests/models/runner.py', start_line=220, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze_2',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 2, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 10506240},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 855, 32, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c0'), sympify('0')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[32, 855, 2, 64],
                    device_coordinates=[sympify('c0'), sympify('c1'), sympify('floor(c2/64)'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 6},
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
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        buf0 = spyre_constant_tensor(0.0, torch.device("spyre:0"), torch.float16)
        buf2 = spyre_constant_tensor(-inf, torch.device("spyre:0"), torch.float16)
        buf24 = spyre_constant_tensor(0.29730177875068026, torch.device("spyre:0"), torch.float16)
        assert_size_stride(arg0_1, (1, 32, 855, 128), (3502080, 109440, 128, 1), 'input')
        assert_size_stride(arg1_1, (1, 32, 2048, 128), (8388608, 262144, 128, 1), 'input')
        assert_size_stride(arg2_1, (1, 32, 2048, 128), (8388608, 262144, 128, 1), 'input')
        buf23 = spyre_empty_with_layout((1, 32, 855, 128), (3502080, 109440, 128, 1), torch.float16, SpyreTensorLayout(device_size=[32, 855, 2, 1, 64], stride_map =[109440, 128, 64, -1, 1], device_dtype=DataFormats.SEN169_FP16))
        # [Provenance debug handles] sdsc_fused__scaled_dot_product_fused_attention_overrideable_0:1
        _pool_sdsc_fused__scaled_dot_product_fused_attention_overrideable_0 = spyre_empty_with_layout((268193792,), (1,), torch.uint8, SpyreTensorLayout(device_size=[268193792], stride_map=[1], device_dtype=DataFormats.SENINT8))
        sdsc_fused__scaled_dot_product_fused_attention_overrideable_0.run(_pool_sdsc_fused__scaled_dot_product_fused_attention_overrideable_0, buf0, buf2, arg0_1, buf24, arg1_1, arg2_1, buf23)
        del _pool_sdsc_fused__scaled_dot_product_fused_attention_overrideable_0
        del arg0_1
        del arg1_1
        del arg2_1
        return (buf23, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns
