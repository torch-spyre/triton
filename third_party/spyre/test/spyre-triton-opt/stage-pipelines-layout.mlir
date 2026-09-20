// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir | FileCheck %s --check-prefix=KTIR
// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir --spyre-prepare-spyrecode | FileCheck %s --check-prefix=PHYS

// Where an annotated descriptor is physicalized, driven by the two stage flags.
//
// This is the one claim the whole layout migration rests on, and no single pass
// can make it: the layout is authored on a descriptor, becomes a discardable
// attribute inside the `ktir` stage, SURVIVES that stage's artifact, and is
// consumed in the `spyrecode` stage. An op could not have crossed that boundary --
// no reader of the artifact registers the Triton dialect, so a surviving `tt.*`
// op fails at parse -- which is why the annotation is an attribute whose values
// are builtin.
//
// stage-pipelines.mlir covers the unannotated kernel and the argument-passing
// modes. This file is only about the layout, so its kernel is the smallest thing
// that carries one: a 2-D copy, stick-tiled on N. No reduce and no contraction,
// because what is under test is which stage rewrites the view, not what the
// rewrite does to a compute op -- that is
// Dialect/KTDP/Transforms/RewriteDescriptorLayoutGeneric/'s.

module {
  tt.func public @copy_kernel(%in_ptr: !tt.ptr<f16>, %out_ptr: !tt.ptr<f16>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %m = arith.constant 64 : i32
    %n = arith.constant 128 : i32
    %sm = arith.constant 128 : i64
    %sn = arith.constant 1 : i64
    %in_desc = tt.make_tensor_descriptor %in_ptr, [%m, %n], [%sm, %sn] : <f16>, <64x128xf16>
    tts.tensor_layout %in_desc {phys_src = array<i64: 1, 0, 1>,
                                phys_op = array<i64: 1, 0, 2>,
                                phys_arg = array<i64: 64, 0, 64>} : <64x128xf16>
    %out_desc = tt.make_tensor_descriptor %out_ptr, [%m, %n], [%sm, %sn] : <f16>, <64x128xf16>
    tts.tensor_layout %out_desc {phys_src = array<i64: 1, 0, 1>,
                                 phys_op = array<i64: 1, 0, 2>,
                                 phys_arg = array<i64: 64, 0, 64>} : <64x128xf16>
    %t = tt.descriptor_load %in_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
    tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %t : !tt.tensordesc<64x128xf16>, tensor<64x128xf16>
    tt.return
  }
}

// The `ktir` stage. The view is LOGICAL -- sizes [64, 128] and the strides the
// kernel declared -- and the layout rides on it as the `tts.tensor_layout`
// attribute that LowerTTSMarkers moved there from the op.
//
// KTIR-LABEL:   func.func @copy_kernel(
// KTIR:           ktdp.construct_memory_view
// KTIR-SAME:        sizes: [64, 128], strides: [128, 1]
// KTIR-SAME:        tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}
// KTIR:           ktdp.construct_memory_view
// KTIR-SAME:        sizes: [64, 128], strides: [128, 1]
// KTIR-SAME:        tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}
//
// Neither the authoring op nor the bridge cast it needed reaches the artifact. The
// first would fail at parse in any reader that does not register `tt`; the second
// exists only to hold the op's `!tt.tensordesc` operand and goes with it.
//
// KTIR-NOT:       tts.tensor_layout %
// KTIR-NOT:       unrealized_conversion_cast
// KTIR-NOT:       tt.make_tensor_descriptor

// The `spyrecode` stage. Now the view is PHYSICAL: [M, N] stick-on-N at a 64-lane
// stick is [ceil(N/64), M, 64] = [2, 64, 64], with strides laid out row-major over
// those physical sizes rather than derived from the logical ones. The attribute is
// gone from the rebuilt view, which is what makes the pass idempotent.
//
// PHYS-LABEL:   func.func @copy_kernel(
// PHYS:           ktdp.construct_memory_view
// PHYS-SAME:        sizes: [2, 64, 64], strides: [4096, 64, 1]
// PHYS:           ktdp.construct_memory_view
// PHYS-SAME:        sizes: [2, 64, 64], strides: [4096, 64, 1]
// PHYS-NOT:       tts.tensor_layout
