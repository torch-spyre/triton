// RUN: spyre-triton-opt %s --split-input-file --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout | FileCheck %s

// Three cases exercising Phase 2's forward propagation through an
// elementwise chain, seeded only from values Phase 1 physicalized.
//
// Positive anchors are kept minimal on purpose: CHECK-NOT only searches the
// gap up to the next positive match, so it is repeated in every gap.

// Two annotated inputs feeding an annotated store through arith.addf: the
// multi-tensor-operand case RewriteElementwisePattern's local shape rule
// covers. Both operands physicalize to the same stick shape, addf's result
// is retyped to match, and the store's destination marker lets it
// physicalize too -- so the whole chain stays physical with no bridging
// loop, unlike the unannotated-store case below.
module {
// CHECK-LABEL: tt.func @elementwise_annotated_addf
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.load
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.load
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       arith.addf {{.*}} : tensor<2x64x64xf32>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK-NOT:   scf.for
// CHECK:       ktdp.store {{.*}} : tensor<2x64x64xf32>, <2x64x64xindex>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
tt.func @elementwise_annotated_addf(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64
  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %b_desc = tt.make_tensor_descriptor %b_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %b_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %b = tt.descriptor_load %b_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %c_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %s = arith.addf %a, %b : tensor<64x128xf32>
  tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %s : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}
}

// -----

// Annotated input feeding an unannotated store through arith.negf. Phase 2
// decides the store itself, sees there is no destination marker to
// physicalize into, and emits a bridging loop (emitBridgeToLogical) that
// reassembles the logical shape from the physical data tile's stick slices
// before the store.
module {
// CHECK-LABEL: tt.func @elementwise_unannotated_store
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.load
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       arith.negf {{.*}} : tensor<2x64x64xf32>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tensor.empty() : tensor<64x128xf32>
// CHECK:       scf.for
// CHECK:         tensor.extract_slice
// CHECK:         tensor.insert_slice
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.store {{.*}} : tensor<64x128xf32>, <64x128xindex>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
tt.func @elementwise_unannotated_store(%a_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64
  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  %n = arith.negf %a : tensor<64x128xf32>
  tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %n : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}
}

// -----

// Regression test: RewriteElementwisePattern's local shape rule (single
// RankedTensorType result + tensor operands sharing one shape + a differently
// shaped result) must NOT retype an op that merely satisfies that shape
// pattern by coincidence when it is not reachable from any physicalized
// ktdp.load. tt.expand_dims is exactly such an op: rank-changing, one tensor
// operand, shape mismatch by construction -- but here it sits downstream of a
// tt.reduce over an ANNOTATED load, on a path with no marker of its own, so it
// must be left as a plain logical reshape.
//
// A is annotated (stick-on-K(64), phys [K/64, M, K%64] = [2, 64, 64]) and
// feeds arith.negf + tt.reduce; the physicalized reduce loop only covers that
// part of the chain. Downstream of the reduce -- expand_dims, then a
// collapse_shape/broadcast idiom -- runs on logical tensor<64xf32> shapes the
// whole way to an unannotated store, exercising exactly the reachability gap
// ctx.physicalValues closes: membership in that set stops propagating once
// the reduce's plain (non-elementwise) result is produced, so expand_dims
// downstream of it is simply never a candidate.
module {
// CHECK-LABEL: tt.func @elementwise_expand_dims_unannotated
// CHECK:       ktdp.load {{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:       arith.negf {{.*}} : tensor<2x64x64xf32>
// CHECK:       linalg.reduce{{.*}}dimensions = [0, 2]
// CHECK:       tensor.expand_shape %{{.*}} {{.*}} output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
// CHECK:       tensor.collapse_shape %{{.*}} : tensor<64x1xf32> into tensor<64xf32>
// CHECK:       linalg.broadcast ins(%{{.*}} : tensor<64xf32>) outs(%{{.*}} : tensor<64x128xf32>)
// CHECK:       ktdp.store %{{.*}} : tensor<64x128xf32>, <64x128xindex>
// CHECK:       tt.return
tt.func @elementwise_expand_dims_unannotated(%a_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64
  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %n = arith.negf %a : tensor<64x128xf32>
  %r = "tt.reduce"(%n) <{axis = 1 : i32}> ({
  ^bb0(%arg5: f32, %arg6: f32):
    %s = arith.addf %arg5, %arg6 : f32
    tt.reduce.return %s : f32
  }) : (tensor<64x128xf32>) -> tensor<64xf32>
  %e = tt.expand_dims %r {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
  %b = tt.broadcast %e : tensor<64x1xf32> -> tensor<64x128xf32>
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %b : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}
}
