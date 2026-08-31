// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout | FileCheck %s

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
//
// CHECK-LABEL: tt.func @elementwise_expand_dims_unannotated
// CHECK:       ktdp.load {{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:       arith.negf {{.*}} : tensor<2x64x64xf32>
// CHECK:       scf.for
// CHECK:       linalg.reduce
// CHECK:       tensor.expand_shape %{{.*}} {{.*}} output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
// CHECK:       tensor.collapse_shape %{{.*}} : tensor<64x1xf32> into tensor<64xf32>
// CHECK:       linalg.broadcast ins(%{{.*}} : tensor<64xf32>) outs(%{{.*}} : tensor<64x128xf32>)
// CHECK:       ktdp.store %{{.*}} : tensor<64x128xf32>, <64x128xindex>
// CHECK:       tt.return
module {
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
