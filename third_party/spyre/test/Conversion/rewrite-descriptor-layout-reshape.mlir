// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout -split-input-file | FileCheck %s

// Crossing size-1 reshapes when physicalizing a layout.
//
// A `tt.spyre_tensor_layout` annotation asks for the Spyre "stick" layout: one
// logical axis splits into a stick index (floordiv) and a lane (mod), so a
// logical <64x128> becomes a physical <64x2x64> and the rank goes 2 -> 3.
// phys_src says which logical axis each physical dim comes from, phys_op is
// 0=identity / 1=floordiv / 2=mod, and phys_arg is the stick WIDTH (64), not a
// dim index.
//
// `tt.expand_dims` inserts a length-1 axis and `tt.broadcast` fans out along
// it. LowerComputeOps lowers them independently, so expand-then-broadcast
// leaves a CANCELLING PAIR: expand 64 -> 64x1, then collapse 64x1 -> 64 right
// back. No real extent changes -- only size-1 axes are added and dropped.
//
// WHAT THIS TEST ASKS OF THE PASS: the layout must survive that round trip. A
// reshape that only adds or drops size-1 axes moves no data, so it must not stop
// the layout propagating -- if it does, the chain breaks before the broadcast
// that needs it and the kernel fails to compile.

// -----

// The cancelling pair, reached from a store that wants a physical layout.
// Axis 1 (length 128) is split into 2 sticks of 64 and is also the reduced
// axis, so the reduce absorbs both physical dims and the chain
// reduce -> collapse -> broadcast must stay connected across the reshape.

// CHECK-LABEL: func @reshape_cancelling_pair_crossed
// CHECK: ktdp.load {{.*}} -> tensor<64x2x64xf32>
// CHECK: linalg.reduce
// CHECK-SAME: outs(%{{.*}} : tensor<64xf32>)
// CHECK-SAME: dimensions = [1, 2]
// The reshape survives as a collapse of the size-1 axis and does NOT block the
// layout: the broadcast downstream of it is physical.
// CHECK: linalg.broadcast
// CHECK-SAME: outs(%{{.*}} : tensor<64x2x64xf32>)
// CHECK-SAME: dimensions = [1, 2]
// CHECK: ktdp.store %{{.*}}, %{{.*}} : tensor<64x2x64xf32>
module {
tt.func @reshape_cancelling_pair_crossed(%a_ptr: !tt.ptr<f32>, %o_ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %c128 = arith.constant 128 : i32
  %s128 = arith.constant 128 : i64
  %s1 = arith.constant 1 : i64
  %ad = tt.make_tensor_descriptor %a_ptr, [%c64, %c128], [%s128, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 1, 2>, phys_arg = array<i64: 0, 64, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %ad[%c0, %c0] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %m = "tt.reduce"(%a) ({ ^bb0(%x: f32, %y: f32): %sm = arith.addf %x, %y : f32
    tt.reduce.return %sm : f32 }) {axis = 1 : i32} : (tensor<64x128xf32>) -> tensor<64xf32>
  // expand then broadcast: the expand's 64x1 is collapsed straight back to 64.
  %me = tt.expand_dims %m {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
  %bc = tt.broadcast %me : tensor<64x1xf32> -> tensor<64x128xf32>
  %d = arith.subf %a, %bc : tensor<64x128xf32>
  %od = tt.make_tensor_descriptor %o_ptr, [%c64, %c128], [%s128, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %od {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 1, 2>, phys_arg = array<i64: 0, 64, 64>} : !tt.tensordesc<64x128xf32>
  tt.descriptor_store %od[%c0, %c0], %d : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}
}
