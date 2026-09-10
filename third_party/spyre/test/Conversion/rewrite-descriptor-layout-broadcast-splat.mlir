// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout -split-input-file | FileCheck %s

// Seeding a splat constant so layernorm can carry a layout.
// Design: docs/designs/rewrite-broadcast-pattern.md.
//
// WHAT THESE TESTS ASK OF THE PASS: in the layernorm shape -- reduce, broadcast,
// then divide by the element count N -- the divisor must reach the physical shape
// along with the broadcast, so the divide's operands agree.
//
// The remaining cases bound that: which constants qualify, and what must happen
// when one is shared with a consumer that still wants the logical shape.
//
// WHY THE DIVISOR NEEDED SEPARATE WORK: `tl.splat`/a Python scalar becomes an
// `arith.constant dense<...>` -- a value with ZERO operands. The forward layout
// analysis walks along operands, asking "given this physical operand, what type
// does the result take?", so an op with no operands is unreachable by
// construction and no analysis rule can claim a type for it. Left logical while
// its sibling went physical, it failed `arith.divf`'s same-type verifier.
//
// The fix is a SEED, not a propagation rule: when an elementwise op has one
// physical operand and one splat-constant operand, the constant is rebuilt at
// the physical shape. Sound because every element is identical -- there is no
// data to move and no coordinate map to rewrite.

// CHECK-LABEL: func @layernorm_splat_divisor
// The divisor carries the same repeated value at the physical shape. It feeds
// only this divide, so it is retyped IN PLACE -- no canonicalizer needed to
// clear a leftover logical copy, and CHECK-NOT below proves none is left.
// CHECK: %[[N:.*]] = arith.constant dense<1.280000e+02> : tensor<64x2x64xf32>
// CHECK-NOT: arith.constant dense<1.280000e+02> : tensor<64x128xf32>
// CHECK: ktdp.load {{.*}} -> tensor<64x2x64xf32>
// CHECK: linalg.reduce
// CHECK-SAME: dimensions = [1, 2]
// CHECK: linalg.broadcast
// CHECK-SAME: outs(%{{.*}} : tensor<64x2x64xf32>)
// CHECK-SAME: dimensions = [1, 2]
// Both arithmetic ops now have agreeing operands.
// CHECK: arith.divf %{{.*}}, %[[N]] : tensor<64x2x64xf32>
// CHECK: arith.subf %{{.*}}, %{{.*}} : tensor<64x2x64xf32>
// CHECK: ktdp.store %{{.*}}, %{{.*}} : tensor<64x2x64xf32>
module {
tt.func @layernorm_splat_divisor(%a_ptr: !tt.ptr<f32>, %o_ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %c128 = arith.constant 128 : i32
  %s128 = arith.constant 128 : i64
  %s1 = arith.constant 1 : i64
  // The element count along the reduced axis, as a splat: 128 columns.
  %n = arith.constant dense<1.280000e+02> : tensor<64x128xf32>
  %ad = tt.make_tensor_descriptor %a_ptr, [%c64, %c128], [%s128, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 1, 2>, phys_arg = array<i64: 0, 64, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %ad[%c0, %c0] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  // Sum along the columns, then fan the row-sums back out: the mean's numerator.
  %s = "tt.reduce"(%a) ({ ^bb0(%x: f32, %y: f32): %sm = arith.addf %x, %y : f32
    tt.reduce.return %sm : f32 }) {axis = 1 : i32} : (tensor<64x128xf32>) -> tensor<64xf32>
  %se = tt.expand_dims %s {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
  %sb = tt.broadcast %se : tensor<64x1xf32> -> tensor<64x128xf32>
  %mean = arith.divf %sb, %n : tensor<64x128xf32>
  %d = arith.subf %a, %mean : tensor<64x128xf32>
  %od = tt.make_tensor_descriptor %o_ptr, [%c64, %c128], [%s128, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %od {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 1, 2>, phys_arg = array<i64: 0, 64, 64>} : !tt.tensordesc<64x128xf32>
  tt.descriptor_store %od[%c0, %c0], %d : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}
}

// -----

// Non-divisible: 130 columns become 3 sticks of 64, i.e. 192 lanes for 130 real
// ones. Every lane including the 62 padding ones gets the same value, which is
// what "splat" means -- so unlike the store bridge, seeding a splat is correct
// on a non-divisible split. This also exercises resizeSplat rather than
// reshape: the element count CHANGES (130 -> 192), which reshape forbids.

// CHECK-LABEL: func @splat_non_divisible
// CHECK: arith.constant dense<1.300000e+02> : tensor<64x3x64xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<64x3x64xf32>
module {
tt.func @splat_non_divisible(%a_ptr: !tt.ptr<f32>, %o_ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %c130 = arith.constant 130 : i32
  %s130 = arith.constant 130 : i64
  %s1 = arith.constant 1 : i64
  %n = arith.constant dense<1.300000e+02> : tensor<64x130xf32>
  %ad = tt.make_tensor_descriptor %a_ptr, [%c64, %c130], [%s130, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x130xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 1, 2>, phys_arg = array<i64: 0, 64, 64>} : !tt.tensordesc<64x130xf32>
  %a = tt.descriptor_load %ad[%c0, %c0] : !tt.tensordesc<64x130xf32> -> tensor<64x130xf32>
  %d = arith.divf %a, %n : tensor<64x130xf32>
  %od = tt.make_tensor_descriptor %o_ptr, [%c64, %c130], [%s130, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x130xf32>
  tt.spyre_tensor_layout %od {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 1, 2>, phys_arg = array<i64: 0, 64, 64>} : !tt.tensordesc<64x130xf32>
  tt.descriptor_store %od[%c0, %c0], %d : !tt.tensordesc<64x130xf32>, tensor<64x130xf32>
  tt.return
}
}

// -----

// One splat feeding TWO consumers: an annotated path that goes physical and an
// unannotated one that must stay logical. The seed mints a FRESH constant
// rather than retyping in place, so the logical use is untouched. Retyping in
// place would corrupt it -- which is why sharing is checked.

// CHECK-LABEL: func @shared_splat_two_consumers
// The physical consumer gets a new constant at the physical shape...
// CHECK: arith.constant dense<1.280000e+02> : tensor<64x2x64xf32>
// ...and the original survives for the logical consumer.
// CHECK: arith.constant dense<1.280000e+02> : tensor<64x128xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : tensor<64x2x64xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : tensor<64x128xf32>
module {
tt.func @shared_splat_two_consumers(%a_ptr: !tt.ptr<f32>, %o_ptr: !tt.ptr<f32>, %p_ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %c128 = arith.constant 128 : i32
  %s128 = arith.constant 128 : i64
  %s1 = arith.constant 1 : i64
  %n = arith.constant dense<1.280000e+02> : tensor<64x128xf32>
  %ad = tt.make_tensor_descriptor %a_ptr, [%c64, %c128], [%s128, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 1, 2>, phys_arg = array<i64: 0, 64, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %ad[%c0, %c0] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %d = arith.divf %a, %n : tensor<64x128xf32>
  %od = tt.make_tensor_descriptor %o_ptr, [%c64, %c128], [%s128, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %od {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 1, 2>, phys_arg = array<i64: 0, 64, 64>} : !tt.tensordesc<64x128xf32>
  tt.descriptor_store %od[%c0, %c0], %d : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  %pd = tt.make_tensor_descriptor %p_ptr, [%c64, %c128], [%s128, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  %q = tt.descriptor_load %pd[%c0, %c0] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %r = arith.mulf %q, %n : tensor<64x128xf32>
  tt.descriptor_store %pd[%c0, %c0], %r : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}
}

// -----

// The layernorm shape under the STICK-OUTERMOST marker, phys_src=[1,0,1], which
// is the ordering most fixtures in this suite use. The split axis owns physical
// dims 0 and 2 here rather than 1 and 2, so the seeded constant has to land at
// <2x64x64> and not <64x2x64>.
//
// A splat is indifferent to that order -- every element is the same value, so
// resizeSplat only needs the total shape -- which is exactly what makes this
// worth pinning: the seed must take its shape from the physical operand beside
// it, never reconstruct one from the marker.

// CHECK-LABEL: func @layernorm_splat_stick_outermost
// CHECK: %[[N:.*]] = arith.constant dense<1.280000e+02> : tensor<2x64x64xf32>
// CHECK-NOT: arith.constant dense<1.280000e+02> : tensor<64x128xf32>
// CHECK: linalg.reduce
// CHECK-SAME: dimensions = [0, 2]
// CHECK: arith.divf %{{.*}}, %[[N]] : tensor<2x64x64xf32>
// CHECK: ktdp.store %{{.*}}, %{{.*}} : tensor<2x64x64xf32>
module {
tt.func @layernorm_splat_stick_outermost(%a_ptr: !tt.ptr<f32>, %o_ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %c128 = arith.constant 128 : i32
  %s128 = arith.constant 128 : i64
  %s1 = arith.constant 1 : i64
  %n = arith.constant dense<1.280000e+02> : tensor<64x128xf32>
  %ad = tt.make_tensor_descriptor %a_ptr, [%c64, %c128], [%s128, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %ad[%c0, %c0] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %s = "tt.reduce"(%a) ({ ^bb0(%x: f32, %y: f32): %sm = arith.addf %x, %y : f32
    tt.reduce.return %sm : f32 }) {axis = 1 : i32} : (tensor<64x128xf32>) -> tensor<64xf32>
  %se = tt.expand_dims %s {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
  %sb = tt.broadcast %se : tensor<64x1xf32> -> tensor<64x128xf32>
  %mean = arith.divf %sb, %n : tensor<64x128xf32>
  %d = arith.subf %a, %mean : tensor<64x128xf32>
  %od = tt.make_tensor_descriptor %o_ptr, [%c64, %c128], [%s128, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %od {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  tt.descriptor_store %od[%c0, %c0], %d : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}
}
