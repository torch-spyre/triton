// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout --canonicalize -split-input-file | FileCheck %s

// A surviving stick-index dim as a BATCH dim of the reduce.
//
// When a reduce folds away whole physical dimensions and the descriptor its
// result is stored to declares exactly the layout the operand's surviving
// physical dims induce, the reduce's output is physicalized: the surviving stick
// index rides along as a batch dim of the one linalg.reduce and ktdp.store
// consumes the result directly. That is the Physical OutputAxisSpace.
//
// The three cases here are the decision, not just its happy path. Case 1 takes
// the form, case 2 is the same input layout folding the stick axis instead — so
// nothing stick-split survives, the induced layout stops matching, and the old
// logical form is still what comes out — and case 3 carries two batch dims at
// once, one of them a plain untouched dim.
//
// CHECKs are written rather than generated: what is pinned is a shape and an
// absence, and a whole-function transcript would pin every constant beside them.
//
// --canonicalize is in the RUN line because this pass never runs alone: the
// backend follows it with canonicalize + CSE in the same pass manager
// (SpyreBackend._make_ktir), so the module *after* folding is the one that
// becomes KTIR, and it is the one worth pinning. One thing it removes that a
// standalone run would leave behind, and this file would then have to assert as
// if it mattered:
//
//   - an identity tensor.extract_slice. The widen stage emits one per stick, and
//     at a single stick that is a slice of the whole thing into its own type --
//     a no-op the folder drops (case 2).
//
// It used to be two. The other was the dead logical init: rebuildPhysicalInit
// minted the accumulator at the physical shape instead of retyping the original,
// leaving a userless tensor.empty per Physical-space case. It now retypes in
// place, so the standalone output carries no dead init -- tensor.empty over this
// file goes 8 -> 5. The CHECK-NOT: tensor.empty below would hold without
// --canonicalize now; case 2's extract_slice is what still needs it.
//
// The CHECK-NOTs are interleaved between consecutive positive checks rather than
// gathered at the end: a CHECK-NOT's range runs from the previous match to the
// next one, so a block of them after the last positive check would only cover the
// tail. Tiling the ranges is what makes "no loop anywhere in this function" true
// of the whole function.

// Case 1: the batch-dim form.
//   A[M=64, N=128] f16 stick-on-N(64) -> phys [N/64, M, N%64] = [2, 64, 64]
//   reduce logical dim 0 (M, a whole physical dim) -> logical [128]
//   C[128] stick(64) -> phys [C/64, C%64] = [2, 64]
// The induced output layout is (N-floordiv-64, N-mod-64) with extents [2, 64],
// which is exactly what C declares, so:
//   - ONE linalg.reduce, over the loaded tile whole, dimensions = [1];
//   - the surviving stick index is dim 0 of both ins and outs — the batch dim;
//   - no scf.for and no extract_slice/insert_slice around it;
//   - ktdp.store takes the reduce result itself.
module {
// CHECK-LABEL:   tt.func @reduce_surviving_stick_is_a_batch_dim(
// CHECK:           %[[LOAD:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf16>
// The accumulator is at the PHYSICAL shape: rank 2, stick index first -- the
// original logical init retyped in place, not a second one minted beside it.
// The neutral-element fill survives here on purpose — dropping it is
// DropReductionInitFill's job, later, in the spyrecode stage only.
// CHECK-NOT:       scf.for
// CHECK-NOT:       tensor.extract_slice
// One tensor.empty in the whole function -- the accumulator. A second one here
// would be a minted accumulator with the logical init left dead beside it.
// CHECK-NOT:       tensor.empty
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK:           %[[ACC:.*]] = linalg.fill ins(%{{.*}} : f16) outs(%[[EMPTY]] : tensor<2x64xf16>) -> tensor<2x64xf16>
// CHECK-NOT:       scf.for
// CHECK-NOT:       tensor.extract_slice
// CHECK:           %[[RED:.*]] = linalg.reduce ins(%[[LOAD]] : tensor<2x64x64xf16>) outs(%[[ACC]] : tensor<2x64xf16>) dimensions = [1]
// One reduce, and nothing tiling it.
// CHECK-NOT:       linalg.reduce
// CHECK-NOT:       scf.for
// CHECK-NOT:       tensor.extract_slice
// CHECK-NOT:       tensor.insert_slice
// CHECK:           ktdp.store %[[RED]], %{{.*}} : tensor<2x64xf16>, <2x64xindex>
tt.func @reduce_surviving_stick_is_a_batch_dim(%a_ptr: !tt.ptr<f16>, %c_ptr: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64

  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<64x128xf16>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf16>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>

  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c128_i32], [%c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<128xf16>
  tt.spyre_tensor_layout %c_desc {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : !tt.tensordesc<128xf16>

  %r = "tt.reduce"(%a) ({
  ^bb0(%arg0: f16, %arg1: f16):
    %add = arith.addf %arg0, %arg1 : f16
    tt.reduce.return %add : f16
  }) {axis = 0 : i32} : (tensor<64x128xf16>) -> tensor<128xf16>

  tt.descriptor_store %c_desc[%c0_i32], %r : !tt.tensordesc<128xf16>, tensor<128xf16>
  tt.return
}
}

// -----

// Case 2: the same input layout, folding the STICK axis instead. Reducing N
// consumes both of its physical dims, so nothing stick-split survives and the
// induced output layout is a single identity dim — not the split C declares.
// The Logical output-axis space is the answer, and it is the form it always was:
// one reduce naming both stick dims, then the store's widen stage re-tiling the
// rank-1 result into the rank-2 physical block. Guards the non-regression.
module {
// CHECK-LABEL:   tt.func @reduce_folds_the_stick_axis(
// CHECK:           %[[LOAD:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf16>
// CHECK:           %[[ACC:.*]] = linalg.fill ins(%{{.*}} : f16) outs(%{{.*}} : tensor<64xf16>) -> tensor<64xf16>
// CHECK:           %[[RED:.*]] = linalg.reduce ins(%[[LOAD]] : tensor<2x64x64xf16>) outs(%[[ACC]] : tensor<64xf16>) dimensions = [0, 2]
// The widen stage: the rank-1 result re-tiled into the rank-2 physical block.
// Its per-stick extract_slice is gone -- at one stick it slices the whole tensor
// into its own type, and the folder drops it -- so the insert_slice takes the
// reduce result directly. Nothing is scattered because there is one stick to
// scatter into; the rank change is the whole of the widen here.
// CHECK:           %[[TILE:.*]] = tensor.insert_slice %[[RED]] into %{{.*}} : tensor<64xf16> into tensor<1x64xf16>
// CHECK:           ktdp.store %[[TILE]], %{{.*}} : tensor<1x64xf16>, <1x64xindex>
tt.func @reduce_folds_the_stick_axis(%a_ptr: !tt.ptr<f16>, %c_ptr: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64

  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<64x128xf16>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf16>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>

  // C[64] stick(64) -> phys [1, 64].
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c64_i32], [%c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<64xf16>
  tt.spyre_tensor_layout %c_desc {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : !tt.tensordesc<64xf16>

  %r = "tt.reduce"(%a) ({
  ^bb0(%arg0: f16, %arg1: f16):
    %add = arith.addf %arg0, %arg1 : f16
    tt.reduce.return %add : f16
  }) {axis = 1 : i32} : (tensor<64x128xf16>) -> tensor<64xf16>

  tt.descriptor_store %c_desc[%c0_i32], %r : !tt.tensordesc<64xf16>, tensor<64xf16>
  tt.return
}
}

// -----

// Case 3: a rank-3 reduce whose surviving axes carry a stick split AND a plain
// identity dim, folding the middle one.
//   A[D0=2, D1=64, D2=128] f16 stick-on-D2(64)
//     -> phys [D2/64, D0, D1, D2%64] = [2, 2, 64, 64]
//   reduce logical dim 1 (D1) -> logical [2, 128]
//   C[2, 128] stick-on-dim1(64) -> phys [128/64, 2, 128%64] = [2, 2, 64]
// Three surviving physical dims, so three output axes and TWO batch dims: the
// stick index and the untouched D0. This is what a role numbered per logical dim
// cannot express — physical dims 0 and 3 both carry logical D2 — and it is why
// the accumulator is keyed by output axis instead.
module {
// CHECK-LABEL:   tt.func @reduce_two_batch_dims(
// CHECK:           %[[LOAD:.*]] = ktdp.load %{{.*}} : <2x2x64x64xindex> -> tensor<2x2x64x64xf16>
// CHECK-NOT:       scf.for
// CHECK-NOT:       tensor.extract_slice
// One tensor.empty, as in case 1 -- and named here rather than left as a
// wildcard so the CHECK-NOT above has somewhere to stop and the accumulator's
// rank-3 shape is stated where the fill consumes it.
// CHECK-NOT:       tensor.empty
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x2x64xf16>
// CHECK:           %[[ACC:.*]] = linalg.fill ins(%{{.*}} : f16) outs(%[[EMPTY]] : tensor<2x2x64xf16>) -> tensor<2x2x64xf16>
// CHECK-NOT:       scf.for
// CHECK-NOT:       tensor.extract_slice
// CHECK:           %[[RED:.*]] = linalg.reduce ins(%[[LOAD]] : tensor<2x2x64x64xf16>) outs(%[[ACC]] : tensor<2x2x64xf16>) dimensions = [2]
// CHECK-NOT:       linalg.reduce
// CHECK-NOT:       scf.for
// CHECK-NOT:       tensor.extract_slice
// CHECK:           ktdp.store %[[RED]], %{{.*}} : tensor<2x2x64xf16>, <2x2x64xindex>
tt.func @reduce_two_batch_dims(%a_ptr: !tt.ptr<f16>, %c_ptr: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %c128_i64 = arith.constant 128 : i64
  %c8192_i64 = arith.constant 8192 : i64

  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c2_i32, %c64_i32, %c128_i32], [%c8192_i64, %c128_i64, %c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<2x64x128xf16>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 2, 0, 1, 2>, phys_op = array<i64: 1, 0, 0, 2>, phys_arg = array<i64: 64, 0, 0, 64>} : !tt.tensordesc<2x64x128xf16>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<2x64x128xf16> -> tensor<2x64x128xf16>

  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c2_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<2x128xf16>
  tt.spyre_tensor_layout %c_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<2x128xf16>

  %r = "tt.reduce"(%a) ({
  ^bb0(%arg0: f16, %arg1: f16):
    %add = arith.addf %arg0, %arg1 : f16
    tt.reduce.return %add : f16
  }) {axis = 1 : i32} : (tensor<2x64x128xf16>) -> tensor<2x128xf16>

  tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %r : !tt.tensordesc<2x128xf16>, tensor<2x128xf16>
  tt.return
}
}

// -----

// Case 4: the same batch-dim form with an elementwise op between the reduce and
// the store. Nothing about the reduce changes -- what this adds is that the
// layout has to reach it across another op.
//
// This is the narrow guard on the backward analysis's reach. ReducePropagation
// selects the Physical space by comparing what its operand induces against the
// requirement AT ITS OWN RESULT, and the only thing that puts a requirement there
// is the backward elementwise rule carrying it back across the math.exp. If that
// rule ever terminates here instead, the reduce falls back to Logical and the
// CHECK-NOTs below catch the stick loop that replaces it.
module {
// CHECK-LABEL:   tt.func @reduce_batch_dim_through_elementwise(
// CHECK:           %[[LOAD:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf16>
// CHECK-NOT:       scf.for
// CHECK-NOT:       tensor.extract_slice
// CHECK-NOT:       tensor.empty
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK:           %[[ACC:.*]] = linalg.fill ins(%{{.*}} : f16) outs(%[[EMPTY]] : tensor<2x64xf16>) -> tensor<2x64xf16>
// CHECK:           %[[RED:.*]] = linalg.reduce ins(%[[LOAD]] : tensor<2x64x64xf16>) outs(%[[ACC]] : tensor<2x64xf16>) dimensions = [1]
// The elementwise op is retyped to the physical shape and the store takes it
// directly: the reduce's result never returns to logical rank on the way out.
// CHECK:           %[[EXP:.*]] = math.exp %[[RED]] : tensor<2x64xf16>
// CHECK-NOT:       linalg.reduce
// CHECK-NOT:       scf.for
// CHECK-NOT:       tensor.extract_slice
// CHECK-NOT:       tensor.insert_slice
// CHECK:           ktdp.store %[[EXP]], %{{.*}} : tensor<2x64xf16>, <2x64xindex>
tt.func @reduce_batch_dim_through_elementwise(%a_ptr: !tt.ptr<f16>, %c_ptr: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64

  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<64x128xf16>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf16>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>

  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c128_i32], [%c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<128xf16>
  tt.spyre_tensor_layout %c_desc {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : !tt.tensordesc<128xf16>

  %r = "tt.reduce"(%a) ({
  ^bb0(%arg0: f16, %arg1: f16):
    %add = arith.addf %arg0, %arg1 : f16
    tt.reduce.return %add : f16
  }) {axis = 0 : i32} : (tensor<64x128xf16>) -> tensor<128xf16>

  %e = math.exp %r : tensor<128xf16>

  tt.descriptor_store %c_desc[%c0_i32], %e : !tt.tensordesc<128xf16>, tensor<128xf16>
  tt.return
}
}
