// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s

// Which domain the subscript arithmetic lands in.
//
// A ktdp.construct_access_tile subscript is `index` by the op's own definition,
// so every one of these kernels reaches the pass with its Triton-side i32
// arithmetic already terminated by one arith.index_cast. The question is whether
// the pass leaves that cast between the arithmetic and the split it emits, or
// rebuilds the arithmetic above it in `index` -- so every case below comes out as
// one of exactly two shapes:
//
//   arrives:   %o = arith.muli %pid, 64 : i32
//              %x = arith.index_cast %o : i32 to index
//   rebuilt:   %p = arith.index_cast %pid : i32 to index
//              %x = arith.muli %p, 64 : index      <- the split reads this
//
// The scheduler's symbolic start-address analysis treats a cast as opaque and
// rejects an address computed through one, so a grid-derived subscript must be
// rebuilt -- and anything whose value a rebuild would change must not be. Case 1
// is "rebuilt"; cases 2 to 4 are "arrives, unchanged", each for its own reason
// why lifting would not preserve the value.
//
// Checks are hand-written and minimal on purpose: the claim is which domain the
// subscript arithmetic lands in, not the whole module.

// Case 1 -- rebuilt: the subscript is pid-derived.
//
// Triton computes offsets in i32, so the subscript arrives as
// index_cast(muli(pid, c) : i32). The multiply is re-emitted in `index` over a
// single cast of the pid itself, and the split reads that. The i32 chain is left
// in place (dead once nothing reads it); canonicalization/CSE later collapse the
// duplicates.

#id = affine_map<(d0) -> (d0)>
#sview = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
#sblock = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
// CHECK-LABEL:   tt.func @pid_offset_lifted_to_index(
// CHECK:           %[[PID:.*]] = tt.get_program_id x : i32
// The floordiv subscript: pid cast once, then multiplied in `index`.
// CHECK:           %[[PIDX:.*]] = arith.index_cast %[[PID]] : i32 to index
// CHECK:           %[[C64:.*]] = arith.constant 64 : index
// CHECK:           %[[OFF:.*]] = arith.muli %[[PIDX]], %[[C64]] : index
// Nothing casts back into the subscript between the multiply and the split.
// CHECK-NOT:       arith.index_cast
// CHECK:           arith.divsi %[[OFF]], %{{.*}} : index
// The mod subscript, same shape.
// CHECK:           %[[PIDX2:.*]] = arith.index_cast %[[PID]] : i32 to index
// CHECK:           %[[OFF2:.*]] = arith.muli %[[PIDX2]], %{{.*}} : index
// CHECK:           arith.remsi %[[OFF2]], %{{.*}} : index
tt.func @pid_offset_lifted_to_index(%ptr: !tt.ptr<f16>) {
  %c64_i32 = arith.constant 64 : i32
  %pid = tt.get_program_id x : i32
  %off = arith.muli %pid, %c64_i32 : i32
  %bi = builtin.unrealized_conversion_cast %ptr : !tt.ptr<f16> to index
  // [n=128] stick-on-n at width 64 -> physical [n/64, n%64] = [2, 64]
  %v = ktdp.construct_memory_view %bi, sizes: [128], strides: [1] {coordinate_set = #sview, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %d = builtin.unrealized_conversion_cast %v : memref<128xf16> to !tt.tensordesc<128xf16>
  tt.spyre_tensor_layout %d {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : <128xf16>
  %offx = arith.index_cast %off : i32 to index
  %lt = ktdp.construct_access_tile %v[%offx] {access_tile_order = #id, access_tile_set = #sblock} : memref<128xf16> -> !ktdp.access_tile<64xindex>
  %l = ktdp.load %lt : <64xindex> -> tensor<64xf16>
  %offx2 = arith.index_cast %off : i32 to index
  %st = ktdp.construct_access_tile %v[%offx2] {access_tile_order = #id, access_tile_set = #sblock} : memref<128xf16> -> !ktdp.access_tile<64xindex>
  ktdp.store %l, %st : tensor<64xf16>, <64xindex>
  tt.return
}
}

// -----

// Case 2 -- arrives, unchanged: a run-time i32 scalar.
//
// It is not a grid coordinate, so its arithmetic keeps the width Triton gave it
// -- rebuilding in 64-bit `index` would change what the expression means on
// overflow. It reaches the split through the single cast it arrived with.

#id = affine_map<(d0) -> (d0)>
#sview = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
#sblock = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
// CHECK-LABEL:   tt.func @runtime_scalar_offset_unchanged(
// CHECK-SAME:      %{{.*}}: !tt.ptr<f16>, %[[N:.*]]: i32)
// CHECK:           %[[OFF:.*]] = arith.muli %[[N]], %{{.*}} : i32
// CHECK:           %[[OFFX:.*]] = arith.index_cast %[[OFF]] : i32 to index
// CHECK:           arith.divsi %[[OFFX]], %{{.*}} : index
// CHECK:           arith.remsi %[[OFFX]], %{{.*}} : index
tt.func @runtime_scalar_offset_unchanged(%ptr: !tt.ptr<f16>, %n: i32) {
  %c64_i32 = arith.constant 64 : i32
  %off = arith.muli %n, %c64_i32 : i32
  %bi = builtin.unrealized_conversion_cast %ptr : !tt.ptr<f16> to index
  %v = ktdp.construct_memory_view %bi, sizes: [128], strides: [1] {coordinate_set = #sview, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %d = builtin.unrealized_conversion_cast %v : memref<128xf16> to !tt.tensordesc<128xf16>
  tt.spyre_tensor_layout %d {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : <128xf16>
  %offx = arith.index_cast %off : i32 to index
  %lt = ktdp.construct_access_tile %v[%offx] {access_tile_order = #id, access_tile_set = #sblock} : memref<128xf16> -> !ktdp.access_tile<64xindex>
  %l = ktdp.load %lt : <64xindex> -> tensor<64xf16>
  %st = ktdp.construct_access_tile %v[%offx] {access_tile_order = #id, access_tile_set = #sblock} : memref<128xf16> -> !ktdp.access_tile<64xindex>
  ktdp.store %l, %st : tensor<64xf16>, <64xindex>
  tt.return
}
}

// -----

// Case 3 -- arrives, unchanged: a truncation sits in the chain.
//
// A truncation is not value-preserving, so lifting past it would feed the
// *untruncated* 64-bit product to the subscript and address a different tile
// than the i32 expression named. The subscript must keep reading the truncated
// value through a single cast, with the wide multiply left in place.

#id = affine_map<(d0) -> (d0)>
#sview = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
#sblock = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
// CHECK-LABEL:   tt.func @trunc_not_lifted(
// CHECK:           %[[WIDE:.*]] = arith.muli %{{.*}}, %{{.*}} : i64
// CHECK:           %[[TR:.*]] = arith.trunci %[[WIDE]] : i64 to i32
// CHECK:           %[[IDX:.*]] = arith.index_cast %[[TR]] : i32 to index
// The multiply stays in i64 above the trunc; nothing re-multiplies in `index`.
// CHECK-NOT:       arith.muli %{{.*}} : index
// CHECK:           arith.divsi %[[IDX]], %{{.*}} : index
tt.func @trunc_not_lifted(%ptr: !tt.ptr<f16>) {
  %c64_i64 = arith.constant 64 : i64
  %pid = tt.get_program_id x : i32
  %pid64 = arith.extsi %pid : i32 to i64
  %big = arith.muli %pid64, %c64_i64 : i64
  %off = arith.trunci %big : i64 to i32
  %bi = builtin.unrealized_conversion_cast %ptr : !tt.ptr<f16> to index
  %v = ktdp.construct_memory_view %bi, sizes: [128], strides: [1] {coordinate_set = #sview, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %d = builtin.unrealized_conversion_cast %v : memref<128xf16> to !tt.tensordesc<128xf16>
  tt.spyre_tensor_layout %d {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : <128xf16>
  %offx = arith.index_cast %off : i32 to index
  %lt = ktdp.construct_access_tile %v[%offx] {access_tile_order = #id, access_tile_set = #sblock} : memref<128xf16> -> !ktdp.access_tile<64xindex>
  %l = ktdp.load %lt : <64xindex> -> tensor<64xf16>
  %st = ktdp.construct_access_tile %v[%offx] {access_tile_order = #id, access_tile_set = #sblock} : memref<128xf16> -> !ktdp.access_tile<64xindex>
  ktdp.store %l, %st : tensor<64xf16>, <64xindex>
  tt.return
}
}

// -----

// Case 4 -- arrives, unchanged: an unsigned widening feeds a signed division.
//
// A zero-extended negative i32 is a large positive i64, so `divsi` on the wide
// value and on a rebuilt narrow value disagree. The whole chain reaches the
// subscript in the width it was written in.

#id = affine_map<(d0) -> (d0)>
#sview = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
#sblock = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
// CHECK-LABEL:   tt.func @extui_into_signed_div_not_lifted(
// CHECK:           %[[NEG:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK:           %[[W:.*]] = arith.extui %[[NEG]] : i32 to i64
// CHECK:           %[[D:.*]] = arith.divsi %[[W]], %{{.*}} : i64
// CHECK:           %[[TR:.*]] = arith.trunci %[[D]] : i64 to i32
// CHECK:           %[[IDX:.*]] = arith.index_cast %[[TR]] : i32 to index
// No part of that chain is re-emitted in `index`.
// CHECK-NOT:       arith.muli %{{.*}} : index
// CHECK:           arith.divsi %[[IDX]], %{{.*}} : index
tt.func @extui_into_signed_div_not_lifted(%ptr: !tt.ptr<f16>) {
  %cneg = arith.constant -3 : i32
  %c64_i64 = arith.constant 64 : i64
  %pid = tt.get_program_id x : i32
  %neg = arith.muli %pid, %cneg : i32
  %wide = arith.extui %neg : i32 to i64
  %off64 = arith.divsi %wide, %c64_i64 : i64
  %off = arith.trunci %off64 : i64 to i32
  %bi = builtin.unrealized_conversion_cast %ptr : !tt.ptr<f16> to index
  %v = ktdp.construct_memory_view %bi, sizes: [128], strides: [1] {coordinate_set = #sview, memory_space = #ktdp.memory_space<global>} : memref<128xf16>
  %d = builtin.unrealized_conversion_cast %v : memref<128xf16> to !tt.tensordesc<128xf16>
  tt.spyre_tensor_layout %d {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : <128xf16>
  %offx = arith.index_cast %off : i32 to index
  %lt = ktdp.construct_access_tile %v[%offx] {access_tile_order = #id, access_tile_set = #sblock} : memref<128xf16> -> !ktdp.access_tile<64xindex>
  %l = ktdp.load %lt : <64xindex> -> tensor<64xf16>
  %st = ktdp.construct_access_tile %v[%offx] {access_tile_order = #id, access_tile_set = #sblock} : memref<128xf16> -> !ktdp.access_tile<64xindex>
  ktdp.store %l, %st : tensor<64xf16>, <64xindex>
  tt.return
}
}
