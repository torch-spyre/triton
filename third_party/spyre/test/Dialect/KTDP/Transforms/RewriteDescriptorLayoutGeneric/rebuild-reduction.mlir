// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s
// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s --check-prefix=NORANK

// A reduction is rebuilt by the same rule as anything else, and the iterator
// kinds follow the dims.
//
// Nothing in the rewrite knows what a reduction is. The generic's own
// iterator_types say which loop dims are reductions, and the rebuild inherits
// them: a stick-split logical dim contributes two loop dims, and those two
// inherit the kind the one dim had. So splitting the REDUCED dim gives two
// reduction loop dims -- which is what an scf.for over sticks would have been,
// expressed where the scheduler can still see it as one op -- while splitting
// some other dim leaves the reduction alone and the stick structure alone. A
// "loop" here is a dim of the generic's iteration space, never an scf.for; the
// pass file's vocabulary note spells that out. The neutral-element
// linalg.fill that LowerComputeOps puts on every reduction is an ordinary producer
// on the chain, so it follows the reduce to physical shape along with the
// tensor.empty underneath it.
//
// Four cases: the reduction off the stick axis, the reduction on it (where the
// output has to be re-stuck by a broadcast because reducing the split dim
// destroys the stick structure), that same re-stuck output with the broadcast
// axis FIRST in its physical order rather than last, and a three-generic chain
// that composes the first two -- a reduce whose broadcast statistic is read back
// by a rank-3 elementwise which feeds another. rebuild-contraction.mlir is this
// same rule with a second input.
//
// Captures are hand-named and this file is hand-maintained: do not regenerate it
// with generate-test-checks.py, which emits only positive CHECKs and would drop
// the NORANK block and the CHECK-NOT in case 3.

// Case 1 -- the reduction is OFF the stick axis.
//
// The reduced logical dim is not the split one, so removing it leaves the stick
// structure untouched: the split dim contributes its two parallel loops as in any
// other case, and the reduced dim is simply absent from the output map.

// CHECK: #[[$OFF_ID4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$OFF_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$OFF_OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[$OFF_SET4:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$OFF_SET3:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>

#in  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#out = affine_map<(d0, d1, d2) -> (d0, d2)>
#sin  = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 127 >= 0)>
#sout = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   tt.func @red_off(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_4:.*]] = ktdp.construct_memory_view %[[VAL_3]], sizes: [2, 2, 64, 64], strides: [8192, 4096, 64, 1] {coordinate_set = #[[$OFF_SET4]], memory_space = #ktdp.memory_space<global>} : memref<2x2x64x64xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_6:.*]] = arith.divsi %[[VAL_2]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_8:.*]] = arith.remsi %[[VAL_2]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = ktdp.construct_access_tile %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_2]], %[[VAL_2]], %[[VAL_8]]] {access_tile_order = #[[$OFF_ID4]], access_tile_set = #[[$OFF_SET4]]} : memref<2x2x64x64xf32> -> !ktdp.access_tile<2x2x64x64xindex>
// CHECK:           %[[VAL_10:.*]] = ktdp.load %[[VAL_9]] : <2x2x64x64xindex> -> tensor<2x2x64x64xf32>
// CHECK:           %[[VAL_11:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_12:.*]] = ktdp.construct_memory_view %[[VAL_11]], sizes: [2, 2, 64], strides: [128, 64, 1] {coordinate_set = #[[$OFF_SET3]], memory_space = #ktdp.memory_space<global>} : memref<2x2x64xf32>
// CHECK:           %[[VAL_13:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_14:.*]] = arith.divsi %[[VAL_2]], %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_16:.*]] = arith.remsi %[[VAL_2]], %[[VAL_15]] : index
// CHECK:           %[[VAL_17:.*]] = ktdp.construct_access_tile %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_2]], %[[VAL_16]]] {access_tile_order = #[[$OFF_ID3]], access_tile_set = #[[$OFF_SET3]]} : memref<2x2x64xf32> -> !ktdp.access_tile<2x2x64xindex>
// The linalg.fill init and its tensor.empty, both at physical shape.
// CHECK:           %[[VAL_18:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_19:.*]] = tensor.empty() : tensor<2x2x64xf32>
// CHECK:           %[[VAL_20:.*]] = linalg.fill ins(%[[VAL_18]] : f32) outs(%[[VAL_19]] : tensor<2x2x64xf32>) -> tensor<2x2x64xf32>
// One reduction loop still, and the split dim's two loops both parallel.
// CHECK:           %[[VAL_21:.*]] = linalg.generic {indexing_maps = [#[[$OFF_ID4]], #[[$OFF_OUT]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%[[VAL_10]] : tensor<2x2x64x64xf32>) outs(%[[VAL_20]] : tensor<2x2x64xf32>) {
// CHECK:           ^bb0(%[[VAL_22:.*]]: f32, %[[VAL_23:.*]]: f32):
// CHECK:             %[[VAL_24:.*]] = arith.addf %[[VAL_22]], %[[VAL_23]] : f32
// CHECK:             linalg.yield %[[VAL_24]] : f32
// CHECK:           } -> tensor<2x2x64xf32>
// CHECK:           ktdp.store %[[VAL_21]], %[[VAL_17]] : tensor<2x2x64xf32>, <2x2x64xindex>
// CHECK:           tt.return
// CHECK:         }
tt.func @red_off(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [2, 64, 128], strides: [8192, 128, 1] {coordinate_set = #sin, memory_space = #ktdp.memory_space<global>} : memref<2x64x128xf32>
  %ad = builtin.unrealized_conversion_cast %av : memref<2x64x128xf32> to !tt.tensordesc<2x64x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 2, 0, 1, 2>, phys_op = array<i64: 1, 0, 0, 2>, phys_arg = array<i64: 64, 0, 0, 64>} : <2x64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0, %c0] {access_tile_order = #id3, access_tile_set = #sin} : memref<2x64x128xf32> -> !ktdp.access_tile<2x64x128xindex>
  %al = ktdp.load %at : <2x64x128xindex> -> tensor<2x64x128xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [2, 128], strides: [128, 1] {coordinate_set = #sout, memory_space = #ktdp.memory_space<global>} : memref<2x128xf32>
  %od = builtin.unrealized_conversion_cast %ov : memref<2x128xf32> to !tt.tensordesc<2x128xf32>
  tt.spyre_tensor_layout %od {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <2x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sout} : memref<2x128xf32> -> !ktdp.access_tile<2x128xindex>
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<2x128xf32>
  %e = linalg.fill ins(%zero : f32) outs(%e0 : tensor<2x128xf32>) -> tensor<2x128xf32>
  %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction", "parallel"]} ins(%al : tensor<2x64x128xf32>) outs(%e : tensor<2x128xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %s = arith.addf %x, %acc : f32
    linalg.yield %s : f32
  } -> tensor<2x128xf32>
  ktdp.store %r, %ot : tensor<2x128xf32>, <2x128xindex>
  tt.return
}
}

// -----

// Case 2 -- the reduction is ON the stick axis, and the output is re-stuck.
//
// Four loop dims: d0 = row (parallel), d1/d2 = the split K's stick and lane
// halves (both reduction), d3 = the broadcast lane axis. d3 is absent from the
// input map and present in the output map -- that asymmetry IS the case.
//
// The reduce runs over logical 256x128 reducing dim 1, so its logical result is
// rank 1: tensor<256xf32>. Reducing away the split dim destroys the stick
// structure, so the output descriptor's marker rebuilds one -- phys_op
// [identity, broadcast] over phys_src [0, 0] replicates logical dim 0 across 64
// lanes, giving physical 256x64.
//
// The broadcast is the one physical dim no logical loop dim accounts for, so the
// domain gives it a loop of its own, after the refinement. That loop therefore
// appears in the OUTPUT map only, which is what makes it parallel and what lets
// the rank change live entirely in the maps -- no extract_slice, no expand_shape,
// no leading size-1 dim.
//
// The output is rank 2, not rank 3. Giving the reduce a rank-2 logical output and
// stick-splitting it produced a spurious leading size-1 dim; a rank-1 logical
// output re-stuck by a broadcast cannot, and the NORANK block below is what pins
// that. It has its own prefix so the CHECK-NOT covers the whole function rather
// than the gap between two positive checks.

// NORANK-LABEL: tt.func @red_on_restick(
// NORANK-NOT:     1x256x64
// NORANK:         tt.return

// CHECK: #[[$ON_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ON_ID2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ON_IN:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: #[[$ON_OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK: #[[$ON_SET3:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ON_SET2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>

#in  = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
#sin  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#sout = affine_set<(d0) : (d0 >= 0, -d0 + 255 >= 0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#id1 = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   tt.func @red_on_restick(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_4:.*]] = ktdp.construct_memory_view %[[VAL_3]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ON_SET3]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_6:.*]] = arith.divsi %[[VAL_2]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_8:.*]] = arith.remsi %[[VAL_2]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = ktdp.construct_access_tile %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_2]], %[[VAL_8]]] {access_tile_order = #[[$ON_ID3]], access_tile_set = #[[$ON_SET3]]} : memref<2x256x64xf32> -> !ktdp.access_tile<2x256x64xindex>
// CHECK:           %[[VAL_10:.*]] = ktdp.load %[[VAL_9]] : <2x256x64xindex> -> tensor<2x256x64xf32>
// CHECK:           %[[VAL_11:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_12:.*]] = ktdp.construct_memory_view %[[VAL_11]], sizes: [256, 64], strides: [64, 1] {coordinate_set = #[[$ON_SET2]], memory_space = #ktdp.memory_space<global>} : memref<256x64xf32>
// The broadcast dim's subscript is the axis's own origin, a constant 0.
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_14:.*]] = ktdp.construct_access_tile %[[VAL_12]]{{\[}}%[[VAL_2]], %[[VAL_13]]] {access_tile_order = #[[$ON_ID2]], access_tile_set = #[[$ON_SET2]]} : memref<256x64xf32> -> !ktdp.access_tile<256x64xindex>
// CHECK:           %[[VAL_15:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_16:.*]] = tensor.empty() : tensor<256x64xf32>
// CHECK:           %[[VAL_17:.*]] = linalg.fill ins(%[[VAL_15]] : f32) outs(%[[VAL_16]] : tensor<256x64xf32>) -> tensor<256x64xf32>
// Two reduction loop dims for the split K, and the broadcast one is parallel.
// CHECK:           %[[VAL_18:.*]] = linalg.generic {indexing_maps = [#[[$ON_IN]], #[[$ON_OUT]]], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%[[VAL_10]] : tensor<2x256x64xf32>) outs(%[[VAL_17]] : tensor<256x64xf32>) {
// CHECK:           ^bb0(%[[VAL_19:.*]]: f32, %[[VAL_20:.*]]: f32):
// CHECK:             %[[VAL_21:.*]] = arith.addf %[[VAL_19]], %[[VAL_20]] : f32
// CHECK:             linalg.yield %[[VAL_21]] : f32
// CHECK:           } -> tensor<256x64xf32>
// CHECK:           ktdp.store %[[VAL_18]], %[[VAL_14]] : tensor<256x64xf32>, <256x64xindex>
// CHECK:           tt.return
// CHECK:         }
tt.func @red_on_restick(%a: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [256, 128], strides: [128, 1] {coordinate_set = #sin, memory_space = #ktdp.memory_space<global>} : memref<256x128xf32>
  %ad = builtin.unrealized_conversion_cast %av : memref<256x128xf32> to !tt.tensordesc<256x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <256x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sin} : memref<256x128xf32> -> !ktdp.access_tile<256x128xindex>
  %al = ktdp.load %at : <256x128xindex> -> tensor<256x128xf32>

  %ci = builtin.unrealized_conversion_cast %c : !tt.ptr<f32> to index
  %cv = ktdp.construct_memory_view %ci, sizes: [256], strides: [1] {coordinate_set = #sout, memory_space = #ktdp.memory_space<global>} : memref<256xf32>
  %cd = builtin.unrealized_conversion_cast %cv : memref<256xf32> to !tt.tensordesc<256xf32>
  tt.spyre_tensor_layout %cd {phys_src = array<i64: 0, 0>, phys_op = array<i64: 0, 3>, phys_arg = array<i64: 0, 64>} : <256xf32>
  %ct = ktdp.construct_access_tile %cv[%c0] {access_tile_order = #id1, access_tile_set = #sout} : memref<256xf32> -> !ktdp.access_tile<256xindex>

  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<256xf32>
  %e = linalg.fill ins(%zero : f32) outs(%e0 : tensor<256xf32>) -> tensor<256xf32>
  %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction"]} ins(%al : tensor<256x128xf32>) outs(%e : tensor<256xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %s = arith.addf %x, %acc : f32
    linalg.yield %s : f32
  } -> tensor<256xf32>
  ktdp.store %r, %ct : tensor<256xf32>, <256xindex>
  tt.return
}
}

// -----

// Case 3 -- case 2 with the broadcast axis FIRST in the output's physical order.
//
// One field differs from case 2: the output marker's phys_op is [broadcast,
// identity] over phys_src [0, 0] rather than [identity, broadcast], so the
// replication axis is physical dim 0 and the logical dim rides behind it --
// physical 64x256, not 256x64.
//
// The loop domain does not see that field. A broadcast names no domain piece, so
// it takes no part in the ordering and gets its loop AFTER the pieces are
// numbered -- d3, the last loop dim, whatever position the broadcast occupies in
// the operand's physical order. The output map therefore names d3 before d1 and
// is NOT monotone: (d0, d1, d2, d3) -> (d3, d1). That is correct and it verifies.
// A projected permutation need not take the loop dims in increasing order, and
// each of the output's physical dims still names the loop carrying the half it
// holds -- lane first, row second, exactly as its own physical type lays them
// out. Case 2, where the broadcast is last, is the same emission with the two
// results the other way round.
//
// The one thing to hold on to: every OTHER positive case in this directory has a
// monotone output map, and this is the only shape that breaks that -- so a
// monotonicity claim about the output map has to exclude the broadcast axis. The
// invariant asserted in the pass is stated that way.

// CHECK: #[[$BF_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$BF_ID2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$BF_IN:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: #[[$BF_OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d1)>
// CHECK: #[[$BF_SET3:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$BF_SET2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 255 >= 0)>

#in  = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
#sin  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#sout = affine_set<(d0) : (d0 >= 0, -d0 + 255 >= 0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#id1 = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   tt.func @red_on_restick_broadcast_first(
// The input is unchanged from case 2: stick-on-K, two sticks.
// CHECK:           ktdp.construct_memory_view %{{.*}}, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$BF_SET3]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf32>
// CHECK:           %[[AL:.*]] = ktdp.load %{{.*}} : <2x256x64xindex> -> tensor<2x256x64xf32>
// The output's physical shape is the marker's order: the 64 lanes first.
// CHECK:           %[[OV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [64, 256], strides: [256, 1] {coordinate_set = #[[$BF_SET2]], memory_space = #ktdp.memory_space<global>} : memref<64x256xf32>
// The broadcast dim's subscript is the axis's own origin, and it comes first.
// CHECK:           %[[BZERO:.*]] = arith.constant 0 : index
// CHECK:           %[[OT:.*]] = ktdp.construct_access_tile %[[OV]]{{\[}}%[[BZERO]], %{{.*}}] {access_tile_order = #[[$BF_ID2]], access_tile_set = #[[$BF_SET2]]} : memref<64x256xf32> -> !ktdp.access_tile<64x256xindex>
// CHECK:           %[[FILL:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<64x256xf32>) -> tensor<64x256xf32>
// The non-monotone output map, and the iterators unchanged from case 2.
// CHECK:           %[[R:.*]] = linalg.generic {indexing_maps = [#[[$BF_IN]], #[[$BF_OUT]]], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%[[AL]] : tensor<2x256x64xf32>) outs(%[[FILL]] : tensor<64x256xf32>) {
// CHECK:           } -> tensor<64x256xf32>
// CHECK:           ktdp.store %[[R]], %[[OT]] : tensor<64x256xf32>, <64x256xindex>
// CHECK:           tt.return
tt.func @red_on_restick_broadcast_first(%a: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [256, 128], strides: [128, 1] {coordinate_set = #sin, memory_space = #ktdp.memory_space<global>} : memref<256x128xf32>
  %ad = builtin.unrealized_conversion_cast %av : memref<256x128xf32> to !tt.tensordesc<256x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <256x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sin} : memref<256x128xf32> -> !ktdp.access_tile<256x128xindex>
  %al = ktdp.load %at : <256x128xindex> -> tensor<256x128xf32>

  %ci = builtin.unrealized_conversion_cast %c : !tt.ptr<f32> to index
  %cv = ktdp.construct_memory_view %ci, sizes: [256], strides: [1] {coordinate_set = #sout, memory_space = #ktdp.memory_space<global>} : memref<256xf32>
  %cd = builtin.unrealized_conversion_cast %cv : memref<256xf32> to !tt.tensordesc<256xf32>
  // The one field that differs from case 2: broadcast first, identity second.
  tt.spyre_tensor_layout %cd {phys_src = array<i64: 0, 0>, phys_op = array<i64: 3, 0>, phys_arg = array<i64: 64, 0>} : <256xf32>
  %ct = ktdp.construct_access_tile %cv[%c0] {access_tile_order = #id1, access_tile_set = #sout} : memref<256xf32> -> !ktdp.access_tile<256xindex>

  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<256xf32>
  %e = linalg.fill ins(%zero : f32) outs(%e0 : tensor<256xf32>) -> tensor<256xf32>
  %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction"]} ins(%al : tensor<256x128xf32>) outs(%e : tensor<256xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %s = arith.addf %x, %acc : f32
    linalg.yield %s : f32
  } -> tensor<256xf32>
  ktdp.store %r, %ct : tensor<256xf32>, <256xindex>
  tt.return
}
}

// -----

// Case 4 -- cases 1 and 2 composed: a reduce whose broadcast statistic is read
// back by a rank-3 elementwise, which feeds another.
//
// The only three-generic chain in this directory, and the only place a rank-2
// statistic PRODUCED BY a reduce in the same kernel is consumed at rank 3.
// rebuild-passthrough.mlir mixes those ranks too, but its statistic is an
// unannotated function argument with nothing upstream of it; cases 1 and 2 above
// are single generics. What this case adds is that the Phase 2 worklist
// collectAdjacentGenerics builds is three long and the ranks along it differ, so
// a layout belonging to one link cannot be read at the next: G3 is elementwise at
// physical rank 3, and a rank-2 layout reaching it would make it inconsistent
// rather than merely wrong. The symptom that shape once produced was a domain one
// loop too wide with the extra loop named by no map, reported several passes away
// by the linalg verifier as
//
//   'linalg.generic' op invalid indexing maps are non-invertible:
//   ((d0, d1, d2, d3) -> (d0, d1, d2, d0, d1, d2))
//
// Input is the already-lowered form the pass sees, so it does not depend on what
// the frontend happens to emit. It is `max_shift_exp_on_stick` in
// fixtures/reduce/kernel.py: out = exp(x - max(x, axis=1)), fp32, 32-lane sticks.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 >= 0)>

// The four maps the output states, checked where they are printed -- at the top,
// ahead of the function.
// CHECK-DAG:   #[[$CHAIN_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$CHAIN_RIN:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$CHAIN_ROUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG:   #[[$CHAIN_LANE0:.+]] = affine_map<(d0, d1, d2) -> (d1, 0)>

// CHECK-LABEL: tt.func public @max_shift_exp_on_stick
// CHECK-NOT:   tt.spyre_tensor_layout

// G1: the max reduce. Its input is stick-on-N, so the reduced dim is split across
// two loops and the surviving dim rides as the third; the broadcast output adds a
// fourth loop that only its own map names. Four loops here is CORRECT -- it is
// case 2's shape.
//
// Its init is the neutral-element linalg.fill, and it follows the reduce to
// physical rank 2 along with the tensor.empty underneath it.
// CHECK:       %[[MAXINIT:.*]] = tensor.empty() : tensor<64x32xf32>
// CHECK:       %[[MAXFILL:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[MAXINIT]] : tensor<64x32xf32>) -> tensor<64x32xf32>
// CHECK:       linalg.generic {indexing_maps = [#[[$CHAIN_RIN:.+]], #[[$CHAIN_ROUT:.+]]], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%{{.*}} : tensor<2x64x32xf32>) outs(%[[MAXFILL]] : tensor<64x32xf32>)
// CHECK:         arith.maximumf
// CHECK:       ktdp.store %{{.*}} : tensor<64x32xf32>, <64x32xindex>

// G2: the shift. Three loops, the statistic read at a constant lane -- the rank
// mixing, at the middle of the chain.
// CHECK:       linalg.generic {indexing_maps = [#[[$CHAIN_ID3:.+]], #[[$CHAIN_LANE0:.+]], #[[$CHAIN_ID3]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%{{.*}}, %{{.*}} : tensor<2x64x32xf32>, tensor<64x1xf32>) outs(%{{.*}} : tensor<2x64x32xf32>)
// CHECK:         arith.subf
// CHECK:       ktdp.store %{{.*}} : tensor<2x64x32xf32>, <2x64x32xindex>

// G3: the exp. THREE parallel loops and two identity maps -- this is the one that
// came out with four loops and an unnamed d3 when a rank-2 layout reached it.
// CHECK:       linalg.generic {indexing_maps = [#[[$CHAIN_ID3]], #[[$CHAIN_ID3]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%{{.*}} : tensor<2x64x32xf32>) outs(%{{.*}} : tensor<2x64x32xf32>)
// CHECK:         math.exp
// CHECK:       ktdp.store %{{.*}} : tensor<2x64x32xf32>, <2x64x32xindex>

module {
  tt.func public @max_shift_exp_on_stick(%x: !tt.ptr<f32>, %max: !tt.ptr<f32>,
                                         %diff: !tt.ptr<f32>, %out: !tt.ptr<f32>)
      attributes {noinline = false} {
    %c0 = arith.constant 0 : index

    %x_i = builtin.unrealized_conversion_cast %x : !tt.ptr<f32> to index
    %x_view = ktdp.construct_memory_view %x_i, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %x_desc = builtin.unrealized_conversion_cast %x_view : memref<64x64xf32> to !tt.tensordesc<64x64xf32>

    %diff_i = builtin.unrealized_conversion_cast %diff : !tt.ptr<f32> to index
    %diff_view = ktdp.construct_memory_view %diff_i, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %diff_desc = builtin.unrealized_conversion_cast %diff_view : memref<64x64xf32> to !tt.tensordesc<64x64xf32>

    %out_i = builtin.unrealized_conversion_cast %out : !tt.ptr<f32> to index
    %out_view = ktdp.construct_memory_view %out_i, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %out_desc = builtin.unrealized_conversion_cast %out_view : memref<64x64xf32> to !tt.tensordesc<64x64xf32>

    // The statistic, written rank-1 through a broadcast layout...
    %max_i = builtin.unrealized_conversion_cast %max : !tt.ptr<f32> to index
    %max_view = ktdp.construct_memory_view %max_i, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf32>
    %max_w = builtin.unrealized_conversion_cast %max_view : memref<64xf32> to !tt.tensordesc<64xf32>
    // ...and read back as the [64, 32] that broadcast made, one lane wide. No
    // layout: its logical shape already is its physical one.
    %max_r_i = builtin.unrealized_conversion_cast %max : !tt.ptr<f32> to index
    %max_r_view = ktdp.construct_memory_view %max_r_i, sizes: [64, 32], strides: [32, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<64x32xf32>

    tt.spyre_tensor_layout %x_desc {phys_arg = array<i64: 32, 0, 32>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
    tt.spyre_tensor_layout %diff_desc {phys_arg = array<i64: 32, 0, 32>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
    tt.spyre_tensor_layout %out_desc {phys_arg = array<i64: 32, 0, 32>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
    tt.spyre_tensor_layout %max_w {phys_arg = array<i64: 0, 32>, phys_op = array<i64: 0, 3>, phys_src = array<i64: 0, 0>} : <64xf32>

    // G1: max(x, axis=1) stored stick-wide.
    %t0 = ktdp.construct_access_tile %x_view[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %x0 = ktdp.load %t0 : <64x64xindex> -> tensor<64x64xf32>
    %ninf = arith.constant 0xFF800000 : f32
    %e0empty = tensor.empty() : tensor<64xf32>
    %e0 = linalg.fill ins(%ninf : f32) outs(%e0empty : tensor<64xf32>) -> tensor<64xf32>
    %maxes = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%x0 : tensor<64x64xf32>) outs(%e0 : tensor<64xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %m = arith.maximumf %in, %acc : f32
      linalg.yield %m : f32
    } -> tensor<64xf32>
    %t1 = ktdp.construct_access_tile %max_view[%c0] {access_tile_order = #map2, access_tile_set = #set1} : memref<64xf32> -> !ktdp.access_tile<64xindex>
    ktdp.store %maxes, %t1 : tensor<64xf32>, <64xindex>

    // G2: x - max, the statistic read at lane 0.
    %t2 = ktdp.construct_access_tile %x_view[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %x1 = ktdp.load %t2 : <64x64xindex> -> tensor<64x64xf32>
    %t3 = ktdp.construct_access_tile %max_r_view[%c0, %c0] {access_tile_order = #map, access_tile_set = #set3} : memref<64x32xf32> -> !ktdp.access_tile<64x1xindex>
    %m0 = ktdp.load %t3 : <64x1xindex> -> tensor<64x1xf32>
    %e1 = tensor.empty() : tensor<64x64xf32>
    %shifted = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%x1, %m0 : tensor<64x64xf32>, tensor<64x1xf32>) outs(%e1 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_m: f32, %acc: f32):
      %d = arith.subf %in, %in_m : f32
      linalg.yield %d : f32
    } -> tensor<64x64xf32>
    %t4 = ktdp.construct_access_tile %diff_view[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %shifted, %t4 : tensor<64x64xf32>, <64x64xindex>

    // G3: exp of that.
    %t5 = ktdp.construct_access_tile %diff_view[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %d0 = ktdp.load %t5 : <64x64xindex> -> tensor<64x64xf32>
    %e2 = tensor.empty() : tensor<64x64xf32>
    %exps = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%d0 : tensor<64x64xf32>) outs(%e2 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %ex = math.exp %in : f32
      linalg.yield %ex : f32
    } -> tensor<64x64xf32>
    %t6 = ktdp.construct_access_tile %out_view[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %exps, %t6 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}
