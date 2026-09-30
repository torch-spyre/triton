// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s

// CHARACTERIZATION. The maps below are pinned exactly, on purpose.
//
// Read this before deciding they are over-specified. buildLoopDomain numbers the
// rebuilt loop dims in a particular order, and the order exists to satisfy
// dbo-opt, the consumer several stages downstream. Its constraints are on the
// SHAPE of the emitted indexing maps -- which map may carry which expression, in
// which position -- and none of that is expressible in FileCheck, or anywhere
// else in this repo. So there is nothing here to assert the property against;
// what these tests can do is pin the maps the pass emits today, so that a
// refactor which changes them fails loudly instead of changing what dbo-opt is
// handed in silence.
//
// The set spans the branches of buildLoopDomain, which is what makes it a guard
// rather than a sample. Four of the five are already pinned elsewhere in this
// directory and are not duplicated here; the same rule above applies to those
// maps:
//
//   1. the result names every piece -- the merge finds everything and inserts
//      nothing: rebuild-passthrough.mlir case 1, rebuild-composite.mlir case 1.
//   2. an input names a piece the result does not -- the insert-at-cursor path:
//      rebuild-reduction.mlir case 1, rebuild-composite.mlir cases 2 and 3.
//   3. a second input whose pieces are already placed -- the cursor-advance
//      path, including a piece found BEHIND the cursor, which is where a
//      non-monotone input map comes from: rebuild-contraction.mlir cases 1
//      and 3.
//   5. a splat, whose loop is allocated after the ordering is fixed:
//      rebuild-reduction.mlir cases 2, 3 and 4 -- case 3 being the one where the
//      splat is not last in its operand's physical order, so the output map
//      comes out non-monotone.
//
// Branch 4 is the case below, which nothing else reached.
//
// A DECLINE LAYER. There is no negative case in this file, and there cannot be
// one while the rule constrains only the result: any set of inputs can be
// ordered somehow, because the result is seeded first and the inputs merely
// merge into it. An input for which NO ordering satisfies the rule only exists
// once the rule constrains inputs too -- "identity input map on a reduce" would
// do it, since two marked inputs demanding incompatible identities could then
// admit no valid order, and that is the point at which this pass needs a
// diagnostic rather than an assertion.
//
// LINEARIZATION. composeStickSplit emits `stick * width + lane` -- the
// expression affine.linearize_index computes -- on the happy path, in nearly
// every passing test in this directory. What is recorded elsewhere of dbo-opt's
// rules is that it REJECTS a linearizing map. Both cannot be true as stated, so
// either that rule is narrower than recorded (an output map only, say, or a map
// that is an op's only one) or there is a tension these tests are not catching.
// Neither invariant asserted in the pass mentions linearization, so this round
// does not resolve it; the question is recorded here so it is not lost.

// Branch 4 -- two inputs naming a shared pair of pieces in OPPOSITE relative
// order.
//
// Both inputs split the reduced dim at the same width and differ only in where
// the stick dim sits in their physical order: A is [K/64, M, lane] and B is
// [M, K/64, lane]. So A's walk names the M piece after the stick piece and B's
// names it before -- the one shape where the two inputs cannot both have a
// monotone map, since the domain has to pick one order for the pair.
//
// The output carries no layout, so it stays logical and pins nothing: it names
// only the M piece, and the pair's relative order is decided entirely by the
// inputs. The pass resolves it by first-merged-wins -- A, the earlier operand,
// gets the plain projected permutation and B gets the legal non-monotone map
// (d0, d1, d2) -> (d1, d0, d2). Nothing prefers A on the merits; it is merged
// first because it is operand 0.
//
// A projected permutation need not take the loop dims in increasing order, so
// B's map is correct as emitted and each of B's physical dims still names the
// loop carrying the half it holds. The result operand is the one whose map the
// ordering exists to keep monotone, and it is: (d0, d1, d2) -> (d1).

// CHECK: #[[$ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ID1:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$B_SWAPPED:.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK: #[[$OUT:.+]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK: #[[$SET_A:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$SET_B:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$SET_OUT:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

#ina = affine_map<(d0, d1) -> (d0, d1)>
#inb = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
#s2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#s1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#id1 = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   tt.func @two_inputs_opposite_order(
// A's physical order puts the stick dim first.
// CHECK:           ktdp.construct_memory_view %{{.*}}, sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$SET_A]], memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
// CHECK:           %[[AL:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// B's puts it second, over the same logical shape and the same width.
// CHECK:           ktdp.construct_memory_view %{{.*}}, sizes: [64, 2, 64], strides: [128, 64, 1] {coordinate_set = #[[$SET_B]], memory_space = #ktdp.memory_space<global>} : memref<64x2x64xf32>
// CHECK:           %[[BL:.*]] = ktdp.load %{{.*}} : <64x2x64xindex> -> tensor<64x2x64xf32>
// The unmarked output stays rank 1, and its tile with it.
// CHECK:           %[[OT:.*]] = ktdp.construct_access_tile %{{.*}}{{\[}}%{{.*}}] {access_tile_order = #[[$ID1]], access_tile_set = #[[$SET_OUT]]} : memref<64xf32> -> !ktdp.access_tile<64xindex>
// CHECK:           %[[FILL:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<64xf32>) -> tensor<64xf32>
// A gets the identity, B the swapped map, and the output map is monotone.
// CHECK:           %[[R:.*]] = linalg.generic {indexing_maps = [#[[$ID3]], #[[$B_SWAPPED]], #[[$OUT]]], iterator_types = ["reduction", "parallel", "reduction"]} ins(%[[AL]], %[[BL]] : tensor<2x64x64xf32>, tensor<64x2x64xf32>) outs(%[[FILL]] : tensor<64xf32>) {
// CHECK:           } -> tensor<64xf32>
// CHECK:           ktdp.store %[[R]], %[[OT]] : tensor<64xf32>, <64xindex>
// CHECK:           tt.return
tt.func @two_inputs_opposite_order(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  // Stick dim first.
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s2, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id2, access_tile_set = #s2} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>

  %bi = builtin.unrealized_conversion_cast %b : !tt.ptr<f32> to index
  // The one line that differs from A: stick dim second.
  %bv = ktdp.construct_memory_view %bi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s2, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 1, 2>, phys_arg = array<i64: 0, 64, 64>}} : memref<64x128xf32>
  %bt = ktdp.construct_access_tile %bv[%c0, %c0] {access_tile_order = #id2, access_tile_set = #s2} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %bl = ktdp.load %bt : <64x128xindex> -> tensor<64x128xf32>

  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [64], strides: [1] {coordinate_set = #s1, memory_space = #ktdp.memory_space<global>} : memref<64xf32>
  %ot = ktdp.construct_access_tile %ov[%c0] {access_tile_order = #id1, access_tile_set = #s1} : memref<64xf32> -> !ktdp.access_tile<64xindex>
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<64xf32>
  %e = linalg.fill ins(%zero : f32) outs(%e0 : tensor<64xf32>) -> tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#ina, #inb, #out], iterator_types = ["parallel", "reduction"]} ins(%al, %bl : tensor<64x128xf32>, tensor<64x128xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %m = arith.mulf %x, %y : f32
    %s = arith.addf %acc, %m : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  ktdp.store %r, %ot : tensor<64xf32>, <64xindex>
  tt.return
}
}
