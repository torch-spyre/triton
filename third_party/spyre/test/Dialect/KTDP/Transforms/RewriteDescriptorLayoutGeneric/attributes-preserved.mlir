// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s

// Attributes this pass does not own must survive physicalization.
//
// Physicalizing an op is a clone-and-retype, not a rebuild: the original op is
// cloned and only the shape it states is replaced. So every field the pass never
// enumerated rides along, including one added to either op after this pass was
// written. The one op the pass cannot clone -- the indirect access tile, whose
// region's block arguments are the variable space it is changing the arity of --
// has to carry those fields across by hand instead, and case 2 holds it to the
// same standard.
//
// That is stated as a regression test because the alternative was tried and
// reported as a bug. RewriteDescriptorLayout, the named-op pass, builds each
// physical op fresh from an enumerated subset of the original's fields, so a
// field nobody listed is dropped with no diagnostic anywhere -- running it on
// this same input drops both attributes below. The two attributes are invented
// for the test precisely because neither pass has any reason to know them.
//
// THE ONE EXCEPTION, and it is not a hole in the claim. `tts.tensor_layout` is
// dropped from the physical view on purpose: it is the instruction this pass
// consumes, not a property of the data, so a physical view carrying it would be
// an instruction to physicalize something already physical, and the second run
// in no-op.mlir would do exactly that. The claim above is about attributes the
// pass DOES NOT OWN; the layout is the one it owns, and it owns it by deleting
// rather than by recomputing. isShapeOwnedAttr is where that is declared, which
// is what lets the pass's own verifyAttributesCarried permit the drop -- so the
// exception is stated in the code and not smuggled past the check. Its absence
// from the physical views below is asserted by the CHECK lines naming their
// attribute dictionaries in full, and by the CHECK-NOT in subscripts-indirect.mlir.
//
// Captures are hand-named and this file is hand-maintained: do not regenerate it
// with generate-test-checks.py. The generated lines would still match if the
// attributes were dropped from only one of the two ops, which is why both are
// spelled out in full below.

// Case 1 -- the two cloned ops: a memory view and a direct access tile.

// CHECK: #[[$ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$SET3:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#id = affine_map<(d0, d1) -> (d0, d1)>
#s2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// The two assertions: spyre.provenance on the physical construct_memory_view and
// spyre.tile_note on the physical construct_access_tile, each named in full on the
// op's own line below.
// CHECK-LABEL:   tt.func @unowned_attributes_survive(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>, %[[VAL_2:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_5:.*]] = ktdp.construct_memory_view %[[VAL_4]], sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$SET3]], memory_space = #ktdp.memory_space<global>, spyre.provenance = "kept"} : memref<2x64x64xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_7:.*]] = arith.divsi %[[VAL_3]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_9:.*]] = arith.remsi %[[VAL_3]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = ktdp.construct_access_tile %[[VAL_5]]{{\[}}%[[VAL_7]], %[[VAL_3]], %[[VAL_9]]] {access_tile_order = #[[$ID3]], access_tile_set = #[[$SET3]], spyre.tile_note = 7 : i64} : memref<2x64x64xf32> -> !ktdp.access_tile<2x64x64xindex>
// CHECK:           %[[VAL_11:.*]] = ktdp.load %[[VAL_10]] : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:           %[[VAL_12:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_13:.*]] = ktdp.construct_memory_view %[[VAL_12]], sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$SET3]], memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
// CHECK:           %[[VAL_14:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_15:.*]] = arith.divsi %[[VAL_3]], %[[VAL_14]] : index
// CHECK:           %[[VAL_16:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_17:.*]] = arith.remsi %[[VAL_3]], %[[VAL_16]] : index
// CHECK:           %[[VAL_18:.*]] = ktdp.construct_access_tile %[[VAL_13]]{{\[}}%[[VAL_15]], %[[VAL_3]], %[[VAL_17]]] {access_tile_order = #[[$ID3]], access_tile_set = #[[$SET3]]} : memref<2x64x64xf32> -> !ktdp.access_tile<2x64x64xindex>
// CHECK:           %[[VAL_19:.*]] = ktdp.load %[[VAL_18]] : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:           %[[VAL_20:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_21:.*]] = ktdp.construct_memory_view %[[VAL_20]], sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$SET3]], memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
// CHECK:           %[[VAL_22:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_23:.*]] = arith.divsi %[[VAL_3]], %[[VAL_22]] : index
// CHECK:           %[[VAL_24:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_25:.*]] = arith.remsi %[[VAL_3]], %[[VAL_24]] : index
// CHECK:           %[[VAL_26:.*]] = ktdp.construct_access_tile %[[VAL_21]]{{\[}}%[[VAL_23]], %[[VAL_3]], %[[VAL_25]]] {access_tile_order = #[[$ID3]], access_tile_set = #[[$SET3]]} : memref<2x64x64xf32> -> !ktdp.access_tile<2x64x64xindex>
// CHECK:           %[[VAL_27:.*]] = tensor.empty() : tensor<2x64x64xf32>
// CHECK:           %[[VAL_28:.*]] = linalg.generic {indexing_maps = [#[[$ID3]], #[[$ID3]], #[[$ID3]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_11]], %[[VAL_19]] : tensor<2x64x64xf32>, tensor<2x64x64xf32>) outs(%[[VAL_27]] : tensor<2x64x64xf32>) {
// CHECK:           ^bb0(%[[VAL_29:.*]]: f32, %[[VAL_30:.*]]: f32, %[[VAL_31:.*]]: f32):
// CHECK:             %[[VAL_32:.*]] = arith.addf %[[VAL_29]], %[[VAL_30]] : f32
// CHECK:             linalg.yield %[[VAL_32]] : f32
// CHECK:           } -> tensor<2x64x64xf32>
// CHECK:           ktdp.store %[[VAL_28]], %[[VAL_26]] : tensor<2x64x64xf32>, <2x64x64xindex>
// CHECK:           tt.return
// CHECK:         }
tt.func @unowned_attributes_survive(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s2, memory_space = #ktdp.memory_space<global>, spyre.provenance = "kept",
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s2, spyre.tile_note = 7 : i64} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>
  %bi = builtin.unrealized_conversion_cast %b : !tt.ptr<f32> to index
  %bv = ktdp.construct_memory_view %bi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s2, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %bt = ktdp.construct_access_tile %bv[%c0, %c0] {access_tile_order = #id, access_tile_set = #s2} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %bl = ktdp.load %bt : <64x128xindex> -> tensor<64x128xf32>
  %ci = builtin.unrealized_conversion_cast %c : !tt.ptr<f32> to index
  %cv = ktdp.construct_memory_view %ci, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s2, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %ct = ktdp.construct_access_tile %cv[%c0, %c0] {access_tile_order = #id, access_tile_set = #s2} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %e = tensor.empty() : tensor<64x128xf32>
  %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%al, %bl : tensor<64x128xf32>, tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  ktdp.store %r, %ct : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}

// -----

// Case 2 -- the op that cannot be cloned: an indirect access tile.
//
// Physicalizing this one changes how many intermediate variables its variable
// space has, and those are its region's block arguments, so it is built fresh
// rather than cloned and retyped. Nothing the builder takes as a parameter can
// therefore be an attribute the pass has never heard of, and spyre.gather_note
// below is exactly that -- carried across explicitly, and checked here because
// the explicit step is what a later edit can forget.

#varorder = affine_map<(d0, d1) -> (d0, d1)>
#sidx = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#sdata = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#stile = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// The assertion: spyre.gather_note on the physical construct_indirect_access_tile,
// named in full on the op's own line below.
// CHECK-LABEL:   tt.func @unowned_attribute_survives_on_indirect_tile(
// CHECK-SAME:      %[[DATA:.*]]: !tt.ptr<f32>, %[[IDX:.*]]: !tt.ptr<i32>) {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[IDXV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [32], strides: [1]
// CHECK-SAME:        : memref<32xi32>
// CHECK:           %[[DATAV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [2, 512, 64], strides: [32768, 64, 1]
// CHECK-SAME:        : memref<2x512x64xf32>
// CHECK:           %[[TILE:.*]] = ktdp.construct_indirect_access_tile intermediate_variables(%[[V0:.*]], %[[V1:.*]], %[[V2:.*]]) %[[DATAV]]{{\[}}((%[[C0]] + %[[V0]] * 64 + %[[V2]]) floordiv 64), ind(%[[IDXV]]{{\[}}%[[C0]] + %[[V1]]]), ((%[[C0]] + %[[V0]] * 64 + %[[V2]]) mod 64)] {spyre.gather_note = "kept", variables_space_order = #{{.*}}, variables_space_set = #{{.*}}} : memref<2x512x64xf32>, memref<32xi32> -> !ktdp.access_tile<2x32x64xindex>
// CHECK:           ktdp.load %[[TILE]] : <2x32x64xindex> -> tensor<2x32x64xf32>
tt.func @unowned_attribute_survives_on_indirect_tile(%data: !tt.ptr<f32>, %idx: !tt.ptr<i32>) {
  %c0 = arith.constant 0 : index
  %ii = builtin.unrealized_conversion_cast %idx : !tt.ptr<i32> to index
  %iv = ktdp.construct_memory_view %ii, sizes: [32], strides: [1] {coordinate_set = #sidx, memory_space = #ktdp.memory_space<global>} : memref<32xi32>
  %di = builtin.unrealized_conversion_cast %data : !tt.ptr<f32> to index
  %dv = ktdp.construct_memory_view %di, sizes: [512, 128], strides: [128, 1] {coordinate_set = #sdata, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<512x128xf32>
  %dt = ktdp.construct_indirect_access_tile intermediate_variables(%v0, %v1) %dv[ind(%iv[%c0 + %v0]), (%c0 + %v1)] {variables_space_order = #varorder, variables_space_set = #stile, spyre.gather_note = "kept"} : memref<512x128xf32>, memref<32xi32> -> !ktdp.access_tile<32x128xindex>
  %dl = ktdp.load %dt : <32x128xindex> -> tensor<32x128xf32>
  tt.return
}
}

