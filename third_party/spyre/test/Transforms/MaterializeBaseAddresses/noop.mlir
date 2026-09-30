// RUN: spyre-triton-opt %s --materialize-base-addresses | FileCheck %s

// --materialize-base-addresses with no addresses supplied. The pass is opt-in:
// with an empty list it returns before touching anything, so the default
// argument-passing path stays byte-identical. Split out from
// materialize-base-addresses.mlir because a pass option is fixed for the whole
// spyre-triton-opt invocation; see that file's header for the full case map.

// Every `index` argument survives and no arith.constant is created. The
// CHECK-LABEL transcribes the full parameter list, so a dropped argument breaks
// the match; CHECK-NOT covers the op that must never be created, which a
// transcript of what *was* created cannot express.

// CHECK-LABEL:   func.func @noop(
// CHECK-SAME:      %[[A:.*]]: index,
// CHECK-SAME:      %[[N:.*]]: i32,
// CHECK-SAME:      %[[B:.*]]: index) {
// CHECK:           %[[VAL_0:.*]] = ktdp.construct_memory_view %[[A]], sizes: [4], strides: [1] {coordinate_set = #{{.*}}, memory_space = #ktdp.memory_space<global>} : memref<4xf16>
// CHECK:           %[[VAL_1:.*]] = ktdp.construct_memory_view %[[B]], sizes: [4], strides: [1] {coordinate_set = #{{.*}}, memory_space = #ktdp.memory_space<global>} : memref<4xf16>
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[N]], %[[N]] : i32
// CHECK:           return
// CHECK:         }
// CHECK-NOT:     arith.constant
#set1d = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>
func.func @noop(%a: index, %n: i32, %b: index) {
  %va = ktdp.construct_memory_view %a, sizes: [4], strides: [1] {
      coordinate_set = #set1d, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  %vb = ktdp.construct_memory_view %b, sizes: [4], strides: [1] {
      coordinate_set = #set1d, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  %w = arith.addi %n, %n : i32
  return
}
