// RUN: spyre-triton-opt %s --materialize-base-addresses='base-addresses=1024,12288' -split-input-file | FileCheck %s

// --materialize-base-addresses with a TWO-address list. Split out from
// materialize-base-addresses.mlir because a pass option is fixed for the whole
// spyre-triton-opt invocation; see that file's header for the full case map.
//
// Two cases live here: the interleaved signature (the load-bearing one) and a
// list shorter than the `index`-argument count.

// -----

// Interleaved signature — the case this design is most exposed to.
// `(%a: index, %n: i32, %b: index)` with two addresses leaves `(%n: i32)`,
// with %a → 1024 and %b → 12288.
//
// Two things fail here if the implementation is wrong. Indexing the address
// list over *all* arguments instead of over `index` arguments would try to give
// %n (an i32 runtime scalar) address 12288 and leave the real base address %b
// live. And conflating the list ordinal i with the signature position pos would
// mis-target %b, which is list entry 1 but signature position 2.

// CHECK-LABEL:   func.func @interleaved(
// CHECK-SAME:      %[[N:.*]]: i32) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1024 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 12288 : index
// CHECK:           %[[VAL_2:.*]] = ktdp.construct_memory_view %[[VAL_0]], sizes: [4], strides: [1] {coordinate_set = #{{.*}}, memory_space = #ktdp.memory_space<global>} : memref<4xf16>
// CHECK:           %[[VAL_3:.*]] = ktdp.construct_memory_view %[[VAL_1]], sizes: [4], strides: [1] {coordinate_set = #{{.*}}, memory_space = #ktdp.memory_space<global>} : memref<4xf16>
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[N]], %[[N]] : i32
// CHECK:           return
// CHECK:         }
#set1d = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>
func.func @interleaved(%a: index, %n: i32, %b: index) {
  %va = ktdp.construct_memory_view %a, sizes: [4], strides: [1] {
      coordinate_set = #set1d, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  %vb = ktdp.construct_memory_view %b, sizes: [4], strides: [1] {
      coordinate_set = #set1d, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  %w = arith.addi %n, %n : i32
  return
}

// -----

// A short list materializes a prefix. Three `index` args, two addresses → two
// constants and one surviving `index` argument (the third). Shorter than the
// `index`-arg count is legal, and useful for incremental bring-up; only a
// longer list is an error.

// CHECK-LABEL:   func.func @short_list(
// CHECK-SAME:      %[[C:.*]]: index) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1024 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 12288 : index
// CHECK:           %[[VAL_2:.*]] = ktdp.construct_memory_view %[[VAL_0]], sizes: [4], strides: [1] {coordinate_set = #{{.*}}, memory_space = #ktdp.memory_space<global>} : memref<4xf16>
// CHECK:           %[[VAL_3:.*]] = ktdp.construct_memory_view %[[VAL_1]], sizes: [4], strides: [1] {coordinate_set = #{{.*}}, memory_space = #ktdp.memory_space<global>} : memref<4xf16>
// CHECK:           %[[VAL_4:.*]] = ktdp.construct_memory_view %[[C]], sizes: [4], strides: [1] {coordinate_set = #{{.*}}, memory_space = #ktdp.memory_space<global>} : memref<4xf16>
// CHECK:           return
// CHECK:         }
#set1db = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>
func.func @short_list(%a: index, %b: index, %c: index) {
  %va = ktdp.construct_memory_view %a, sizes: [4], strides: [1] {
      coordinate_set = #set1db, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  %vb = ktdp.construct_memory_view %b, sizes: [4], strides: [1] {
      coordinate_set = #set1db, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  %vc = ktdp.construct_memory_view %c, sizes: [4], strides: [1] {
      coordinate_set = #set1db, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  return
}
