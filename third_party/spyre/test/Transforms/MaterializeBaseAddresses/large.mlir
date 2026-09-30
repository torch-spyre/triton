// RUN: spyre-triton-opt %s --materialize-base-addresses='base-addresses=8589934592' | FileCheck %s

// --materialize-base-addresses with a single large address, pinning the
// int64_t carrier across the ListOption / pybind boundary. Split out from
// materialize-base-addresses.mlir because a pass option is fixed for the whole
// spyre-triton-opt invocation; see that file's header for the full case map.

// 8589934592 = 2^33 exceeds int32_t. It must
// appear intact in the emitted arith.constant: a 32-bit carrier anywhere on the
// path would truncate it silently to 0 and still produce IR that verifies.
//
// Note the operand is an element index, not a byte address (see the pass
// header), so this is 2^33 elements. Either way it is past 2^31 and
// exercises the same carrier, which is all this case is pinning.

// CHECK-LABEL:   func.func @large_address() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 8589934592 : index
// CHECK:           %[[VAL_1:.*]] = ktdp.construct_memory_view %[[VAL_0]], sizes: [4], strides: [1] {coordinate_set = #{{.*}}, memory_space = #ktdp.memory_space<global>} : memref<4xf16>
// CHECK:           return
// CHECK:         }
#set1d = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>
func.func @large_address(%a: index) {
  %va = ktdp.construct_memory_view %a, sizes: [4], strides: [1] {
      coordinate_set = #set1d, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  return
}
