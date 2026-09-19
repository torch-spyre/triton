// RUN: spyre-triton-opt %s --materialize-base-addresses='base-addresses=1024,-8' -verify-diagnostics

// Negative test for --materialize-base-addresses: the negative-address
// diagnostic. Needs its own file because the offending value is a pass option,
// fixed for the whole invocation; the count-mismatch case lives in
// materialize-base-addresses-invalid.mlir.
//
// Written by hand — utils/generate-test-checks.py has no generator for
// expected-error annotations.

// An address cannot be negative. int64_t is the carrier because realistic
// exceed int32_t, and ListOption cannot express unsignedness, so admitting
// negatives is a property of the carrier that the pass has to check itself —
// otherwise `arith.constant -8 : index` is emitted silently and verifies. The
// diagnostic names the list position and the value.

#set1d = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>
// expected-error @below {{MaterializeBaseAddresses: base address 1 is negative (-8); addresses must be >= 0}}
func.func @negative_address(%a: index, %b: index) {
  %va = ktdp.construct_memory_view %a, sizes: [4], strides: [1] {
      coordinate_set = #set1d, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  %vb = ktdp.construct_memory_view %b, sizes: [4], strides: [1] {
      coordinate_set = #set1d, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  return
}
