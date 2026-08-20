// RUN: spyre-triton-opt %s --materialize-base-addresses='base-addresses=1024,12288,18432' -split-input-file -verify-diagnostics

// Negative test for --materialize-base-addresses: the count-mismatch
// diagnostic. Three addresses are supplied for every case in this file; the
// negative-address diagnostic needs a different list and lives in
// materialize-base-addresses-invalid-negative.mlir.
//
// Written by hand — utils/generate-test-checks.py has no generator for
// expected-error annotations.

// -----

// More addresses than the kernel has base-address arguments. There is no
// correct reading of that input: the caller named three base addresses for a
// kernel with two pointers. The diagnostic names both counts so the caller can
// see which side to fix. (A *shorter* list is legal — see
// materialize-base-addresses-two.mlir @short_list.)

#set1d = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>
// expected-error @below {{MaterializeBaseAddresses: 3 base addresses supplied but the function has only 2 index argument(s) to materialize them into}}
func.func @too_many_addresses(%a: index, %b: index) {
  %va = ktdp.construct_memory_view %a, sizes: [4], strides: [1] {
      coordinate_set = #set1d, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  %vb = ktdp.construct_memory_view %b, sizes: [4], strides: [1] {
      coordinate_set = #set1d, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  return
}

// -----

// The interleaved signature counts `index` arguments, not all arguments, on the
// error path too: `(%a: index, %n: i32, %b: index)` has two base-address
// arguments, so three addresses is still a mismatch and the reported count is 2
// rather than 3.

#set1db = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>
// expected-error @below {{MaterializeBaseAddresses: 3 base addresses supplied but the function has only 2 index argument(s) to materialize them into}}
func.func @too_many_interleaved(%a: index, %n: i32, %b: index) {
  %va = ktdp.construct_memory_view %a, sizes: [4], strides: [1] {
      coordinate_set = #set1db, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  %vb = ktdp.construct_memory_view %b, sizes: [4], strides: [1] {
      coordinate_set = #set1db, memory_space = #ktdp.memory_space<global>
  } : memref<4xf16>
  %w = arith.addi %n, %n : i32
  return
}
