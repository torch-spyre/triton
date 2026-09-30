// RUN: spyre-triton-opt %s --lower-compute-ops | FileCheck %s

// Pins an unsupported state, not a lowering: --lower-compute-ops has no pattern
// for tt.make_range, so the op survives the pass untouched.
//
// tt.make_range produces a 1-D tensor [start, start+1, ..., end-1]. In the raw-
// pointer Triton idiom it usually feeds tt.addptr to build a vector of element
// pointers (`ptr + tl.arange(0, BLOCK)`), and in that role it is eliminated
// upstream when descriptor lowering rewrites the whole address chain. But
// tl.arange can also appear in pure tensor arithmetic with no descriptor
// consumer -- a per-lane index vector compared against a runtime bound, say
// (`bh_idx = bh_offset + tl.arange(0, BLOCK_BH); bh_idx < BH`).
//
// In that arithmetic role tt.make_range survives every existing pass and reaches
// the final KTIR with no replacement: neither LowerComputeOps nor
// LowerDescriptorMemory has a pattern for it. The natural lowering would be
// arith.constant dense<[0, 1, ..., end-1]>, since the values are known at compile
// time for fixed start/end, but that pattern does not exist today. The wider rule
// this violates -- no tt.* ops or types surviving end-to-end lowering -- is
// pinned by test_ktir_examples.py::test_no_raw_ptr_ops / ::test_no_tt_ptr_type.
//
// Until a lowering exists, kernels must avoid tl.arange unless the result feeds a
// descriptor index. kernels/decode_softmax_reducev/spyre_rewrite.py works around
// this by pushing per-lane masking decisions into the wrapper rather than
// computing them in-kernel from a tl.arange-derived index vector.
//
// When the lowering is added, replace the surviving-op check below with a guard
// against tt.make_range and rename this file to say the op is lowered away.
// -split-input-file is deliberately absent: one case, and the claim is about
// what the whole module looks like afterwards.

// -----
// The pass succeeds -- no diagnostic, exit 0 -- and tt.make_range is still there
// in the output. The tt.splat alongside it *is* lowered, to tensor.empty plus
// linalg.fill, which is what shows the pass ran at all rather than bailing out on
// the unrecognized op and leaving the module wholly untouched.
//
// Triton source pattern:
//
//   # NOT supported: tl.arange in pure tensor arithmetic, no descriptor consumer
//   bh_idx    = bh_offset + tl.arange(0, BLOCK_BH)   # tt.make_range survives
//   bh_active = bh_idx < BH                          # ... into arith.cmpi

// CHECK-LABEL:   tt.func @make_range_in_arithmetic_survives(
// CHECK-SAME:  %[[VAL_0:.*]]: i32) -> tensor<8xi32> {
// CHECK:           %[[VAL_1:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8xi32>
// CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_0]] : i32) outs(%[[VAL_2]] : tensor<8xi32>) -> tensor<8xi32>
// CHECK-NOT:       tt.splat
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_1]], %[[VAL_3]] : tensor<8xi32>
// CHECK:           tt.return %[[VAL_4]] : tensor<8xi32>
// CHECK:         }
tt.func @make_range_in_arithmetic_survives(%scalar: i32) -> tensor<8xi32> {
  %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %s = tt.splat %scalar : i32 -> tensor<8xi32>
  %0 = arith.addi %r, %s : tensor<8xi32>
  tt.return %0 : tensor<8xi32>
}
