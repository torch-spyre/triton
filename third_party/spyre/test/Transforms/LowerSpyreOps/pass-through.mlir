// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file | FileCheck %s

// AN OP WITH NO DEVICE FORM FLOWS THROUGH, unchanged and unreported.
//
// The pass is a greedy rewrite: selection claims what it can and leaves the rest
// alone. Whether an f64 divide is compilable is the backend's judgement, so the
// pass does not duplicate it.
//
// So these cases are not "invalid" inputs. They are inputs this pass has nothing
// to say about, and the test is that it says nothing. What the pass does report,
// an `i1` left in a body, is in invalid.mlir and not-yet-lowered.mlir.

// CHECK-LABEL:   tt.func @sqrt_f64(
// CHECK-NOT:       spyreop
// CHECK:           math.sqrt
tt.func @sqrt_f64(%s: f64) -> f64 {
  %0 = math.sqrt %s : f64
  tt.return %0 : f64
}

// -----

// CHECK-LABEL:   tt.func @exp_f64(
// CHECK-NOT:       spyreop
// CHECK:           math.exp
tt.func @exp_f64(%s: f64) -> f64 {
  %0 = math.exp %s : f64
  tt.return %0 : f64
}

// -----

// CHECK-LABEL:   tt.func @rsqrt_f64(
// CHECK-NOT:       spyreop
// CHECK:           math.rsqrt
tt.func @rsqrt_f64(%s: f64) -> f64 {
  %0 = math.rsqrt %s : f64
  tt.return %0 : f64
}

// -----

// CHECK-LABEL:   tt.func @divf_f64(
// CHECK-NOT:       spyreop
// CHECK:           arith.divf
tt.func @divf_f64(%a: f64, %b: f64) -> f64 {
  %0 = arith.divf %a, %b : f64
  tt.return %0 : f64
}

// -----

// bf16 is the other float width with no intrinsic, and it is worth its own case
// because it is a width a kernel author might plausibly reach for, where f64 is
// not.
// CHECK-LABEL:   tt.func @sqrt_bf16(
// CHECK-NOT:       spyreop
// CHECK:           math.sqrt
tt.func @sqrt_bf16(%s: bf16) -> bf16 {
  %0 = math.sqrt %s : bf16
  tt.return %0 : bf16
}

// -----

// An integer width with no intrinsic, inside a body, is the same story on the
// integer side: i16 add has no spyreop form and is left alone.
// CHECK-LABEL:   func.func @addi_i16_in_body(
// CHECK-NOT:       spyreop
// CHECK:           arith.addi
func.func @addi_i16_in_body(%t: tensor<4xi16>) -> tensor<4xi16> {
  %init = tensor.empty() : tensor<4xi16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%t : tensor<4xi16>) outs(%init : tensor<4xi16>) {
  ^bb0(%in: i16, %out: i16):
    %1 = arith.addi %in, %in : i16
    linalg.yield %1 : i16
  } -> tensor<4xi16>
  return %0 : tensor<4xi16>
}

// -----

// The scope is the generic's own body: an arith.addi nested deeper, in an
// scf.if inside it, is left as it is. No pass puts one there today; this pins
// the stated scope so widening it is deliberate.
// CHECK-LABEL:   tt.func @addi_nested_in_scf_if(
// CHECK:           scf.if
// CHECK-NOT:         spyreop
// CHECK:             arith.addi
tt.func @addi_nested_in_scf_if(%t: tensor<4xi32>, %flag: i1) -> tensor<4xi32> {
  %init = tensor.empty() : tensor<4xi32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%t : tensor<4xi32>) outs(%init : tensor<4xi32>) {
  ^bb0(%in: i32, %out: i32):
    %r = scf.if %flag -> i32 {
      %1 = arith.addi %in, %in : i32
      scf.yield %1 : i32
    } else {
      scf.yield %in : i32
    }
    linalg.yield %r : i32
  } -> tensor<4xi32>
  tt.return %0 : tensor<4xi32>
}

// -----

// An `i1` made by an op no rule is anchored on is not refused: only the rules
// anchored on an op say it is illegal after selection, and the compare rules
// are the ones that do. The integer compare and its select, in one body, stay
// as they are for the backend to judge. FuseGenerics still reports such an `i1`
// between two generics, whatever op made it.
// CHECK-LABEL:   func.func @cmpi_select_in_one_body(
// CHECK-NOT:       spyreop
// CHECK:           arith.cmpi
// CHECK:           arith.select
func.func @cmpi_select_in_one_body(%x: tensor<4xi32>, %y: tensor<4xi32>, %p: tensor<4xi32>, %q: tensor<4xi32>) -> tensor<4xi32> {
  %init = tensor.empty() : tensor<4xi32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y, %p, %q : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
      outs(%init : tensor<4xi32>) {
  ^bb0(%a: i32, %b: i32, %t: i32, %f: i32, %out: i32):
    %c = arith.cmpi slt, %a, %b : i32
    %s = arith.select %c, %t, %f : i32
    linalg.yield %s : i32
  } -> tensor<4xi32>
  return %0 : tensor<4xi32>
}
