// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file | FileCheck %s

// ktdp.gelu on a scalar f16 operand -> spyreop.gelu. Same shape as
// math.sqrt in basic.mlir: no body scope, no discriminator -- by the time
// this pass runs, ConvertElementwiseToLinalg has already scalarized the
// tensor-level `ktdp.gelu` (traced from `tl.spyre_op("gelu", x)`) into
// exactly this form, inside a linalg.generic body this pass does not need
// to see to select on.
module {
// CHECK-LABEL:   tt.func @gelu_f16(
// CHECK-SAME:  %[[VAL_0:.*]]: f16) -> f16 {
// CHECK:           %[[VAL_1:.*]] = spyreop.gelu %[[VAL_0]] : f16
// CHECK:           tt.return %[[VAL_1]] : f16
// CHECK:         }
tt.func @gelu_f16(%s: f16) -> f16 {
  %0 = ktdp.gelu %s : f16
  tt.return %0 : f16
}
}
