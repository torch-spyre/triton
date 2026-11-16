// RUN: spyre-triton-opt %s | spyre-triton-opt | FileCheck %s

// Parse-print-parse round trip for the tts.spyre_op OP.
//
// Four cases, chosen to cover the three shapes `extra_attrs`/results/`axis`
// give this one op: a plain elementwise request with no attributes
// (`gelu`), one carrying real attributes under the nested `extra_attrs`
// dictionary (`softplus`'s `beta`/`threshold`), one with more than one
// operand (`layernormnorm`'s five), and the one reduction-shaped intrinsic
// in scope (`exx2`'s `axis` and two results).
//
// `extra_attrs` prints as a NESTED dictionary -- `{extra_attrs = {beta =
// ..., threshold = ...}, name = "..."}` -- rather than inlining `beta`/
// `threshold` directly into the op's own attr-dict. That is `attr-dict`'s
// ordinary behaviour for an `OptionalAttr<DictionaryAttr>` field, not
// something this op's assembly format hand-specifies; asserting anything
// about attribute PRINT ORDER within either dictionary would be asserting
// MLIR's sorting, not this op's own contract, so CHECK lines below pin only
// the one order attr-dict is observed to use today, and existing `tts.pin`
// roundtrip tests take the same stance.

// CHECK-LABEL: tt.func @elementwise_no_attrs(
// CHECK: tts.spyre_op {name = "gelu"}(%{{.*}}) : (tensor<128xf16>) -> tensor<128xf16>
tt.func @elementwise_no_attrs(%x: tensor<128xf16>) -> tensor<128xf16> {
  %y = tts.spyre_op {name = "gelu"} (%x) : (tensor<128xf16>) -> tensor<128xf16>
  tt.return %y : tensor<128xf16>
}

// CHECK-LABEL: tt.func @elementwise_with_attrs(
// CHECK: tts.spyre_op {extra_attrs = {beta = 1.000000e+00 : f32, threshold = 2.000000e+01 : f32}, name = "softplus"}(%{{.*}}) : (tensor<128xf16>) -> tensor<128xf16>
tt.func @elementwise_with_attrs(%x: tensor<128xf16>) -> tensor<128xf16> {
  %y = tts.spyre_op {name = "softplus", extra_attrs = {beta = 1.0 : f32, threshold = 20.0 : f32}} (%x) : (tensor<128xf16>) -> tensor<128xf16>
  tt.return %y : tensor<128xf16>
}

// CHECK-LABEL: tt.func @five_operands(
// CHECK: tts.spyre_op {name = "layernormnorm"}(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (tensor<4x8xf16>, tensor<4x8xf16>, tensor<4x8xf16>, tensor<4x8xf16>, tensor<4x8xf16>) -> tensor<4x8xf16>
tt.func @five_operands(%x: tensor<4x8xf16>, %mean: tensor<4x8xf16>,
                       %scale: tensor<4x8xf16>, %weight: tensor<4x8xf16>,
                       %bias: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %y = tts.spyre_op {name = "layernormnorm"} (%x, %mean, %scale, %weight, %bias)
      : (tensor<4x8xf16>, tensor<4x8xf16>, tensor<4x8xf16>, tensor<4x8xf16>, tensor<4x8xf16>) -> tensor<4x8xf16>
  tt.return %y : tensor<4x8xf16>
}

// CHECK-LABEL: tt.func @reduction_shaped(
// CHECK: tts.spyre_op {axis = 1 : i32, name = "exx2"}(%{{.*}}) : (tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>)
tt.func @reduction_shaped(%x: tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>) {
  %mean, %msq = tts.spyre_op {name = "exx2", axis = 1 : i32} (%x)
      : (tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>)
  tt.return %mean, %msq : tensor<4xf16>, tensor<4xf16>
}
