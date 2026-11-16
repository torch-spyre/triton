// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics

// The tts.spyre_op OP's verifier. No pass runs here.
//
// Its whole job is holding a call to the REAL `spyreop.<name>` op it names,
// not to a parallel table restating that op's shape -- see
// `verifySpyreIntrinsicCall` (Dialect.h/Ops.cpp) for the mechanism. So most
// cases below exist to pin a specific real-op constraint actually firing
// through this path, not to test a rule this op states itself: `gelu`'s
// element-type restriction, `softplus`'s confined `beta`, and so on all come
// from `SpyreOp.td`, read live through `RegisteredOperationName::lookup` and
// `InferTypeOpInterface`, then `mlir::verify()`.
//
// A case that reaches the real op's own verifier reports ONE diagnostic, not
// two: `verifySpyreIntrinsicCall` captures the real op's own message through
// a `ScopedDiagnosticHandler` and folds it into this op's own summary
// (`tts.spyre_op: 'spyreop.<name>' rejects this call: <captured message>`)
// before the handler is uninstalled, so only the combined text ever reaches
// the real diagnostic engine. Each case below pins that combined text.
//
// `operands` are TENSORS and every `spyreop.*` op in scope is SCALAR, so
// what gets checked against the real op is each operand's ELEMENT type, not
// the tensor type -- see @wrong_element_type for the one case that
// distinguishes this from "checks the tensor type directly".
//
// `axis` is checked against the INTRINSIC, not against the real op's own
// attributes: no `spyreop.*` op in scope declares one (confirmed against
// SpyreOp.td), so it never reaches the real op's `extra_attrs` check or its
// `mlir::verify()` at all. See @reduction_shaped_needs_axis and
// @elementwise_refuses_axis.

// An unknown name. The lookup fails before anything else runs, so there is
// exactly one diagnostic -- nothing downstream of the lookup to add a second.
tt.func @unknown_name(%x: tensor<128xf16>) -> tensor<128xf16> {
  // expected-error @+1 {{'gleu' does not name a registered spyreop intrinsic}}
  %y = tts.spyre_op {name = "gleu"} (%x) : (tensor<128xf16>) -> tensor<128xf16>
  tt.return %y : tensor<128xf16>
}

// -----
// `topk` is staged (see spyreop-frontend-api.md) but not yet a real
// `spyreop.*` op -- SpyreOp.td does not define one -- so it fails the same
// lookup an outright typo does, with the same one diagnostic.
tt.func @topk_not_yet_a_real_op(%x: tensor<128xf16>) -> tensor<128xf16> {
  // expected-error @+1 {{'topk' does not name a registered spyreop intrinsic}}
  %y = tts.spyre_op {name = "topk"} (%x) : (tensor<128xf16>) -> tensor<128xf16>
  tt.return %y : tensor<128xf16>
}

// -----
// `eps` is not one of `spyreop.layernormscale`'s own attributes (it takes
// none at all) -- the exact shape the design doc once wrote by mistake for
// this op, caught here against `getAttributeNames()` before the real op is
// even built, so there is one diagnostic, not two.
tt.func @unknown_attribute(%mean: tensor<4xf16>, %msq: tensor<4xf16>) -> tensor<4xf16> {
  // expected-error @+1 {{'"eps"' is not an attribute of spyreop.layernormscale}}
  %y = tts.spyre_op {name = "layernormscale", extra_attrs = {eps = 1.0 : f32}} (%mean, %msq)
      : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xf16>
  tt.return %y : tensor<4xf16>
}

// -----
// `gelu` is F16/DF16 only (SpyreOp.td); this checks the ELEMENT type, not
// the tensor type -- `tensor<128xf32>` is a perfectly good ranked tensor,
// and the rejection is specifically about what is INSIDE it.
tt.func @wrong_element_type(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{tts.spyre_op: 'spyreop.gelu' rejects this call: 'spyreop.gelu' op operand #0 must be 16-bit float or IBM df16 float, but got 'f32'}}
  %y = tts.spyre_op {name = "gelu"} (%x) : (tensor<128xf32>) -> tensor<128xf32>
  tt.return %y : tensor<128xf32>
}

// -----
// `softplus` requires `beta` -- declaring `threshold` alone is an arity
// mismatch on the real op's attributes, reported by its own verifier.
tt.func @softplus_missing_beta(%x: tensor<128xf16>) -> tensor<128xf16> {
  // expected-error @+1 {{tts.spyre_op: 'spyreop.softplus' rejects this call: 'spyreop.softplus' op requires attribute 'beta'}}
  %y = tts.spyre_op {name = "softplus", extra_attrs = {threshold = 20.0 : f32}} (%x)
      : (tensor<128xf16>) -> tensor<128xf16>
  tt.return %y : tensor<128xf16>
}

// -----
// `softplus`'s `beta` is `ConfinedAttr<F32Attr, [FloatNonZero]>` -- the
// predicate only a real `mlir::verify()` call checks, not a plain
// attribute-name/type comparison. This is the case a lighter-weight
// verifier (comparing names and types without building the real op) would
// miss.
tt.func @softplus_beta_zero(%x: tensor<128xf16>) -> tensor<128xf16> {
  // expected-error @+1 {{tts.spyre_op: 'spyreop.softplus' rejects this call: 'spyreop.softplus' op attribute 'beta' failed to satisfy constraint: 32-bit float attribute which is non-zero}}
  %y = tts.spyre_op {name = "softplus", extra_attrs = {beta = 0.0 : f32, threshold = 20.0 : f32}} (%x)
      : (tensor<128xf16>) -> tensor<128xf16>
  tt.return %y : tensor<128xf16>
}

// -----
// `exx2` is reduction-shaped and needs `axis` to know which tensor
// dimension its real (scalar) op is being asked about -- without it there
// is no result shape to lift the real op's inferred scalar type onto.
tt.func @reduction_shaped_needs_axis(%x: tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>) {
  // expected-error @+1 {{'spyreop.exx2' is reduction-shaped and needs an 'axis'}}
  %mean, %msq = tts.spyre_op {name = "exx2"} (%x) : (tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>)
  tt.return %mean, %msq : tensor<4xf16>, tensor<4xf16>
}

// -----
// `gelu` is elementwise; naming `axis` on it is refused rather than
// silently ignored, the same way an attribute the real op does not declare
// is refused -- both are "this field names something this intrinsic does
// not have".
tt.func @elementwise_refuses_axis(%x: tensor<128xf16>) -> tensor<128xf16> {
  // expected-error @+1 {{'spyreop.gelu' is not reduction-shaped; it takes no 'axis'}}
  %y = tts.spyre_op {name = "gelu", axis = 0 : i32} (%x) : (tensor<128xf16>) -> tensor<128xf16>
  tt.return %y : tensor<128xf16>
}

// -----
// `axis` out of range for the first operand's rank.
tt.func @axis_out_of_range(%x: tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>) {
  // expected-error @+1 {{'axis' 2 is out of range for rank 2}}
  %mean, %msq = tts.spyre_op {name = "exx2", axis = 2 : i32} (%x) : (tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>)
  tt.return %mean, %msq : tensor<4xf16>, tensor<4xf16>
}

// -----
// The declared result count disagrees with the real op's -- `gelu` has one
// result, not two.
tt.func @wrong_result_count(%x: tensor<128xf16>) -> (tensor<128xf16>, tensor<128xf16>) {
  // expected-error @+1 {{'spyreop.gelu' has 1 result(s), but this call declares 2}}
  %y, %z = tts.spyre_op {name = "gelu"} (%x) : (tensor<128xf16>) -> (tensor<128xf16>, tensor<128xf16>)
  tt.return %y, %z : tensor<128xf16>, tensor<128xf16>
}

// -----
// The declared result TYPE disagrees with what the real op infers --
// `gelu`'s result is `SameOperandsAndResultType`, so it has to match the
// operand's element type at the operand's own shape, not some other shape.
tt.func @wrong_result_type(%x: tensor<128xf16>) -> tensor<4x8xf16> {
  // expected-error @+1 {{'spyreop.gelu' infers result type 'tensor<128xf16>', but this call declares 'tensor<4x8xf16>'}}
  %y = tts.spyre_op {name = "gelu"} (%x) : (tensor<128xf16>) -> tensor<4x8xf16>
  tt.return %y : tensor<4x8xf16>
}

// The forms that DO verify, so the rejections above are read as rules and
// not as the op being hard to satisfy.
//
// Note there are no rules of dashes anywhere in this file, and no comment
// quotes the split marker either: -split-input-file matches the marker as a
// substring, so both would start a new chunk and the text after them would
// be parsed as IR.

// -----
tt.func @accepted_forms(%x: tensor<128xf16>, %x2: tensor<4x8xf16>) {
  // Elementwise, no attributes.
  %gelu = tts.spyre_op {name = "gelu"} (%x) : (tensor<128xf16>) -> tensor<128xf16>

  // Elementwise with real attributes, under `extra_attrs`.
  %sp = tts.spyre_op {name = "softplus", extra_attrs = {beta = 1.0 : f32, threshold = 20.0 : f32}} (%x)
      : (tensor<128xf16>) -> tensor<128xf16>

  // Reduction-shaped, with `axis`, two results, each with the axis dropped.
  %mean, %msq = tts.spyre_op {name = "exx2", axis = 1 : i32} (%x2)
      : (tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>)

  tt.return
}
